"""Explicit JSON persistence for immutable agent task snapshots with SQLite CAS."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import unicodedata
import weakref
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Protocol
from uuid import uuid4

from sovereign_api.agent_task_state import (
    AgentTaskState, StageExecutionState, StageOutputKind, StageStatus, TaskStatus,
)
from sovereign_api.errors import SovereignAPIError
from sovereign_api.task_classification import TaskClass
from sovereign_api.task_plan_runner import revalidate_agent_task_state
from sovereign_api.task_planning import StageExecutionKind, TaskPlan, TaskStage, TaskStageType
from sovereign_api.tool_approval import ApprovalRequest
from sovereign_api.tool_contracts import ToolPermission, ToolRiskLevel, ToolSideEffectLevel


SCHEMA_VERSION = 1
DEFAULT_STAGE_LEASE_DURATION = timedelta(minutes=5)
MAX_STAGE_LEASE_DURATION = timedelta(minutes=30)
_LEASE_COLUMN_MIGRATIONS = (
    ("lease_id", "ALTER TABLE agent_task_states ADD COLUMN lease_id TEXT"),
    ("lease_stage_id", "ALTER TABLE agent_task_states ADD COLUMN lease_stage_id TEXT"),
    ("lease_claimed_at", "ALTER TABLE agent_task_states ADD COLUMN lease_claimed_at TEXT"),
    ("lease_expires_at", "ALTER TABLE agent_task_states ADD COLUMN lease_expires_at TEXT"),
    ("lease_proof", "ALTER TABLE agent_task_states ADD COLUMN lease_proof TEXT"),
    ("lease_migration_state", "ALTER TABLE agent_task_states ADD COLUMN "
     "lease_migration_state TEXT NOT NULL DEFAULT 'current'"),
)
_CURRENT_LEASE_STATE = "current"
_LEGACY_UNLEASED_RUNNING = "legacy_unleased_running"
_LEGACY_UNPROVEN_LEASE = "legacy_unproven_lease"


class TaskPersistenceError(SovereignAPIError):
    code = "task_persistence_error"


class InvalidTaskDatabaseConfigurationError(TaskPersistenceError):
    code = "invalid_task_database_configuration"


class InvalidRepositoryClockError(TaskPersistenceError):
    code = "invalid_repository_clock"


class TaskNotFoundError(TaskPersistenceError):
    code = "task_not_found"


class TaskAlreadyExistsError(TaskPersistenceError):
    code = "task_already_exists"


class StaleTaskStateError(TaskPersistenceError):
    code = "stale_task_state"


class CorruptTaskStateError(TaskPersistenceError):
    code = "corrupt_task_state"


class StageClaimError(TaskPersistenceError):
    code = "stage_claim_error"


class StageAlreadyClaimedError(StageClaimError):
    code = "stage_already_claimed"


class StageLeaseExpiredError(StageClaimError):
    code = "stage_lease_expired"


class StageLeaseMismatchError(StageClaimError):
    code = "stage_lease_mismatch"


class InvalidLeaseConfigurationError(StageClaimError):
    code = "invalid_stage_lease_configuration"


@dataclass(frozen=True, slots=True)
class StageExecutionLease:
    lease_id: str
    task_id: str
    stage_id: str
    claimed_version: int
    claimed_at: datetime
    lease_expires_at: datetime

    def __post_init__(self) -> None:
        if (
            type(self.lease_id) is not str or len(self.lease_id) != 32
            or any(char not in "0123456789abcdef" for char in self.lease_id)
            or any(type(value) is not str or not value.strip()
                   for value in (self.task_id, self.stage_id))
            or type(self.claimed_version) is not int or self.claimed_version < 2
            or not _utc(self.claimed_at) or not _utc(self.lease_expires_at)
            or self.lease_expires_at <= self.claimed_at
            or self.lease_expires_at - self.claimed_at > MAX_STAGE_LEASE_DURATION
        ):
            raise CorruptTaskStateError("Stored stage lease is invalid")


class ValidatedStageExecutionLease:
    """Opaque, non-copyable authority backed by one claim-specific secret."""

    __slots__ = ("lease_id", "task_id", "stage_id", "claimed_version", "__weakref__")

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise StageLeaseMismatchError("Stage execution lease is invalid")

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Stage execution lease is immutable")

    def __repr__(self) -> str:
        return (
            "ValidatedStageExecutionLease("
            f"lease_id={self.lease_id!r}, task_id={self.task_id!r}, "
            f"stage_id={self.stage_id!r}, claimed_version={self.claimed_version!r})"
        )

    def __copy__(self) -> object:
        raise TypeError("Stage execution lease cannot be copied")

    def __deepcopy__(self, memo: object) -> object:
        raise TypeError("Stage execution lease cannot be copied")

    def __reduce__(self) -> object:
        raise TypeError("Stage execution lease cannot be serialized")

    def __reduce_ex__(self, protocol: int) -> object:
        raise TypeError("Stage execution lease cannot be serialized")


_LEASE_SECRETS: dict[int, tuple[weakref.ReferenceType[ValidatedStageExecutionLease], bytes]] = {}


def _lease_authority(
    lease: StageExecutionLease, secret: bytes,
) -> ValidatedStageExecutionLease:
    authority = object.__new__(ValidatedStageExecutionLease)
    for name, value in (
        ("lease_id", lease.lease_id), ("task_id", lease.task_id),
        ("stage_id", lease.stage_id), ("claimed_version", lease.claimed_version),
    ):
        object.__setattr__(authority, name, value)
    identity = id(authority)

    def discard(reference: object) -> None:
        current = _LEASE_SECRETS.get(identity)
        if current is not None and current[0] is reference:
            _LEASE_SECRETS.pop(identity, None)

    _LEASE_SECRETS[identity] = (weakref.ref(authority, discard), secret)
    return authority


def _lease_authority_secret(value: object) -> bytes | None:
    if type(value) is not ValidatedStageExecutionLease:
        return None
    registered = _LEASE_SECRETS.get(id(value))
    if registered is None or registered[0]() is not value:
        return None
    return registered[1]


def _claim_binding(lease: StageExecutionLease) -> bytes:
    return json.dumps(
        {
            "claimed_version": lease.claimed_version,
            "lease_id": lease.lease_id,
            "stage_id": lease.stage_id,
            "task_id": lease.task_id,
        },
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")


def _claim_proof(lease: StageExecutionLease, secret: bytes) -> str:
    return hmac.new(secret, _claim_binding(lease), hashlib.sha256).hexdigest()


def _valid_lease_proof(value: object) -> bool:
    return (
        type(value) is str and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _utc(value: datetime) -> bool:
    return (
        type(value) is datetime and value.tzinfo is not None
        and value.utcoffset() == timedelta(0)
    )


def _timestamp(value: object) -> datetime:
    if type(value) is not str:
        raise ValueError
    result = datetime.fromisoformat(value)
    if not _utc(result):
        raise ValueError
    return result


def _validate_lease_duration(lease_duration: timedelta) -> None:
    if (
        type(lease_duration) is not timedelta or lease_duration <= timedelta(0)
        or lease_duration > MAX_STAGE_LEASE_DURATION
    ):
        raise InvalidLeaseConfigurationError("Stage lease configuration is invalid")


def _lease_from_row(
    task_id: str, version: int, values: tuple[object, object, object, object],
) -> StageExecutionLease | None:
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise CorruptTaskStateError("Stored stage lease is invalid")
    return StageExecutionLease(
        lease_id=values[0], task_id=task_id, stage_id=values[1],
        claimed_version=version, claimed_at=_timestamp(values[2]),
        lease_expires_at=_timestamp(values[3]),
    )


def _is_valid_running_state(state: AgentTaskState) -> bool:
    running = tuple(
        stage for stage in state.stage_states if stage.status is StageStatus.RUNNING
    )
    return (
        state.task_status is TaskStatus.RUNNING
        and state.current_stage_id is not None
        and len(running) == 1
        and running[0].stage_id == state.current_stage_id
    )


def _object(value: object, fields: set[str]) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError
    return value


def _items(value: object) -> list[object]:
    if type(value) is not list:
        raise ValueError
    return value


def _strings(value: object) -> tuple[str, ...]:
    items = _items(value)
    if any(type(item) is not str for item in items):
        raise ValueError
    return tuple(items)


def _enum(enum_type: type, value: object):
    if type(value) is not str:
        raise ValueError
    return enum_type(value)


def _plain_json(value: object) -> object:
    if type(value) in (dict, MappingProxyType):
        return {key: _plain_json(item) for key, item in value.items()}
    if type(value) is tuple:
        return [_plain_json(item) for item in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise ValueError


def _stage_payload(stage: TaskStage) -> dict[str, object]:
    return {
        "stage_id": stage.stage_id,
        "stage_type": stage.stage_type.value,
        "required_capabilities": list(stage.required_capabilities),
        "execution_kind": stage.execution_kind.value,
        "tool_id": stage.tool_id,
        "tool_arguments": (
            _plain_json(stage.tool_arguments) if stage.tool_arguments is not None else None
        ),
    }


def _stage_from_payload(value: object) -> TaskStage:
    data = _object(value, {
        "stage_id", "stage_type", "required_capabilities", "execution_kind",
        "tool_id", "tool_arguments",
    })
    return TaskStage(
        stage_id=data["stage_id"],
        stage_type=_enum(TaskStageType, data["stage_type"]),
        required_capabilities=_strings(data["required_capabilities"]),
        execution_kind=_enum(StageExecutionKind, data["execution_kind"]),
        tool_id=data["tool_id"],
        tool_arguments=data["tool_arguments"],
    )


def _execution_payload(stage: StageExecutionState) -> dict[str, object]:
    return {
        "stage_id": stage.stage_id,
        "stage_type": stage.stage_type.value,
        "required_capabilities": list(stage.required_capabilities),
        "status": stage.status.value,
        "selected_model_id": stage.selected_model_id,
        "selected_tool_id": stage.selected_tool_id,
        "output_reference": stage.output_reference,
        "output_kind": stage.output_kind.value if stage.output_kind is not None else None,
        "error_code": stage.error_code,
        "safe_message": stage.safe_message,
        "execution_kind": stage.execution_kind.value,
    }


def _execution_from_payload(value: object) -> StageExecutionState:
    data = _object(value, {
        "stage_id", "stage_type", "required_capabilities", "status",
        "selected_model_id", "selected_tool_id", "output_reference", "output_kind",
        "error_code", "safe_message", "execution_kind",
    })
    return StageExecutionState(
        stage_id=data["stage_id"],
        stage_type=_enum(TaskStageType, data["stage_type"]),
        required_capabilities=_strings(data["required_capabilities"]),
        status=_enum(StageStatus, data["status"]),
        selected_model_id=data["selected_model_id"],
        selected_tool_id=data["selected_tool_id"],
        output_reference=data["output_reference"],
        output_kind=(
            _enum(StageOutputKind, data["output_kind"])
            if data["output_kind"] is not None else None
        ),
        error_code=data["error_code"],
        safe_message=data["safe_message"],
        execution_kind=_enum(StageExecutionKind, data["execution_kind"]),
    )


def _approval_payload(approval: ApprovalRequest | None) -> dict[str, object] | None:
    if approval is None:
        return None
    return {
        "approval_id": approval.approval_id,
        "request_id": approval.request_id,
        "task_id": approval.task_id,
        "stage_id": approval.stage_id,
        "tool_id": approval.tool_id,
        "operation": approval.operation,
        "requested_permissions": [item.identifier for item in approval.requested_permissions],
        "risk_level": approval.risk_level.value,
        "side_effect_level": approval.side_effect_level.value,
        "created_at": approval.created_at.isoformat(),
        "safe_summary": approval.safe_summary,
        "request_fingerprint": approval.request_fingerprint,
    }


def _approval_from_payload(value: object) -> ApprovalRequest | None:
    if value is None:
        return None
    data = _object(value, {
        "approval_id", "request_id", "task_id", "stage_id", "tool_id", "operation",
        "requested_permissions", "risk_level", "side_effect_level", "created_at",
        "safe_summary", "request_fingerprint",
    })
    return ApprovalRequest(
        approval_id=data["approval_id"],
        request_id=data["request_id"],
        task_id=data["task_id"],
        stage_id=data["stage_id"],
        tool_id=data["tool_id"],
        operation=data["operation"],
        requested_permissions=tuple(
            ToolPermission(item) for item in _strings(data["requested_permissions"])
        ),
        risk_level=_enum(ToolRiskLevel, data["risk_level"]),
        side_effect_level=_enum(ToolSideEffectLevel, data["side_effect_level"]),
        created_at=_timestamp(data["created_at"]),
        safe_summary=data["safe_summary"],
        request_fingerprint=data["request_fingerprint"],
    )


def serialize_task_state(state: AgentTaskState) -> str:
    """Canonical schema-v1 JSON; reject tampered nested domain values."""
    try:
        checked = revalidate_agent_task_state(state)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "state": {
                "task_id": checked.task_id,
                "original_prompt": checked.original_prompt,
                "task_class": checked.task_class.value,
                "required_capabilities": list(checked.required_capabilities),
                "plan": {
                    "task_class": checked.plan.task_class.value,
                    "stages": [_stage_payload(item) for item in checked.plan.stages],
                },
                "task_status": checked.task_status.value,
                "current_stage_id": checked.current_stage_id,
                "stage_states": [_execution_payload(item) for item in checked.stage_states],
                "created_at": checked.created_at.isoformat(),
                "updated_at": checked.updated_at.isoformat(),
                "approval_request": _approval_payload(checked.approval_request),
            },
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False)
    except Exception:
        raise TaskPersistenceError("Task state is invalid") from None


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def deserialize_task_state(payload: str) -> AgentTaskState:
    """Parse only known fields and reconstruct every nested domain constructor."""
    try:
        if type(payload) is not str:
            raise ValueError
        envelope = _object(json.loads(
            payload, object_pairs_hook=_unique_object,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError()),
        ), {"schema_version", "state"})
        if type(envelope["schema_version"]) is not int or envelope["schema_version"] != SCHEMA_VERSION:
            raise ValueError
        data = _object(envelope["state"], {
            "task_id", "original_prompt", "task_class", "required_capabilities",
            "plan", "task_status", "current_stage_id", "stage_states",
            "created_at", "updated_at", "approval_request",
        })
        plan_data = _object(data["plan"], {"task_class", "stages"})
        plan = TaskPlan(
            task_class=_enum(TaskClass, plan_data["task_class"]),
            stages=tuple(_stage_from_payload(item) for item in _items(plan_data["stages"])),
        )
        state = AgentTaskState(
            task_id=data["task_id"],
            original_prompt=data["original_prompt"],
            task_class=_enum(TaskClass, data["task_class"]),
            required_capabilities=_strings(data["required_capabilities"]),
            plan=plan,
            task_status=_enum(TaskStatus, data["task_status"]),
            current_stage_id=data["current_stage_id"],
            stage_states=tuple(
                _execution_from_payload(item) for item in _items(data["stage_states"])
            ),
            created_at=_timestamp(data["created_at"]),
            updated_at=_timestamp(data["updated_at"]),
            approval_request=_approval_from_payload(data["approval_request"]),
        )
        return revalidate_agent_task_state(state)
    except Exception:
        raise CorruptTaskStateError("Stored task state is invalid") from None


@dataclass(frozen=True, slots=True)
class PersistedTaskState:
    state: AgentTaskState
    version: int
    persisted_at: datetime
    lease: StageExecutionLease | None = None
    legacy_unleased_running: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.state) is not AgentTaskState
            or type(self.version) is not int or self.version < 1
            or not _utc(self.persisted_at)
            or (self.lease is not None and type(self.lease) is not StageExecutionLease)
            or type(self.legacy_unleased_running) is not bool
            or (self.lease is not None and self.legacy_unleased_running)
            or (self.lease is not None and (
                self.lease.task_id != self.state.task_id
                or self.lease.claimed_version != self.version
                or self.state.task_status is not TaskStatus.RUNNING
                or self.state.current_stage_id != self.lease.stage_id
                or sum(stage.status is StageStatus.RUNNING for stage in self.state.stage_states) != 1
            ))
            or (self.legacy_unleased_running and not _is_valid_running_state(self.state))
            or (
                self.lease is None and not self.legacy_unleased_running
                and any(stage.status is StageStatus.RUNNING for stage in self.state.stage_states)
            )
        ):
            raise TaskPersistenceError("Persisted task state is invalid")


@dataclass(frozen=True, slots=True)
class ClaimedTaskState:
    persisted: PersistedTaskState
    lease: StageExecutionLease
    execution_authority: ValidatedStageExecutionLease

    def __post_init__(self) -> None:
        if (
            type(self.persisted) is not PersistedTaskState
            or type(self.lease) is not StageExecutionLease
            or self.persisted.lease != self.lease
            or _lease_authority_secret(self.execution_authority) is None
            or (self.execution_authority.lease_id, self.execution_authority.task_id,
                self.execution_authority.stage_id, self.execution_authority.claimed_version)
            != (self.lease.lease_id, self.lease.task_id, self.lease.stage_id,
                self.lease.claimed_version)
        ):
            raise StageLeaseMismatchError("Stage execution lease is invalid")


class TaskStateRepository(Protocol):
    def create(self, state: AgentTaskState) -> PersistedTaskState: ...

    def get(self, task_id: str) -> PersistedTaskState: ...

    def replace(
        self, task_id: str, state: AgentTaskState, *, expected_version: int,
    ) -> PersistedTaskState: ...

    def claim_next_stage(
        self, task_id: str, stage_id: str, *, expected_version: int,
        lease_duration: timedelta = DEFAULT_STAGE_LEASE_DURATION,
    ) -> ClaimedTaskState: ...

    def claim_legacy_running_stage(
        self, task_id: str, stage_id: str, *, expected_version: int,
        lease_duration: timedelta = DEFAULT_STAGE_LEASE_DURATION,
    ) -> ClaimedTaskState: ...

    def complete_claim(
        self, task_id: str, stage_id: str, lease_id: str,
        execution_authority: ValidatedStageExecutionLease,
        new_state: AgentTaskState, *, expected_version: int,
    ) -> PersistedTaskState: ...

    def close(self) -> None: ...


class SQLiteTaskStateRepository:
    """One thread-affine connection per instance; SQLite serializes CAS updates."""

    def __init__(self, db_path: str | Path | None, *, clock=None) -> None:
        if type(db_path) is not str and not isinstance(db_path, Path):
            raise InvalidTaskDatabaseConfigurationError("Task database path is invalid")
        try:
            raw = os.fspath(db_path)
            if (
                type(raw) is not str or not raw or
                any(unicodedata.category(char) == "Cc" for char in raw)
            ):
                raise ValueError
            path = Path(raw)
            if not path.is_absolute() or not path.parent.is_dir() or path.is_dir():
                raise ValueError
            connection = sqlite3.connect(str(path), timeout=5.0)
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA synchronous = FULL")
            connection.execute("""
                CREATE TABLE IF NOT EXISTS agent_task_states (
                    task_id TEXT PRIMARY KEY,
                    state_json TEXT NOT NULL,
                    version INTEGER NOT NULL CHECK (version >= 1),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    persisted_at TEXT NOT NULL,
                    lease_id TEXT,
                    lease_stage_id TEXT,
                    lease_claimed_at TEXT,
                    lease_expires_at TEXT,
                    lease_proof TEXT,
                    lease_migration_state TEXT NOT NULL DEFAULT 'current'
                )
            """)
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(agent_task_states)")
            }
            migration_needed = "lease_migration_state" not in columns
            for column, statement in _LEASE_COLUMN_MIGRATIONS[:-1]:
                if column not in columns:
                    connection.execute(statement)
                    columns.add(column)
            legacy_unleased: list[str] = []
            legacy_unproven: list[str] = []
            if migration_needed:
                rows = connection.execute(
                    "SELECT task_id, state_json, lease_id, lease_stage_id, "
                    "lease_claimed_at, lease_expires_at, lease_proof "
                    "FROM agent_task_states"
                ).fetchall()
                for row in rows:
                    try:
                        state = deserialize_task_state(row[1])
                        if state.task_id != row[0] or not _is_valid_running_state(state):
                            continue
                        lease_values = row[2:6]
                        if all(value is None for value in lease_values) and row[6] is None:
                            legacy_unleased.append(row[0])
                        elif all(value is not None for value in lease_values) and row[6] is None:
                            legacy_unproven.append(row[0])
                    except TaskPersistenceError:
                        continue
                connection.execute(_LEASE_COLUMN_MIGRATIONS[-1][1])
                connection.executemany(
                    "UPDATE agent_task_states SET lease_migration_state = ? "
                    "WHERE task_id = ?",
                    ((_LEGACY_UNLEASED_RUNNING, task_id) for task_id in legacy_unleased),
                )
                connection.executemany(
                    "UPDATE agent_task_states SET lease_migration_state = ? "
                    "WHERE task_id = ?",
                    ((_LEGACY_UNPROVEN_LEASE, task_id) for task_id in legacy_unproven),
                )
            connection.commit()
        except (ValueError, OSError, sqlite3.Error):
            if "connection" in locals():
                connection.close()
            raise InvalidTaskDatabaseConfigurationError("Task database path is invalid") from None
        self._connection: sqlite3.Connection | None = connection
        self._clock = clock if clock is not None else lambda: datetime.now(UTC)

    @classmethod
    def from_environment(cls) -> SQLiteTaskStateRepository:
        return cls(os.environ.get("SOVEREIGN_TASK_DB_PATH"))

    def __enter__(self) -> SQLiteTaskStateRepository:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def _db(self) -> sqlite3.Connection:
        if self._connection is None:
            raise TaskPersistenceError("Task repository is closed")
        return self._connection

    def _now(self) -> datetime:
        try:
            value = self._clock()
            if not _utc(value):
                raise ValueError
            return value
        except Exception:
            raise InvalidRepositoryClockError("Task repository clock is invalid") from None

    def create(self, state: AgentTaskState) -> PersistedTaskState:
        payload = serialize_task_state(state)
        if any(stage.status is StageStatus.RUNNING for stage in state.stage_states):
            raise TaskPersistenceError("An executing stage requires a durable lease")
        persisted_at = self._now()
        db = self._db()
        try:
            with db:
                db.execute(
                    "INSERT INTO agent_task_states "
                    "(task_id, state_json, version, created_at, updated_at, persisted_at) "
                    "VALUES (?, ?, 1, ?, ?, ?)",
                    (state.task_id, payload, state.created_at.isoformat(),
                     state.updated_at.isoformat(), persisted_at.isoformat()),
                )
        except sqlite3.IntegrityError:
            raise TaskAlreadyExistsError("Task already exists") from None
        except sqlite3.Error:
            raise TaskPersistenceError("Task state could not be stored") from None
        return PersistedTaskState(deserialize_task_state(payload), 1, persisted_at)

    def get(self, task_id: str) -> PersistedTaskState:
        if type(task_id) is not str or not task_id.strip():
            raise TaskPersistenceError("Task ID is invalid")
        try:
            row = self._db().execute(
                "SELECT state_json, version, created_at, updated_at, persisted_at, "
                "lease_id, lease_stage_id, lease_claimed_at, lease_expires_at, "
                "lease_proof, lease_migration_state "
                "FROM agent_task_states WHERE task_id = ?", (task_id,),
            ).fetchone()
        except sqlite3.Error:
            raise TaskPersistenceError("Task state could not be loaded") from None
        if row is None:
            raise TaskNotFoundError("Task was not found")
        try:
            state = deserialize_task_state(row[0])
            if (
                state.task_id != task_id or type(row[1]) is not int or row[1] < 1
                or _timestamp(row[2]) != state.created_at
                or _timestamp(row[3]) != state.updated_at
            ):
                raise ValueError
            lease = _lease_from_row(task_id, row[1], (row[5], row[6], row[7], row[8]))
            migration_state = row[10]
            if migration_state not in (
                _CURRENT_LEASE_STATE, _LEGACY_UNLEASED_RUNNING,
                _LEGACY_UNPROVEN_LEASE,
            ):
                raise ValueError
            if lease is None:
                if row[9] is not None or migration_state == _LEGACY_UNPROVEN_LEASE:
                    raise ValueError
                legacy = migration_state == _LEGACY_UNLEASED_RUNNING
                if legacy != _is_valid_running_state(state):
                    raise ValueError
            else:
                legacy = False
                if (
                    state.current_stage_id != lease.stage_id
                    or not any(stage.stage_id == lease.stage_id
                               and stage.status is StageStatus.RUNNING
                               for stage in state.stage_states)
                    or migration_state == _LEGACY_UNLEASED_RUNNING
                    or (
                        migration_state == _CURRENT_LEASE_STATE
                        and not _valid_lease_proof(row[9])
                    )
                    or (
                        migration_state == _LEGACY_UNPROVEN_LEASE
                        and row[9] is not None
                    )
                ):
                    raise ValueError
            return PersistedTaskState(
                state, row[1], _timestamp(row[4]), lease, legacy,
            )
        except Exception:
            raise CorruptTaskStateError("Stored task state is invalid") from None

    def replace(
        self, task_id: str, state: AgentTaskState, *, expected_version: int,
    ) -> PersistedTaskState:
        if type(task_id) is not str or type(state) is not AgentTaskState or task_id != state.task_id:
            raise TaskPersistenceError("Replacement task identity is invalid")
        if type(expected_version) is not int or expected_version < 1:
            raise TaskPersistenceError("Expected task version is invalid")
        payload = serialize_task_state(state)
        persisted_at = self._now()
        db = self._db()
        try:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                "SELECT state_json, version, created_at, updated_at, lease_id, "
                "lease_stage_id, lease_claimed_at, lease_expires_at, lease_proof, "
                "lease_migration_state "
                "FROM agent_task_states WHERE task_id = ?", (task_id,),
            ).fetchone()
            if existing is None:
                raise TaskNotFoundError("Task was not found")
            if type(existing[1]) is not int or existing[1] != expected_version:
                raise StaleTaskStateError("Task version is stale")
            previous = deserialize_task_state(existing[0])
            if (
                previous.task_id != task_id
                or _timestamp(existing[2]) != previous.created_at
                or _timestamp(existing[3]) != previous.updated_at
                or previous.created_at != state.created_at
                or previous.original_prompt != state.original_prompt
                or previous.plan != state.plan
                or any(value is not None for value in existing[4:9])
                or existing[9] != _CURRENT_LEASE_STATE
                or any(stage.status is StageStatus.RUNNING for stage in previous.stage_states)
                or any(stage.status is StageStatus.RUNNING for stage in state.stage_states)
            ):
                raise TaskPersistenceError("Replacement task identity is invalid")
            changed = db.execute(
                "UPDATE agent_task_states SET state_json = ?, version = version + 1, "
                "updated_at = ?, persisted_at = ? WHERE task_id = ? AND version = ?",
                (payload, state.updated_at.isoformat(), persisted_at.isoformat(),
                 task_id, expected_version),
            )
            if changed.rowcount != 1:
                raise StaleTaskStateError("Task version is stale")
            db.commit()
        except TaskPersistenceError:
            db.rollback()
            raise
        except (sqlite3.Error, ValueError):
            db.rollback()
            raise TaskPersistenceError("Task state could not be replaced") from None
        return PersistedTaskState(deserialize_task_state(payload), expected_version + 1, persisted_at)

    def claim_next_stage(
        self, task_id: str, stage_id: str, *, expected_version: int,
        lease_duration: timedelta = DEFAULT_STAGE_LEASE_DURATION,
    ) -> ClaimedTaskState:
        if (
            type(task_id) is not str or not task_id.strip()
            or type(stage_id) is not str or not stage_id.strip()
            or type(expected_version) is not int or expected_version < 1
        ):
            raise StageClaimError("Stage claim identity is invalid")
        _validate_lease_duration(lease_duration)
        db = self._db()
        try:
            db.execute("BEGIN IMMEDIATE")
            now = self._now()
            row = db.execute(
                "SELECT state_json, version, created_at, updated_at, "
                "lease_id, lease_stage_id, lease_claimed_at, lease_expires_at, "
                "lease_proof, lease_migration_state "
                "FROM agent_task_states WHERE task_id = ?", (task_id,),
            ).fetchone()
            if row is None:
                raise TaskNotFoundError("Task was not found")
            if type(row[1]) is not int or row[1] != expected_version:
                raise StaleTaskStateError("Task version is stale")
            state = deserialize_task_state(row[0])
            if (
                state.task_id != task_id
                or _timestamp(row[2]) != state.created_at
                or _timestamp(row[3]) != state.updated_at
            ):
                raise CorruptTaskStateError("Stored task state is invalid")
            existing_lease = _lease_from_row(
                task_id, row[1], (row[4], row[5], row[6], row[7]),
            )
            if existing_lease is not None:
                if (
                    state.task_status is not TaskStatus.RUNNING
                    or state.current_stage_id != existing_lease.stage_id
                    or not any(item.stage_id == existing_lease.stage_id
                               and item.status is StageStatus.RUNNING
                               for item in state.stage_states)
                ):
                    raise CorruptTaskStateError("Stored stage lease is invalid")
                if (
                    row[9] == _CURRENT_LEASE_STATE
                    and not _valid_lease_proof(row[8])
                ) or (
                    row[9] == _LEGACY_UNPROVEN_LEASE and row[8] is not None
                ) or row[9] not in (_CURRENT_LEASE_STATE, _LEGACY_UNPROVEN_LEASE):
                    raise CorruptTaskStateError("Stored stage lease is invalid")
                if existing_lease.lease_expires_at > now:
                    raise StageAlreadyClaimedError("Stage already has an active lease")
                if stage_id != existing_lease.stage_id:
                    raise StageClaimError("Requested stage is not executable")
                claimed_state = state
            else:
                if row[9] == _LEGACY_UNLEASED_RUNNING:
                    raise StageClaimError("Legacy running stage requires recovery claim")
                if row[9] != _CURRENT_LEASE_STATE or row[8] is not None:
                    raise CorruptTaskStateError("Stored stage lease is invalid")
                if state.task_status not in (TaskStatus.PENDING, TaskStatus.RUNNING):
                    raise StageClaimError("Task cannot claim a stage")
                if state.current_stage_id is not None or any(
                    item.status is StageStatus.RUNNING for item in state.stage_states
                ):
                    raise CorruptTaskStateError("Stored task has unleased execution state")
                index = next((i for i, item in enumerate(state.stage_states)
                              if item.status is StageStatus.PENDING), None)
                if (
                    index is None or state.stage_states[index].stage_id != stage_id
                    or any(item.status is not StageStatus.COMPLETED
                           for item in state.stage_states[:index])
                    or any(item.status is not StageStatus.PENDING
                           for item in state.stage_states[index:])
                ):
                    raise StageClaimError("Requested stage is not executable")
                if now <= state.updated_at:
                    raise InvalidLeaseConfigurationError("Claim time must follow task state")
                claimed_state = state
                stage_time = now
                if claimed_state.task_status is TaskStatus.PENDING:
                    claimed_state = claimed_state.start(updated_at=now)
                    stage_time = now + timedelta(microseconds=1)
                claimed_state = claimed_state.update_stage(
                    claimed_state.stage_states[index].start(), updated_at=stage_time,
                )
            new_version = expected_version + 1
            lease = StageExecutionLease(
                lease_id=uuid4().hex, task_id=task_id, stage_id=stage_id,
                claimed_version=new_version, claimed_at=now,
                lease_expires_at=now + lease_duration,
            )
            secret = secrets.token_bytes(32)
            proof = _claim_proof(lease, secret)
            payload = serialize_task_state(claimed_state)
            changed = db.execute(
                "UPDATE agent_task_states SET state_json = ?, version = version + 1, "
                "updated_at = ?, persisted_at = ?, lease_id = ?, lease_stage_id = ?, "
                "lease_claimed_at = ?, lease_expires_at = ?, lease_proof = ?, "
                "lease_migration_state = ? "
                "WHERE task_id = ? AND version = ?",
                (payload, claimed_state.updated_at.isoformat(), now.isoformat(),
                 lease.lease_id, lease.stage_id, lease.claimed_at.isoformat(),
                 lease.lease_expires_at.isoformat(), proof, _CURRENT_LEASE_STATE,
                 task_id, expected_version),
            )
            if changed.rowcount != 1:
                raise StaleTaskStateError("Task version is stale")
            db.commit()
        except TaskPersistenceError:
            db.rollback()
            raise
        except Exception:
            db.rollback()
            raise StageClaimError("Stage could not be claimed") from None
        persisted = PersistedTaskState(claimed_state, new_version, now, lease)
        return ClaimedTaskState(persisted, lease, _lease_authority(lease, secret))

    def claim_legacy_running_stage(
        self, task_id: str, stage_id: str, *, expected_version: int,
        lease_duration: timedelta = DEFAULT_STAGE_LEASE_DURATION,
    ) -> ClaimedTaskState:
        """Recover only a migration-marked, pre-lease running stage."""
        if (
            type(task_id) is not str or not task_id.strip()
            or type(stage_id) is not str or not stage_id.strip()
            or type(expected_version) is not int or expected_version < 1
        ):
            raise StageClaimError("Stage claim identity is invalid")
        _validate_lease_duration(lease_duration)
        db = self._db()
        try:
            db.execute("BEGIN IMMEDIATE")
            now = self._now()
            row = db.execute(
                "SELECT state_json, version, created_at, updated_at, lease_id, "
                "lease_stage_id, lease_claimed_at, lease_expires_at, lease_proof, "
                "lease_migration_state FROM agent_task_states WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if row is None:
                raise TaskNotFoundError("Task was not found")
            if type(row[1]) is not int or row[1] != expected_version:
                raise StaleTaskStateError("Task version is stale")
            state = deserialize_task_state(row[0])
            if (
                state.task_id != task_id
                or _timestamp(row[2]) != state.created_at
                or _timestamp(row[3]) != state.updated_at
                or row[9] != _LEGACY_UNLEASED_RUNNING
                or any(value is not None for value in row[4:9])
                or not _is_valid_running_state(state)
                or state.current_stage_id != stage_id
            ):
                raise StageClaimError("Legacy running stage is not recoverable")
            new_version = expected_version + 1
            lease = StageExecutionLease(
                lease_id=uuid4().hex, task_id=task_id, stage_id=stage_id,
                claimed_version=new_version, claimed_at=now,
                lease_expires_at=now + lease_duration,
            )
            secret = secrets.token_bytes(32)
            proof = _claim_proof(lease, secret)
            changed = db.execute(
                "UPDATE agent_task_states SET version = version + 1, persisted_at = ?, "
                "lease_id = ?, lease_stage_id = ?, lease_claimed_at = ?, "
                "lease_expires_at = ?, lease_proof = ?, lease_migration_state = ? "
                "WHERE task_id = ? AND version = ? AND lease_migration_state = ?",
                (now.isoformat(), lease.lease_id, lease.stage_id,
                 lease.claimed_at.isoformat(), lease.lease_expires_at.isoformat(),
                 proof, _CURRENT_LEASE_STATE, task_id, expected_version,
                 _LEGACY_UNLEASED_RUNNING),
            )
            if changed.rowcount != 1:
                raise StaleTaskStateError("Task version is stale")
            db.commit()
        except TaskPersistenceError:
            db.rollback()
            raise
        except Exception:
            db.rollback()
            raise StageClaimError("Legacy running stage could not be claimed") from None
        persisted = PersistedTaskState(state, new_version, now, lease)
        return ClaimedTaskState(persisted, lease, _lease_authority(lease, secret))

    @staticmethod
    def _validate_claim_completion(
        claimed: AgentTaskState, stage_id: str, new_state: AgentTaskState,
    ) -> None:
        try:
            checked = revalidate_agent_task_state(new_state)
            index = next(i for i, item in enumerate(claimed.stage_states)
                         if item.stage_id == stage_id)
            replacement = checked.stage_states[index]
            if (
                claimed.stage_states[index].status is not StageStatus.RUNNING
                or replacement.status not in (
                    StageStatus.COMPLETED, StageStatus.FAILED,
                    StageStatus.CANCELLED, StageStatus.AWAITING_APPROVAL,
                )
                or checked.stage_states[:index] != claimed.stage_states[:index]
                or checked.stage_states[index + 1:] != claimed.stage_states[index + 1:]
            ):
                raise ValueError
            if checked.task_status is TaskStatus.COMPLETED:
                stage_time = checked.updated_at - timedelta(microseconds=1)
                expected = claimed.update_stage(replacement, updated_at=stage_time)
                expected = expected.complete(updated_at=checked.updated_at)
            else:
                expected = claimed.update_stage(
                    replacement, updated_at=checked.updated_at,
                    approval_request=checked.approval_request,
                )
            if expected != checked:
                raise ValueError
        except Exception:
            raise StageClaimError("Claim completion state is invalid") from None

    def complete_claim(
        self, task_id: str, stage_id: str, lease_id: str,
        execution_authority: ValidatedStageExecutionLease,
        new_state: AgentTaskState, *, expected_version: int,
    ) -> PersistedTaskState:
        if (
            type(task_id) is not str or type(stage_id) is not str
            or type(lease_id) is not str or type(expected_version) is not int
            or expected_version < 1 or type(new_state) is not AgentTaskState
        ):
            raise StageLeaseMismatchError("Stage execution lease is invalid")
        db = self._db()
        try:
            db.execute("BEGIN IMMEDIATE")
            now = self._now()
            row = db.execute(
                "SELECT state_json, version, lease_id, lease_stage_id, "
                "lease_claimed_at, lease_expires_at, lease_proof, "
                "lease_migration_state FROM agent_task_states "
                "WHERE task_id = ?", (task_id,),
            ).fetchone()
            if row is None:
                raise TaskNotFoundError("Task was not found")
            if type(row[1]) is not int or row[1] != expected_version:
                raise StaleTaskStateError("Task version is stale")
            claimed = deserialize_task_state(row[0])
            lease = _lease_from_row(task_id, row[1], (row[2], row[3], row[4], row[5]))
            secret = _lease_authority_secret(execution_authority)
            if (
                lease is None or lease.lease_id != lease_id or lease.stage_id != stage_id
                or row[7] != _CURRENT_LEASE_STATE
                or not _valid_lease_proof(row[6]) or secret is None
                or (execution_authority.lease_id, execution_authority.task_id,
                    execution_authority.stage_id, execution_authority.claimed_version)
                != (lease_id, task_id, stage_id, expected_version)
                or not hmac.compare_digest(row[6], _claim_proof(lease, secret))
            ):
                raise StageLeaseMismatchError("Stage execution lease is invalid")
            if lease.lease_expires_at <= now:
                raise StageLeaseExpiredError("Stage execution lease has expired")
            if new_state.task_id != task_id:
                raise StageLeaseMismatchError("Stage execution lease is invalid")
            self._validate_claim_completion(claimed, stage_id, new_state)
            payload = serialize_task_state(new_state)
            changed = db.execute(
                "UPDATE agent_task_states SET state_json = ?, version = version + 1, "
                "updated_at = ?, persisted_at = ?, lease_id = NULL, "
                "lease_stage_id = NULL, lease_claimed_at = NULL, lease_expires_at = NULL, "
                "lease_proof = NULL, lease_migration_state = ? "
                "WHERE task_id = ? AND version = ? AND lease_id = ? AND lease_stage_id = ?",
                (payload, new_state.updated_at.isoformat(), now.isoformat(),
                 _CURRENT_LEASE_STATE, task_id, expected_version, lease_id, stage_id),
            )
            if changed.rowcount != 1:
                raise StageLeaseMismatchError("Stage execution lease is invalid")
            db.commit()
            _LEASE_SECRETS.pop(id(execution_authority), None)
        except TaskPersistenceError:
            db.rollback()
            raise
        except Exception:
            db.rollback()
            raise TaskPersistenceError("Claim completion could not be stored") from None
        return PersistedTaskState(
            deserialize_task_state(payload), expected_version + 1, now,
        )
