"""Explicit JSON persistence for immutable agent task snapshots with SQLite CAS."""

from __future__ import annotations

import json
import os
import sqlite3
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

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


class TaskPersistenceError(SovereignAPIError):
    code = "task_persistence_error"


class InvalidTaskDatabaseConfigurationError(TaskPersistenceError):
    code = "invalid_task_database_configuration"


class TaskNotFoundError(TaskPersistenceError):
    code = "task_not_found"


class TaskAlreadyExistsError(TaskPersistenceError):
    code = "task_already_exists"


class StaleTaskStateError(TaskPersistenceError):
    code = "stale_task_state"


class CorruptTaskStateError(TaskPersistenceError):
    code = "corrupt_task_state"


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

    def __post_init__(self) -> None:
        if type(self.state) is not AgentTaskState or type(self.version) is not int or self.version < 1 or not _utc(self.persisted_at):
            raise TaskPersistenceError("Persisted task state is invalid")


class TaskStateRepository(Protocol):
    def create(self, state: AgentTaskState) -> PersistedTaskState: ...

    def get(self, task_id: str) -> PersistedTaskState: ...

    def replace(
        self, task_id: str, state: AgentTaskState, *, expected_version: int,
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
                    persisted_at TEXT NOT NULL
                )
            """)
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
            raise TaskPersistenceError("Task repository clock is invalid") from None

    def create(self, state: AgentTaskState) -> PersistedTaskState:
        payload = serialize_task_state(state)
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
                "SELECT state_json, version, created_at, updated_at, persisted_at "
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
            return PersistedTaskState(state, row[1], _timestamp(row[4]))
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
                "SELECT state_json, created_at, updated_at "
                "FROM agent_task_states WHERE task_id = ?", (task_id,),
            ).fetchone()
            if existing is None:
                raise TaskNotFoundError("Task was not found")
            previous = deserialize_task_state(existing[0])
            if (
                previous.task_id != task_id
                or _timestamp(existing[1]) != previous.created_at
                or _timestamp(existing[2]) != previous.updated_at
                or previous.created_at != state.created_at
                or previous.original_prompt != state.original_prompt
                or previous.plan != state.plan
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
