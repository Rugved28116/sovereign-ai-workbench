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
from sovereign_api.artifact_reference import valid_artifact_reference
from sovereign_api.errors import SovereignAPIError
from sovereign_api.registry.models import valid_model_id
from sovereign_api.task_classification import TaskClass
from sovereign_api.task_plan_runner import revalidate_agent_task_state
from sovereign_api.task_planning import StageExecutionKind, TaskPlan, TaskStage, TaskStageType
from sovereign_api.stage_execution_records import (
    ExecutionResultDurability,
    StageExecutionAlreadyInProgressError, StageExecutionAttemptMismatchError,
    StageExecutionAwaitingApprovalError,
    StageExecutionOutcomeUnknownError, StageExecutionPreparationDecision,
    StageExecutionPreviouslyFailedError, StageExecutionRecord,
    StageExecutionRecordError, StageExecutionRecordStatus, StageIdempotencyKey,
    derive_stage_idempotency_key, stage_text_digest,
)
from sovereign_api.stage_output_store import MAX_STAGE_OUTPUT_BYTES
from sovereign_api.tool_approval import ApprovalRequest
from sovereign_api.tool_contracts import (
    ToolPermission, ToolResultStatus, ToolRiskLevel, ToolSideEffectLevel,
)
from sovereign_api.workspace_read_file import WORKSPACE_READ_FILE_TOOL_ID
from sovereign_api.workspace_write_artifact import WORKSPACE_WRITE_ARTIFACT_TOOL_ID


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
_LEASE_MIGRATION_NAME = "lease_state_classification_v1"
_EXECUTION_RECORD_COLUMN_MIGRATIONS = (
    ("result_durability", "ALTER TABLE stage_execution_records "
     "ADD COLUMN result_durability TEXT"),
    ("result_content_digest", "ALTER TABLE stage_execution_records "
     "ADD COLUMN result_content_digest TEXT"),
    ("side_effect_attempt_count", "ALTER TABLE stage_execution_records "
     "ADD COLUMN side_effect_attempt_count INTEGER"),
    ("legacy_approval_only", "ALTER TABLE stage_execution_records "
     "ADD COLUMN legacy_approval_only INTEGER NOT NULL DEFAULT 0"),
    ("terminal_error_code", "ALTER TABLE stage_execution_records "
     "ADD COLUMN terminal_error_code TEXT"),
    ("terminal_safe_message", "ALTER TABLE stage_execution_records "
     "ADD COLUMN terminal_safe_message TEXT"),
)
_EXECUTION_ATTEMPT_COLUMN_MIGRATIONS = (
    ("invocation_admitted_at", "ALTER TABLE stage_execution_attempts "
     "ADD COLUMN invocation_admitted_at TEXT"),
)


class ValidatedStageExecutionAttempt:
    """Opaque authority for one repository-issued physical execution attempt."""

    __slots__ = ("attempt_id", "idempotency_key", "task_id", "stage_id", "__weakref__")

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise StageExecutionAttemptMismatchError("Stage execution attempt is invalid")

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Stage execution attempt is immutable")

    def __repr__(self) -> str:
        return (
            "ValidatedStageExecutionAttempt("
            f"attempt_id={self.attempt_id!r}, idempotency_key={self.idempotency_key!r}, "
            f"task_id={self.task_id!r}, stage_id={self.stage_id!r})"
        )

    def __copy__(self) -> object:
        raise TypeError("Stage execution attempt cannot be copied")

    def __deepcopy__(self, memo: object) -> object:
        raise TypeError("Stage execution attempt cannot be copied")

    def __reduce__(self) -> object:
        raise TypeError("Stage execution attempt cannot be serialized")

    def __reduce_ex__(self, protocol: int) -> object:
        raise TypeError("Stage execution attempt cannot be serialized")


_ATTEMPT_SECRETS: dict[
    int, tuple[
        weakref.ReferenceType[ValidatedStageExecutionAttempt], bytes, bool,
        str, str, str, str,
    ]
] = {}


def _attempt_authority(
    *, attempt_id: str, idempotency_key: StageIdempotencyKey,
    task_id: str, stage_id: str, secret: bytes,
) -> ValidatedStageExecutionAttempt:
    authority = object.__new__(ValidatedStageExecutionAttempt)
    for name, value in (
        ("attempt_id", attempt_id), ("idempotency_key", idempotency_key.value),
        ("task_id", task_id), ("stage_id", stage_id),
    ):
        object.__setattr__(authority, name, value)
    identity = id(authority)

    def discard(reference: object) -> None:
        current = _ATTEMPT_SECRETS.get(identity)
        if current is not None and current[0] is reference:
            _ATTEMPT_SECRETS.pop(identity, None)

    _ATTEMPT_SECRETS[identity] = (
        weakref.ref(authority, discard), secret, False,
        attempt_id, idempotency_key.value, task_id, stage_id,
    )
    return authority


def _attempt_authority_secret(
    value: object, *, invoked: bool | None = None,
) -> bytes | None:
    if type(value) is not ValidatedStageExecutionAttempt:
        return None
    registered = _ATTEMPT_SECRETS.get(id(value))
    if (
        registered is None or registered[0]() is not value
        or (invoked is not None and registered[2] is not invoked)
        or (value.attempt_id, value.idempotency_key, value.task_id, value.stage_id)
        != registered[3:]
    ):
        return None
    return registered[1]


class ValidatedStageInvocationPermit:
    """Opaque one-use permit issued after durable immediate revalidation."""

    __slots__ = (
        "attempt_id", "idempotency_key", "task_id", "stage_id", "tool_id",
        "lease_id", "claimed_version", "__weakref__",
    )

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise StageExecutionAttemptMismatchError("Stage invocation permit is invalid")

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Stage invocation permit is immutable")

    def __copy__(self) -> object:
        raise TypeError("Stage invocation permit cannot be copied")

    def __deepcopy__(self, memo: object) -> object:
        raise TypeError("Stage invocation permit cannot be copied")

    def __reduce__(self) -> object:
        raise TypeError("Stage invocation permit cannot be serialized")

    def __reduce_ex__(self, protocol: int) -> object:
        raise TypeError("Stage invocation permit cannot be serialized")


class ValidatedModelInvocationPermit:
    """Opaque one-use permit for one exact admitted model request."""

    __slots__ = (
        "attempt_id", "idempotency_key", "task_id", "stage_id", "model_id",
        "request_digest", "lease_id", "claimed_version", "__weakref__",
    )

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise StageExecutionAttemptMismatchError("Model invocation permit is invalid")

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Model invocation permit is immutable")

    def __copy__(self) -> object:
        raise TypeError("Model invocation permit cannot be copied")

    def __deepcopy__(self, memo: object) -> object:
        raise TypeError("Model invocation permit cannot be copied")

    def __reduce__(self) -> object:
        raise TypeError("Model invocation permit cannot be serialized")

    def __reduce_ex__(self, protocol: int) -> object:
        raise TypeError("Model invocation permit cannot be serialized")


_MODEL_INVOCATION_PERMITS: dict[int, tuple[object, ...]] = {}


def _model_invocation_permit(
    *, attempt_id: str, idempotency_key: str, task_id: str, stage_id: str,
    model_id: str, request_digest: str, lease_id: str, claimed_version: int,
) -> ValidatedModelInvocationPermit:
    permit = object.__new__(ValidatedModelInvocationPermit)
    values = (
        attempt_id, idempotency_key, task_id, stage_id, model_id,
        request_digest, lease_id, claimed_version,
    )
    for name, value in zip(
        ("attempt_id", "idempotency_key", "task_id", "stage_id", "model_id",
         "request_digest", "lease_id", "claimed_version"), values, strict=True,
    ):
        object.__setattr__(permit, name, value)
    identity = id(permit)
    def discard(reference: object) -> None:
        current = _MODEL_INVOCATION_PERMITS.get(identity)
        if current is not None and current[0] is reference:
            _MODEL_INVOCATION_PERMITS.pop(identity, None)
    _MODEL_INVOCATION_PERMITS[identity] = (weakref.ref(permit, discard), *values)
    return permit


def _model_permit_identity(
    value: object, *, consume: bool,
) -> tuple[str, str, str, str, str, str, str, int]:
    if type(value) is not ValidatedModelInvocationPermit:
        raise StageExecutionAttemptMismatchError("Model invocation permit is invalid")
    stored = _MODEL_INVOCATION_PERMITS.get(id(value))
    visible = (
        value.attempt_id, value.idempotency_key, value.task_id, value.stage_id,
        value.model_id, value.request_digest, value.lease_id, value.claimed_version,
    )
    if stored is None or stored[0]() is not value or stored[1:] != visible:
        raise StageExecutionAttemptMismatchError("Model invocation permit is invalid")
    if consume:
        _MODEL_INVOCATION_PERMITS.pop(id(value), None)
    return visible


_INVOCATION_PERMITS: dict[
    int, tuple[
        weakref.ReferenceType[ValidatedStageInvocationPermit],
        str, str, str, str, str, str, int,
    ]
] = {}


def _invocation_permit(
    *, attempt_id: str, idempotency_key: str, task_id: str, stage_id: str,
    tool_id: str, lease_id: str, claimed_version: int,
) -> ValidatedStageInvocationPermit:
    permit = object.__new__(ValidatedStageInvocationPermit)
    values = (
        attempt_id, idempotency_key, task_id, stage_id, tool_id,
        lease_id, claimed_version,
    )
    for name, value in zip(
        ("attempt_id", "idempotency_key", "task_id", "stage_id", "tool_id",
         "lease_id", "claimed_version"), values, strict=True,
    ):
        object.__setattr__(permit, name, value)
    identity = id(permit)

    def discard(reference: object) -> None:
        current = _INVOCATION_PERMITS.get(identity)
        if current is not None and current[0] is reference:
            _INVOCATION_PERMITS.pop(identity, None)

    reference = weakref.ref(permit, discard)
    _INVOCATION_PERMITS[identity] = (reference, *values)
    return permit


def _stage_invocation_permit_identity(
    value: object, *, consume: bool,
) -> tuple[str, str, str, str, str, str, int]:
    if type(value) is not ValidatedStageInvocationPermit:
        raise StageExecutionAttemptMismatchError("Stage invocation permit is invalid")
    registered = _INVOCATION_PERMITS.get(id(value))
    visible = (
        value.attempt_id, value.idempotency_key, value.task_id, value.stage_id,
        value.tool_id, value.lease_id, value.claimed_version,
    )
    if (
        registered is None or registered[0]() is not value
        or registered[1:] != visible
    ):
        raise StageExecutionAttemptMismatchError("Stage invocation permit is invalid")
    if consume:
        _INVOCATION_PERMITS.pop(id(value), None)
    return visible


def _attempt_binding(
    *, attempt_id: str, key: str, task_id: str, stage_id: str,
    lease_id: str, claimed_version: int,
) -> bytes:
    return json.dumps({
        "attempt_id": attempt_id, "claimed_version": claimed_version,
        "idempotency_key": key, "lease_id": lease_id,
        "stage_id": stage_id, "task_id": task_id,
    }, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _attempt_proof(*, secret: bytes, **identity: object) -> str:
    return hmac.new(secret, _attempt_binding(**identity), hashlib.sha256).hexdigest()


@dataclass(frozen=True, slots=True)
class PreparedStageExecution:
    decision: StageExecutionPreparationDecision
    record: StageExecutionRecord
    execution_authority: ValidatedStageExecutionAttempt | None = None

    def __post_init__(self) -> None:
        if (
            type(self.decision) is not StageExecutionPreparationDecision
            or type(self.record) is not StageExecutionRecord
            or (
                self.decision is StageExecutionPreparationDecision.EXECUTE
                and _attempt_authority_secret(
                    self.execution_authority, invoked=False,
                ) is None
            )
            or (
                self.decision is StageExecutionPreparationDecision.KNOWN_SUCCESS
                and self.execution_authority is not None
            )
        ):
            raise StageExecutionRecordError("Stage execution preparation is invalid")


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


def _execution_record_from_row(row: tuple[object, ...]) -> StageExecutionRecord:
    try:
        return StageExecutionRecord(
            idempotency_key=StageIdempotencyKey(row[0]),
            task_id=row[1], stage_id=row[2],
            execution_kind=StageExecutionKind(row[3]),
            status=StageExecutionRecordStatus(row[4]), attempt_count=row[5],
            current_attempt_id=row[6], first_started_at=_timestamp(row[7]),
            last_started_at=_timestamp(row[8]),
            completed_at=_timestamp(row[9]) if row[9] is not None else None,
            safe_result_reference=row[10],
            output_kind=(StageOutputKind(row[11]) if row[11] is not None else None),
            result_durability=(
                ExecutionResultDurability(row[12])
                if row[12] is not None else ExecutionResultDurability.NONE
            ),
            result_content_digest=row[13],
            selected_model_id=row[14], selected_tool_id=row[15],
            terminal_error_code=row[16], terminal_safe_message=row[17],
            created_at=_timestamp(row[18]), updated_at=_timestamp(row[19]),
        )
    except Exception:
        raise CorruptTaskStateError("Stored stage execution record is invalid") from None


_EXECUTION_RECORD_COLUMNS = (
    "idempotency_key, task_id, stage_id, execution_kind, status, "
    "side_effect_attempt_count, "
    "current_attempt_id, first_started_at, last_started_at, completed_at, "
    "safe_result_reference, output_kind, result_durability, "
    "result_content_digest, selected_model_id, selected_tool_id, "
    "terminal_error_code, terminal_safe_message, created_at, updated_at"
)


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

    def validate_claimed_execution(
        self, claimed: ClaimedTaskState,
        execution_authority: ValidatedStageExecutionLease,
    ) -> PersistedTaskState: ...

    def prepare_stage_execution(
        self, claimed: ClaimedTaskState,
        execution_authority: ValidatedStageExecutionLease,
        *, admit_attempt: bool = True,
    ) -> PreparedStageExecution | None: ...

    def authorize_attempt_invocation(
        self, claimed: ClaimedTaskState,
        lease_authority: ValidatedStageExecutionLease,
        prepared: PreparedStageExecution,
        attempt_authority: ValidatedStageExecutionAttempt,
    ) -> ValidatedStageInvocationPermit: ...

    def consume_stage_invocation_permit(
        self, permit: ValidatedStageInvocationPermit,
    ) -> tuple[str, str, str, str, str, str, int]: ...

    def authorize_model_attempt_invocation(
        self, claimed: ClaimedTaskState,
        lease_authority: ValidatedStageExecutionLease,
        prepared: PreparedStageExecution,
        attempt_authority: ValidatedStageExecutionAttempt, *,
        model_id: str, request_digest: str,
    ) -> ValidatedModelInvocationPermit: ...

    def consume_model_invocation_permit(
        self, permit: ValidatedModelInvocationPermit,
    ) -> tuple[str, str, str, str, str, str, str, int]: ...

    def suspend_claim_for_approval(
        self, claimed: ClaimedTaskState,
        execution_authority: ValidatedStageExecutionLease,
        new_state: AgentTaskState,
    ) -> PersistedTaskState: ...

    def record_stage_execution_outcome(
        self, claimed: ClaimedTaskState,
        lease_authority: ValidatedStageExecutionLease,
        prepared: PreparedStageExecution,
        attempt_authority: ValidatedStageExecutionAttempt,
        new_state: AgentTaskState,
        *, invocation_receipt: object | None = None,
        result_content_digest: str | None = None,
    ) -> StageExecutionRecord: ...

    def get_stage_execution_record(
        self, task_id: str, stage_id: str,
    ) -> StageExecutionRecord: ...

    def complete_claim(
        self, task_id: str, stage_id: str, lease_id: str,
        execution_authority: ValidatedStageExecutionLease,
        new_state: AgentTaskState, *, expected_version: int,
    ) -> PersistedTaskState: ...

    def close(self) -> None: ...


class SQLiteTaskStateRepository:
    """One thread-affine connection per instance; SQLite serializes CAS updates."""

    def __init__(self, db_path: str | Path | None) -> None:
        self._initialize(db_path, clock=lambda: datetime.now(UTC))

    @classmethod
    def _for_test(
        cls, db_path: str | Path | None, *, clock,
    ) -> SQLiteTaskStateRepository:
        """Construct with a deterministic clock; intentionally test-only."""
        instance = cls.__new__(cls)
        instance._initialize(db_path, clock=clock)
        return instance

    def _initialize(self, db_path: str | Path | None, *, clock) -> None:
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
            # SQLite DDL participates in this explicit transaction.  The
            # durable migration marker is written only after row
            # classification, so a reopen never treats a new column alone as
            # proof that legacy rows were handled.
            connection.execute("BEGIN IMMEDIATE")
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
            connection.execute("""
                CREATE TABLE IF NOT EXISTS stage_execution_records (
                    idempotency_key TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    stage_id TEXT NOT NULL,
                    execution_kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL CHECK (attempt_count >= 0),
                    current_attempt_id TEXT,
                    first_started_at TEXT NOT NULL,
                    last_started_at TEXT NOT NULL,
                    completed_at TEXT,
                    safe_result_reference TEXT,
                    output_kind TEXT,
                    result_durability TEXT,
                    result_content_digest TEXT,
                    selected_model_id TEXT,
                    selected_tool_id TEXT,
                    terminal_error_code TEXT,
                    terminal_safe_message TEXT,
                    side_effect_attempt_count INTEGER NOT NULL DEFAULT 0
                        CHECK (side_effect_attempt_count >= 0),
                    legacy_approval_only INTEGER NOT NULL DEFAULT 0
                        CHECK (legacy_approval_only IN (0, 1)),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE (task_id, stage_id)
                )
            """)
            execution_columns = {
                row[1] for row in connection.execute(
                    "PRAGMA table_info(stage_execution_records)"
                )
            }
            for column, statement in _EXECUTION_RECORD_COLUMN_MIGRATIONS:
                if column not in execution_columns:
                    connection.execute(statement)
                    execution_columns.add(column)
            connection.execute(
                "UPDATE stage_execution_records SET side_effect_attempt_count = "
                "attempt_count WHERE side_effect_attempt_count IS NULL"
            )
            connection.execute("""
                CREATE TABLE IF NOT EXISTS stage_execution_attempts (
                    attempt_id TEXT PRIMARY KEY,
                    idempotency_key TEXT NOT NULL,
                    lease_id TEXT NOT NULL,
                    claimed_version INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    invocation_admitted_at TEXT,
                    completed_at TEXT,
                    attempt_proof TEXT NOT NULL,
                    FOREIGN KEY (idempotency_key)
                        REFERENCES stage_execution_records(idempotency_key)
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS task_repository_migrations (
                    migration_name TEXT PRIMARY KEY,
                    completed_at TEXT NOT NULL
                )
            """)
            attempt_columns = {
                row[1] for row in connection.execute(
                    "PRAGMA table_info(stage_execution_attempts)"
                )
            }
            for column, statement in _EXECUTION_ATTEMPT_COLUMN_MIGRATIONS:
                if column not in attempt_columns:
                    connection.execute(statement)
                    attempt_columns.add(column)
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(agent_task_states)")
            }
            for column, statement in _LEASE_COLUMN_MIGRATIONS[:-1]:
                if column not in columns:
                    connection.execute(statement)
                    columns.add(column)
            if "lease_migration_state" not in columns:
                connection.execute(_LEASE_COLUMN_MIGRATIONS[-1][1])
                columns.add("lease_migration_state")
            legacy_unleased: list[str] = []
            legacy_unproven: list[str] = []
            migration_complete = connection.execute(
                "SELECT 1 FROM task_repository_migrations "
                "WHERE migration_name = ?", (_LEASE_MIGRATION_NAME,),
            ).fetchone()
            if migration_complete is None:
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
                connection.execute(
                    "INSERT INTO task_repository_migrations "
                    "(migration_name, completed_at) VALUES (?, ?)",
                    (_LEASE_MIGRATION_NAME, datetime.now(UTC).isoformat()),
                )
            legacy_approval_rows = connection.execute(
                "SELECT records.idempotency_key, records.task_id, records.stage_id, "
                "records.current_attempt_id, tasks.state_json, attempts.status, "
                "attempts.idempotency_key, attempts.invocation_admitted_at, "
                "records.side_effect_attempt_count, records.attempt_count "
                "FROM stage_execution_records AS records "
                "JOIN agent_task_states AS tasks ON tasks.task_id = records.task_id "
                "LEFT JOIN stage_execution_attempts AS attempts "
                "ON attempts.attempt_id = records.current_attempt_id "
                "WHERE records.status = ? AND records.result_durability IS NULL "
                "AND records.legacy_approval_only = 1 "
                "AND records.execution_kind = ? "
                "AND records.safe_result_reference IS NULL "
                "AND records.output_kind IS NULL "
                "AND records.result_content_digest IS NULL "
                "AND records.selected_model_id IS NULL "
                "AND records.selected_tool_id IS NULL",
                (StageExecutionRecordStatus.FAILED.value,
                 StageExecutionKind.TOOL.value),
            ).fetchall()
            for (
                key, record_task_id, record_stage_id, attempt_id, state_json,
                attempt_status, attempt_key, invocation_admitted_at,
                side_effect_attempt_count, compatibility_attempt_count,
            ) in legacy_approval_rows:
                try:
                    legacy_state = deserialize_task_state(state_json)
                    if (
                        legacy_state.task_id != record_task_id
                        or legacy_state.task_status is not TaskStatus.AWAITING_APPROVAL
                        or legacy_state.current_stage_id != record_stage_id
                    ):
                        continue
                    blocked = next(
                        item for item in legacy_state.stage_states
                        if item.stage_id == record_stage_id
                    )
                    planned_stage = next(
                        item for item in legacy_state.plan.stages
                        if item.stage_id == record_stage_id
                    )
                    if blocked.status is not StageStatus.AWAITING_APPROVAL:
                        continue
                    if derive_stage_idempotency_key(
                        legacy_state, planned_stage,
                    ).value != key:
                        continue
                    if (
                        attempt_id is None
                        or attempt_status != StageExecutionRecordStatus.FAILED.value
                        or attempt_key != key
                        or
                        invocation_admitted_at is not None
                        or side_effect_attempt_count != 0
                    ):
                        continue
                except Exception:
                    continue
                table_sql = connection.execute(
                    "SELECT sql FROM sqlite_master WHERE type = 'table' "
                    "AND name = 'stage_execution_records'"
                ).fetchone()
                normalized_table_sql = "".join(
                    str(table_sql[0] or "").lower().split()
                )
                # Very old schemas constrained this compatibility counter to
                # one or more.  It is non-authoritative there; modern schemas
                # store the valid zero alongside the authoritative count.
                compatibility_value = (
                    0 if "attempt_count>=0" in normalized_table_sql
                    else compatibility_attempt_count
                )
                updated = connection.execute(
                    "UPDATE stage_execution_records SET status = ?, "
                    "attempt_count = ?, current_attempt_id = NULL, "
                    "side_effect_attempt_count = 0, "
                    "result_durability = ?, legacy_approval_only = 0 "
                    "WHERE idempotency_key = ? AND task_id = ? AND stage_id = ? "
                    "AND legacy_approval_only = 1 "
                    "AND side_effect_attempt_count = 0",
                    (StageExecutionRecordStatus.AWAITING_APPROVAL.value,
                     compatibility_value, ExecutionResultDurability.NONE.value, key,
                     record_task_id, record_stage_id),
                )
                if updated.rowcount != 1:
                    continue
                if attempt_id is not None:
                    deleted = connection.execute(
                        "DELETE FROM stage_execution_attempts WHERE attempt_id = ? "
                        "AND idempotency_key = ? AND status = ? "
                        "AND invocation_admitted_at IS NULL",
                        (attempt_id, key, StageExecutionRecordStatus.FAILED.value),
                    )
                    if deleted.rowcount != 1:
                        raise sqlite3.IntegrityError
            connection.commit()
        except (ValueError, OSError, sqlite3.Error):
            if "connection" in locals():
                connection.close()
            raise InvalidTaskDatabaseConfigurationError("Task database path is invalid") from None
        self._connection: sqlite3.Connection | None = connection
        self._clock = clock

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
        except (TaskPersistenceError, StageExecutionRecordError):
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

    def _live_claim_in_transaction(
        self, claimed: ClaimedTaskState,
        execution_authority: ValidatedStageExecutionLease,
        *, now: datetime, require_unexpired: bool,
    ) -> tuple[PersistedTaskState, StageExecutionLease, str]:
        if type(claimed) is not ClaimedTaskState:
            raise StageLeaseMismatchError("Stage execution lease is invalid")
        row = self._db().execute(
            "SELECT state_json, version, created_at, updated_at, persisted_at, "
            "lease_id, lease_stage_id, lease_claimed_at, lease_expires_at, "
            "lease_proof, lease_migration_state FROM agent_task_states "
            "WHERE task_id = ?", (claimed.persisted.state.task_id,),
        ).fetchone()
        if row is None:
            raise TaskNotFoundError("Task was not found")
        if type(row[1]) is not int or row[1] != claimed.persisted.version:
            raise StaleTaskStateError("Task version is stale")
        state = deserialize_task_state(row[0])
        lease = _lease_from_row(
            state.task_id, row[1], (row[5], row[6], row[7], row[8]),
        )
        secret = _lease_authority_secret(execution_authority)
        persisted = PersistedTaskState(state, row[1], _timestamp(row[4]), lease)
        if (
            _timestamp(row[2]) != state.created_at
            or _timestamp(row[3]) != state.updated_at
            or row[10] != _CURRENT_LEASE_STATE
            or lease is None or not _valid_lease_proof(row[9])
            or secret is None or persisted != claimed.persisted
            or lease != claimed.lease
            or (execution_authority.lease_id, execution_authority.task_id,
                execution_authority.stage_id, execution_authority.claimed_version)
            != (lease.lease_id, lease.task_id, lease.stage_id, lease.claimed_version)
            or not hmac.compare_digest(row[9], _claim_proof(lease, secret))
        ):
            raise StageLeaseMismatchError("Stage execution lease is invalid")
        if require_unexpired and lease.lease_expires_at <= now:
            raise StageLeaseExpiredError("Stage execution lease has expired")
        return persisted, lease, row[9]

    def validate_claimed_execution(
        self, claimed: ClaimedTaskState,
        execution_authority: ValidatedStageExecutionLease,
    ) -> PersistedTaskState:
        """Validate live claim ownership in a short transaction before side effects."""
        if type(claimed) is not ClaimedTaskState:
            raise StageLeaseMismatchError("Stage execution lease is invalid")
        db = self._db()
        try:
            db.execute("BEGIN IMMEDIATE")
            now = self._now()
            persisted, _, _ = self._live_claim_in_transaction(
                claimed, execution_authority, now=now, require_unexpired=True,
            )
            db.commit()
            return persisted
        except TaskPersistenceError:
            db.rollback()
            raise
        except Exception:
            db.rollback()
            raise StageLeaseMismatchError("Stage execution lease is invalid") from None

    def get_stage_execution_record(
        self, task_id: str, stage_id: str,
    ) -> StageExecutionRecord:
        if (
            type(task_id) is not str or not task_id.strip()
            or type(stage_id) is not str or not stage_id.strip()
        ):
            raise StageExecutionRecordError("Stage execution identity is invalid")
        try:
            row = self._db().execute(
                f"SELECT {_EXECUTION_RECORD_COLUMNS} FROM stage_execution_records "
                "WHERE task_id = ? AND stage_id = ?", (task_id, stage_id),
            ).fetchone()
        except sqlite3.Error:
            raise StageExecutionRecordError(
                "Stage execution record could not be loaded"
            ) from None
        if row is None:
            raise StageExecutionRecordError("Stage execution record was not found")
        return _execution_record_from_row(row)

    @staticmethod
    def _require_terminal_attempt_evidence(
        db: sqlite3.Connection, record: StageExecutionRecord, *,
        expected_status: StageExecutionRecordStatus,
    ) -> None:
        """Reject a terminal record that lacks one admitted terminal attempt."""
        if (
            record.status is not expected_status
            or record.current_attempt_id is None
            or record.attempt_count < 1
        ):
            raise CorruptTaskStateError("Stored terminal execution evidence is invalid")
        attempt = db.execute(
            "SELECT attempt_id, idempotency_key, lease_id, claimed_version, "
            "status, started_at, invocation_admitted_at, completed_at, attempt_proof "
            "FROM stage_execution_attempts WHERE attempt_id = ? "
            "AND idempotency_key = ?",
            (record.current_attempt_id, record.idempotency_key.value),
        ).fetchone()
        try:
            if (
                attempt is None
                or attempt[0] != record.current_attempt_id
                or attempt[1] != record.idempotency_key.value
                or type(attempt[2]) is not str or len(attempt[2]) != 32
                or any(character not in "0123456789abcdef" for character in attempt[2])
                or type(attempt[3]) is not int or attempt[3] < 1
                or attempt[4] != expected_status.value
                or attempt[6] is None or attempt[7] is None
                or not _valid_lease_proof(attempt[8])
                or _timestamp(attempt[6]) < _timestamp(attempt[5])
                or _timestamp(attempt[7]) < _timestamp(attempt[6])
            ):
                raise ValueError
        except Exception:
            raise CorruptTaskStateError(
                "Stored terminal execution evidence is invalid"
            ) from None

    def prepare_stage_execution(
        self, claimed: ClaimedTaskState,
        execution_authority: ValidatedStageExecutionLease,
        *, admit_attempt: bool = True,
    ) -> PreparedStageExecution | None:
        """Authorize one physical attempt or return a known completed outcome."""
        if type(admit_attempt) is not bool:
            raise StageExecutionRecordError("Stage execution admission is invalid")
        db = self._db()
        unknown_record: StageExecutionRecord | None = None
        try:
            db.execute("BEGIN IMMEDIATE")
            now = self._now()
            persisted, lease, _ = self._live_claim_in_transaction(
                claimed, execution_authority, now=now, require_unexpired=True,
            )
            stage = next(
                item for item in persisted.state.plan.stages
                if item.stage_id == lease.stage_id
            )
            key = derive_stage_idempotency_key(persisted.state, stage)
            row = db.execute(
                f"SELECT {_EXECUTION_RECORD_COLUMNS} FROM stage_execution_records "
                "WHERE task_id = ? AND stage_id = ?",
                (lease.task_id, lease.stage_id),
            ).fetchone()
            if row is not None:
                record = _execution_record_from_row(row)
                if record.idempotency_key != key:
                    raise CorruptTaskStateError(
                        "Stored stage execution identity does not match the task"
                    )
                if record.status is StageExecutionRecordStatus.SUCCEEDED:
                    self._require_terminal_attempt_evidence(
                        db, record, expected_status=StageExecutionRecordStatus.SUCCEEDED,
                    )
                    db.commit()
                    return PreparedStageExecution(
                        StageExecutionPreparationDecision.KNOWN_SUCCESS, record,
                    )
                if record.status is StageExecutionRecordStatus.FAILED:
                    raise StageExecutionPreviouslyFailedError(
                        "Stage execution previously failed"
                    )
                if record.status is StageExecutionRecordStatus.AWAITING_APPROVAL:
                    raise StageExecutionAwaitingApprovalError(
                        "Stage execution is awaiting approval"
                    )
                if record.status is StageExecutionRecordStatus.UNKNOWN:
                    raise StageExecutionOutcomeUnknownError(
                        "Stage execution outcome is unknown"
                    )
                attempt = db.execute(
                    "SELECT attempt_id, lease_id, claimed_version, status "
                    "FROM stage_execution_attempts WHERE attempt_id = ?",
                    (record.current_attempt_id,),
                ).fetchone()
                if (
                    attempt is None or attempt[0] != record.current_attempt_id
                    or attempt[3] != StageExecutionRecordStatus.IN_PROGRESS.value
                ):
                    raise CorruptTaskStateError(
                        "Stored stage execution attempt is invalid"
                    )
                if attempt[1] == lease.lease_id and attempt[2] == lease.claimed_version:
                    raise StageExecutionAlreadyInProgressError(
                        "Stage execution is already in progress"
                    )
                db.execute(
                    "UPDATE stage_execution_attempts SET status = ?, completed_at = ? "
                    "WHERE attempt_id = ? AND status = ?",
                    (StageExecutionRecordStatus.UNKNOWN.value, now.isoformat(),
                     attempt[0], StageExecutionRecordStatus.IN_PROGRESS.value),
                )
                db.execute(
                    "UPDATE stage_execution_records SET status = ?, completed_at = ?, "
                    "result_durability = ?, "
                    "updated_at = ? WHERE idempotency_key = ? AND status = ?",
                    (StageExecutionRecordStatus.UNKNOWN.value, now.isoformat(),
                     ExecutionResultDurability.NONE.value, now.isoformat(), key.value,
                     StageExecutionRecordStatus.IN_PROGRESS.value),
                )
                unknown_record = _execution_record_from_row(db.execute(
                    f"SELECT {_EXECUTION_RECORD_COLUMNS} FROM stage_execution_records "
                    "WHERE idempotency_key = ?", (key.value,),
                ).fetchone())
                db.commit()
            else:
                if not admit_attempt:
                    db.commit()
                    return None
                attempt_id = uuid4().hex
                secret = secrets.token_bytes(32)
                identity = {
                    "attempt_id": attempt_id, "key": key.value,
                    "task_id": lease.task_id, "stage_id": lease.stage_id,
                    "lease_id": lease.lease_id,
                    "claimed_version": lease.claimed_version,
                }
                proof = _attempt_proof(secret=secret, **identity)
                db.execute(
                    "INSERT INTO stage_execution_records "
                    "(idempotency_key, task_id, stage_id, execution_kind, status, "
                    "attempt_count, current_attempt_id, first_started_at, "
                    "last_started_at, result_durability, side_effect_attempt_count, "
                    "created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, ?, 1, ?, ?)",
                    (key.value, lease.task_id, lease.stage_id,
                     stage.execution_kind.value,
                     StageExecutionRecordStatus.IN_PROGRESS.value, attempt_id,
                     now.isoformat(), now.isoformat(),
                     ExecutionResultDurability.NONE.value,
                     now.isoformat(), now.isoformat()),
                )
                db.execute(
                    "INSERT INTO stage_execution_attempts "
                    "(attempt_id, idempotency_key, lease_id, claimed_version, status, "
                    "started_at, attempt_proof) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (attempt_id, key.value, lease.lease_id, lease.claimed_version,
                     StageExecutionRecordStatus.IN_PROGRESS.value,
                     now.isoformat(), proof),
                )
                record = _execution_record_from_row(db.execute(
                    f"SELECT {_EXECUTION_RECORD_COLUMNS} FROM stage_execution_records "
                    "WHERE idempotency_key = ?", (key.value,),
                ).fetchone())
                db.commit()
                authority = _attempt_authority(
                    attempt_id=attempt_id, idempotency_key=key,
                    task_id=lease.task_id, stage_id=lease.stage_id, secret=secret,
                )
                return PreparedStageExecution(
                    StageExecutionPreparationDecision.EXECUTE, record, authority,
                )
        except (TaskPersistenceError, StageExecutionRecordError):
            if db.in_transaction:
                db.rollback()
            raise
        except Exception:
            if db.in_transaction:
                db.rollback()
            raise StageExecutionRecordError(
                "Stage execution could not be prepared"
            ) from None
        if unknown_record is not None:
            raise StageExecutionOutcomeUnknownError(
                "Stage execution outcome is unknown"
            )
        raise StageExecutionRecordError("Stage execution could not be prepared")

    def authorize_attempt_invocation(
        self, claimed: ClaimedTaskState,
        lease_authority: ValidatedStageExecutionLease,
        prepared: PreparedStageExecution,
        attempt_authority: ValidatedStageExecutionAttempt,
    ) -> ValidatedStageInvocationPermit:
        """Revalidate one admitted attempt and issue a provisional permit."""
        if (
            type(prepared) is not PreparedStageExecution
            or prepared.decision is not StageExecutionPreparationDecision.EXECUTE
            or prepared.execution_authority is not attempt_authority
        ):
            raise StageExecutionAttemptMismatchError(
                "Stage execution attempt is invalid"
            )
        db = self._db()
        try:
            db.execute("BEGIN IMMEDIATE")
            now = self._now()
            persisted, lease, _ = self._live_claim_in_transaction(
                claimed, lease_authority, now=now, require_unexpired=True,
            )
            stage = next(
                item for item in persisted.state.plan.stages
                if item.stage_id == lease.stage_id
            )
            if (
                stage.execution_kind is not StageExecutionKind.TOOL
                or stage.tool_id is None
            ):
                raise StageExecutionAttemptMismatchError(
                    "Stage execution attempt is invalid"
                )
            secret = _attempt_authority_secret(attempt_authority, invoked=False)
            attempt = db.execute(
                "SELECT attempts.idempotency_key, attempts.lease_id, "
                "attempts.claimed_version, attempts.status, attempts.attempt_proof, "
                "attempts.invocation_admitted_at, records.current_attempt_id, "
                "records.status "
                "FROM stage_execution_attempts AS attempts "
                "JOIN stage_execution_records AS records "
                "ON records.idempotency_key = attempts.idempotency_key "
                "WHERE attempts.attempt_id = ? AND records.task_id = ? "
                "AND records.stage_id = ?",
                (attempt_authority.attempt_id, lease.task_id, lease.stage_id),
            ).fetchone()
            identity = {
                "attempt_id": attempt_authority.attempt_id,
                "key": prepared.record.idempotency_key.value,
                "task_id": lease.task_id, "stage_id": lease.stage_id,
                "lease_id": lease.lease_id,
                "claimed_version": lease.claimed_version,
            }
            if (
                secret is None or attempt is None
                or attempt[0] != prepared.record.idempotency_key.value
                or attempt[1] != lease.lease_id
                or attempt[2] != lease.claimed_version
                or attempt[3] != StageExecutionRecordStatus.IN_PROGRESS.value
                or not _valid_lease_proof(attempt[4])
                or attempt[5] is not None
                or attempt[6] != attempt_authority.attempt_id
                or attempt[7] != StageExecutionRecordStatus.IN_PROGRESS.value
                or not hmac.compare_digest(
                    attempt[4], _attempt_proof(secret=secret, **identity),
                )
                or (
                    attempt_authority.attempt_id,
                    attempt_authority.idempotency_key,
                    attempt_authority.task_id,
                    attempt_authority.stage_id,
                ) != (
                    identity["attempt_id"], identity["key"],
                    identity["task_id"], identity["stage_id"],
                )
            ):
                raise StageExecutionAttemptMismatchError(
                    "Stage execution attempt is invalid"
                )
            db.commit()
            registered = _ATTEMPT_SECRETS.get(id(attempt_authority))
            if registered is None or registered[0]() is not attempt_authority:
                raise StageExecutionAttemptMismatchError(
                    "Stage execution attempt is invalid"
                )
            _ATTEMPT_SECRETS[id(attempt_authority)] = (
                registered[0], registered[1], True, *registered[3:],
            )
            return _invocation_permit(
                attempt_id=attempt_authority.attempt_id,
                idempotency_key=prepared.record.idempotency_key.value,
                task_id=lease.task_id, stage_id=lease.stage_id,
                tool_id=stage.tool_id, lease_id=lease.lease_id,
                claimed_version=lease.claimed_version,
            )
        except (TaskPersistenceError, StageExecutionRecordError):
            if db.in_transaction:
                db.rollback()
            raise
        except Exception:
            if db.in_transaction:
                db.rollback()
            raise StageExecutionRecordError(
                "Stage invocation could not be authorized"
            ) from None

    def consume_stage_invocation_permit(
        self, permit: ValidatedStageInvocationPermit,
    ) -> tuple[str, str, str, str, str, str, int]:
        """Atomically consume one permit immediately before external invocation."""
        identity = _stage_invocation_permit_identity(permit, consume=False)
        (
            attempt_id, idempotency_key, task_id, stage_id, tool_id,
            lease_id, claimed_version,
        ) = identity
        db = self._db()
        try:
            db.execute("BEGIN IMMEDIATE")
            now = self._now()
            row = db.execute(
                "SELECT tasks.version, tasks.lease_id, tasks.lease_stage_id, "
                "tasks.lease_expires_at, attempts.status, "
                "attempts.invocation_admitted_at, records.current_attempt_id, "
                "records.status, records.execution_kind "
                "FROM agent_task_states AS tasks "
                "JOIN stage_execution_records AS records "
                "ON records.task_id = tasks.task_id AND records.stage_id = ? "
                "JOIN stage_execution_attempts AS attempts "
                "ON attempts.attempt_id = records.current_attempt_id "
                "WHERE tasks.task_id = ? AND records.idempotency_key = ? "
                "AND attempts.attempt_id = ? AND attempts.idempotency_key = ? "
                "AND attempts.lease_id = ? AND attempts.claimed_version = ?",
                (stage_id, task_id, idempotency_key, attempt_id,
                 idempotency_key, lease_id, claimed_version),
            ).fetchone()
            if (
                row is None or row[0] != claimed_version
                or row[1] != lease_id or row[2] != stage_id
                or _timestamp(row[3]) <= now
                or row[4] != StageExecutionRecordStatus.IN_PROGRESS.value
                or row[5] is not None or row[6] != attempt_id
                or row[7] != StageExecutionRecordStatus.IN_PROGRESS.value
                or row[8] != StageExecutionKind.TOOL.value
            ):
                raise StageExecutionAttemptMismatchError(
                    "Stage invocation permit is invalid"
                )
            stage_state = deserialize_task_state(db.execute(
                "SELECT state_json FROM agent_task_states WHERE task_id = ?",
                (task_id,),
            ).fetchone()[0])
            planned = next(
                item for item in stage_state.plan.stages if item.stage_id == stage_id
            )
            running = next(
                item for item in stage_state.stage_states
                if item.stage_id == stage_id
            )
            if (
                stage_state.task_status is not TaskStatus.RUNNING
                or stage_state.current_stage_id != stage_id
                or running.status is not StageStatus.RUNNING
                or planned.execution_kind is not StageExecutionKind.TOOL
                or planned.tool_id != tool_id
                or derive_stage_idempotency_key(stage_state, planned).value
                != idempotency_key
            ):
                raise StageExecutionAttemptMismatchError(
                    "Stage invocation permit is invalid"
                )
            changed = db.execute(
                "UPDATE stage_execution_attempts SET invocation_admitted_at = ? "
                "WHERE attempt_id = ? AND idempotency_key = ? AND lease_id = ? "
                "AND claimed_version = ? AND status = ? "
                "AND invocation_admitted_at IS NULL",
                (now.isoformat(), attempt_id, idempotency_key, lease_id,
                 claimed_version, StageExecutionRecordStatus.IN_PROGRESS.value),
            )
            if changed.rowcount != 1:
                raise StageExecutionAttemptMismatchError(
                    "Stage invocation permit is invalid"
                )
            db.commit()
            return _stage_invocation_permit_identity(permit, consume=True)
        except (TaskPersistenceError, StageExecutionRecordError):
            if db.in_transaction:
                db.rollback()
            raise
        except Exception:
            if db.in_transaction:
                db.rollback()
            raise StageExecutionAttemptMismatchError(
                "Stage invocation permit is invalid"
            ) from None

    def authorize_model_attempt_invocation(
        self, claimed: ClaimedTaskState,
        lease_authority: ValidatedStageExecutionLease,
        prepared: PreparedStageExecution,
        attempt_authority: ValidatedStageExecutionAttempt, *,
        model_id: str, request_digest: str,
    ) -> ValidatedModelInvocationPermit:
        """Revalidate an admitted model attempt and bind its exact request."""
        if (
            type(prepared) is not PreparedStageExecution
            or prepared.decision is not StageExecutionPreparationDecision.EXECUTE
            or prepared.execution_authority is not attempt_authority
            or not valid_model_id(model_id)
            or not _valid_lease_proof(request_digest)
        ):
            raise StageExecutionAttemptMismatchError("Model attempt is invalid")
        db = self._db()
        try:
            db.execute("BEGIN IMMEDIATE")
            now = self._now()
            persisted, lease, _ = self._live_claim_in_transaction(
                claimed, lease_authority, now=now, require_unexpired=True,
            )
            stage = next(
                item for item in persisted.state.plan.stages
                if item.stage_id == lease.stage_id
            )
            secret = _attempt_authority_secret(attempt_authority, invoked=False)
            attempt = db.execute(
                "SELECT attempts.idempotency_key, attempts.lease_id, "
                "attempts.claimed_version, attempts.status, attempts.attempt_proof, "
                "attempts.invocation_admitted_at, records.current_attempt_id, "
                "records.status FROM stage_execution_attempts AS attempts "
                "JOIN stage_execution_records AS records "
                "ON records.idempotency_key = attempts.idempotency_key "
                "WHERE attempts.attempt_id = ? AND records.task_id = ? "
                "AND records.stage_id = ?",
                (attempt_authority.attempt_id, lease.task_id, lease.stage_id),
            ).fetchone()
            identity = {
                "attempt_id": attempt_authority.attempt_id,
                "key": prepared.record.idempotency_key.value,
                "task_id": lease.task_id, "stage_id": lease.stage_id,
                "lease_id": lease.lease_id,
                "claimed_version": lease.claimed_version,
            }
            if (
                stage.execution_kind is not StageExecutionKind.MODEL
                or secret is None or attempt is None
                or attempt[0] != identity["key"] or attempt[1] != lease.lease_id
                or attempt[2] != lease.claimed_version
                or attempt[3] != StageExecutionRecordStatus.IN_PROGRESS.value
                or not _valid_lease_proof(attempt[4]) or attempt[5] is not None
                or attempt[6] != attempt_authority.attempt_id
                or attempt[7] != StageExecutionRecordStatus.IN_PROGRESS.value
                or not hmac.compare_digest(
                    attempt[4], _attempt_proof(secret=secret, **identity),
                )
            ):
                raise StageExecutionAttemptMismatchError("Model attempt is invalid")
            db.commit()
            registered = _ATTEMPT_SECRETS.get(id(attempt_authority))
            if registered is None or registered[0]() is not attempt_authority:
                raise StageExecutionAttemptMismatchError("Model attempt is invalid")
            _ATTEMPT_SECRETS[id(attempt_authority)] = (
                registered[0], registered[1], True, *registered[3:],
            )
            return _model_invocation_permit(
                attempt_id=attempt_authority.attempt_id,
                idempotency_key=prepared.record.idempotency_key.value,
                task_id=lease.task_id, stage_id=lease.stage_id,
                model_id=model_id, request_digest=request_digest,
                lease_id=lease.lease_id, claimed_version=lease.claimed_version,
            )
        except (TaskPersistenceError, StageExecutionRecordError):
            if db.in_transaction:
                db.rollback()
            raise
        except Exception:
            if db.in_transaction:
                db.rollback()
            raise StageExecutionRecordError("Model invocation could not be authorized") from None

    def consume_model_invocation_permit(
        self, permit: ValidatedModelInvocationPermit,
    ) -> tuple[str, str, str, str, str, str, str, int]:
        """Durably consume a model permit immediately before provider invocation."""
        identity = _model_permit_identity(permit, consume=False)
        (attempt_id, key, task_id, stage_id, model_id, request_digest,
         lease_id, claimed_version) = identity
        db = self._db()
        try:
            db.execute("BEGIN IMMEDIATE")
            now = self._now()
            row = db.execute(
                "SELECT tasks.version, tasks.lease_id, tasks.lease_stage_id, "
                "tasks.lease_expires_at, tasks.state_json, attempts.status, "
                "attempts.invocation_admitted_at, records.current_attempt_id, "
                "records.status, records.execution_kind "
                "FROM agent_task_states AS tasks "
                "JOIN stage_execution_records AS records "
                "ON records.task_id = tasks.task_id AND records.stage_id = ? "
                "JOIN stage_execution_attempts AS attempts "
                "ON attempts.attempt_id = records.current_attempt_id "
                "WHERE tasks.task_id = ? AND records.idempotency_key = ? "
                "AND attempts.attempt_id = ? AND attempts.idempotency_key = ? "
                "AND attempts.lease_id = ? AND attempts.claimed_version = ?",
                (stage_id, task_id, key, attempt_id, key, lease_id, claimed_version),
            ).fetchone()
            if (
                row is None or row[0] != claimed_version or row[1] != lease_id
                or row[2] != stage_id or _timestamp(row[3]) <= now
                or row[5] != StageExecutionRecordStatus.IN_PROGRESS.value
                or row[6] is not None or row[7] != attempt_id
                or row[8] != StageExecutionRecordStatus.IN_PROGRESS.value
                or row[9] != StageExecutionKind.MODEL.value
            ):
                raise StageExecutionAttemptMismatchError("Model invocation permit is invalid")
            state = deserialize_task_state(row[4])
            planned = next(item for item in state.plan.stages if item.stage_id == stage_id)
            running = next(item for item in state.stage_states if item.stage_id == stage_id)
            if (
                state.task_status is not TaskStatus.RUNNING
                or state.current_stage_id != stage_id
                or running.status is not StageStatus.RUNNING
                or planned.execution_kind is not StageExecutionKind.MODEL
                or derive_stage_idempotency_key(state, planned).value != key
            ):
                raise StageExecutionAttemptMismatchError("Model invocation permit is invalid")
            changed = db.execute(
                "UPDATE stage_execution_attempts SET invocation_admitted_at = ? "
                "WHERE attempt_id = ? AND idempotency_key = ? AND lease_id = ? "
                "AND claimed_version = ? AND status = ? "
                "AND invocation_admitted_at IS NULL",
                (now.isoformat(), attempt_id, key, lease_id, claimed_version,
                 StageExecutionRecordStatus.IN_PROGRESS.value),
            )
            if changed.rowcount != 1:
                raise StageExecutionAttemptMismatchError("Model invocation permit is invalid")
            db.commit()
            return _model_permit_identity(permit, consume=True)
        except (TaskPersistenceError, StageExecutionRecordError):
            if db.in_transaction:
                db.rollback()
            raise
        except Exception:
            if db.in_transaction:
                db.rollback()
            raise StageExecutionAttemptMismatchError("Model invocation permit is invalid") from None

    def suspend_claim_for_approval(
        self, claimed: ClaimedTaskState,
        execution_authority: ValidatedStageExecutionLease,
        new_state: AgentTaskState,
    ) -> PersistedTaskState:
        """Atomically suspend a claimed tool stage without admitting an attempt."""
        if type(new_state) is not AgentTaskState:
            raise StageLeaseMismatchError("Stage execution lease is invalid")
        db = self._db()
        try:
            db.execute("BEGIN IMMEDIATE")
            now = self._now()
            persisted, lease, _ = self._live_claim_in_transaction(
                claimed, execution_authority, now=now, require_unexpired=True,
            )
            stage = next(
                item for item in persisted.state.plan.stages
                if item.stage_id == lease.stage_id
            )
            if stage.execution_kind is not StageExecutionKind.TOOL:
                raise StageClaimError("Only tool execution may await approval")
            self._validate_claim_completion(
                persisted.state, lease.stage_id, new_state,
            )
            outcome = next(
                item for item in new_state.stage_states
                if item.stage_id == lease.stage_id
            )
            if (
                outcome.status is not StageStatus.AWAITING_APPROVAL
                or new_state.task_status is not TaskStatus.AWAITING_APPROVAL
                or new_state.current_stage_id != lease.stage_id
                or new_state.approval_request is None
            ):
                raise StageClaimError("Approval suspension state is invalid")
            key = derive_stage_idempotency_key(persisted.state, stage)
            if db.execute(
                "SELECT 1 FROM stage_execution_records "
                "WHERE task_id = ? AND stage_id = ?",
                (lease.task_id, lease.stage_id),
            ).fetchone() is not None:
                raise StageExecutionRecordError(
                    "Stage execution record already exists"
                )
            table_sql = db.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' "
                "AND name = 'stage_execution_records'"
            ).fetchone()
            normalized_table_sql = "".join(
                str(table_sql[0]).lower().split()
            ) if table_sql is not None else ""
            # Databases created before zero-attempt suspension retain a legacy
            # CHECK >= 1 column. The canonical external-attempt count is the
            # side_effect_attempt_count column; keep old schemas writable.
            compatibility_count = (
                0 if "attempt_count>=0" in normalized_table_sql else 1
            )
            db.execute(
                "INSERT INTO stage_execution_records "
                "(idempotency_key, task_id, stage_id, execution_kind, status, "
                "attempt_count, current_attempt_id, first_started_at, "
                "last_started_at, completed_at, result_durability, "
                "side_effect_attempt_count, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, 0, ?, ?)",
                (key.value, lease.task_id, lease.stage_id,
                 stage.execution_kind.value,
                 StageExecutionRecordStatus.AWAITING_APPROVAL.value,
                 compatibility_count,
                 now.isoformat(), now.isoformat(), now.isoformat(),
                 ExecutionResultDurability.NONE.value,
                 now.isoformat(), now.isoformat()),
            )
            payload = serialize_task_state(new_state)
            changed = db.execute(
                "UPDATE agent_task_states SET state_json = ?, version = version + 1, "
                "updated_at = ?, persisted_at = ?, lease_id = NULL, "
                "lease_stage_id = NULL, lease_claimed_at = NULL, "
                "lease_expires_at = NULL, lease_proof = NULL, "
                "lease_migration_state = ? "
                "WHERE task_id = ? AND version = ? AND lease_id = ? "
                "AND lease_stage_id = ?",
                (payload, new_state.updated_at.isoformat(), now.isoformat(),
                 _CURRENT_LEASE_STATE, lease.task_id, persisted.version,
                 lease.lease_id, lease.stage_id),
            )
            if changed.rowcount != 1:
                raise StageLeaseMismatchError("Stage execution lease is invalid")
            db.commit()
            _LEASE_SECRETS.pop(id(execution_authority), None)
        except (TaskPersistenceError, StageExecutionRecordError):
            db.rollback()
            raise
        except Exception:
            db.rollback()
            raise TaskPersistenceError(
                "Approval suspension could not be stored"
            ) from None
        return PersistedTaskState(
            deserialize_task_state(payload), persisted.version + 1, now,
        )

    def record_stage_execution_outcome(
        self, claimed: ClaimedTaskState,
        lease_authority: ValidatedStageExecutionLease,
        prepared: PreparedStageExecution,
        attempt_authority: ValidatedStageExecutionAttempt,
        new_state: AgentTaskState,
        *, invocation_receipt: object | None = None,
        result_content_digest: str | None = None,
    ) -> StageExecutionRecord:
        """Persist one terminal attempt outcome before lease completion."""
        if (
            type(prepared) is not PreparedStageExecution
            or prepared.decision is not StageExecutionPreparationDecision.EXECUTE
            or prepared.execution_authority is not attempt_authority
            or type(new_state) is not AgentTaskState
        ):
            raise StageExecutionAttemptMismatchError(
                "Stage execution attempt is invalid"
            )
        db = self._db()
        try:
            db.execute("BEGIN IMMEDIATE")
            now = self._now()
            persisted, lease, _ = self._live_claim_in_transaction(
                claimed, lease_authority, now=now, require_unexpired=False,
            )
            stage_definition = next(
                item for item in persisted.state.plan.stages
                if item.stage_id == lease.stage_id
            )
            secret = _attempt_authority_secret(
                attempt_authority,
                invoked=True,
            )
            attempt = db.execute(
                "SELECT idempotency_key, lease_id, claimed_version, status, "
                "attempt_proof FROM stage_execution_attempts WHERE attempt_id = ?",
                (attempt_authority.attempt_id,),
            ).fetchone()
            identity = {
                "attempt_id": attempt_authority.attempt_id,
                "key": prepared.record.idempotency_key.value,
                "task_id": lease.task_id, "stage_id": lease.stage_id,
                "lease_id": lease.lease_id,
                "claimed_version": lease.claimed_version,
            }
            if (
                secret is None or attempt is None
                or attempt[0] != prepared.record.idempotency_key.value
                or attempt[1] != lease.lease_id or attempt[2] != lease.claimed_version
                or attempt[3] != StageExecutionRecordStatus.IN_PROGRESS.value
                or not _valid_lease_proof(attempt[4])
                or not hmac.compare_digest(
                    attempt[4], _attempt_proof(secret=secret, **identity),
                )
                or (attempt_authority.idempotency_key, attempt_authority.task_id,
                    attempt_authority.stage_id)
                != (prepared.record.idempotency_key.value, lease.task_id, lease.stage_id)
            ):
                raise StageExecutionAttemptMismatchError(
                    "Stage execution attempt is invalid"
                )
            self._validate_claim_completion(persisted.state, lease.stage_id, new_state)
            index = next(
                i for i, item in enumerate(new_state.stage_states)
                if item.stage_id == lease.stage_id
            )
            outcome = new_state.stage_states[index]
            if stage_definition.execution_kind is StageExecutionKind.TOOL:
                from sovereign_api.tool_execution import (
                    _consume_trusted_tool_invocation_receipt,
                )
                if stage_definition.tool_id is None:
                    raise StageExecutionAttemptMismatchError(
                        "Stage execution attempt is invalid"
                    )
                trusted = _consume_trusted_tool_invocation_receipt(
                    invocation_receipt,
                    task_id=lease.task_id, stage_id=lease.stage_id,
                    tool_id=stage_definition.tool_id,
                    attempt_id=attempt_authority.attempt_id,
                    idempotency_key=prepared.record.idempotency_key.value,
                    lease_id=lease.lease_id,
                    claimed_version=lease.claimed_version,
                )
                trusted_result = trusted.result
                trusted_error = trusted.error_code
                expected_error = trusted_error
                if expected_error is None:
                    if trusted_result is None:
                        expected_error = "tool_invalid_result"
                    elif trusted_result.status is ToolResultStatus.FAILED:
                        expected_error = "tool_failed"
                    elif stage_definition.tool_id == WORKSPACE_READ_FILE_TOOL_ID:
                        if (
                            trusted_result.text_content is None
                            or trusted_result.output_reference is not None
                        ):
                            expected_error = "tool_invalid_result"
                        else:
                            try:
                                size = len(trusted_result.text_content.encode(
                                    "utf-8", errors="strict",
                                ))
                            except UnicodeEncodeError:
                                expected_error = "tool_output_too_large"
                            else:
                                if size > MAX_STAGE_OUTPUT_BYTES:
                                    expected_error = "tool_output_too_large"
                    elif stage_definition.tool_id == WORKSPACE_WRITE_ARTIFACT_TOOL_ID:
                        if (
                            trusted_result.output_reference is None
                            or trusted_result.text_content is not None
                            or not valid_artifact_reference(
                                trusted_result.output_reference
                            )
                        ):
                            expected_error = "tool_invalid_result"
                    else:
                        expected_error = "tool_invalid_result"
                if expected_error is not None:
                    if (
                        outcome.status is not StageStatus.FAILED
                        or outcome.error_code != expected_error
                        or outcome.selected_tool_id != stage_definition.tool_id
                        or outcome.output_reference is not None
                    ):
                        raise StageExecutionAttemptMismatchError(
                            "Tool outcome does not match invocation receipt"
                        )
                elif stage_definition.tool_id == WORKSPACE_READ_FILE_TOOL_ID:
                    expected_digest = stage_text_digest(trusted_result.text_content)
                    if (
                        outcome.status is not StageStatus.COMPLETED
                        or outcome.output_kind is not StageOutputKind.TEXT
                        or outcome.selected_tool_id != stage_definition.tool_id
                        or outcome.output_reference
                        != trusted.stage_output_reference
                        or result_content_digest != expected_digest
                        or trusted.result_content_digest != expected_digest
                    ):
                        raise StageExecutionAttemptMismatchError(
                            "Tool outcome does not match invocation receipt"
                        )
                elif (
                    outcome.status is not StageStatus.COMPLETED
                    or outcome.output_kind is not StageOutputKind.ARTIFACT
                    or outcome.selected_tool_id != stage_definition.tool_id
                    or outcome.output_reference != trusted_result.output_reference
                    or trusted.stage_output_reference
                    != trusted_result.output_reference
                    or result_content_digest != stage_text_digest(
                        stage_definition.tool_arguments.get("content")
                    )
                    or trusted.result_content_digest != result_content_digest
                ):
                    raise StageExecutionAttemptMismatchError(
                        "Tool outcome does not match invocation receipt"
                    )
            else:
                from sovereign_api.agent_stage_execution import _model_receipt_outcome
                if outcome.selected_model_id is None:
                    raise StageExecutionAttemptMismatchError(
                        "Model outcome requires a trusted invocation receipt"
                    )
                trusted_content, trusted_error = _model_receipt_outcome(
                    invocation_receipt,
                    task_id=lease.task_id, stage_id=lease.stage_id,
                    model_id=outcome.selected_model_id,
                    attempt_id=attempt_authority.attempt_id,
                    idempotency_key=prepared.record.idempotency_key.value,
                    request_digest=invocation_receipt.request_digest,
                    lease_id=lease.lease_id,
                    claimed_version=lease.claimed_version,
                    consume=True,
                )
                if trusted_error is None:
                    if (
                        trusted_content is None
                        or outcome.status is not StageStatus.COMPLETED
                        or outcome.output_kind is not StageOutputKind.TEXT
                        or result_content_digest != stage_text_digest(trusted_content)
                    ):
                        raise StageExecutionAttemptMismatchError(
                            "Model outcome does not match invocation receipt"
                        )
                elif (
                    outcome.status is not StageStatus.FAILED
                    or outcome.error_code != trusted_error
                    or outcome.safe_message != "Stage execution failed"
                    or outcome.output_reference is not None
                ):
                    raise StageExecutionAttemptMismatchError(
                        "Model outcome does not match invocation receipt"
                    )
            if outcome.status is StageStatus.COMPLETED:
                status = StageExecutionRecordStatus.SUCCEEDED
                reference = outcome.output_reference
                output_kind = outcome.output_kind.value
                model_id = outcome.selected_model_id
                tool_id = outcome.selected_tool_id
                if outcome.output_kind is StageOutputKind.TEXT:
                    durability = ExecutionResultDurability.EPHEMERAL
                    if (
                        type(result_content_digest) is not str
                        or not _valid_lease_proof(result_content_digest)
                    ):
                        raise StageExecutionRecordError(
                            "Stage text result digest is invalid"
                        )
                    content_digest = result_content_digest
                elif outcome.output_kind is StageOutputKind.ARTIFACT:
                    durability = ExecutionResultDurability.DURABLE
                    if (
                        type(result_content_digest) is not str
                        or not _valid_lease_proof(result_content_digest)
                    ):
                        raise StageExecutionRecordError(
                            "Artifact execution result digest is invalid"
                        )
                    content_digest = result_content_digest
                else:
                    raise StageExecutionRecordError(
                        "Stage execution result durability is invalid"
                    )
                current_attempt_id = attempt_authority.attempt_id
                terminal_error_code = terminal_safe_message = None
            elif outcome.status is StageStatus.AWAITING_APPROVAL:
                raise StageExecutionAttemptMismatchError(
                    "Approval cannot consume a physical execution attempt"
                )
            else:
                status = StageExecutionRecordStatus.FAILED
                reference = output_kind = content_digest = None
                model_id = outcome.selected_model_id
                tool_id = outcome.selected_tool_id
                terminal_error_code = outcome.error_code
                terminal_safe_message = outcome.safe_message
                durability = ExecutionResultDurability.NONE
                current_attempt_id = attempt_authority.attempt_id
            changed_attempt = db.execute(
                "UPDATE stage_execution_attempts SET status = ?, completed_at = ? "
                "WHERE attempt_id = ? AND status = ?",
                (status.value, now.isoformat(), attempt_authority.attempt_id,
                 StageExecutionRecordStatus.IN_PROGRESS.value),
            )
            changed_record = db.execute(
                "UPDATE stage_execution_records SET status = ?, completed_at = ?, "
                "safe_result_reference = ?, output_kind = ?, selected_model_id = ?, "
                "selected_tool_id = ?, result_durability = ?, "
                "result_content_digest = ?, terminal_error_code = ?, "
                "terminal_safe_message = ?, current_attempt_id = ?, updated_at = ? "
                "WHERE idempotency_key = ? AND current_attempt_id = ? AND status = ?",
                (status.value, now.isoformat(), reference, output_kind, model_id,
                 tool_id, durability.value, content_digest,
                 terminal_error_code, terminal_safe_message, current_attempt_id,
                 now.isoformat(),
                 prepared.record.idempotency_key.value,
                 attempt_authority.attempt_id,
                 StageExecutionRecordStatus.IN_PROGRESS.value),
            )
            if changed_attempt.rowcount != 1 or changed_record.rowcount != 1:
                raise StageExecutionAttemptMismatchError(
                    "Stage execution attempt is invalid"
                )
            row = db.execute(
                f"SELECT {_EXECUTION_RECORD_COLUMNS} FROM stage_execution_records "
                "WHERE idempotency_key = ?",
                (prepared.record.idempotency_key.value,),
            ).fetchone()
            record = _execution_record_from_row(row)
            db.commit()
            _ATTEMPT_SECRETS.pop(id(attempt_authority), None)
            return record
        except (TaskPersistenceError, StageExecutionRecordError):
            db.rollback()
            raise
        except Exception:
            db.rollback()
            raise StageExecutionRecordError(
                "Stage execution outcome could not be stored"
            ) from None

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
            stage_index = next(
                index for index, item in enumerate(claimed.plan.stages)
                if item.stage_id == stage_id
            )
            stage_definition = claimed.plan.stages[stage_index]
            stage_outcome = new_state.stage_states[stage_index]
            if stage_definition.execution_kind is StageExecutionKind.TOOL:
                execution_row = db.execute(
                    f"SELECT {_EXECUTION_RECORD_COLUMNS} "
                    "FROM stage_execution_records WHERE task_id = ? AND stage_id = ?",
                    (task_id, stage_id),
                ).fetchone()
                execution_record = (
                    _execution_record_from_row(execution_row)
                    if execution_row is not None else None
                )
                canonical_key = derive_stage_idempotency_key(
                    claimed, stage_definition,
                )
                if execution_record is None:
                    if (
                        stage_outcome.status is not StageStatus.FAILED
                        or stage_outcome.error_code not in (
                            "tool_denied", "tool_unavailable",
                        )
                        or stage_outcome.safe_message != "Stage execution failed"
                        or stage_outcome.selected_tool_id != stage_definition.tool_id
                    ):
                        raise StageExecutionAttemptMismatchError(
                            "Tool completion requires a trusted execution receipt"
                        )
                elif (
                    execution_record.idempotency_key != canonical_key
                    or execution_record.selected_tool_id != stage_definition.tool_id
                ):
                    raise StageExecutionAttemptMismatchError(
                        "Tool completion requires a trusted execution receipt"
                    )
                elif execution_record.status is StageExecutionRecordStatus.SUCCEEDED:
                    if (
                        stage_outcome.status is not StageStatus.COMPLETED
                        or execution_record.safe_result_reference
                        != stage_outcome.output_reference
                        or execution_record.output_kind is not stage_outcome.output_kind
                        or stage_outcome.selected_tool_id != stage_definition.tool_id
                    ):
                        raise StageExecutionAttemptMismatchError(
                            "Tool completion requires a trusted execution receipt"
                        )
                elif execution_record.status is StageExecutionRecordStatus.FAILED:
                    if (
                        stage_outcome.status is not StageStatus.FAILED
                        or stage_outcome.error_code
                        != execution_record.terminal_error_code
                        or stage_outcome.safe_message
                        != execution_record.terminal_safe_message
                        or stage_outcome.selected_tool_id
                        != execution_record.selected_tool_id
                    ):
                        raise StageExecutionAttemptMismatchError(
                            "Tool failure requires a trusted execution receipt"
                        )
                else:
                    raise StageExecutionAttemptMismatchError(
                        "Tool completion requires a trusted execution receipt"
                    )
            else:
                execution_row = db.execute(
                    f"SELECT {_EXECUTION_RECORD_COLUMNS} "
                    "FROM stage_execution_records WHERE task_id = ? AND stage_id = ?",
                    (task_id, stage_id),
                ).fetchone()
                execution_record = (
                    _execution_record_from_row(execution_row)
                    if execution_row is not None else None
                )
                if execution_record is None:
                    valid_outcome = False
                elif (
                    execution_record.idempotency_key != derive_stage_idempotency_key(
                        claimed, stage_definition,
                    )
                    or execution_record.execution_kind is not StageExecutionKind.MODEL
                    or execution_record.selected_tool_id is not None
                ):
                    raise StageExecutionAttemptMismatchError(
                        "Model completion requires a trusted execution receipt"
                    )
                elif execution_record.status is StageExecutionRecordStatus.SUCCEEDED:
                    self._require_terminal_attempt_evidence(
                        db, execution_record,
                        expected_status=StageExecutionRecordStatus.SUCCEEDED,
                    )
                    valid_outcome = (
                        stage_outcome.status is StageStatus.COMPLETED
                        and stage_outcome.selected_model_id
                        == execution_record.selected_model_id
                        and stage_outcome.selected_tool_id is None
                        and stage_outcome.output_reference
                        == execution_record.safe_result_reference
                        and stage_outcome.output_kind is execution_record.output_kind
                        and execution_record.result_durability
                        is ExecutionResultDurability.EPHEMERAL
                        and execution_record.result_content_digest is not None
                    )
                elif execution_record.status is StageExecutionRecordStatus.FAILED:
                    self._require_terminal_attempt_evidence(
                        db, execution_record,
                        expected_status=StageExecutionRecordStatus.FAILED,
                    )
                    valid_outcome = (
                        stage_outcome.status is StageStatus.FAILED
                        and stage_outcome.selected_model_id
                        == execution_record.selected_model_id
                        and stage_outcome.selected_tool_id is None
                        and stage_outcome.error_code
                        == execution_record.terminal_error_code
                        and stage_outcome.safe_message
                        == execution_record.terminal_safe_message
                    )
                else:
                    valid_outcome = False
                if not valid_outcome:
                    raise StageExecutionAttemptMismatchError(
                        "Model completion requires a trusted execution receipt"
                    )
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
        except (TaskPersistenceError, StageExecutionRecordError):
            db.rollback()
            raise
        except Exception:
            db.rollback()
            raise TaskPersistenceError("Claim completion could not be stored") from None
        return PersistedTaskState(
            deserialize_task_state(payload), expected_version + 1, now,
        )
