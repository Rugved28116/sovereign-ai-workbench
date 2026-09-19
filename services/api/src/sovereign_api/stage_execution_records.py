"""Stable logical identities and safe metadata for persisted stage attempts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from types import MappingProxyType

from sovereign_api.agent_task_state import AgentTaskState, StageOutputKind
from sovereign_api.artifact_reference import valid_artifact_reference
from sovereign_api.errors import SovereignAPIError
from sovereign_api.task_planning import StageExecutionKind, TaskStage
from sovereign_api.workspace_read_file import (
    FILESYSTEM_READ_PERMISSION, WORKSPACE_READ_FILE_TOOL_ID, WORKSPACE_READ_OPERATION,
)
from sovereign_api.workspace_write_artifact import (
    ARTIFACT_WRITE_PERMISSION, WORKSPACE_WRITE_ARTIFACT_TOOL_ID,
    WORKSPACE_WRITE_OPERATION,
)


IDEMPOTENCY_SCHEMA_VERSION = 1
_TOOL_OPERATIONS = {
    WORKSPACE_READ_FILE_TOOL_ID: WORKSPACE_READ_OPERATION,
    WORKSPACE_WRITE_ARTIFACT_TOOL_ID: WORKSPACE_WRITE_OPERATION,
}
_TOOL_PERMISSIONS = {
    WORKSPACE_READ_FILE_TOOL_ID: (FILESYSTEM_READ_PERMISSION.identifier,),
    WORKSPACE_WRITE_ARTIFACT_TOOL_ID: (ARTIFACT_WRITE_PERMISSION.identifier,),
}


class StageExecutionRecordError(SovereignAPIError):
    code = "stage_execution_record_error"


class StageExecutionAlreadyInProgressError(StageExecutionRecordError):
    code = "stage_execution_already_in_progress"


class StageExecutionPreviouslyFailedError(StageExecutionRecordError):
    code = "stage_execution_previously_failed"


class StageExecutionAwaitingApprovalError(StageExecutionRecordError):
    code = "stage_execution_awaiting_approval"


class StageExecutionOutcomeUnknownError(StageExecutionRecordError):
    code = "stage_execution_outcome_unknown"


class SuccessfulResultUnavailableError(StageExecutionRecordError):
    code = "successful_result_unavailable"


class StageExecutionAttemptMismatchError(StageExecutionRecordError):
    code = "stage_execution_attempt_mismatch"


class StageExecutionRecordStatus(StrEnum):
    IN_PROGRESS = "in_progress"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNKNOWN = "unknown"
    AWAITING_APPROVAL = "awaiting_approval"


class ExecutionResultDurability(StrEnum):
    DURABLE = "durable"
    EPHEMERAL = "ephemeral"
    NONE = "none"


class StageExecutionPreparationDecision(StrEnum):
    EXECUTE = "execute"
    KNOWN_SUCCESS = "known_success"


@dataclass(frozen=True, slots=True)
class StageIdempotencyKey:
    value: str

    def __post_init__(self) -> None:
        if (
            type(self.value) is not str or len(self.value) != 64
            or any(character not in "0123456789abcdef" for character in self.value)
        ):
            raise StageExecutionRecordError("Stage idempotency key is invalid")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class StageExecutionRecord:
    idempotency_key: StageIdempotencyKey
    task_id: str
    stage_id: str
    execution_kind: StageExecutionKind
    status: StageExecutionRecordStatus
    attempt_count: int
    current_attempt_id: str | None
    first_started_at: datetime
    last_started_at: datetime
    completed_at: datetime | None
    safe_result_reference: str | None
    output_kind: StageOutputKind | None
    result_durability: ExecutionResultDurability
    result_content_digest: str | None
    selected_model_id: str | None
    selected_tool_id: str | None
    terminal_error_code: str | None
    terminal_safe_message: str | None
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        times = (self.first_started_at, self.last_started_at, self.created_at,
                 self.updated_at)
        if (
            type(self.idempotency_key) is not StageIdempotencyKey
            or any(type(value) is not str or not value.strip()
                   for value in (self.task_id, self.stage_id))
            or type(self.execution_kind) is not StageExecutionKind
            or type(self.status) is not StageExecutionRecordStatus
            or type(self.attempt_count) is not int or self.attempt_count < 0
            or type(self.result_durability) is not ExecutionResultDurability
            or any(type(value) is not datetime or value.tzinfo is None
                   or value.utcoffset() != timedelta(0) for value in times)
            or (self.completed_at is not None and (
                type(self.completed_at) is not datetime
                or self.completed_at.tzinfo is None
                or self.completed_at.utcoffset() != timedelta(0)
            ))
            or self.last_started_at < self.first_started_at
            or self.updated_at < self.created_at
        ):
            raise StageExecutionRecordError("Stage execution record is invalid")
        for value in (
            self.current_attempt_id, self.safe_result_reference,
            self.result_content_digest, self.selected_model_id, self.selected_tool_id,
            self.terminal_error_code, self.terminal_safe_message,
        ):
            if value is not None and (type(value) is not str or not value.strip()):
                raise StageExecutionRecordError("Stage execution record is invalid")
        if self.current_attempt_id is not None and (
            len(self.current_attempt_id) != 32
            or any(character not in "0123456789abcdef"
                   for character in self.current_attempt_id)
        ):
            raise StageExecutionRecordError("Stage execution record is invalid")
        if self.output_kind is not None and type(self.output_kind) is not StageOutputKind:
            raise StageExecutionRecordError("Stage execution record is invalid")
        if self.result_content_digest is not None and (
            len(self.result_content_digest) != 64
            or any(character not in "0123456789abcdef"
                   for character in self.result_content_digest)
        ):
            raise StageExecutionRecordError("Stage execution record is invalid")
        if self.status is StageExecutionRecordStatus.IN_PROGRESS:
            if (
                self.current_attempt_id is None or self.completed_at is not None
                or self.attempt_count != 1
                or self.result_durability is not ExecutionResultDurability.NONE
                or self.result_content_digest is not None
                or self.terminal_error_code is not None
                or self.terminal_safe_message is not None
            ):
                raise StageExecutionRecordError("Stage execution record is invalid")
        elif self.status is StageExecutionRecordStatus.SUCCEEDED:
            if (
                self.current_attempt_id is None or self.completed_at is None
                or self.safe_result_reference is None or self.output_kind is None
            ):
                raise StageExecutionRecordError("Stage execution record is invalid")
            if self.attempt_count < 1:
                raise StageExecutionRecordError("Stage execution record is invalid")
            if self.result_durability is ExecutionResultDurability.EPHEMERAL:
                if (
                    self.output_kind is not StageOutputKind.TEXT
                    or self.result_content_digest is None
                ):
                    raise StageExecutionRecordError("Stage execution record is invalid")
            elif self.result_durability is ExecutionResultDurability.DURABLE:
                if (
                    self.output_kind is not StageOutputKind.ARTIFACT
                    or not valid_artifact_reference(self.safe_result_reference)
                ):
                    raise StageExecutionRecordError("Stage execution record is invalid")
            elif self.result_content_digest is not None:
                raise StageExecutionRecordError("Stage execution record is invalid")
            if self.execution_kind is StageExecutionKind.MODEL:
                if self.selected_model_id is None or self.selected_tool_id is not None:
                    raise StageExecutionRecordError("Stage execution record is invalid")
            elif self.selected_tool_id is None or self.selected_model_id is not None:
                raise StageExecutionRecordError("Stage execution record is invalid")
        elif self.status is StageExecutionRecordStatus.AWAITING_APPROVAL:
            if (
                self.attempt_count != 0 or self.current_attempt_id is not None
                or self.completed_at is None or self.safe_result_reference is not None
                or self.output_kind is not None
                or self.result_durability is not ExecutionResultDurability.NONE
                or self.result_content_digest is not None
                or self.selected_model_id is not None or self.selected_tool_id is not None
                or self.terminal_error_code is not None
                or self.terminal_safe_message is not None
            ):
                raise StageExecutionRecordError("Stage execution record is invalid")
        elif (
            self.completed_at is None or self.attempt_count < 1
            or self.safe_result_reference is not None or self.output_kind is not None
            or self.result_durability is not ExecutionResultDurability.NONE
            or self.result_content_digest is not None
        ):
            raise StageExecutionRecordError("Stage execution record is invalid")
        if self.status is StageExecutionRecordStatus.SUCCEEDED and (
            self.terminal_error_code is not None
            or self.terminal_safe_message is not None
        ):
            raise StageExecutionRecordError("Stage execution record is invalid")
        if self.status is StageExecutionRecordStatus.FAILED and (
            (self.terminal_error_code is None)
            != (self.terminal_safe_message is None)
        ):
            raise StageExecutionRecordError("Stage execution record is invalid")


def _plain_json(value: object) -> object:
    if type(value) in (dict, MappingProxyType):
        return {key: _plain_json(item) for key, item in value.items()}
    if type(value) in (tuple, list):
        return [_plain_json(item) for item in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise ValueError


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        _plain_json(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stage_text_digest(text: str) -> str:
    if type(text) is not str:
        raise StageExecutionRecordError("Stage text result is invalid")
    try:
        return hashlib.sha256(text.encode("utf-8", errors="strict")).hexdigest()
    except UnicodeEncodeError:
        raise StageExecutionRecordError("Stage text result is invalid") from None


def derive_stage_idempotency_key(
    task: AgentTaskState, stage: TaskStage,
) -> StageIdempotencyKey:
    """Derive a non-secret stable key from validated logical stage semantics."""
    if type(task) is not AgentTaskState or type(stage) is not TaskStage:
        raise StageExecutionRecordError("Stage execution identity is invalid")
    try:
        index = next(
            position for position, planned in enumerate(task.plan.stages)
            if planned.stage_id == stage.stage_id
        )
        if task.plan.stages[index] != stage:
            raise ValueError
        base: dict[str, object] = {
            "schema_version": IDEMPOTENCY_SCHEMA_VERSION,
            "task_id": task.task_id,
            "stage_id": stage.stage_id,
            "execution_kind": stage.execution_kind.value,
            "stage_type": stage.stage_type.value,
        }
        if stage.execution_kind is StageExecutionKind.MODEL:
            base.update({
                "required_capabilities": list(stage.required_capabilities),
                "task_prompt_sha256": hashlib.sha256(
                    task.original_prompt.encode("utf-8", errors="strict")
                ).hexdigest(),
                "previous_output_reference": (
                    task.stage_states[index - 1].output_reference if index else None
                ),
            })
        elif stage.execution_kind is StageExecutionKind.TOOL:
            operation = _TOOL_OPERATIONS.get(stage.tool_id)
            if operation is None or stage.tool_arguments is None:
                raise ValueError
            base.update({
                "tool_id": stage.tool_id,
                "operation": operation,
                "required_permissions": sorted(_TOOL_PERMISSIONS[stage.tool_id]),
                "tool_arguments_sha256": _canonical_digest(stage.tool_arguments),
            })
        else:
            raise ValueError
        return StageIdempotencyKey(_canonical_digest(base))
    except Exception:
        raise StageExecutionRecordError("Stage execution identity is invalid") from None
