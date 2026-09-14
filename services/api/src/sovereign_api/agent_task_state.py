"""Immutable provider-neutral state for future multi-stage agent tasks."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from enum import StrEnum

from pydantic import TypeAdapter

from sovereign_api.errors import (
    InvalidExecutionStateError,
    InvalidStepTransitionError,
    InvalidTaskTransitionError,
)
from sovereign_api.prompt_validation import Prompt
from sovereign_api.artifact_reference import valid_artifact_reference
from sovereign_api.registry.models import valid_model_id
from sovereign_api.task_classification import TaskClass
from sovereign_api.task_planning import StageExecutionKind, TaskPlan, TaskStageType
from sovereign_api.tool_approval import ApprovalRequest

_PROMPT_VALIDATOR = TypeAdapter(Prompt)
_UTC_OFFSET = timedelta(0)


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    AWAITING_APPROVAL = "awaiting_approval"


class StageStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"
    AWAITING_APPROVAL = "awaiting_approval"


class StageOutputKind(StrEnum):
    TEXT = "text"
    ARTIFACT = "artifact"


@dataclass(frozen=True, slots=True)
class StageExecutionState:
    """Immutable lifecycle state for one logical plan stage."""

    stage_id: str
    stage_type: TaskStageType
    required_capabilities: tuple[str, ...]
    status: StageStatus = StageStatus.PENDING
    selected_model_id: str | None = None
    output_reference: str | None = None
    error_code: str | None = None
    safe_message: str | None = None
    execution_kind: StageExecutionKind = StageExecutionKind.MODEL
    selected_tool_id: str | None = None
    output_kind: StageOutputKind | None = None

    def __post_init__(self) -> None:
        capabilities = tuple(self.required_capabilities)
        object.__setattr__(self, "required_capabilities", capabilities)

        if type(self.stage_id) is not str or not self.stage_id.strip():
            raise InvalidExecutionStateError("stage_id must be a non-empty string")
        if type(self.stage_type) is not TaskStageType:
            raise InvalidExecutionStateError("stage_type must be a TaskStageType")
        if any(
            type(capability) is not str or not capability
            for capability in capabilities
        ) or (self.execution_kind is StageExecutionKind.MODEL and not capabilities):
            raise InvalidExecutionStateError(
                "stage capabilities must be non-empty strings"
            )
        if type(self.status) is not StageStatus:
            raise InvalidExecutionStateError("status must be a StageStatus")
        if type(self.execution_kind) is not StageExecutionKind:
            raise InvalidExecutionStateError("stage execution kind is invalid")
        if self.execution_kind is StageExecutionKind.TOOL:
            if self.stage_type is not TaskStageType.TOOL or capabilities or self.selected_model_id is not None:
                raise InvalidExecutionStateError("tool stage identity is invalid")
        elif self.stage_type is TaskStageType.TOOL or self.selected_tool_id is not None:
            raise InvalidExecutionStateError("model stage identity is invalid")
        if self.output_kind is not None and type(self.output_kind) is not StageOutputKind:
            raise InvalidExecutionStateError("stage output kind is invalid")
        if self.execution_kind is StageExecutionKind.MODEL and self.output_kind is StageOutputKind.ARTIFACT:
            raise InvalidExecutionStateError("model stage cannot contain an artifact reference")
        if self.output_kind is StageOutputKind.ARTIFACT and not valid_artifact_reference(self.output_reference):
            raise InvalidExecutionStateError("artifact reference is invalid")
        for name, value in (
            ("selected_model_id", self.selected_model_id),
            ("selected_tool_id", self.selected_tool_id),
            ("output_reference", self.output_reference),
            ("error_code", self.error_code),
            ("safe_message", self.safe_message),
        ):
            if value is not None and (
                type(value) is not str or not value.strip()
            ):
                raise InvalidExecutionStateError(
                    f"{name} must be None or a non-empty string"
                )
        if self.selected_model_id is not None and not valid_model_id(self.selected_model_id):
            raise InvalidExecutionStateError("selected_model_id is invalid")

        if self.status is StageStatus.COMPLETED:
            if self.output_reference is None:
                raise InvalidExecutionStateError(
                    "a completed stage requires an output reference"
                )
            if self.execution_kind is StageExecutionKind.TOOL and self.selected_tool_id is None:
                raise InvalidExecutionStateError("a completed tool stage requires tool identity")
            if self.error_code is not None or self.safe_message is not None:
                raise InvalidExecutionStateError(
                    "a completed stage cannot contain failure information"
                )
        elif self.status is StageStatus.FAILED:
            if self.error_code is None or self.safe_message is None:
                raise InvalidExecutionStateError(
                    "a failed stage requires an error code and safe message"
                )
            if self.output_reference is not None or self.output_kind is not None:
                raise InvalidExecutionStateError(
                    "a failed stage cannot contain an output reference"
                )
        elif (
            self.output_reference is not None or self.output_kind is not None
            or self.error_code is not None
            or self.safe_message is not None
        ):
            raise InvalidExecutionStateError(
                "output and failure fields require their matching terminal state"
            )
        if self.status in (StageStatus.PENDING, StageStatus.SKIPPED) and (
            self.selected_model_id is not None or self.selected_tool_id is not None
        ):
            raise InvalidExecutionStateError(
                "pending and skipped stages cannot contain a selected model"
            )

    def start(self, *, selected_model_id: str | None = None, selected_tool_id: str | None = None) -> StageExecutionState:
        self._require_status(StageStatus.PENDING, StageStatus.RUNNING)
        return replace(
            self,
            status=StageStatus.RUNNING,
            selected_model_id=selected_model_id,
            selected_tool_id=selected_tool_id,
        )

    def complete(
        self, *, output_reference: str, selected_model_id: str | None = None,
        selected_tool_id: str | None = None, output_kind: StageOutputKind = StageOutputKind.TEXT,
    ) -> StageExecutionState:
        self._require_status(StageStatus.RUNNING, StageStatus.COMPLETED)
        self._require_same_selected_model(selected_model_id)
        self._require_same_selected_tool(selected_tool_id)
        return replace(
            self,
            status=StageStatus.COMPLETED,
            output_reference=output_reference,
            output_kind=output_kind,
            selected_model_id=(
                self.selected_model_id if selected_model_id is None else selected_model_id
            ),
            selected_tool_id=(
                self.selected_tool_id if selected_tool_id is None else selected_tool_id
            ),
        )

    def fail(
        self, *, error_code: str, safe_message: str,
        selected_model_id: str | None = None,
        selected_tool_id: str | None = None,
    ) -> StageExecutionState:
        if self.status not in (StageStatus.RUNNING, StageStatus.AWAITING_APPROVAL):
            raise InvalidStepTransitionError("stage cannot fail from this status")
        self._require_same_selected_model(selected_model_id)
        self._require_same_selected_tool(selected_tool_id)
        return replace(
            self,
            status=StageStatus.FAILED,
            error_code=error_code,
            safe_message=safe_message,
            selected_model_id=(
                self.selected_model_id if selected_model_id is None else selected_model_id
            ),
            selected_tool_id=(
                self.selected_tool_id if selected_tool_id is None else selected_tool_id
            ),
        )

    def await_approval(self, *, selected_tool_id: str) -> StageExecutionState:
        self._require_status(StageStatus.RUNNING, StageStatus.AWAITING_APPROVAL)
        if self.execution_kind is not StageExecutionKind.TOOL:
            raise InvalidStepTransitionError("only a tool stage may await approval")
        self._require_same_selected_tool(selected_tool_id)
        return replace(self, status=StageStatus.AWAITING_APPROVAL,
                       selected_tool_id=selected_tool_id)

    def resume_approval(self) -> StageExecutionState:
        self._require_status(StageStatus.AWAITING_APPROVAL, StageStatus.RUNNING)
        return replace(self, status=StageStatus.RUNNING)

    def cancel(self) -> StageExecutionState:
        self._require_status(StageStatus.RUNNING, StageStatus.CANCELLED)
        return replace(self, status=StageStatus.CANCELLED)

    def skip(self) -> StageExecutionState:
        self._require_status(StageStatus.PENDING, StageStatus.SKIPPED)
        return replace(self, status=StageStatus.SKIPPED)

    def _require_status(
        self, expected: StageStatus, destination: StageStatus
    ) -> None:
        if self.status is not expected:
            raise InvalidStepTransitionError(
                f"cannot transition stage from {self.status.value} "
                f"to {destination.value}"
            )

    def _require_same_selected_model(self, selected_model_id: str | None) -> None:
        if (
            self.selected_model_id is not None
            and selected_model_id is not None
            and selected_model_id != self.selected_model_id
        ):
            raise InvalidStepTransitionError("selected model ID cannot change")

    def _require_same_selected_tool(self, selected_tool_id: str | None) -> None:
        if self.selected_tool_id is not None and selected_tool_id is not None and selected_tool_id != self.selected_tool_id:
            raise InvalidStepTransitionError("selected tool ID cannot change")


@dataclass(frozen=True, slots=True)
class AgentTaskState:
    """Immutable control-plane state derived exactly from a TaskPlan."""

    task_id: str
    original_prompt: str
    task_class: TaskClass
    required_capabilities: tuple[str, ...]
    plan: TaskPlan
    task_status: TaskStatus
    current_stage_id: str | None
    stage_states: tuple[StageExecutionState, ...]
    created_at: datetime
    updated_at: datetime
    approval_request: ApprovalRequest | None = None

    def __post_init__(self) -> None:
        capabilities = tuple(self.required_capabilities)
        stages = tuple(self.stage_states)
        object.__setattr__(self, "required_capabilities", capabilities)
        object.__setattr__(self, "stage_states", stages)
        object.__setattr__(
            self,
            "original_prompt",
            _PROMPT_VALIDATOR.validate_python(self.original_prompt),
        )

        if type(self.task_id) is not str or not self.task_id.strip():
            raise InvalidExecutionStateError("task_id must be a non-empty string")
        if type(self.task_class) is not TaskClass:
            raise InvalidExecutionStateError("task_class must be a TaskClass")
        if type(self.plan) is not TaskPlan:
            raise InvalidExecutionStateError("plan must be a TaskPlan")
        if type(self.task_status) is not TaskStatus:
            raise InvalidExecutionStateError("task_status must be a TaskStatus")
        if self.current_stage_id is not None and (
            type(self.current_stage_id) is not str
            or not self.current_stage_id.strip()
        ):
            raise InvalidExecutionStateError(
                "current_stage_id must be None or a non-empty string"
            )
        if not stages or any(
            type(stage) is not StageExecutionState for stage in stages
        ):
            raise InvalidExecutionStateError(
                "stage_states must contain StageExecutionState values"
            )
        self._validate_timestamps()
        self._validate_plan_projection()
        self._validate_current_stage()
        if self.task_status is TaskStatus.AWAITING_APPROVAL:
            if (
                type(self.approval_request) is not ApprovalRequest
                or self.approval_request.task_id != self.task_id
                or self.approval_request.stage_id != self.current_stage_id
                or self.approval_request.tool_id != next(
                    stage.tool_id for stage in self.plan.stages
                    if stage.stage_id == self.current_stage_id
                )
            ):
                raise InvalidExecutionStateError("approval request does not match task")
        elif self.approval_request is not None:
            raise InvalidExecutionStateError("approval request requires suspended task")

    @classmethod
    def from_plan(
        cls,
        *,
        task_id: str,
        original_prompt: str,
        plan: TaskPlan,
        timestamp: datetime,
    ) -> AgentTaskState:
        stage_states = tuple(
            StageExecutionState(
                stage_id=stage.stage_id,
                stage_type=stage.stage_type,
                required_capabilities=stage.required_capabilities,
                execution_kind=stage.execution_kind,
            )
            for stage in plan.stages
        )
        required_capabilities = tuple(
            capability
            for stage in plan.stages
            for capability in stage.required_capabilities
        )
        return cls(
            task_id=task_id,
            original_prompt=original_prompt,
            task_class=plan.task_class,
            required_capabilities=required_capabilities,
            plan=plan,
            task_status=TaskStatus.PENDING,
            current_stage_id=None,
            stage_states=stage_states,
            created_at=timestamp,
            updated_at=timestamp,
        )

    def start(self, *, updated_at: datetime) -> AgentTaskState:
        self._require_task_status(TaskStatus.PENDING, TaskStatus.RUNNING)
        self._require_later_time(updated_at)
        return replace(
            self,
            task_status=TaskStatus.RUNNING,
            updated_at=updated_at,
        )

    def complete(self, *, updated_at: datetime) -> AgentTaskState:
        self._require_task_status(TaskStatus.RUNNING, TaskStatus.COMPLETED)
        self._require_later_time(updated_at)
        if any(
            stage.status not in (StageStatus.COMPLETED, StageStatus.SKIPPED)
            for stage in self.stage_states
        ):
            raise InvalidTaskTransitionError(
                "a task can complete only after every stage completes or is skipped"
            )
        return replace(
            self,
            task_status=TaskStatus.COMPLETED,
            current_stage_id=None,
            updated_at=updated_at,
        )

    def fail(self, *, updated_at: datetime) -> AgentTaskState:
        self._require_task_status(TaskStatus.RUNNING, TaskStatus.FAILED)
        self._require_later_time(updated_at)
        failure_index = self._active_stage_index()
        if failure_index is None:
            failure_index = next(
                (
                    index
                    for index, stage in enumerate(self.stage_states)
                    if stage.status is StageStatus.PENDING
                ),
                None,
            )
        if failure_index is None:
            raise InvalidTaskTransitionError(
                "task failure requires an active or pending stage"
            )
        failed_stage = self.stage_states[failure_index]
        if failed_stage.status is StageStatus.PENDING:
            failed_stage = failed_stage.start()
        failed_stage = failed_stage.fail(
            error_code="task_failed",
            safe_message="Task failed",
        )
        stages = (
            self.stage_states[:failure_index]
            + (failed_stage,)
            + self.stage_states[failure_index + 1 :]
        )
        return replace(
            self,
            task_status=TaskStatus.FAILED,
            current_stage_id=None,
            stage_states=stages,
            updated_at=updated_at,
        )

    def cancel(self, *, updated_at: datetime) -> AgentTaskState:
        self._require_task_status(TaskStatus.RUNNING, TaskStatus.CANCELLED)
        self._require_later_time(updated_at)
        stages = self.stage_states
        active_index = self._active_stage_index()
        if active_index is not None:
            stages = (
                stages[:active_index]
                + (stages[active_index].cancel(),)
                + stages[active_index + 1 :]
            )
        return replace(
            self,
            task_status=TaskStatus.CANCELLED,
            current_stage_id=None,
            stage_states=stages,
            updated_at=updated_at,
        )

    def update_stage(
        self,
        stage: StageExecutionState,
        *,
        updated_at: datetime,
        approval_request: ApprovalRequest | None = None,
    ) -> AgentTaskState:
        if self.task_status not in (TaskStatus.RUNNING, TaskStatus.AWAITING_APPROVAL):
            raise InvalidTaskTransitionError(
                "stages can change only while the task is active"
            )
        self._require_later_time(updated_at)
        try:
            index = next(
                index
                for index, current in enumerate(self.stage_states)
                if current.stage_id == stage.stage_id
            )
        except StopIteration as error:
            raise InvalidStepTransitionError(
                f"task has no stage {stage.stage_id!r}"
            ) from error

        current = self.stage_states[index]
        if (
            stage.stage_type is not current.stage_type
            or stage.required_capabilities != current.required_capabilities
            or stage.execution_kind is not current.execution_kind
        ):
            raise InvalidExecutionStateError(
                "stage updates cannot reinterpret plan requirements"
            )
        if (
            current.selected_model_id is not None
            and stage.selected_model_id != current.selected_model_id
        ):
            raise InvalidExecutionStateError(
                "stage updates cannot change an already selected model"
            )
        if current.selected_tool_id is not None and stage.selected_tool_id != current.selected_tool_id:
            raise InvalidExecutionStateError("stage updates cannot change an already selected tool")
        expected_stage = self._expected_stage_transition(current, stage)
        if stage != expected_stage:
            raise InvalidExecutionStateError(
                "stage updates may change only fields controlled by the transition"
            )

        stages = self.stage_states[:index] + (stage,) + self.stage_states[index + 1 :]
        if stage.status is StageStatus.AWAITING_APPROVAL:
            if type(approval_request) is not ApprovalRequest:
                raise InvalidExecutionStateError("approval request is required")
        elif approval_request is not None:
            raise InvalidExecutionStateError("approval request is not expected")
        current_stage_id = (
            stage.stage_id if stage.status in (StageStatus.RUNNING, StageStatus.AWAITING_APPROVAL)
            else None
        )
        task_status = self.task_status
        if stage.status is StageStatus.FAILED:
            task_status = TaskStatus.FAILED
        elif stage.status is StageStatus.CANCELLED:
            task_status = TaskStatus.CANCELLED
        elif stage.status is StageStatus.AWAITING_APPROVAL:
            task_status = TaskStatus.AWAITING_APPROVAL
        elif current.status is StageStatus.AWAITING_APPROVAL and stage.status is StageStatus.RUNNING:
            task_status = TaskStatus.RUNNING
        return replace(
            self,
            task_status=task_status,
            current_stage_id=current_stage_id,
            stage_states=stages,
            updated_at=updated_at,
            approval_request=approval_request,
        )

    @staticmethod
    def _expected_stage_transition(
        current: StageExecutionState,
        replacement: StageExecutionState,
    ) -> StageExecutionState:
        transition = (current.status, replacement.status)
        if transition == (StageStatus.PENDING, StageStatus.RUNNING):
            return current.start(
                selected_model_id=replacement.selected_model_id,
                selected_tool_id=replacement.selected_tool_id,
            )
        if transition == (StageStatus.RUNNING, StageStatus.COMPLETED):
            assert replacement.output_reference is not None
            return current.complete(
                output_reference=replacement.output_reference,
                selected_model_id=replacement.selected_model_id,
                selected_tool_id=replacement.selected_tool_id,
                output_kind=replacement.output_kind or StageOutputKind.TEXT,
            )
        if transition == (StageStatus.RUNNING, StageStatus.FAILED):
            assert replacement.error_code is not None
            assert replacement.safe_message is not None
            return current.fail(
                error_code=replacement.error_code,
                safe_message=replacement.safe_message,
                selected_model_id=replacement.selected_model_id,
                selected_tool_id=replacement.selected_tool_id,
            )
        if transition == (StageStatus.RUNNING, StageStatus.CANCELLED):
            return current.cancel()
        if transition == (StageStatus.RUNNING, StageStatus.AWAITING_APPROVAL):
            assert replacement.selected_tool_id is not None
            return current.await_approval(selected_tool_id=replacement.selected_tool_id)
        if transition == (StageStatus.AWAITING_APPROVAL, StageStatus.RUNNING):
            return current.resume_approval()
        if transition == (StageStatus.AWAITING_APPROVAL, StageStatus.FAILED):
            assert replacement.error_code is not None
            assert replacement.safe_message is not None
            return current.fail(error_code=replacement.error_code,
                                safe_message=replacement.safe_message,
                                selected_tool_id=replacement.selected_tool_id)
        if transition == (StageStatus.PENDING, StageStatus.SKIPPED):
            return current.skip()
        else:
            raise InvalidStepTransitionError(
                f"cannot update stage from {current.status.value} "
                f"to {replacement.status.value}"
            )

    def _active_stage_index(self) -> int | None:
        if self.current_stage_id is None:
            return None
        return next(
            index
            for index, stage in enumerate(self.stage_states)
            if stage.stage_id == self.current_stage_id
        )

    def _require_task_status(
        self, expected: TaskStatus, destination: TaskStatus
    ) -> None:
        if self.task_status is not expected:
            raise InvalidTaskTransitionError(
                f"cannot transition task from {self.task_status.value} "
                f"to {destination.value}"
            )

    def _require_later_time(self, timestamp: datetime) -> None:
        _require_utc(timestamp)
        if timestamp <= self.updated_at:
            raise InvalidTaskTransitionError(
                "transition timestamp must be later than updated_at"
            )

    def _validate_timestamps(self) -> None:
        _require_utc(self.created_at)
        _require_utc(self.updated_at)
        if self.updated_at < self.created_at:
            raise InvalidExecutionStateError(
                "updated_at cannot precede created_at"
            )

    def _validate_plan_projection(self) -> None:
        stage_ids = tuple(stage.stage_id for stage in self.plan.stages)
        if len(stage_ids) != len(set(stage_ids)):
            raise InvalidExecutionStateError(
                "task plans must contain unique stage IDs"
            )
        expected_stages = tuple(
            (stage.stage_id, stage.stage_type, stage.required_capabilities, stage.execution_kind)
            for stage in self.plan.stages
        )
        actual_stages = tuple(
            (stage.stage_id, stage.stage_type, stage.required_capabilities, stage.execution_kind)
            for stage in self.stage_states
        )
        expected_capabilities = tuple(
            capability
            for stage in self.plan.stages
            for capability in stage.required_capabilities
        )
        if (
            self.task_class is not self.plan.task_class
            or expected_capabilities != self.required_capabilities
            or actual_stages != expected_stages
            or any(
                state.selected_tool_id is not None and state.selected_tool_id != plan_stage.tool_id
                for state, plan_stage in zip(self.stage_states, self.plan.stages)
            )
        ):
            raise InvalidExecutionStateError(
                "task state must exactly preserve its plan projection"
            )

    def _validate_current_stage(self) -> None:
        running_ids = tuple(
            stage.stage_id
            for stage in self.stage_states
            if stage.status is StageStatus.RUNNING
        )
        waiting_ids = tuple(
            stage.stage_id for stage in self.stage_states
            if stage.status is StageStatus.AWAITING_APPROVAL
        )
        active_ids = running_ids + waiting_ids
        if len(active_ids) > 1 or (
            active_ids and self.current_stage_id != active_ids[0]
        ) or (not active_ids and self.current_stage_id is not None):
            raise InvalidExecutionStateError(
                "current_stage_id must identify the sole active stage"
            )
        if self.task_status is TaskStatus.AWAITING_APPROVAL:
            if len(waiting_ids) != 1 or running_ids:
                raise InvalidExecutionStateError("suspended task requires one waiting stage")
            index = next(i for i, stage in enumerate(self.stage_states)
                         if stage.status is StageStatus.AWAITING_APPROVAL)
            if any(stage.status is not StageStatus.COMPLETED for stage in self.stage_states[:index]) or any(
                stage.status is not StageStatus.PENDING for stage in self.stage_states[index + 1:]
            ):
                raise InvalidExecutionStateError("suspended stages must preserve plan order")
        elif waiting_ids:
            raise InvalidExecutionStateError("waiting stage requires suspended task")
        if self.task_status is TaskStatus.PENDING and any(
            stage.status is not StageStatus.PENDING
            for stage in self.stage_states
        ):
            raise InvalidExecutionStateError(
                "a pending task requires every stage to be pending"
            )
        if self.task_status is TaskStatus.COMPLETED and any(
            stage.status not in (StageStatus.COMPLETED, StageStatus.SKIPPED)
            for stage in self.stage_states
        ):
            raise InvalidExecutionStateError(
                "a completed task requires every stage to be completed or skipped"
            )
        if self.task_status is TaskStatus.RUNNING and any(
            stage.status in (StageStatus.FAILED, StageStatus.CANCELLED)
            for stage in self.stage_states
        ):
            raise InvalidExecutionStateError(
                "a running task cannot contain a failed or cancelled stage"
            )
        if self.task_status is TaskStatus.FAILED and (
            not any(
                stage.status is StageStatus.FAILED
                for stage in self.stage_states
            )
            or any(
                stage.status is StageStatus.CANCELLED
                for stage in self.stage_states
            )
        ):
            raise InvalidExecutionStateError(
                "a failed task requires a failed stage and no cancelled stages"
            )
        if self.task_status is TaskStatus.CANCELLED and any(
            stage.status is StageStatus.FAILED
            for stage in self.stage_states
        ):
            raise InvalidExecutionStateError(
                "a cancelled task cannot contain a failed stage"
            )
        if self.task_status in (
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        ) and running_ids:
            raise InvalidExecutionStateError(
                "a terminal task cannot contain a running stage"
            )


def _require_utc(value: object) -> None:
    if (
        type(value) is not datetime
        or value.tzinfo is None
        or value.utcoffset() != _UTC_OFFSET
    ):
        raise InvalidExecutionStateError("timestamps must be timezone-aware UTC")
