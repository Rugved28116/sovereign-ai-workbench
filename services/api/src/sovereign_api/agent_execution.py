"""Provider-neutral immutable state for future agent execution."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum

from sovereign_api.errors import (
    InvalidExecutionStateError,
    InvalidStepTransitionError,
    InvalidTaskTransitionError,
    StaleAgentTaskRevisionError,
)
from sovereign_api.task_planning import TaskPlan, TaskStageType


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class StepOutputType(StrEnum):
    TEXT = "text"
    STRUCTURED = "structured"


@dataclass(frozen=True, slots=True)
class StepResult:
    """Small provider-neutral output value with no arbitrary object channel."""

    output_type: StepOutputType
    content: str

    def __post_init__(self) -> None:
        if type(self.output_type) is not StepOutputType:
            raise InvalidExecutionStateError(
                "step result output_type must be a StepOutputType"
            )
        if type(self.content) is not str or not self.content:
            raise InvalidExecutionStateError(
                "step result content must be a non-empty string"
            )


@dataclass(frozen=True, slots=True)
class AgentStep:
    """Immutable state for one planned stage."""

    stage_id: str
    stage_type: TaskStageType
    required_capabilities: tuple[str, ...]
    status: StepStatus = StepStatus.PENDING
    result: StepResult | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        capabilities = tuple(self.required_capabilities)
        object.__setattr__(self, "required_capabilities", capabilities)

        if type(self.stage_id) is not str or not self.stage_id.strip():
            raise InvalidExecutionStateError("agent step stage_id must be non-empty")
        if type(self.stage_type) is not TaskStageType:
            raise InvalidExecutionStateError(
                "agent step stage_type must be a TaskStageType"
            )
        if not capabilities or any(
            type(capability) is not str or not capability
            for capability in capabilities
        ):
            raise InvalidExecutionStateError(
                "agent step capabilities must be non-empty strings"
            )
        if type(self.status) is not StepStatus:
            raise InvalidExecutionStateError(
                "agent step status must be a StepStatus"
            )
        if self.result is not None and type(self.result) is not StepResult:
            raise InvalidExecutionStateError(
                "agent step result must be a StepResult"
            )
        if self.error is not None and type(self.error) is not str:
            raise InvalidExecutionStateError("agent step error must be a string")

        if self.status is StepStatus.SUCCEEDED:
            if self.result is None:
                raise InvalidExecutionStateError(
                    "a succeeded step requires a result"
                )
            if self.error is not None:
                raise InvalidExecutionStateError(
                    "a succeeded step cannot contain an error"
                )
        elif self.status is StepStatus.FAILED:
            if not self.error:
                raise InvalidExecutionStateError(
                    "a failed step requires a non-empty error"
                )
            if self.result is not None:
                raise InvalidExecutionStateError(
                    "a failed step cannot contain a result"
                )
        elif self.result is not None or self.error is not None:
            raise InvalidExecutionStateError(
                "results are limited to succeeded steps and errors to failed steps"
            )

    def start(self) -> AgentStep:
        self._require_status(StepStatus.PENDING, StepStatus.RUNNING)
        return replace(self, status=StepStatus.RUNNING)

    def succeed(self, result: StepResult) -> AgentStep:
        self._require_status(StepStatus.RUNNING, StepStatus.SUCCEEDED)
        return replace(self, status=StepStatus.SUCCEEDED, result=result)

    def fail(self, error: str) -> AgentStep:
        self._require_status(StepStatus.RUNNING, StepStatus.FAILED)
        return replace(self, status=StepStatus.FAILED, error=error)

    def skip(self) -> AgentStep:
        self._require_status(StepStatus.PENDING, StepStatus.SKIPPED)
        return replace(self, status=StepStatus.SKIPPED)

    def cancel(self) -> AgentStep:
        self._require_status(StepStatus.RUNNING, StepStatus.CANCELLED)
        return replace(self, status=StepStatus.CANCELLED)

    def _require_status(
        self, expected: StepStatus, destination: StepStatus
    ) -> None:
        if self.status is not expected:
            raise InvalidStepTransitionError(
                f"cannot transition step from {self.status.value} "
                f"to {destination.value}"
            )


@dataclass(frozen=True, slots=True)
class AgentTask:
    """Immutable execution snapshot derived from an immutable TaskPlan."""

    task_id: str
    plan: TaskPlan
    status: TaskStatus
    steps: tuple[AgentStep, ...]
    created_at: datetime
    updated_at: datetime
    revision: int

    def __post_init__(self) -> None:
        steps = tuple(self.steps)
        object.__setattr__(self, "steps", steps)

        if type(self.task_id) is not str or not self.task_id.strip():
            raise InvalidExecutionStateError("agent task task_id must be non-empty")
        if type(self.plan) is not TaskPlan:
            raise InvalidExecutionStateError("agent task plan must be a TaskPlan")
        if type(self.status) is not TaskStatus:
            raise InvalidExecutionStateError(
                "agent task status must be a TaskStatus"
            )
        if type(self.revision) is not int or self.revision < 0:
            raise InvalidExecutionStateError(
                "agent task revision must be a non-negative integer"
            )
        if not steps or any(type(step) is not AgentStep for step in steps):
            raise InvalidExecutionStateError(
                "agent task steps must contain AgentStep values"
            )
        self._validate_timestamps()
        self._validate_steps_match_plan()
        self._validate_status_consistency()

    @classmethod
    def from_plan(
        cls, *, task_id: str, plan: TaskPlan, timestamp: datetime
    ) -> AgentTask:
        """Create a deterministic pending task using an explicit timestamp."""
        steps = tuple(
            AgentStep(
                stage_id=stage.stage_id,
                stage_type=stage.stage_type,
                required_capabilities=stage.required_capabilities,
            )
            for stage in plan.stages
        )
        return cls(
            task_id=task_id,
            plan=plan,
            status=TaskStatus.PENDING,
            steps=steps,
            created_at=timestamp,
            updated_at=timestamp,
            revision=0,
        )

    def start(self, *, updated_at: datetime) -> AgentTask:
        self._require_status(TaskStatus.PENDING, TaskStatus.RUNNING)
        self._require_valid_transition_time(updated_at)
        return replace(
            self,
            status=TaskStatus.RUNNING,
            updated_at=updated_at,
            revision=self.revision + 1,
        )

    def start_step(self, stage_id: str, *, updated_at: datetime) -> AgentTask:
        self._require_running()
        return self._replace_step(
            stage_id, self._step(stage_id).start(), updated_at=updated_at
        )

    def succeed_step(
        self, stage_id: str, result: StepResult, *, updated_at: datetime
    ) -> AgentTask:
        self._require_running()
        return self._replace_step(
            stage_id,
            self._step(stage_id).succeed(result),
            updated_at=updated_at,
        )

    def fail_step(
        self, stage_id: str, error: str, *, updated_at: datetime
    ) -> AgentTask:
        """Fail one required step and the containing task atomically."""
        self._require_running()
        self._require_valid_transition_time(updated_at)
        failed_step = self._step(stage_id).fail(error)
        terminal_steps = tuple(
            failed_step
            if step.stage_id == stage_id
            else step.cancel()
            if step.status is StepStatus.RUNNING
            else step.skip()
            if step.status is StepStatus.PENDING
            else step
            for step in self.steps
        )
        return replace(
            self,
            status=TaskStatus.FAILED,
            steps=terminal_steps,
            updated_at=updated_at,
            revision=self.revision + 1,
        )

    def skip_step(self, stage_id: str, *, updated_at: datetime) -> AgentTask:
        self._require_running()
        return self._replace_step(
            stage_id, self._step(stage_id).skip(), updated_at=updated_at
        )

    def succeed(self, *, updated_at: datetime) -> AgentTask:
        self._require_status(TaskStatus.RUNNING, TaskStatus.SUCCEEDED)
        self._require_valid_transition_time(updated_at)
        if any(step.status is not StepStatus.SUCCEEDED for step in self.steps):
            raise InvalidTaskTransitionError(
                "task cannot succeed until every required step succeeds"
            )
        return replace(
            self,
            status=TaskStatus.SUCCEEDED,
            updated_at=updated_at,
            revision=self.revision + 1,
        )

    def cancel(self, *, updated_at: datetime) -> AgentTask:
        self._require_status(TaskStatus.RUNNING, TaskStatus.CANCELLED)
        self._require_valid_transition_time(updated_at)
        return replace(
            self,
            status=TaskStatus.CANCELLED,
            updated_at=updated_at,
            revision=self.revision + 1,
        )

    def _step(self, stage_id: str) -> AgentStep:
        try:
            return next(step for step in self.steps if step.stage_id == stage_id)
        except StopIteration as error:
            raise InvalidStepTransitionError(
                f"task has no step with stage_id {stage_id!r}"
            ) from error

    def _replace_step(
        self, stage_id: str, step: AgentStep, *, updated_at: datetime
    ) -> AgentTask:
        self._require_valid_transition_time(updated_at)
        return replace(
            self,
            steps=self._steps_with(stage_id, step),
            updated_at=updated_at,
            revision=self.revision + 1,
        )

    def _steps_with(
        self, stage_id: str, replacement: AgentStep
    ) -> tuple[AgentStep, ...]:
        return tuple(
            replacement if step.stage_id == stage_id else step
            for step in self.steps
        )

    def _require_running(self) -> None:
        if self.status is not TaskStatus.RUNNING:
            raise InvalidTaskTransitionError(
                f"cannot change steps while task is {self.status.value}"
            )

    def _require_status(
        self, expected: TaskStatus, destination: TaskStatus
    ) -> None:
        if self.status is not expected:
            raise InvalidTaskTransitionError(
                f"cannot transition task from {self.status.value} "
                f"to {destination.value}"
            )

    def _require_valid_transition_time(self, timestamp: datetime) -> None:
        if not _is_aware_datetime(timestamp) or timestamp < self.updated_at:
            raise InvalidTaskTransitionError(
                "task transition timestamp must be timezone-aware and monotonic"
            )

    def _validate_timestamps(self) -> None:
        if not _is_aware_datetime(self.created_at) or not _is_aware_datetime(
            self.updated_at
        ):
            raise InvalidExecutionStateError(
                "agent task timestamps must be timezone-aware"
            )
        if self.updated_at < self.created_at:
            raise InvalidExecutionStateError(
                "agent task updated_at cannot precede created_at"
            )

    def _validate_steps_match_plan(self) -> None:
        expected = tuple(
            (stage.stage_id, stage.stage_type, stage.required_capabilities)
            for stage in self.plan.stages
        )
        actual = tuple(
            (step.stage_id, step.stage_type, step.required_capabilities)
            for step in self.steps
        )
        if actual != expected:
            raise InvalidExecutionStateError(
                "agent task steps must exactly preserve plan ordering and requirements"
            )
        if len({step.stage_id for step in self.steps}) != len(self.steps):
            raise InvalidExecutionStateError("agent task stage IDs must be unique")

    def _validate_status_consistency(self) -> None:
        failed_steps = tuple(
            step for step in self.steps if step.status is StepStatus.FAILED
        )
        if failed_steps and self.status is not TaskStatus.FAILED:
            raise InvalidExecutionStateError(
                "a task with a failed required step must be failed"
            )
        if self.status is TaskStatus.FAILED and not failed_steps:
            raise InvalidExecutionStateError(
                "a failed task requires a failed required step"
            )
        if self.status is TaskStatus.FAILED and any(
            step.status in {StepStatus.PENDING, StepStatus.RUNNING}
            for step in self.steps
        ):
            raise InvalidExecutionStateError(
                "a failed task cannot contain pending or running steps"
            )
        if self.status is TaskStatus.PENDING and any(
            step.status is not StepStatus.PENDING for step in self.steps
        ):
            raise InvalidExecutionStateError(
                "a pending task can contain only pending steps"
            )
        if self.status is TaskStatus.SUCCEEDED and any(
            step.status is not StepStatus.SUCCEEDED for step in self.steps
        ):
            raise InvalidExecutionStateError(
                "a succeeded task requires every step to have succeeded"
            )


def _is_aware_datetime(value: object) -> bool:
    return (
        type(value) is datetime
        and value.tzinfo is not None
        and value.utcoffset() is not None
    )


def validate_agent_task_replacement(
    *,
    current: AgentTask,
    replacement: AgentTask,
    expected_revision: int,
) -> None:
    """Validate a prospective compare-and-swap without storing either value."""
    if type(expected_revision) is not int or expected_revision < 0:
        raise InvalidExecutionStateError(
            "expected revision must be a non-negative integer"
        )
    if expected_revision != current.revision:
        raise StaleAgentTaskRevisionError(
            f"expected task revision {expected_revision}, "
            f"but current revision is {current.revision}"
        )
    if replacement.task_id != current.task_id:
        raise InvalidExecutionStateError(
            "replacement must have the same task_id as the current task"
        )
    if replacement.plan != current.plan or replacement.created_at != current.created_at:
        raise InvalidExecutionStateError(
            "replacement must preserve the current task plan and creation time"
        )
    if replacement.revision != current.revision + 1:
        raise InvalidExecutionStateError(
            "replacement revision must increment the current revision exactly once"
        )
