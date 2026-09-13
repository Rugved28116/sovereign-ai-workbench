"""Bounded sequential execution of an existing immutable task plan."""

from __future__ import annotations

from dataclasses import dataclass

from sovereign_api.agent_stage_execution import (
    StageCoordinationReport, StageExecutionCoordinator,
)
from sovereign_api.agent_task_state import (
    AgentTaskState, StageExecutionState, StageStatus, TaskStatus,
)
from sovereign_api.errors import OrchestrationError
from sovereign_api.task_planning import TaskPlan, TaskStage, TaskStageType


MAX_RUN_STAGES = 16
MAX_RUN_MODEL_INVOCATIONS = 16
_TERMINAL = (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
_MODEL_STAGE_TYPES = frozenset({
    TaskStageType.GENERATE, TaskStageType.REASON, TaskStageType.CODE,
    TaskStageType.DOCUMENT, TaskStageType.VISION,
})


class InvalidTaskRunError(OrchestrationError):
    code = "invalid_task_run"


class TaskRunExecutionError(OrchestrationError):
    code = "task_run_execution_failed"

    def __init__(
        self, *, last_confirmed_state: AgentTaskState,
        stages_executed: int, model_invocations: int,
    ) -> None:
        super().__init__("Task run stopped; the last attempted stage may need reconciliation")
        self.last_confirmed_state = last_confirmed_state
        self.stages_executed = stages_executed
        self.model_invocations = model_invocations


def stage_requires_model_invocation(stage: TaskStage) -> bool:
    """Classify only canonical runtime stage types; unknown types fail closed."""

    if type(stage) is not TaskStage or type(stage.stage_type) is not TaskStageType:
        raise InvalidTaskRunError("Task stage type is invalid")
    if stage.stage_type in _MODEL_STAGE_TYPES:
        return True
    raise InvalidTaskRunError("Task stage type is unsupported")


def revalidate_agent_task_state(state: AgentTaskState) -> AgentTaskState:
    """Reconstruct every nested value, rejecting tampering instead of repairing it."""

    if type(state) is not AgentTaskState:
        raise InvalidTaskRunError("Task state is invalid")
    try:
        if (
            type(state.plan) is not TaskPlan
            or type(state.plan.stages) is not tuple
            or any(type(stage) is not TaskStage for stage in state.plan.stages)
            or type(state.stage_states) is not tuple
            or any(type(stage) is not StageExecutionState for stage in state.stage_states)
        ):
            raise InvalidTaskRunError("Task state is invalid")
        plan = TaskPlan(
            task_class=state.plan.task_class,
            stages=tuple(
                TaskStage(
                    stage_id=stage.stage_id,
                    stage_type=stage.stage_type,
                    required_capabilities=stage.required_capabilities,
                )
                for stage in state.plan.stages
            ),
        )
        stages = tuple(
            StageExecutionState(
                stage_id=stage.stage_id,
                stage_type=stage.stage_type,
                required_capabilities=stage.required_capabilities,
                status=stage.status,
                selected_model_id=stage.selected_model_id,
                output_reference=stage.output_reference,
                error_code=stage.error_code,
                safe_message=stage.safe_message,
            )
            for stage in state.stage_states
        )
        validated = AgentTaskState(
            task_id=state.task_id,
            original_prompt=state.original_prompt,
            task_class=state.task_class,
            required_capabilities=state.required_capabilities,
            plan=plan,
            task_status=state.task_status,
            current_stage_id=state.current_stage_id,
            stage_states=stages,
            created_at=state.created_at,
            updated_at=state.updated_at,
        )
        if validated != state:
            raise InvalidTaskRunError("Task state is invalid")
        return validated
    except Exception:
        raise InvalidTaskRunError("Task state is invalid") from None


@dataclass(frozen=True, slots=True)
class TaskRunResult:
    """No raw output: final_state is the authoritative, immutable snapshot."""

    final_state: AgentTaskState
    stages_executed: int
    model_invocations: int
    terminal_status: TaskStatus
    safe_message: str
    error_code: str | None = None

    def __post_init__(self) -> None:
        if (
            type(self.final_state) is not AgentTaskState
            or type(self.stages_executed) is not int
            or type(self.model_invocations) is not int
            or not 0 <= self.stages_executed <= MAX_RUN_STAGES
            or not 0 <= self.model_invocations <= MAX_RUN_MODEL_INVOCATIONS
            or self.model_invocations > self.stages_executed
            or self.terminal_status is not self.final_state.task_status
            or type(self.safe_message) is not str
            or not self.safe_message
            or (self.error_code is not None and (
                type(self.error_code) is not str or not self.error_code
            ))
        ):
            raise InvalidTaskRunError("Task run result is invalid")


class TaskPlanRunner:
    """Call the same one-stage coordinator with each latest state, at most 16 times."""

    def __init__(
        self,
        coordinator: StageExecutionCoordinator,
        *,
        max_stages_per_run: int = MAX_RUN_STAGES,
        max_total_model_invocations_per_run: int = MAX_RUN_MODEL_INVOCATIONS,
    ) -> None:
        if (
            type(max_stages_per_run) is not int
            or not 1 <= max_stages_per_run <= MAX_RUN_STAGES
            or type(max_total_model_invocations_per_run) is not int
            or not 1 <= max_total_model_invocations_per_run <= MAX_RUN_MODEL_INVOCATIONS
        ):
            raise InvalidTaskRunError("Task run limits are invalid")
        self._coordinator = coordinator
        self._max_stages = max_stages_per_run
        self._max_model_invocations = max_total_model_invocations_per_run

    async def run(self, initial_state: AgentTaskState) -> TaskRunResult:
        revalidate_agent_task_state(initial_state)

        state = initial_state
        if state.task_status in _TERMINAL:
            return self._result(state, 0, 0)
        if len(state.plan.stages) > self._max_stages:
            raise InvalidTaskRunError("Task plan exceeds the run stage limit")
        if state.current_stage_id is not None:
            raise InvalidTaskRunError("An active stage cannot be resumed by this runner")
        pending = next(
            (index for index, stage in enumerate(state.stage_states)
             if stage.status is StageStatus.PENDING), None,
        )
        if pending is None or any(
            stage.status is not StageStatus.COMPLETED
            for stage in state.stage_states[:pending]
        ) or any(
            stage.status is not StageStatus.PENDING
            for stage in state.stage_states[pending:]
        ):
            raise InvalidTaskRunError("Task stages are not in executable plan order")

        stages_executed = 0
        model_invocations = 0
        # The immutable plan and both counters bound this loop independently.
        while state.task_status not in _TERMINAL:
            if stages_executed >= self._max_stages:
                return self._result(
                    state, stages_executed, model_invocations,
                    error_code="stage_limit_reached",
                )
            next_index = next(
                (index for index, stage in enumerate(state.stage_states)
                 if stage.status is StageStatus.PENDING), None,
            )
            if next_index is None:
                raise TaskRunExecutionError(
                    last_confirmed_state=state,
                    stages_executed=stages_executed,
                    model_invocations=model_invocations,
                )
            previous = state
            stage = previous.plan.stages[next_index]
            reserved = int(stage_requires_model_invocation(stage))
            if model_invocations + reserved > self._max_model_invocations:
                return self._result(
                    state, stages_executed, model_invocations,
                    error_code="model_invocation_limit_reached",
                )
            # Reserve before entering the coordinator. Never refund based on a
            # pluggable executor's self-report, even if routing fails early.
            model_invocations += reserved
            try:
                report = await self._coordinator.execute_one_with_report(
                    previous, stage,
                )
                if type(report) is not StageCoordinationReport:
                    raise InvalidTaskRunError("Coordinator report is invalid")
                next_state = report.state
                if type(next_state) is not AgentTaskState:
                    raise InvalidTaskRunError("Coordinator state is invalid")
                revalidate_agent_task_state(next_state)
                if (
                    next_state.task_id != previous.task_id
                    or next_state.plan != initial_state.plan
                    or next_state.original_prompt != previous.original_prompt
                    or next_state.created_at != previous.created_at
                    or next_state.updated_at <= previous.updated_at
                    or next_state.stage_states[:next_index] != previous.stage_states[:next_index]
                    or next_state.stage_states[next_index + 1:] != previous.stage_states[next_index + 1:]
                    or next_state.stage_states[next_index].status not in (
                        StageStatus.COMPLETED, StageStatus.FAILED, StageStatus.CANCELLED,
                    )
                ):
                    raise InvalidTaskRunError("Coordinator did not advance exactly one stage")
            except Exception:
                # Exit the exception block before raising: no raw exception chain
                # or provider diagnostic is retained on the caller-visible error.
                pass
            else:
                state = next_state
                stages_executed += 1
                continue
            raise TaskRunExecutionError(
                last_confirmed_state=previous,
                stages_executed=stages_executed,
                model_invocations=model_invocations,
            ) from None
        return self._result(state, stages_executed, model_invocations)

    @staticmethod
    def _result(
        state: AgentTaskState, stages_executed: int, model_invocations: int,
        *, error_code: str | None = None,
    ) -> TaskRunResult:
        if error_code is None and state.task_status is TaskStatus.FAILED:
            error_code = "task_failed"
        message = {
            TaskStatus.COMPLETED: "Task completed",
            TaskStatus.FAILED: "Task failed",
            TaskStatus.CANCELLED: "Task cancelled",
            TaskStatus.RUNNING: "Task run limit reached",
        }.get(state.task_status, "Task not started")
        return TaskRunResult(
            final_state=state,
            stages_executed=stages_executed,
            model_invocations=model_invocations,
            terminal_status=state.task_status,
            safe_message=message,
            error_code=error_code,
        )
