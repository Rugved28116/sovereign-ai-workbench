import asyncio
import socket
from datetime import UTC, datetime, timedelta

import pytest

from sovereign_api.agent_execution import (
    AgentStep,
    AgentTask,
    StepOutputType,
    StepResult,
    StepStatus,
    TaskStatus,
)
from sovereign_api.errors import (
    ExecutionStateError,
    InvalidTaskTransitionError,
    StageExecutionError,
)
from sovereign_api.orchestration import (
    UNEXPECTED_STAGE_FAILURE,
    MockStageExecutor,
    SequentialTaskOrchestrator,
)
from sovereign_api.task_classification import TaskClass, TaskRequirements
from sovereign_api.task_planning import (
    DeterministicTaskRequirementPlanner,
    TaskPlan,
)


NOW = datetime(2026, 9, 11, 12, 0, tzinfo=UTC)


def _clock() -> datetime:
    return NOW


def _plan(capabilities: tuple[str, ...]) -> TaskPlan:
    task_class = {
        "chat": TaskClass.GENERAL,
        "coding": TaskClass.CODING,
        "document": TaskClass.DOCUMENT,
        "vision": TaskClass.VISION,
        "reasoning": TaskClass.REASONING,
    }[capabilities[0]]
    return DeterministicTaskRequirementPlanner().plan(
        TaskRequirements(task_class, capabilities)
    )


def _execute(
    plan: TaskPlan,
    executor,
    *,
    task_id: str = "task-orchestration-1",
    clock=_clock,
) -> AgentTask:
    return asyncio.run(
        SequentialTaskOrchestrator(executor, clock=clock).execute(
            plan, task_id=task_id
        )
    )


def test_single_stage_plan_succeeds_with_a_result() -> None:
    task = _execute(_plan(("chat",)), MockStageExecutor())

    assert task.status is TaskStatus.SUCCEEDED
    assert task.revision == 4
    assert len(task.steps) == 1
    assert task.steps[0].status is StepStatus.SUCCEEDED
    assert task.steps[0].result == StepResult(
        StepOutputType.TEXT,
        "Mock result for task-orchestration-1/stage-1",
    )


def test_multi_stage_plan_executes_in_order_using_latest_snapshots() -> None:
    class RecordingExecutor:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int, StepStatus]] = []

        async def execute(
            self, task: AgentTask, step: AgentStep
        ) -> StepResult:
            self.calls.append((step.stage_id, task.revision, step.status))
            assert step is task.steps[len(self.calls) - 1]
            return StepResult(StepOutputType.TEXT, step.stage_id)

    executor = RecordingExecutor()
    plan = _plan(("document", "vision", "reasoning"))

    task = _execute(plan, executor)

    assert task.status is TaskStatus.SUCCEEDED
    assert task.revision == 8
    assert executor.calls == [
        ("stage-1", 2, StepStatus.RUNNING),
        ("stage-2", 4, StepStatus.RUNNING),
        ("stage-3", 6, StepStatus.RUNNING),
    ]
    assert all(step.result is not None for step in task.steps)


def test_identical_inputs_produce_identical_mock_results() -> None:
    plan = _plan(("vision", "reasoning"))

    first = _execute(plan, MockStageExecutor())
    second = _execute(plan, MockStageExecutor())

    assert first == second


def test_first_stage_failure_stops_execution_and_skips_later_steps() -> None:
    class RecordingFailureExecutor:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def execute(
            self, task: AgentTask, step: AgentStep
        ) -> StepResult:
            self.calls.append(step.stage_id)
            raise StageExecutionError("Known safe stage failure")

    executor = RecordingFailureExecutor()

    task = _execute(
        _plan(("document", "vision", "reasoning")), executor
    )

    assert executor.calls == ["stage-1"]
    assert task.status is TaskStatus.FAILED
    assert task.revision == 3
    assert tuple(step.status for step in task.steps) == (
        StepStatus.FAILED,
        StepStatus.SKIPPED,
        StepStatus.SKIPPED,
    )
    assert task.steps[0].error == "Known safe stage failure"


def test_middle_stage_failure_preserves_earlier_success_and_stops() -> None:
    class MiddleFailureExecutor:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def execute(
            self, task: AgentTask, step: AgentStep
        ) -> StepResult:
            self.calls.append(step.stage_id)
            if step.stage_id == "stage-2":
                raise StageExecutionError("Middle stage failed")
            return StepResult(StepOutputType.TEXT, step.stage_id)

    executor = MiddleFailureExecutor()

    task = _execute(
        _plan(("document", "vision", "reasoning")), executor
    )

    assert executor.calls == ["stage-1", "stage-2"]
    assert task.status is TaskStatus.FAILED
    assert task.revision == 5
    assert tuple(step.status for step in task.steps) == (
        StepStatus.SUCCEEDED,
        StepStatus.FAILED,
        StepStatus.SKIPPED,
    )
    assert task.steps[0].result == StepResult(StepOutputType.TEXT, "stage-1")
    assert all(step.status is not StepStatus.RUNNING for step in task.steps)


def test_unexpected_executor_exception_uses_safe_failure_message() -> None:
    class UnexpectedFailureExecutor:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def execute(
            self, task: AgentTask, step: AgentStep
        ) -> StepResult:
            self.calls.append(step.stage_id)
            if step.stage_id == "stage-2":
                raise RuntimeError("sensitive internal exception detail")
            return StepResult(StepOutputType.TEXT, step.stage_id)

    executor = UnexpectedFailureExecutor()

    task = _execute(
        _plan(("document", "vision", "reasoning")), executor
    )

    assert executor.calls == ["stage-1", "stage-2"]
    assert task.status is TaskStatus.FAILED
    assert task.revision == 5
    assert tuple(step.status for step in task.steps) == (
        StepStatus.SUCCEEDED,
        StepStatus.FAILED,
        StepStatus.SKIPPED,
    )
    assert task.steps[0].result == StepResult(StepOutputType.TEXT, "stage-1")
    assert task.steps[1].error == UNEXPECTED_STAGE_FAILURE
    assert "sensitive" not in task.steps[1].error
    assert all(step.status is not StepStatus.RUNNING for step in task.steps)


def test_executor_execution_state_error_becomes_safe_terminal_failure() -> None:
    class StateFailureExecutor:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int]] = []

        async def execute(
            self, task: AgentTask, step: AgentStep
        ) -> StepResult:
            self.calls.append((step.stage_id, task.revision))
            if step.stage_id == "stage-2":
                raise ExecutionStateError(
                    "sensitive executor-controlled state detail"
                )
            return StepResult(StepOutputType.TEXT, step.stage_id)

    executor = StateFailureExecutor()

    task = _execute(
        _plan(("document", "vision", "reasoning")), executor
    )

    assert executor.calls == [("stage-1", 2), ("stage-2", 4)]
    assert task.status is TaskStatus.FAILED
    assert task.revision == 5
    assert tuple(step.status for step in task.steps) == (
        StepStatus.SUCCEEDED,
        StepStatus.FAILED,
        StepStatus.SKIPPED,
    )
    assert task.steps[0].result == StepResult(StepOutputType.TEXT, "stage-1")
    assert task.steps[1].error == UNEXPECTED_STAGE_FAILURE
    assert "sensitive" not in task.steps[1].error
    assert all(step.status is not StepStatus.RUNNING for step in task.steps)


def test_invalid_state_transition_propagates_as_typed_error() -> None:
    times = iter((NOW, NOW + timedelta(minutes=1), NOW))

    with pytest.raises(InvalidTaskTransitionError):
        _execute(_plan(("chat",)), MockStageExecutor(), clock=lambda: next(times))


def test_task_plan_is_not_mutated() -> None:
    plan = _plan(("vision", "reasoning"))
    original_stages = plan.stages

    _execute(plan, MockStageExecutor())

    assert plan.stages == original_stages


def test_falsey_executor_is_preserved_and_invoked() -> None:
    class FalseyExecutor:
        def __init__(self) -> None:
            self.called = False

        def __bool__(self) -> bool:
            return False

        async def execute(
            self, task: AgentTask, step: AgentStep
        ) -> StepResult:
            self.called = True
            return StepResult(StepOutputType.TEXT, "falsey executor result")

    executor = FalseyExecutor()

    task = _execute(_plan(("chat",)), executor)

    assert executor.called
    assert task.steps[0].result == StepResult(
        StepOutputType.TEXT, "falsey executor result"
    )


def test_mock_executor_requires_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    original_connect = socket.socket.connect

    async def execute_with_network_blocked() -> AgentTask:
        def reject_network(*args, **kwargs):
            raise AssertionError("MockStageExecutor attempted network access")

        monkeypatch.setattr(socket.socket, "connect", reject_network)
        try:
            return await SequentialTaskOrchestrator(
                MockStageExecutor(), clock=_clock
            ).execute(_plan(("chat",)), task_id="task-orchestration-1")
        finally:
            monkeypatch.setattr(socket.socket, "connect", original_connect)

    task = asyncio.run(execute_with_network_blocked())

    assert task.status is TaskStatus.SUCCEEDED


def test_mock_failure_injection_is_deterministic() -> None:
    plan = _plan(("vision", "reasoning"))
    executor = MockStageExecutor(frozenset({"stage-2"}))

    first = _execute(plan, executor)
    second = _execute(plan, executor)

    assert first == second
    assert first.status is TaskStatus.FAILED
    assert first.steps[1].status is StepStatus.FAILED


def test_orchestrator_state_contains_no_infrastructure_identity() -> None:
    task = _execute(_plan(("chat",)), MockStageExecutor())

    assert not hasattr(task, "provider")
    assert not hasattr(task, "model_id")
    assert all(not hasattr(step, "provider") for step in task.steps)
    assert all(not hasattr(step, "model_id") for step in task.steps)
