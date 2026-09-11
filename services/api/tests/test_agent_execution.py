from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta

import pytest

from sovereign_api.agent_execution import (
    AgentStep,
    AgentTask,
    StepOutputType,
    StepResult,
    StepStatus,
    TaskStatus,
    validate_agent_task_replacement,
)
from sovereign_api.errors import (
    InvalidExecutionStateError,
    InvalidStepTransitionError,
    InvalidTaskTransitionError,
    StaleAgentTaskRevisionError,
)
from sovereign_api.task_classification import TaskClass, TaskRequirements
from sovereign_api.task_planning import DeterministicTaskRequirementPlanner


CREATED_AT = datetime(2026, 9, 11, 8, 0, tzinfo=UTC)


def _plan():
    return DeterministicTaskRequirementPlanner().plan(
        TaskRequirements(
            TaskClass.VISION,
            ("document", "vision", "reasoning"),
        )
    )


def _task() -> AgentTask:
    return AgentTask.from_plan(
        task_id="task-fixture-1", plan=_plan(), timestamp=CREATED_AT
    )


def _time(minutes: int) -> datetime:
    return CREATED_AT + timedelta(minutes=minutes)


def _result(content: str = "complete") -> StepResult:
    return StepResult(StepOutputType.TEXT, content)


def _complete_all_steps(task: AgentTask) -> AgentTask:
    current = task.start(updated_at=_time(1))
    minute = 2
    for step in current.steps:
        current = current.start_step(step.stage_id, updated_at=_time(minute))
        minute += 1
        current = current.succeed_step(
            step.stage_id,
            _result(step.stage_id),
            updated_at=_time(minute),
        )
        minute += 1
    return current


def test_task_plan_converts_to_agent_task_without_mutating_plan() -> None:
    plan = _plan()
    original_stages = plan.stages

    task = AgentTask.from_plan(
        task_id="task-fixture-1", plan=plan, timestamp=CREATED_AT
    )

    assert task.task_id == "task-fixture-1"
    assert task.plan is plan
    assert task.status is TaskStatus.PENDING
    assert task.revision == 0
    assert task.created_at == task.updated_at == CREATED_AT
    assert tuple(step.stage_id for step in task.steps) == tuple(
        stage.stage_id for stage in plan.stages
    )
    assert tuple(step.stage_type for step in task.steps) == tuple(
        stage.stage_type for stage in plan.stages
    )
    assert tuple(step.required_capabilities for step in task.steps) == tuple(
        stage.required_capabilities for stage in plan.stages
    )
    assert all(step.status is StepStatus.PENDING for step in task.steps)
    assert plan.stages == original_stages


def test_same_inputs_produce_structurally_identical_tasks() -> None:
    plan = _plan()

    first = AgentTask.from_plan(
        task_id="task-fixture-1", plan=plan, timestamp=CREATED_AT
    )
    second = AgentTask.from_plan(
        task_id="task-fixture-1", plan=plan, timestamp=CREATED_AT
    )

    assert first == second


def test_agent_task_and_steps_are_immutable_snapshots() -> None:
    task = _task()

    with pytest.raises(FrozenInstanceError):
        task.status = TaskStatus.RUNNING
    with pytest.raises(FrozenInstanceError):
        task.steps[0].status = StepStatus.RUNNING


def test_agent_task_defensively_copies_step_collection() -> None:
    original = _task()
    steps = list(original.steps)
    copied = AgentTask(
        task_id=original.task_id,
        plan=original.plan,
        status=original.status,
        steps=steps,  # type: ignore[arg-type]
        created_at=original.created_at,
        updated_at=original.updated_at,
        revision=original.revision,
    )

    steps.clear()

    assert copied.steps == original.steps
    assert isinstance(copied.steps, tuple)


def test_agent_step_defensively_copies_capabilities() -> None:
    capabilities = ["vision"]
    original = _task().steps[1]
    step = AgentStep(
        stage_id=original.stage_id,
        stage_type=original.stage_type,
        required_capabilities=capabilities,  # type: ignore[arg-type]
    )

    capabilities.append("reasoning")

    assert step.required_capabilities == ("vision",)


def test_pending_task_can_start() -> None:
    original = _task()

    running = original.start(updated_at=_time(1))

    assert original.status is TaskStatus.PENDING
    assert running.status is TaskStatus.RUNNING
    assert running.updated_at == _time(1)
    assert running.revision == original.revision + 1


def test_every_task_and_step_transition_increments_revision_once() -> None:
    pending = _task()
    running = pending.start(updated_at=_time(1))
    step_running = running.start_step("stage-1", updated_at=_time(2))
    step_succeeded = step_running.succeed_step(
        "stage-1", _result(), updated_at=_time(3)
    )
    skipped = running.skip_step("stage-2", updated_at=_time(2))
    failed = step_running.fail_step("stage-1", "failed", updated_at=_time(3))
    cancelled = running.cancel(updated_at=_time(2))
    ready_to_succeed = _complete_all_steps(pending)
    succeeded = ready_to_succeed.succeed(updated_at=_time(8))

    assert running.revision == pending.revision + 1
    assert step_running.revision == running.revision + 1
    assert step_succeeded.revision == step_running.revision + 1
    assert skipped.revision == running.revision + 1
    assert failed.revision == step_running.revision + 1
    assert cancelled.revision == running.revision + 1
    assert succeeded.revision == ready_to_succeed.revision + 1


def test_negative_revision_is_rejected_during_direct_construction() -> None:
    task = _task()

    with pytest.raises(InvalidExecutionStateError):
        AgentTask(
            task_id=task.task_id,
            plan=task.plan,
            status=task.status,
            steps=task.steps,
            created_at=task.created_at,
            updated_at=task.updated_at,
            revision=-1,
        )


def test_compare_and_swap_accepts_current_revision_and_next_replacement() -> None:
    current = _task().start(updated_at=_time(1))
    replacement = current.start_step("stage-1", updated_at=_time(2))

    validate_agent_task_replacement(
        current=current,
        replacement=replacement,
        expected_revision=current.revision,
    )


def test_compare_and_swap_rejects_branch_from_stale_revision() -> None:
    base = _task().start(updated_at=_time(1))
    first_branch = base.start_step("stage-1", updated_at=_time(2))
    stale_branch = base.start_step("stage-2", updated_at=_time(3))

    assert first_branch.revision == stale_branch.revision == base.revision + 1
    with pytest.raises(StaleAgentTaskRevisionError):
        validate_agent_task_replacement(
            current=first_branch,
            replacement=stale_branch,
            expected_revision=base.revision,
        )


def test_running_task_can_succeed_when_all_steps_have_outputs() -> None:
    running = _complete_all_steps(_task())

    succeeded = running.succeed(updated_at=_time(8))

    assert succeeded.status is TaskStatus.SUCCEEDED
    assert all(step.result is not None for step in succeeded.steps)


def test_running_task_fails_atomically_with_required_step() -> None:
    running = _task().start(updated_at=_time(1))
    running = running.start_step("stage-1", updated_at=_time(2))

    failed = running.fail_step(
        "stage-1", "stage execution failed", updated_at=_time(3)
    )

    assert failed.status is TaskStatus.FAILED
    assert failed.steps[0].status is StepStatus.FAILED
    assert failed.steps[0].error == "stage execution failed"
    assert all(step.status is StepStatus.SKIPPED for step in failed.steps[1:])


def test_failure_cancels_running_siblings_and_skips_pending_steps() -> None:
    running = _task().start(updated_at=_time(1))
    running = running.start_step("stage-1", updated_at=_time(2))
    running = running.start_step("stage-2", updated_at=_time(3))

    failed = running.fail_step("stage-1", "failed", updated_at=_time(4))

    assert tuple(step.status for step in failed.steps) == (
        StepStatus.FAILED,
        StepStatus.CANCELLED,
        StepStatus.SKIPPED,
    )


def test_failure_preserves_succeeded_steps() -> None:
    running = _task().start(updated_at=_time(1))
    running = running.start_step("stage-1", updated_at=_time(2))
    running = running.succeed_step("stage-1", _result(), updated_at=_time(3))
    running = running.start_step("stage-2", updated_at=_time(4))

    failed = running.fail_step("stage-2", "failed", updated_at=_time(5))

    assert tuple(step.status for step in failed.steps) == (
        StepStatus.SUCCEEDED,
        StepStatus.FAILED,
        StepStatus.SKIPPED,
    )
    assert failed.steps[0].result == _result()


def test_running_task_can_be_cancelled_without_changing_steps() -> None:
    running = _task().start(updated_at=_time(1))
    original_steps = running.steps

    cancelled = running.cancel(updated_at=_time(2))

    assert cancelled.status is TaskStatus.CANCELLED
    assert cancelled.steps == original_steps
    assert all(step.status is not StepStatus.SUCCEEDED for step in cancelled.steps)


def test_pending_step_can_start() -> None:
    step = _task().steps[0]

    running = step.start()

    assert step.status is StepStatus.PENDING
    assert running.status is StepStatus.RUNNING


def test_running_step_can_succeed() -> None:
    result = _result()

    succeeded = _task().steps[0].start().succeed(result)

    assert succeeded.status is StepStatus.SUCCEEDED
    assert succeeded.result is result


def test_running_step_can_fail() -> None:
    failed = _task().steps[0].start().fail("failed locally")

    assert failed.status is StepStatus.FAILED
    assert failed.error == "failed locally"


def test_pending_step_can_be_skipped() -> None:
    skipped = _task().steps[0].skip()

    assert skipped.status is StepStatus.SKIPPED


@pytest.mark.parametrize("terminal_status", list(TaskStatus)[2:])
def test_terminal_task_cannot_return_to_running(
    terminal_status: TaskStatus,
) -> None:
    if terminal_status is TaskStatus.SUCCEEDED:
        terminal = _complete_all_steps(_task()).succeed(updated_at=_time(8))
    elif terminal_status is TaskStatus.FAILED:
        running = _task().start(updated_at=_time(1))
        running = running.start_step("stage-1", updated_at=_time(2))
        terminal = running.fail_step("stage-1", "failed", updated_at=_time(3))
    else:
        terminal = _task().start(updated_at=_time(1)).cancel(
            updated_at=_time(2)
        )

    with pytest.raises(InvalidTaskTransitionError):
        terminal.start(updated_at=_time(9))


@pytest.mark.parametrize(
    ("terminal_step", "transition"),
    [
        (lambda step: step.start().succeed(_result()), lambda step: step.start()),
        (
            lambda step: step.start().fail("failed"),
            lambda step: step.succeed(_result()),
        ),
        (lambda step: step.skip(), lambda step: step.start()),
        (lambda step: step.start().cancel(), lambda step: step.start()),
    ],
)
def test_terminal_step_transitions_are_rejected(terminal_step, transition) -> None:
    step = terminal_step(_task().steps[0])

    with pytest.raises(InvalidStepTransitionError):
        transition(step)


def test_succeeded_step_requires_result() -> None:
    pending = _task().steps[0]

    with pytest.raises(InvalidExecutionStateError):
        AgentStep(
            stage_id=pending.stage_id,
            stage_type=pending.stage_type,
            required_capabilities=pending.required_capabilities,
            status=StepStatus.SUCCEEDED,
        )


def test_failed_step_requires_error() -> None:
    pending = _task().steps[0]

    with pytest.raises(InvalidExecutionStateError):
        AgentStep(
            stage_id=pending.stage_id,
            stage_type=pending.stage_type,
            required_capabilities=pending.required_capabilities,
            status=StepStatus.FAILED,
        )


@pytest.mark.parametrize(
    ("status", "result", "error"),
    [
        (StepStatus.PENDING, _result(), None),
        (StepStatus.RUNNING, None, "error"),
        (StepStatus.SKIPPED, _result(), None),
        (StepStatus.SUCCEEDED, _result(), "error"),
        (StepStatus.FAILED, _result(), "error"),
    ],
)
def test_result_and_error_are_restricted_to_matching_terminal_state(
    status: StepStatus,
    result: StepResult | None,
    error: str | None,
) -> None:
    pending = _task().steps[0]

    with pytest.raises(InvalidExecutionStateError):
        AgentStep(
            stage_id=pending.stage_id,
            stage_type=pending.stage_type,
            required_capabilities=pending.required_capabilities,
            status=status,
            result=result,
            error=error,
        )


@pytest.mark.parametrize("incomplete_status", [StepStatus.PENDING, StepStatus.RUNNING])
def test_task_cannot_succeed_with_incomplete_step(
    incomplete_status: StepStatus,
) -> None:
    task = _task().start(updated_at=_time(1))
    if incomplete_status is StepStatus.RUNNING:
        task = task.start_step("stage-1", updated_at=_time(2))

    with pytest.raises(InvalidTaskTransitionError):
        task.succeed(updated_at=_time(3))


def test_task_cannot_succeed_with_failed_step() -> None:
    running = _task().start(updated_at=_time(1))
    running = running.start_step("stage-1", updated_at=_time(2))
    failed = running.fail_step("stage-1", "failed", updated_at=_time(3))

    with pytest.raises(InvalidTaskTransitionError):
        failed.succeed(updated_at=_time(4))


def test_task_cannot_succeed_with_skipped_required_step() -> None:
    task = _task().start(updated_at=_time(1))
    task = task.skip_step("stage-1", updated_at=_time(2))

    with pytest.raises(InvalidTaskTransitionError):
        task.succeed(updated_at=_time(3))


def test_task_steps_cannot_reinterpret_plan() -> None:
    task = _task()
    altered = list(task.steps)
    altered[0] = AgentStep(
        stage_id=altered[0].stage_id,
        stage_type=altered[0].stage_type,
        required_capabilities=("coding",),
    )

    with pytest.raises(InvalidExecutionStateError):
        AgentTask(
            task_id=task.task_id,
            plan=task.plan,
            status=task.status,
            steps=altered,  # type: ignore[arg-type]
            created_at=task.created_at,
            updated_at=task.updated_at,
            revision=task.revision,
        )


@pytest.mark.parametrize(
    "inconsistent_statuses",
    [
        (StepStatus.FAILED, StepStatus.RUNNING, StepStatus.SKIPPED),
        (StepStatus.FAILED, StepStatus.PENDING, StepStatus.SKIPPED),
        (StepStatus.SKIPPED, StepStatus.SKIPPED, StepStatus.SKIPPED),
    ],
)
def test_direct_construction_rejects_inconsistent_failed_task(
    inconsistent_statuses: tuple[StepStatus, ...],
) -> None:
    task = _task()
    steps = []
    for source, status in zip(task.steps, inconsistent_statuses, strict=True):
        step = source
        if status is StepStatus.RUNNING:
            step = step.start()
        elif status is StepStatus.FAILED:
            step = step.start().fail("failed")
        elif status is StepStatus.SKIPPED:
            step = step.skip()
        steps.append(step)

    with pytest.raises(InvalidExecutionStateError):
        AgentTask(
            task_id=task.task_id,
            plan=task.plan,
            status=TaskStatus.FAILED,
            steps=steps,  # type: ignore[arg-type]
            created_at=task.created_at,
            updated_at=_time(1),
            revision=1,
        )


def test_transition_timestamps_must_be_monotonic() -> None:
    running = _task().start(updated_at=_time(2))

    with pytest.raises(InvalidTaskTransitionError):
        running.start_step("stage-1", updated_at=_time(1))


def test_execution_state_has_no_provider_or_model_identity() -> None:
    task = _task()

    assert not hasattr(task, "provider")
    assert not hasattr(task, "model_id")
    assert all(not hasattr(step, "provider") for step in task.steps)
    assert all(not hasattr(step, "model_id") for step in task.steps)
