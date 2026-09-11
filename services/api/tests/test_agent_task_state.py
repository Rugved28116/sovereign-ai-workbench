from dataclasses import FrozenInstanceError, fields, replace
from datetime import UTC, datetime, timedelta, timezone

import pytest

from sovereign_api.agent_task_state import (
    AgentTaskState,
    StageExecutionState,
    StageStatus,
    TaskStatus,
)
from sovereign_api.errors import (
    InvalidExecutionStateError,
    InvalidStepTransitionError,
    InvalidTaskTransitionError,
)
from sovereign_api.task_classification import TaskClass, TaskRequirements
from sovereign_api.task_planning import (
    DeterministicTaskRequirementPlanner,
    TaskPlan,
    TaskStage,
    TaskStageType,
)


CREATED_AT = datetime(2026, 9, 11, 8, 0, tzinfo=UTC)


def _plan():
    return DeterministicTaskRequirementPlanner().plan(
        TaskRequirements(
            TaskClass.VISION,
            ("document", "vision", "reasoning"),
        )
    )


def _state() -> AgentTaskState:
    return AgentTaskState.from_plan(
        task_id="task-state-fixture",
        original_prompt="  Analyze this inspection report  ",
        plan=_plan(),
        timestamp=CREATED_AT,
    )


def _time(minutes: int) -> datetime:
    return CREATED_AT + timedelta(minutes=minutes)


def _complete_all_stages(state: AgentTaskState) -> AgentTaskState:
    current = state.start(updated_at=_time(1))
    minute = 2
    for index in range(len(current.stage_states)):
        running_stage = current.stage_states[index].start(
            selected_model_id=f"approved-model-{index + 1}"
        )
        current = current.update_stage(
            running_stage,
            updated_at=_time(minute),
        )
        minute += 1
        completed_stage = current.stage_states[index].complete(
            output_reference=f"output-{index + 1}"
        )
        current = current.update_stage(
            completed_stage,
            updated_at=_time(minute),
        )
        minute += 1
    return current


def test_task_plan_initializes_exact_pending_state_without_mutating_plan() -> None:
    plan = _plan()
    original_stages = plan.stages

    state = AgentTaskState.from_plan(
        task_id="task-state-fixture",
        original_prompt="  Analyze this inspection report  ",
        plan=plan,
        timestamp=CREATED_AT,
    )

    assert state.task_id == "task-state-fixture"
    assert state.original_prompt == "Analyze this inspection report"
    assert state.task_class is plan.task_class
    assert state.required_capabilities == (
        "document",
        "vision",
        "reasoning",
    )
    assert state.plan is plan
    assert state.task_status is TaskStatus.PENDING
    assert state.current_stage_id is None
    assert state.created_at == state.updated_at == CREATED_AT
    assert tuple(stage.stage_id for stage in state.stage_states) == tuple(
        stage.stage_id for stage in plan.stages
    )
    assert tuple(stage.stage_type for stage in state.stage_states) == tuple(
        stage.stage_type for stage in plan.stages
    )
    assert tuple(
        stage.required_capabilities for stage in state.stage_states
    ) == tuple(stage.required_capabilities for stage in plan.stages)
    assert all(
        stage.status is StageStatus.PENDING for stage in state.stage_states
    )
    assert plan.stages == original_stages


def test_same_explicit_inputs_produce_identical_state() -> None:
    first = _state()
    second = _state()

    assert first == second


def test_state_and_nested_stages_are_frozen() -> None:
    state = _state()

    with pytest.raises(FrozenInstanceError):
        state.task_status = TaskStatus.RUNNING
    with pytest.raises(FrozenInstanceError):
        state.stage_states[0].status = StageStatus.RUNNING
    with pytest.raises(AttributeError):
        state.required_capabilities.append("coding")
    with pytest.raises(AttributeError):
        state.stage_states[0].required_capabilities.append("coding")


def test_caller_owned_collections_are_defensively_copied() -> None:
    original = _state()
    capabilities = list(original.required_capabilities)
    stage_capabilities = ["document"]
    first_stage = StageExecutionState(
        stage_id="stage-1",
        stage_type=original.stage_states[0].stage_type,
        required_capabilities=stage_capabilities,
    )
    stage_states = [first_stage, *original.stage_states[1:]]
    copied = AgentTaskState(
        task_id=original.task_id,
        original_prompt=original.original_prompt,
        task_class=original.task_class,
        required_capabilities=capabilities,
        plan=original.plan,
        task_status=original.task_status,
        current_stage_id=original.current_stage_id,
        stage_states=stage_states,
        created_at=original.created_at,
        updated_at=original.updated_at,
    )

    capabilities.append("coding")
    stage_capabilities.append("coding")
    stage_states.clear()

    assert copied.required_capabilities == original.required_capabilities
    assert copied.stage_states == original.stage_states
    assert isinstance(copied.required_capabilities, tuple)
    assert isinstance(copied.stage_states, tuple)
    assert isinstance(copied.stage_states[0].required_capabilities, tuple)


def test_pending_task_transitions_to_running_as_new_snapshot() -> None:
    pending = _state()

    running = pending.start(updated_at=_time(1))

    assert pending.task_status is TaskStatus.PENDING
    assert pending.updated_at == CREATED_AT
    assert running.task_status is TaskStatus.RUNNING
    assert running.created_at == CREATED_AT
    assert running.updated_at == _time(1)


def test_running_task_transitions_to_completed_after_all_stages() -> None:
    ready = _complete_all_stages(_state())

    completed = ready.complete(updated_at=_time(8))

    assert completed.task_status is TaskStatus.COMPLETED
    assert completed.current_stage_id is None
    assert completed.created_at == CREATED_AT


@pytest.mark.parametrize(
    ("transition", "expected_status"),
    [
        ("fail", TaskStatus.FAILED),
        ("cancel", TaskStatus.CANCELLED),
    ],
)
def test_running_task_transitions_to_other_terminal_states(
    transition: str,
    expected_status: TaskStatus,
) -> None:
    running = _state().start(updated_at=_time(1))

    terminal = getattr(running, transition)(updated_at=_time(2))

    assert terminal.task_status is expected_status
    assert terminal.created_at == CREATED_AT
    assert terminal.updated_at == _time(2)


def test_task_failure_atomically_fails_active_stage() -> None:
    pending = _state()
    running_task = pending.start(updated_at=_time(1))
    running_stage = running_task.stage_states[0].start(
        selected_model_id="approved-document-model"
    )
    active = running_task.update_stage(running_stage, updated_at=_time(2))

    failed = active.fail(updated_at=_time(3))

    assert failed.task_status is TaskStatus.FAILED
    assert failed.current_stage_id is None
    assert failed.stage_states[0].status is StageStatus.FAILED
    assert failed.stage_states[0].error_code == "task_failed"
    assert failed.stage_states[0].safe_message == "Task failed"
    assert not any(
        stage.status is StageStatus.RUNNING for stage in failed.stage_states
    )
    assert pending.task_status is TaskStatus.PENDING
    assert active.task_status is TaskStatus.RUNNING
    assert active.stage_states[0].status is StageStatus.RUNNING


def test_task_cancellation_atomically_cancels_active_stage() -> None:
    pending = _state()
    running_task = pending.start(updated_at=_time(1))
    running_stage = running_task.stage_states[0].start(
        selected_model_id="approved-document-model"
    )
    active = running_task.update_stage(running_stage, updated_at=_time(2))

    cancelled = active.cancel(updated_at=_time(3))

    assert cancelled.task_status is TaskStatus.CANCELLED
    assert cancelled.current_stage_id is None
    assert cancelled.stage_states[0].status is StageStatus.CANCELLED
    assert not any(
        stage.status in (StageStatus.RUNNING, StageStatus.FAILED)
        for stage in cancelled.stage_states
    )
    assert pending.task_status is TaskStatus.PENDING
    assert active.task_status is TaskStatus.RUNNING
    assert active.stage_states[0].status is StageStatus.RUNNING


def test_task_cannot_complete_directly_from_pending() -> None:
    with pytest.raises(InvalidTaskTransitionError):
        _state().complete(updated_at=_time(1))


def test_task_cannot_complete_with_unfinished_stages() -> None:
    running = _state().start(updated_at=_time(1))

    with pytest.raises(InvalidTaskTransitionError):
        running.complete(updated_at=_time(2))


@pytest.mark.parametrize(
    "terminal_state",
    [
        lambda: _complete_all_stages(_state()).complete(updated_at=_time(8)),
        lambda: _state().start(updated_at=_time(1)).fail(updated_at=_time(2)),
        lambda: _state().start(updated_at=_time(1)).cancel(updated_at=_time(2)),
    ],
)
def test_terminal_task_cannot_return_to_running(terminal_state) -> None:
    with pytest.raises(InvalidTaskTransitionError):
        terminal_state().start(updated_at=_time(9))


def test_pending_stage_transitions_to_running() -> None:
    pending = _state().stage_states[0]

    running = pending.start(selected_model_id="approved-document-model")

    assert pending.status is StageStatus.PENDING
    assert pending.selected_model_id is None
    assert running.status is StageStatus.RUNNING
    assert running.selected_model_id == "approved-document-model"


def test_running_stage_transitions_to_completed() -> None:
    running = _state().stage_states[0].start(
        selected_model_id="approved-document-model"
    )

    completed = running.complete(output_reference="output-stage-1")

    assert completed.status is StageStatus.COMPLETED
    assert completed.output_reference == "output-stage-1"
    assert completed.selected_model_id == "approved-document-model"


def test_running_stage_transitions_to_failed_with_safe_fields() -> None:
    running = _state().stage_states[0].start()

    failed = running.fail(
        error_code="stage_processing_failed",
        safe_message="Stage processing failed",
    )

    assert failed.status is StageStatus.FAILED
    assert failed.error_code == "stage_processing_failed"
    assert failed.safe_message == "Stage processing failed"
    assert failed.output_reference is None


def test_pending_stage_transitions_to_skipped() -> None:
    skipped = _state().stage_states[0].skip()

    assert skipped.status is StageStatus.SKIPPED


def test_running_stage_transitions_to_cancelled() -> None:
    running = _state().stage_states[0].start(
        selected_model_id="approved-document-model"
    )

    cancelled = running.cancel()

    assert cancelled.status is StageStatus.CANCELLED
    assert cancelled.selected_model_id == "approved-document-model"


@pytest.mark.parametrize(
    "invalid_transition",
    [
        lambda stage: stage.complete(output_reference="output"),
        lambda stage: stage.fail(error_code="failed", safe_message="failed"),
        lambda stage: stage.start().start(),
        lambda stage: stage.start().complete(output_reference="output").start(),
        lambda stage: stage.start().fail(
            error_code="failed", safe_message="failed"
        ).complete(output_reference="output"),
        lambda stage: stage.start().cancel().start(),
        lambda stage: stage.skip().start(),
    ],
)
def test_invalid_stage_transitions_fail_closed(invalid_transition) -> None:
    with pytest.raises(InvalidStepTransitionError):
        invalid_transition(_state().stage_states[0])


def test_task_stage_update_uses_controlled_copy_semantics() -> None:
    pending = _state()
    running_task = pending.start(updated_at=_time(1))
    running_stage = running_task.stage_states[0].start(
        selected_model_id="approved-document-model"
    )

    updated = running_task.update_stage(running_stage, updated_at=_time(2))

    assert pending.task_status is TaskStatus.PENDING
    assert running_task.stage_states[0].status is StageStatus.PENDING
    assert running_task.current_stage_id is None
    assert updated.stage_states[0] is running_stage
    assert updated.current_stage_id == "stage-1"
    assert updated.updated_at == _time(2)


def test_stage_failure_atomically_fails_task() -> None:
    running_task = _state().start(updated_at=_time(1))
    running_stage = running_task.stage_states[0].start()
    active = running_task.update_stage(running_stage, updated_at=_time(2))
    failed_stage = active.stage_states[0].fail(
        error_code="stage_failed",
        safe_message="Stage failed",
    )

    failed = active.update_stage(failed_stage, updated_at=_time(3))

    assert failed.task_status is TaskStatus.FAILED
    assert failed.current_stage_id is None
    assert failed.stage_states[0].status is StageStatus.FAILED
    assert not any(
        stage.status is StageStatus.RUNNING for stage in failed.stage_states
    )
    assert active.task_status is TaskStatus.RUNNING
    assert active.stage_states[0].status is StageStatus.RUNNING


def test_stage_cancellation_atomically_cancels_task() -> None:
    running_task = _state().start(updated_at=_time(1))
    running_stage = running_task.stage_states[0].start()
    active = running_task.update_stage(running_stage, updated_at=_time(2))

    cancelled = active.update_stage(
        active.stage_states[0].cancel(),
        updated_at=_time(3),
    )

    assert cancelled.task_status is TaskStatus.CANCELLED
    assert cancelled.current_stage_id is None
    assert cancelled.stage_states[0].status is StageStatus.CANCELLED


def test_failed_task_cannot_later_be_cancelled() -> None:
    running_task = _state().start(updated_at=_time(1))
    active = running_task.update_stage(
        running_task.stage_states[0].start(),
        updated_at=_time(2),
    )
    failed = active.update_stage(
        active.stage_states[0].fail(
            error_code="stage_failed",
            safe_message="Stage failed",
        ),
        updated_at=_time(3),
    )

    with pytest.raises(InvalidTaskTransitionError):
        failed.cancel(updated_at=_time(4))


def test_completed_task_may_contain_only_completed_or_skipped_stages() -> None:
    completed = _complete_all_stages(_state()).complete(updated_at=_time(8))
    invalid_statuses = (
        StageStatus.PENDING,
        StageStatus.RUNNING,
        StageStatus.FAILED,
        StageStatus.CANCELLED,
    )

    for status in invalid_statuses:
        replacement = _state().stage_states[0]
        if status is StageStatus.RUNNING:
            replacement = replacement.start()
        elif status is StageStatus.FAILED:
            replacement = replacement.start().fail(
                error_code="failed",
                safe_message="Stage failed",
            )
        elif status is StageStatus.CANCELLED:
            replacement = replacement.start().cancel()
        invalid_stages = (replacement,) + completed.stage_states[1:]

        with pytest.raises(InvalidExecutionStateError):
            replace(
                completed,
                current_stage_id=(
                    replacement.stage_id
                    if status is StageStatus.RUNNING
                    else None
                ),
                stage_states=invalid_stages,
            )


def test_completed_task_accepts_an_intentionally_skipped_stage() -> None:
    state = _state().start(updated_at=_time(1))
    skipped = state.update_stage(
        state.stage_states[0].skip(),
        updated_at=_time(2),
    )
    current = skipped
    minute = 3
    for index in range(1, len(current.stage_states)):
        current = current.update_stage(
            current.stage_states[index].start(),
            updated_at=_time(minute),
        )
        minute += 1
        current = current.update_stage(
            current.stage_states[index].complete(
                output_reference=f"output-{index + 1}"
            ),
            updated_at=_time(minute),
        )
        minute += 1

    completed = current.complete(updated_at=_time(minute))

    assert completed.task_status is TaskStatus.COMPLETED
    assert completed.stage_states[0].status is StageStatus.SKIPPED


def test_direct_construction_rejects_cross_level_status_conflicts() -> None:
    running_task = _state().start(updated_at=_time(1))
    active = running_task.update_stage(
        running_task.stage_states[0].start(),
        updated_at=_time(2),
    )
    failed_stage = active.stage_states[0].fail(
        error_code="failed",
        safe_message="Stage failed",
    )
    failed_stages = (failed_stage,) + active.stage_states[1:]

    with pytest.raises(InvalidExecutionStateError):
        replace(
            active,
            task_status=TaskStatus.RUNNING,
            current_stage_id=None,
            stage_states=failed_stages,
        )
    with pytest.raises(InvalidExecutionStateError):
        replace(
            active,
            task_status=TaskStatus.CANCELLED,
            current_stage_id=None,
            stage_states=failed_stages,
        )
    with pytest.raises(InvalidExecutionStateError):
        replace(
            running_task,
            task_status=TaskStatus.FAILED,
        )


def test_duplicate_plan_stage_ids_are_rejected() -> None:
    duplicate_plan = TaskPlan(
        task_class=TaskClass.GENERAL,
        stages=(
            TaskStage("stage-1", TaskStageType.GENERATE, ("chat",)),
            TaskStage("stage-1", TaskStageType.REASON, ("reasoning",)),
        ),
    )

    with pytest.raises(
        InvalidExecutionStateError,
        match="unique stage IDs",
    ):
        AgentTaskState.from_plan(
            task_id="duplicate-stage-task",
            original_prompt="Explain this",
            plan=duplicate_plan,
            timestamp=CREATED_AT,
        )


def test_current_stage_id_matches_the_unique_running_stage() -> None:
    task = _state().start(updated_at=_time(1))
    active = task.update_stage(
        task.stage_states[1].start(),
        updated_at=_time(2),
    )

    assert active.current_stage_id == active.stage_states[1].stage_id
    assert tuple(
        stage.stage_id
        for stage in active.stage_states
        if stage.status is StageStatus.RUNNING
    ) == (active.current_stage_id,)

    with pytest.raises(InvalidExecutionStateError):
        replace(active, current_stage_id="stage-1")


def test_stage_update_cannot_reinterpret_plan_capabilities() -> None:
    running_task = _state().start(updated_at=_time(1))
    forged = replace(
        running_task.stage_states[0].start(),
        required_capabilities=("chat",),
    )

    with pytest.raises(InvalidExecutionStateError):
        running_task.update_stage(forged, updated_at=_time(2))


def test_stage_update_cannot_change_selected_model_after_start() -> None:
    running_task = _state().start(updated_at=_time(1))
    started = running_task.stage_states[0].start(
        selected_model_id="approved-document-model"
    )
    with_started = running_task.update_stage(started, updated_at=_time(2))
    completed = with_started.stage_states[0].complete(
        output_reference="output-stage-1"
    )
    forged = replace(completed, selected_model_id="unapproved-model")

    with pytest.raises(InvalidExecutionStateError):
        with_started.update_stage(forged, updated_at=_time(3))


def test_direct_construction_rejects_changed_plan_projection() -> None:
    state = _state()

    with pytest.raises(InvalidExecutionStateError):
        replace(
            state,
            required_capabilities=("document", "vision", "reasoning", "chat"),
        )


def test_direct_construction_rejects_inconsistent_current_stage() -> None:
    state = _state()

    with pytest.raises(InvalidExecutionStateError):
        replace(state, current_stage_id="stage-1")


def test_state_contains_no_provider_configuration_or_exception_channel() -> None:
    task_field_names = {field.name for field in fields(AgentTaskState)}
    stage_field_names = {field.name for field in fields(StageExecutionState)}

    assert not task_field_names & {
        "provider",
        "endpoint",
        "runtime_client",
        "network_session",
        "metadata",
        "exception",
    }
    assert not stage_field_names & {
        "provider",
        "endpoint",
        "runtime_configuration",
        "metadata",
        "exception",
    }

    with pytest.raises(InvalidExecutionStateError):
        StageExecutionState(
            stage_id="stage-1",
            stage_type=_state().stage_states[0].stage_type,
            required_capabilities=("document",),
            status=StageStatus.FAILED,
            error_code="failed",
            safe_message=RuntimeError("private detail"),  # type: ignore[arg-type]
        )


def test_timestamps_are_timezone_aware_utc_and_created_at_is_stable() -> None:
    state = _state()
    running = state.start(updated_at=_time(1))

    assert state.created_at.tzinfo is not None
    assert state.created_at.utcoffset() == timedelta(0)
    assert running.updated_at.tzinfo is not None
    assert running.updated_at.utcoffset() == timedelta(0)
    assert running.created_at == state.created_at
    assert running.updated_at > state.updated_at


@pytest.mark.parametrize(
    "timestamp",
    [
        datetime(2026, 9, 11, 8, 0),
        datetime(2026, 9, 11, 8, 0, tzinfo=timezone(timedelta(hours=1))),
    ],
)
def test_non_utc_or_naive_timestamps_are_rejected(timestamp: datetime) -> None:
    with pytest.raises(InvalidExecutionStateError):
        AgentTaskState.from_plan(
            task_id="task-state-fixture",
            original_prompt="Analyze this report",
            plan=_plan(),
            timestamp=timestamp,
        )


def test_transition_timestamp_must_advance() -> None:
    state = _state()

    with pytest.raises(InvalidTaskTransitionError):
        state.start(updated_at=CREATED_AT)
