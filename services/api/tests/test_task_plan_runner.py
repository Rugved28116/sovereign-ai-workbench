"""Bounded full-plan runs delegate every stage to the existing coordinator."""

import asyncio
import traceback
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta

import pytest

from conftest import model_data, registry_data
from sovereign_api.agent_stage_execution import (
    RoutedAgentStageExecutor, StageCoordinationReport, StageExecutionCoordinator,
)
from sovereign_api.agent_task_state import AgentTaskState, StageStatus, TaskStatus
from sovereign_api.config import DeploymentEnvironment
from sovereign_api.contracts import ModelResponse
from sovereign_api.registry import ModelRegistry
from sovereign_api.routing import DeterministicModelRouter
from sovereign_api.stage_output_store import InMemoryStageOutputStore, StageOutputReference
from sovereign_api.task_classification import TaskClass, TaskRequirements
from sovereign_api.task_plan_runner import (
    InvalidTaskRunError, TaskPlanRunner, TaskRunExecutionError, TaskRunResult,
    stage_requires_model_invocation,
)
from sovereign_api.task_planning import (
    DeterministicTaskRequirementPlanner, TaskPlan, TaskStage, TaskStageType,
)


NOW = datetime(2026, 9, 13, tzinfo=UTC)


def _plan(capabilities=("chat",)):
    return DeterministicTaskRequirementPlanner().plan(
        TaskRequirements(TaskClass.GENERAL, capabilities)
    )


def _state(plan=None):
    return AgentTaskState.from_plan(
        task_id="task-1", original_prompt="Original request",
        plan=plan if plan is not None else _plan(), timestamp=NOW,
    )


class RecordingProvider:
    def __init__(self, *, fail_on=None, text_prefix="output"):
        self.requests = []
        self.fail_on = fail_on
        self.text_prefix = text_prefix

    async def generate(self, request):
        self.requests.append(request)
        if len(self.requests) == self.fail_on:
            raise RuntimeError("/private/provider/diagnostic")
        return ModelResponse(
            request.model_id, f"{self.text_prefix}-{len(self.requests)}",
        )


def _runner(*, provider=None, store=None, max_stages=16, max_calls=16, executor_wrapper=None):
    actual_provider = provider if provider is not None else RecordingProvider()
    actual_store = store if store is not None else InMemoryStageOutputStore()
    registry = ModelRegistry.model_validate(registry_data(
        model_data("chat-model", capabilities=["chat"]),
        model_data("document-model", capabilities=["document"]),
        model_data("vision-model", capabilities=["vision"]),
        model_data("reason-model", capabilities=["reasoning"]),
    ))
    routed_executor = RoutedAgentStageExecutor(
        router=DeterministicModelRouter(registry, DeploymentEnvironment.DEVELOPMENT),
        providers={"mock": actual_provider},
    )
    coordinator = StageExecutionCoordinator(
        executor_wrapper(routed_executor) if executor_wrapper is not None else routed_executor,
        output_store=actual_store, clock=lambda: NOW,
    )
    return (
        TaskPlanRunner(
            coordinator,
            max_stages_per_run=max_stages,
            max_total_model_invocations_per_run=max_calls,
        ),
        actual_provider,
        actual_store,
        coordinator,
    )


def _run(runner, state):
    return asyncio.run(runner.run(state))


def test_single_stage_runs_once_and_result_contains_no_raw_output():
    runner, provider, store, _ = _runner()
    initial = _state()
    result = _run(runner, initial)
    assert result.final_state.task_status is TaskStatus.COMPLETED
    assert result.terminal_status is TaskStatus.COMPLETED
    assert (result.stages_executed, result.model_invocations) == (1, 1)
    assert len(provider.requests) == 1
    assert provider.requests[0].model_id == "chat-model"
    assert initial.task_status is TaskStatus.PENDING
    assert initial.stage_states[0].status is StageStatus.PENDING
    assert result.final_state is not initial
    reference = StageOutputReference(result.final_state.stage_states[0].output_reference)
    assert store.get(reference, task_id="task-1", stage_id="stage-1").text_content == "output-1"
    assert "output-1" not in repr(result)
    with pytest.raises(FrozenInstanceError):
        result.stages_executed = 2


def test_three_stages_use_latest_snapshot_exact_order_and_same_output_store():
    plan = _plan(("document", "vision", "reasoning"))
    runner, provider, store, coordinator = _runner()

    class RecordingCoordinator:
        def __init__(self):
            self.inputs = []
            self.outputs = []

        async def execute_one_with_report(self, state, stage):
            self.inputs.append((state, stage))
            report = await coordinator.execute_one_with_report(state, stage)
            self.outputs.append(report.state)
            return report

    recorder = RecordingCoordinator()
    runner = TaskPlanRunner(
        recorder, max_stages_per_run=3,
        max_total_model_invocations_per_run=3,
    )
    initial = _state(plan)
    result = _run(runner, initial)
    assert result.final_state.task_status is TaskStatus.COMPLETED
    assert (result.stages_executed, result.model_invocations) == (3, 3)
    assert [stage.stage_id for _, stage in recorder.inputs] == [
        "stage-1", "stage-2", "stage-3",
    ]
    assert recorder.inputs[0][0] is initial
    assert recorder.inputs[1][0] is recorder.outputs[0]
    assert recorder.inputs[2][0] is recorder.outputs[1]
    assert [request.model_id for request in provider.requests] == [
        "document-model", "vision-model", "reason-model",
    ]
    assert "output-1" in provider.requests[1].prompt
    assert "output-2" in provider.requests[2].prompt
    assert "output-1" not in provider.requests[2].prompt
    assert result.final_state.plan is plan
    assert initial.plan.stages == plan.stages
    assert all(stage.status is StageStatus.COMPLETED for stage in result.final_state.stage_states)
    references = [stage.output_reference for stage in result.final_state.stage_states]
    assert len(set(references)) == 3
    assert store.get(StageOutputReference(references[2]), task_id="task-1", stage_id="stage-3").text_content == "output-3"


def test_second_stage_failure_stops_without_retry_or_third_provider_call():
    plan = _plan(("document", "vision", "reasoning"))
    runner, provider, _, _ = _runner(provider=RecordingProvider(fail_on=2))
    result = _run(runner, _state(plan))
    assert (result.stages_executed, result.model_invocations) == (2, 2)
    assert result.final_state.task_status is TaskStatus.FAILED
    assert [stage.status for stage in result.final_state.stage_states] == [
        StageStatus.COMPLETED, StageStatus.FAILED, StageStatus.PENDING,
    ]
    assert result.final_state.stage_states[1].error_code == "provider_failed"
    assert len(provider.requests) == 2
    assert "/private/provider/diagnostic" not in repr(result)


def test_already_terminal_states_make_zero_additional_calls():
    runner, provider, _, _ = _runner()
    completed = _run(runner, _state()).final_state
    failed = _state().start(updated_at=NOW + timedelta(microseconds=1)).fail(
        updated_at=NOW + timedelta(microseconds=2),
    )
    cancelled = _state().start(updated_at=NOW + timedelta(microseconds=1)).cancel(
        updated_at=NOW + timedelta(microseconds=2),
    )
    for terminal in (completed, failed, cancelled):
        before = len(provider.requests)
        result = _run(runner, terminal)
        assert result.final_state is terminal
        assert result.terminal_status is terminal.task_status
        assert (result.stages_executed, result.model_invocations) == (0, 0)
        assert len(provider.requests) == before


@pytest.mark.parametrize("limit", [0, -1, 17, True, 1.5])
def test_invalid_stage_or_model_limits_fail_before_execution(limit):
    runner, provider, _, coordinator = _runner()
    for name in ("max_stages_per_run", "max_total_model_invocations_per_run"):
        with pytest.raises(InvalidTaskRunError):
            TaskPlanRunner(coordinator, **{name: limit})
    assert provider.requests == []


def test_plan_over_stage_limit_rejected_before_provider_call():
    plan = _plan(("document", "vision", "reasoning"))
    runner, provider, _, _ = _runner(max_stages=2)
    with pytest.raises(InvalidTaskRunError):
        _run(runner, _state(plan))
    assert provider.requests == []


def test_hard_upper_bound_rejects_seventeen_stage_plan_before_provider():
    plan = TaskPlan(
        TaskClass.GENERAL,
        tuple(TaskStage(f"stage-{index}", TaskStageType.GENERATE, ("chat",))
              for index in range(1, 18)),
    )
    runner, provider, _, _ = _runner()
    with pytest.raises(InvalidTaskRunError):
        _run(runner, _state(plan))
    assert provider.requests == []


def test_model_invocation_runtime_limit_stops_before_next_stage_and_can_resume():
    plan = _plan(("document", "vision", "reasoning"))
    runner, provider, _, _ = _runner(max_calls=2)
    initial = _state(plan)
    partial = _run(runner, initial)
    assert partial.final_state.task_status is TaskStatus.RUNNING
    assert partial.error_code == "model_invocation_limit_reached"
    assert (partial.stages_executed, partial.model_invocations) == (2, 2)
    assert len(provider.requests) == 2
    resumed = _run(runner, partial.final_state)
    assert resumed.final_state.task_status is TaskStatus.COMPLETED
    assert (resumed.stages_executed, resumed.model_invocations) == (1, 1)
    assert len(provider.requests) == 3


def test_no_eligible_model_counts_attempted_stage_but_zero_provider_calls():
    plan = _plan(("coding",))
    runner, provider, _, _ = _runner()
    result = _run(runner, _state(plan))
    assert result.final_state.task_status is TaskStatus.FAILED
    assert result.final_state.stage_states[0].error_code == "routing_failed"
    assert (result.stages_executed, result.model_invocations) == (1, 1)
    assert provider.requests == []


@pytest.mark.parametrize("reported", [0, 1])
def test_trusted_budget_blocks_second_call_even_when_executor_underreports(reported):
    class ReportingExecutor:
        def __init__(self, routed):
            self.routed = routed

        async def execute(self, task, stage, prompt):
            actual = await self.routed.execute(task, stage, prompt)
            return replace(actual, model_invocations=reported)

    plan = _plan(("document", "reasoning"))
    runner, provider, _, _ = _runner(max_calls=1, executor_wrapper=ReportingExecutor)
    result = _run(runner, _state(plan))
    assert result.final_state.task_status is TaskStatus.RUNNING
    assert result.error_code == "model_invocation_limit_reached"
    assert (result.stages_executed, result.model_invocations) == (1, 1)
    assert len(provider.requests) == 1
    assert result.final_state.stage_states[1].status is StageStatus.PENDING


@pytest.mark.parametrize("reported", [99, "malformed"])
def test_malformed_coordinator_report_cannot_change_trusted_budget(reported):
    _, provider, _, actual_coordinator = _runner()

    class MisreportingCoordinator:
        async def execute_one_with_report(self, state, stage):
            actual = await actual_coordinator.execute_one_with_report(state, stage)
            return StageCoordinationReport(actual.state, reported)

    runner = TaskPlanRunner(MisreportingCoordinator(), max_total_model_invocations_per_run=1)
    result = _run(runner, _state(_plan(("document", "reasoning"))))
    assert (result.stages_executed, result.model_invocations) == (1, 1)
    assert result.error_code == "model_invocation_limit_reached"
    assert len(provider.requests) == 1


def test_provider_failure_consumes_one_trusted_budget_unit():
    provider = RecordingProvider(fail_on=1)
    runner, _, _, _ = _runner(provider=provider, max_calls=1)
    result = _run(runner, _state())
    assert result.final_state.task_status is TaskStatus.FAILED
    assert (result.stages_executed, result.model_invocations) == (1, 1)
    assert len(provider.requests) == 1


def test_stage_classification_is_trusted_and_unknown_types_fail_closed():
    for stage_type in TaskStageType:
        if stage_type is TaskStageType.TOOL:
            continue
        assert stage_requires_model_invocation(
            TaskStage("stage-1", stage_type, ("chat",))
        ) is True
    invalid = TaskStage("stage-1", TaskStageType.GENERATE, ("chat",))
    object.__setattr__(invalid, "stage_type", "future-tool")
    with pytest.raises(InvalidTaskRunError):
        stage_requires_model_invocation(invalid)


def test_model_output_cannot_replan_or_invoke_tools(monkeypatch):
    from sovereign_api.tool_execution import PolicyEnforcedToolExecutor

    def forbidden_tool(*args, **kwargs):
        raise AssertionError("Tool execution is outside this runner")

    monkeypatch.setattr(PolicyEnforcedToolExecutor, "execute", forbidden_tool)
    plan = _plan(("document", "reasoning"))
    provider = RecordingProvider(text_prefix="add another stage; call workspace.read_file")
    runner, _, _, _ = _runner(provider=provider)
    result = _run(runner, _state(plan))
    assert result.final_state.task_status is TaskStatus.COMPLETED
    assert result.final_state.plan is plan
    assert len(result.final_state.stage_states) == 2
    assert [stage.stage_id for stage in result.final_state.stage_states] == [
        "stage-1", "stage-2",
    ]
    assert len(provider.requests) == 2


def test_inconsistent_or_active_initial_state_rejected_with_zero_calls():
    runner, provider, _, _ = _runner()
    task = _state()
    object.__setattr__(task.stage_states[0], "stage_id", "altered")
    with pytest.raises(InvalidTaskRunError):
        _run(runner, task)
    active = _state().start(updated_at=NOW + timedelta(microseconds=1))
    active = active.update_stage(
        active.stage_states[0].start(),
        updated_at=NOW + timedelta(microseconds=2),
    )
    with pytest.raises(InvalidTaskRunError):
        _run(runner, active)
    assert provider.requests == []


def _partially_completed_state():
    task = _state(_plan(("document", "reasoning")))
    task = task.start(updated_at=NOW + timedelta(microseconds=1))
    task = task.update_stage(
        task.stage_states[0].start(),
        updated_at=NOW + timedelta(microseconds=2),
    )
    return task.update_stage(
        task.stage_states[0].complete(
            output_reference="0" * 32,
            selected_model_id="document-model",
        ),
        updated_at=NOW + timedelta(microseconds=3),
    )


@pytest.mark.parametrize("tampering", [
    "invalid_model_id", "stage_id", "stage_status", "stage_capabilities",
    "missing_output", "current_stage_id", "plan_stage_id", "plan_capabilities",
])
def test_deep_preflight_rejects_tampered_nested_state_without_coordinator_call(tampering):
    class CountingCoordinator:
        def __init__(self):
            self.calls = 0

        async def execute_one_with_report(self, state, stage):
            self.calls += 1
            raise AssertionError("Invalid state must not reach coordinator")

    state = _partially_completed_state()
    if tampering == "invalid_model_id":
        object.__setattr__(state.stage_states[0], "selected_model_id", "model\u0080id")
    elif tampering == "stage_id":
        object.__setattr__(state.stage_states[0], "stage_id", "forged-stage")
    elif tampering == "stage_status":
        object.__setattr__(state.stage_states[0], "status", StageStatus.RUNNING)
    elif tampering == "stage_capabilities":
        object.__setattr__(state.stage_states[0], "required_capabilities", ("coding",))
    elif tampering == "missing_output":
        object.__setattr__(state.stage_states[0], "output_reference", None)
    elif tampering == "current_stage_id":
        object.__setattr__(state, "current_stage_id", "stage-1")
    elif tampering == "plan_stage_id":
        object.__setattr__(state.plan.stages[0], "stage_id", "forged-stage")
    elif tampering == "plan_capabilities":
        object.__setattr__(state.plan.stages[0], "required_capabilities", ["document"])
    coordinator = CountingCoordinator()
    with pytest.raises(InvalidTaskRunError):
        _run(TaskPlanRunner(coordinator), state)
    assert coordinator.calls == 0


def test_out_of_order_completed_stage_is_rejected_before_provider():
    plan = _plan(("document", "reasoning"))
    original = _state(plan)
    completed_later = original.stage_states[1].start().complete(
        output_reference="0" * 32,
    )
    inconsistent = replace(
        original,
        task_status=TaskStatus.RUNNING,
        stage_states=(original.stage_states[0], completed_later),
        updated_at=NOW + timedelta(microseconds=1),
    )
    runner, provider, _, _ = _runner()
    with pytest.raises(InvalidTaskRunError):
        _run(runner, inconsistent)
    assert provider.requests == []


def test_missing_previous_output_raises_safe_run_error_before_provider():
    plan = _plan(("document", "reasoning"))
    provider = RecordingProvider()
    _, _, _, first_coordinator = _runner(provider=provider)
    first = asyncio.run(first_coordinator.execute_one(
        _state(plan), plan.stages[0],
    ))
    restarted_runner, _, _, _ = _runner(
        provider=provider, store=InMemoryStageOutputStore(),
    )
    with pytest.raises(TaskRunExecutionError) as captured:
        _run(restarted_runner, first)
    assert captured.value.last_confirmed_state is first
    assert (captured.value.stages_executed, captured.value.model_invocations) == (0, 1)
    assert len(provider.requests) == 1


def test_unexpected_coordinator_error_is_typed_safe_and_does_not_retry():
    class BrokenCoordinator:
        def __init__(self):
            self.calls = 0

        async def execute_one_with_report(self, state, stage):
            self.calls += 1
            raise OSError("/private/host/path")

    coordinator = BrokenCoordinator()
    runner = TaskPlanRunner(coordinator)
    initial = _state()
    with pytest.raises(TaskRunExecutionError) as captured:
        _run(runner, initial)
    error = captured.value
    assert coordinator.calls == 1
    assert error.last_confirmed_state is initial
    assert (error.stages_executed, error.model_invocations) == (0, 1)
    assert error.__cause__ is None
    assert error.__context__ is None
    assert "/private/host/path" not in str(error)
    assert "/private/host/path" not in "".join(traceback.format_exception(error))


def test_nonadvancing_coordinator_cannot_repeat_a_stage():
    class NonadvancingCoordinator:
        def __init__(self):
            self.calls = 0

        async def execute_one_with_report(self, state, stage):
            self.calls += 1
            return StageCoordinationReport(state, 0)

    coordinator = NonadvancingCoordinator()
    runner = TaskPlanRunner(coordinator)
    with pytest.raises(TaskRunExecutionError):
        _run(runner, _state())
    assert coordinator.calls == 1


def test_runner_stops_if_a_coordinator_returns_cancelled_task():
    class CancellingCoordinator:
        def __init__(self):
            self.calls = 0

        async def execute_one_with_report(self, state, stage):
            self.calls += 1
            started = state.start(updated_at=NOW + timedelta(microseconds=1))
            started = started.update_stage(
                started.stage_states[0].start(),
                updated_at=NOW + timedelta(microseconds=2),
            )
            cancelled = started.update_stage(
                started.stage_states[0].cancel(),
                updated_at=NOW + timedelta(microseconds=3),
            )
            return StageCoordinationReport(cancelled, 0)

    coordinator = CancellingCoordinator()
    runner = TaskPlanRunner(coordinator)
    result = _run(runner, _state(_plan(("document", "reasoning"))))
    assert result.final_state.task_status is TaskStatus.CANCELLED
    assert (result.stages_executed, result.model_invocations) == (1, 1)
    assert coordinator.calls == 1


def test_run_result_rejects_invalid_counts():
    state = _state()
    with pytest.raises(InvalidTaskRunError):
        TaskRunResult(state, -1, 0, TaskStatus.PENDING, "Invalid")
    with pytest.raises(InvalidTaskRunError):
        TaskRunResult(state, 1, -1, TaskStatus.PENDING, "Invalid")
    with pytest.raises(InvalidTaskRunError):
        TaskRunResult(state, 1, 2, TaskStatus.PENDING, "Invalid")
