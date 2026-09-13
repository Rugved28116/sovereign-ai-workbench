"""One-call, one-stage coordination over sovereign routing and immutable state."""

import asyncio
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from pathlib import Path

import pytest

from conftest import model_data, registry_data, write_registry
from sovereign_api.agent_stage_execution import (
    MAX_STAGE_OUTPUT_BYTES, InvalidStageCoordinationError,
    InvalidStageExecutionResultError,
    RoutedAgentStageExecutor, StageExecutionCoordinator, StageExecutionResult,
)
from sovereign_api.agent_task_state import AgentTaskState, StageStatus, TaskStatus
from sovereign_api.config import DeploymentEnvironment
from sovereign_api.contracts import ModelResponse
from sovereign_api.errors import (
    InvalidExecutionStateError, InvalidStepTransitionError,
    ProviderConnectionError, RegistryValidationError,
)
from sovereign_api.providers.mock import MockProvider
from sovereign_api.registry import ModelRegistry, load_registry
from sovereign_api.registry.models import MAX_MODEL_ID_LENGTH
from sovereign_api.routing import DeterministicModelRouter, RoutingCandidate
from sovereign_api.task_classification import TaskClass, TaskRequirements
from sovereign_api.task_planning import DeterministicTaskRequirementPlanner, TaskStage


NOW = datetime(2026, 9, 13, tzinfo=UTC)


def _plan(capabilities=("chat",)):
    return DeterministicTaskRequirementPlanner().plan(
        TaskRequirements(TaskClass.GENERAL, capabilities)
    )


def _task(plan=None):
    actual_plan = plan if plan is not None else _plan()
    return AgentTaskState.from_plan(
        task_id="task-1", original_prompt="Original prompt",
        plan=actual_plan, timestamp=NOW,
    )


def _router(*models, environment=DeploymentEnvironment.DEVELOPMENT, optimizer=None):
    registry = ModelRegistry.model_validate(registry_data(*models))
    return DeterministicModelRouter(registry, environment, optimizer=optimizer)


class RecordingProvider:
    def __init__(self, *, response="answer", failure=None):
        self.requests = []
        self.response = response
        self.failure = failure

    async def generate(self, request):
        self.requests.append(request)
        if self.failure is not None:
            raise self.failure
        return ModelResponse(request.model_id, self.response)


def _coordinator(router=None, provider=None):
    actual_router = router if router is not None else _router(model_data("model-chat"))
    actual_provider = provider if provider is not None else RecordingProvider()
    executor = RoutedAgentStageExecutor(
        router=actual_router, providers={"mock": actual_provider}
    )
    return StageExecutionCoordinator(executor, clock=lambda: NOW), actual_provider


def _run(coordinator, task, stage):
    return asyncio.run(coordinator.execute_one(task, stage))


def test_single_stage_routes_once_completes_task_and_preserves_input():
    task = _task()
    original_plan = task.plan
    coordinator, provider = _coordinator()

    result = _run(coordinator, task, task.plan.stages[0])

    assert len(provider.requests) == 1
    assert provider.requests[0].model_id == "model-chat"
    assert provider.requests[0].prompt == "Original prompt"
    assert result.task_status is TaskStatus.COMPLETED
    assert result.stage_states[0].status is StageStatus.COMPLETED
    assert result.stage_states[0].selected_model_id == "model-chat"
    assert result.stage_states[0].output_reference.startswith("sha256:")
    assert result.current_stage_id is None
    assert task.task_status is TaskStatus.PENDING
    assert task.stage_states[0].status is StageStatus.PENDING
    assert task.plan is original_plan and task.plan.stages == original_plan.stages


def test_existing_mock_provider_executes_without_network():
    coordinator, _ = _coordinator(provider=MockProvider())
    task = _task()
    result = _run(coordinator, task, task.plan.stages[0])
    assert result.task_status is TaskStatus.COMPLETED
    assert result.stage_states[0].selected_model_id == "model-chat"


def test_maximum_length_registry_model_id_routes_and_executes():
    model_id = "m" * MAX_MODEL_ID_LENGTH
    provider = RecordingProvider()
    coordinator, _ = _coordinator(_router(model_data(model_id)), provider)
    task = _task()
    result = _run(coordinator, task, task.plan.stages[0])
    assert result.task_status is TaskStatus.COMPLETED
    assert result.stage_states[0].selected_model_id == model_id
    assert len(provider.requests) == 1


@pytest.mark.parametrize("invalid_model_id", [
    "m" * (MAX_MODEL_ID_LENGTH + 1), "model\x80id",
])
def test_invalid_registry_model_id_is_rejected_before_provider(
    tmp_path: Path, invalid_model_id: str,
):
    provider = RecordingProvider()
    path = write_registry(tmp_path, registry_data(
        model_data(invalid_model_id)
    ))
    with pytest.raises(RegistryValidationError):
        registry = load_registry(path)
        _coordinator(DeterministicModelRouter(registry, DeploymentEnvironment.DEVELOPMENT), provider)
    assert provider.requests == []


def test_two_stages_advance_only_one_at_a_time_and_use_original_prompt():
    plan = _plan(("document", "reasoning"))
    provider = RecordingProvider()
    coordinator, _ = _coordinator(
        _router(
            model_data("document", capabilities=["document"]),
            model_data("reason", capabilities=["reasoning"]),
        ), provider,
    )
    initial = _task(plan)
    first = _run(coordinator, initial, plan.stages[0])
    assert first.task_status is TaskStatus.RUNNING
    assert tuple(stage.status for stage in first.stage_states) == (
        StageStatus.COMPLETED, StageStatus.PENDING,
    )
    assert first.stage_states[0].selected_model_id == "document"
    assert len(provider.requests) == 1
    second = _run(coordinator, first, plan.stages[1])
    assert second.task_status is TaskStatus.COMPLETED
    assert tuple(stage.selected_model_id for stage in second.stage_states) == (
        "document", "reason",
    )
    assert [request.prompt for request in provider.requests] == ["Original prompt"] * 2
    assert first.stage_states[1].status is StageStatus.PENDING


@pytest.mark.parametrize("capability", ["chat", "coding", "document", "vision", "reasoning"])
def test_each_model_stage_routes_its_own_capability(capability):
    plan = _plan((capability,))
    provider = RecordingProvider()
    coordinator, _ = _coordinator(
        _router(model_data("matching", capabilities=[capability])), provider,
    )
    result = _run(coordinator, _task(plan), plan.stages[0])
    assert result.stage_states[0].selected_model_id == "matching"
    assert len(provider.requests) == 1


def test_wrong_order_reexecution_forgery_and_terminal_task_make_zero_calls():
    plan = _plan(("document", "reasoning"))
    coordinator, provider = _coordinator(
        _router(model_data("multi", capabilities=["document", "reasoning"]))
    )
    initial = _task(plan)
    with pytest.raises(InvalidStageCoordinationError):
        _run(coordinator, initial, plan.stages[1])
    forged = type(plan.stages[0])("stage-1", plan.stages[0].stage_type, ("chat",))
    with pytest.raises(InvalidStageCoordinationError):
        _run(coordinator, initial, forged)
    wrong_type = TaskStage("stage-1", "document", ("document",))
    with pytest.raises(InvalidStageCoordinationError):
        _run(coordinator, initial, wrong_type)
    first = _run(coordinator, initial, plan.stages[0])
    with pytest.raises(InvalidStageCoordinationError):
        _run(coordinator, first, plan.stages[0])
    second = _run(coordinator, first, plan.stages[1])
    with pytest.raises(InvalidStageCoordinationError):
        _run(coordinator, second, plan.stages[1])
    assert len(provider.requests) == 2


@pytest.mark.parametrize("environment", [
    DeploymentEnvironment.ON_PREM, DeploymentEnvironment.AIR_GAPPED,
])
def test_sovereignty_rejects_development_mock(environment):
    coordinator, provider = _coordinator(
        _router(model_data("mock-only"), environment=environment)
    )
    task = _task()
    result = _run(coordinator, task, task.plan.stages[0])
    assert result.task_status is TaskStatus.FAILED
    assert result.stage_states[0].error_code == "routing_failed"
    assert provider.requests == []


def test_no_capability_eligible_model_fails_without_provider_call():
    coordinator, provider = _coordinator(
        _router(model_data("coding-only", capabilities=["coding"]))
    )
    task = _task()
    result = _run(coordinator, task, task.plan.stages[0])
    assert result.task_status is TaskStatus.FAILED
    assert result.stage_states[0].error_code == "routing_failed"
    assert provider.requests == []


def test_optimizer_cannot_reintroduce_rejected_model():
    class MaliciousOptimizer:
        def select(self, candidates):
            return RoutingCandidate("disabled", 1, frozenset({"chat"}), 8192)

    coordinator, provider = _coordinator(_router(
        model_data("allowed"), model_data("disabled", enabled=False),
        optimizer=MaliciousOptimizer(),
    ))
    task = _task()
    result = _run(coordinator, task, task.plan.stages[0])
    assert result.task_status is TaskStatus.FAILED
    assert result.stage_states[0].error_code == "routing_failed"
    assert provider.requests == []


@pytest.mark.parametrize("failure", [
    ProviderConnectionError("/private/provider/path"), RuntimeError("/private/provider/path"),
])
def test_provider_failures_are_sanitized_and_fail_stage_and_task(failure):
    provider = RecordingProvider(failure=failure)
    coordinator, _ = _coordinator(provider=provider)
    task = _task()
    result = _run(coordinator, task, task.plan.stages[0])
    assert result.task_status is TaskStatus.FAILED
    assert result.stage_states[0].status is StageStatus.FAILED
    assert result.current_stage_id is None
    assert result.stage_states[0].error_code == "provider_failed"
    assert "/private/provider/path" not in repr(result)
    assert len(provider.requests) == 1


def test_unexpected_router_exception_uses_sanitized_failure():
    class BrokenRouter:
        def route(self, capabilities):
            raise RuntimeError("/private/router/path")

    coordinator, provider = _coordinator(BrokenRouter())
    task = _task()
    result = _run(coordinator, task, task.plan.stages[0])
    assert result.task_status is TaskStatus.FAILED
    assert result.stage_states[0].error_code == "stage_unexpected_failure"
    assert result.stage_states[0].safe_message == "Stage execution failed unexpectedly"
    assert "/private/router/path" not in repr(result)
    assert provider.requests == []


def test_malformed_provider_response_and_oversized_output_fail_safely():
    class WrongModelProvider(RecordingProvider):
        async def generate(self, request):
            self.requests.append(request)
            return ModelResponse("wrong", "answer")

    for provider in (WrongModelProvider(), RecordingProvider(response="x" * (MAX_STAGE_OUTPUT_BYTES + 1))):
        coordinator, _ = _coordinator(provider=provider)
        task = _task()
        result = _run(coordinator, task, task.plan.stages[0])
        assert result.task_status is TaskStatus.FAILED
        assert result.stage_states[0].error_code == "provider_response_invalid"


def test_executor_failure_and_malformed_result_fail_closed_without_raw_details():
    class BrokenExecutor:
        async def execute(self, task, stage):
            raise OSError("/private/host/path")

    class MalformedExecutor:
        async def execute(self, task, stage):
            return object()

    for executor, expected_code in (
        (BrokenExecutor(), "stage_unexpected_failure"),
        (MalformedExecutor(), "stage_invalid_result"),
    ):
        task = _task()
        result = _run(StageExecutionCoordinator(executor, clock=lambda: NOW), task, task.plan.stages[0])
        assert result.task_status is TaskStatus.FAILED
        assert result.stage_states[0].error_code == expected_code
        assert "/private/host/path" not in repr(result)


def test_falsey_executor_is_used_and_output_reference_is_immutable():
    class FalseyExecutor:
        def __init__(self):
            self.calls = 0

        def __bool__(self):
            return False

        async def execute(self, task, stage):
            self.calls += 1
            return StageExecutionResult(stage.stage_id, StageStatus.COMPLETED, "result-1")

    executor = FalseyExecutor()
    task = _task()
    result = _run(StageExecutionCoordinator(executor, clock=lambda: NOW), task, task.plan.stages[0])
    assert executor.calls == 1
    assert result.task_status is TaskStatus.COMPLETED
    with pytest.raises(FrozenInstanceError):
        result.stage_states[0].output_reference = "modified"


def test_stage_result_is_frozen_and_rejects_host_output_path():
    value = StageExecutionResult("stage-1", StageStatus.COMPLETED, "result-1")
    with pytest.raises(FrozenInstanceError):
        value.output_reference = "changed"
    with pytest.raises(InvalidStageExecutionResultError):
        StageExecutionResult("stage-1", StageStatus.COMPLETED, "/private/host/path")


def test_existing_selected_model_cannot_be_changed_by_later_transition():
    task = _task().start(updated_at=NOW.replace(microsecond=1))
    running = task.update_stage(
        task.stage_states[0].start(selected_model_id="approved"),
        updated_at=NOW.replace(microsecond=2),
    )
    with pytest.raises(InvalidStepTransitionError):
        running.stage_states[0].complete(
            output_reference="result-1", selected_model_id="other",
        )


class CountingClock:
    def __init__(self, *, fail_on: int | None = None, invalid_on: int | None = None):
        self.calls = 0
        self.fail_on = fail_on
        self.invalid_on = invalid_on

    def __call__(self):
        self.calls += 1
        if self.calls == self.fail_on:
            raise RuntimeError("private clock diagnostic")
        if self.calls == self.invalid_on:
            return datetime(2026, 9, 13)
        return NOW


@pytest.mark.parametrize("fail_on,invalid_on", [
    (1, None), (3, None), (4, None), (None, 2), (None, 4),
])
def test_clock_failure_before_execution_makes_zero_provider_calls(
    fail_on, invalid_on,
):
    provider = RecordingProvider()
    clock = CountingClock(fail_on=fail_on, invalid_on=invalid_on)
    executor = RoutedAgentStageExecutor(
        router=_router(model_data("model-chat")), providers={"mock": provider},
    )
    coordinator = StageExecutionCoordinator(executor, clock=clock)
    task = _task()
    with pytest.raises(InvalidExecutionStateError) as captured:
        _run(coordinator, task, task.plan.stages[0])
    assert "private clock diagnostic" not in str(captured.value)
    assert provider.requests == []
    assert task.task_status is TaskStatus.PENDING


@pytest.mark.parametrize("failure,expected_status", [
    (None, TaskStatus.COMPLETED),
    (ProviderConnectionError("private provider diagnostic"), TaskStatus.FAILED),
])
def test_no_clock_access_after_provider_invocation(failure, expected_status):
    provider = RecordingProvider(failure=failure)
    clock = CountingClock(fail_on=5)
    executor = RoutedAgentStageExecutor(
        router=_router(model_data("model-chat")), providers={"mock": provider},
    )
    coordinator = StageExecutionCoordinator(executor, clock=clock)
    task = _task()
    result = _run(coordinator, task, task.plan.stages[0])
    assert result.task_status is expected_status
    assert len(provider.requests) == 1
    assert clock.calls == 4
    assert result.updated_at > task.updated_at


def test_active_or_cancelled_task_cannot_execute():
    task = _task()
    running = task.start(updated_at=NOW.replace(microsecond=1))
    active = running.update_stage(
        running.stage_states[0].start(), updated_at=NOW.replace(microsecond=2),
    )
    coordinator, provider = _coordinator()
    with pytest.raises(InvalidStageCoordinationError):
        _run(coordinator, active, task.plan.stages[0])
    cancelled = active.cancel(updated_at=NOW.replace(microsecond=3))
    with pytest.raises(InvalidStageCoordinationError):
        _run(coordinator, cancelled, task.plan.stages[0])
    assert provider.requests == []
