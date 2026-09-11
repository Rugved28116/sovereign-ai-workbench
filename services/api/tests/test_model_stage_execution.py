import asyncio
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from conftest import model_data, registry_data
from sovereign_api.agent_execution import AgentTask, StepResult, StepOutputType, StepStatus, TaskStatus
from sovereign_api.api_models import GenerateRequest
from sovereign_api.config import DeploymentEnvironment
from sovereign_api.contracts import ModelResponse
from sovereign_api.errors import (
    StageExecutionError, ProviderConfigurationError, ProviderConnectionError,
    ProviderResponseError,
)
from sovereign_api.model_stage_execution import (
    TaskExecutionInput, ModelStageExecutor, build_stage_context,
    MAX_STAGE_CONTEXT_LENGTH, MODEL_STAGE_FAILURE,
)
from sovereign_api.orchestration import SequentialTaskOrchestrator
from sovereign_api.orchestration import UNEXPECTED_STAGE_FAILURE
from sovereign_api.providers.mock import MockProvider
from sovereign_api.registry import ModelRegistry
from sovereign_api.routing import DeterministicModelRouter, RoutingCandidate
from sovereign_api.task_classification import TaskRequirements, TaskClass
from sovereign_api.task_planning import DeterministicTaskRequirementPlanner

NOW = datetime(2026, 9, 11, tzinfo=UTC)


def plan(caps=("chat",)):
    return DeterministicTaskRequirementPlanner().plan(
        TaskRequirements(TaskClass.GENERAL, caps)
    )


def router(*models, environment=DeploymentEnvironment.DEVELOPMENT, optimizer=None):
    registry = ModelRegistry.model_validate(registry_data(*models))
    return DeterministicModelRouter(registry, environment, optimizer=optimizer)


def executor(route, provider=None):
    return ModelStageExecutor(
        execution_input=TaskExecutionInput("Original question"),
        router=route,
        providers={"mock": provider if provider is not None else MockProvider()},
    )


def run(execute, task_plan=None):
    return asyncio.run(SequentialTaskOrchestrator(execute, clock=lambda: NOW).execute(
        task_plan if task_plan is not None else plan(), task_id="task-1"
    ))


class RecordingProvider:
    def __init__(self):
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)
        return ModelResponse(request.model_id, f"output-{len(self.requests)}")


@pytest.mark.parametrize("capability", ["chat", "coding"])
def test_stage_routes_using_declared_capability(capability):
    route = router(
        model_data("chat-model"),
        model_data("code-model", capabilities=["coding"]),
    )
    result = run(executor(route), plan((capability,)))
    assert result.status is TaskStatus.SUCCEEDED
    expected = "chat-model" if capability == "chat" else "code-model"
    assert result.steps[0].result.content == f"Mock response from {expected}"


def test_each_stage_routes_independently_and_receives_ordered_context():
    provider = RecordingProvider()
    route = router(*(model_data(c, capabilities=[c]) for c in (
        "document", "vision", "reasoning"
    )))
    task_plan = plan(("document", "vision", "reasoning"))
    original = task_plan.stages
    result = run(executor(route, provider), task_plan)
    assert result.status is TaskStatus.SUCCEEDED
    assert [r.model_id for r in provider.requests] == ["document", "vision", "reasoning"]
    assert provider.requests[2].prompt == (
        "Original task:\nOriginal question\n\nPrevious stage results:\n"
        "[stage-1 / document]\noutput-1\n\n"
        "[stage-2 / vision]\noutput-2\n\nCurrent stage:\nreason"
    )
    assert "output-2" not in provider.requests[1].prompt
    assert task_plan.stages == original
    assert result.revision == 8


@pytest.mark.parametrize("environment", [
    DeploymentEnvironment.ON_PREM, DeploymentEnvironment.AIR_GAPPED,
])
def test_mock_is_rejected_outside_development(environment):
    provider = RecordingProvider()
    task = run(executor(router(model_data("mock"), environment=environment), provider))
    assert task.status is TaskStatus.FAILED
    assert not provider.requests


@pytest.mark.parametrize("model", [
    model_data("disabled", enabled=False),
    model_data("mismatch", capabilities=["coding"]),
])
def test_ineligible_models_fail_closed(model):
    provider = RecordingProvider()
    execute = executor(router(model), provider)
    task = AgentTask.from_plan(task_id="x", plan=plan(), timestamp=NOW).start(updated_at=NOW)
    task = task.start_step("stage-1", updated_at=NOW)
    with pytest.raises(StageExecutionError, match=MODEL_STAGE_FAILURE):
        asyncio.run(execute.execute(task, task.steps[0]))
    assert not provider.requests
    assert run(execute).status is TaskStatus.FAILED


def test_optimizer_cannot_reintroduce_rejected_model():
    class MaliciousOptimizer:
        def select(self, candidates):
            return RoutingCandidate("disabled", 1, frozenset({"chat"}), 8192)

    provider = RecordingProvider()
    route = router(model_data("allowed"), model_data("disabled", enabled=False),
                   optimizer=MaliciousOptimizer())
    execute = executor(route, provider)
    task = AgentTask.from_plan(
        task_id="x", plan=plan(), timestamp=NOW
    ).start(updated_at=NOW)
    task = task.start_step("stage-1", updated_at=NOW)
    with pytest.raises(StageExecutionError, match=MODEL_STAGE_FAILURE):
        asyncio.run(execute.execute(task, task.steps[0]))
    assert run(execute).status is TaskStatus.FAILED
    assert not provider.requests


def test_unexpected_router_fault_propagates_from_model_stage_executor():
    class BrokenRouter:
        def route(self, required_capabilities):
            raise RuntimeError("confidential router diagnostic")

    execute = ModelStageExecutor(
        execution_input=TaskExecutionInput("Hello"),
        router=BrokenRouter(),
        providers={},
    )
    task = AgentTask.from_plan(
        task_id="x", plan=plan(), timestamp=NOW
    ).start(updated_at=NOW)
    task = task.start_step("stage-1", updated_at=NOW)

    with pytest.raises(RuntimeError, match="confidential router diagnostic"):
        asyncio.run(execute.execute(task, task.steps[0]))


def test_unexpected_optimizer_fault_propagates_from_routing():
    class BrokenOptimizer:
        def select(self, candidates):
            raise RuntimeError("confidential optimizer diagnostic")

    execute = executor(router(model_data("model"), optimizer=BrokenOptimizer()))
    task = AgentTask.from_plan(
        task_id="x", plan=plan(), timestamp=NOW
    ).start(updated_at=NOW)
    task = task.start_step("stage-1", updated_at=NOW)

    with pytest.raises(RuntimeError, match="confidential optimizer diagnostic"):
        asyncio.run(execute.execute(task, task.steps[0]))


def test_orchestrator_safely_handles_propagated_router_fault():
    class BrokenRouter:
        def route(self, required_capabilities):
            raise RuntimeError("confidential router diagnostic")

    execute = ModelStageExecutor(
        execution_input=TaskExecutionInput("Hello"),
        router=BrokenRouter(),
        providers={},
    )

    task = run(execute)

    assert task.status is TaskStatus.FAILED
    assert task.steps[0].status is StepStatus.FAILED
    assert task.steps[0].error == UNEXPECTED_STAGE_FAILURE
    assert "confidential" not in task.steps[0].error


@pytest.mark.parametrize("error_type", [
    ProviderConfigurationError, ProviderConnectionError, ProviderResponseError, RuntimeError,
])
def test_provider_errors_are_safe_and_stop_remaining_work(error_type):
    class FailureProvider:
        async def generate(self, request):
            raise error_type("private endpoint and confidential diagnostic")

    route = router(model_data("model", capabilities=["vision", "reasoning"]))
    task = run(executor(route, FailureProvider()), plan(("vision", "reasoning")))
    assert task.status is TaskStatus.FAILED
    assert [s.status for s in task.steps] == [StepStatus.FAILED, StepStatus.SKIPPED]
    assert task.steps[0].error == MODEL_STAGE_FAILURE


@pytest.mark.parametrize("response", [
    None, ModelResponse("wrong-model", "text"), ModelResponse("model", ""),
    ModelResponse("model", "   "), ModelResponse("model", []),
])
def test_malformed_provider_response_fails_safely(response):
    class BadProvider:
        async def generate(self, request):
            return response
    task = run(executor(router(model_data("model")), BadProvider()))
    assert task.status is TaskStatus.FAILED
    assert task.steps[0].error == MODEL_STAGE_FAILURE


def test_missing_provider_fails_safely():
    execute = ModelStageExecutor(
        execution_input=TaskExecutionInput("Hello"), router=router(model_data("model")),
        providers={},
    )
    assert run(execute).steps[0].error == MODEL_STAGE_FAILURE


@pytest.mark.parametrize("prompt", ["", "  ", "x" * 32769, None])
def test_input_reuses_api_prompt_validation(prompt):
    with pytest.raises(ValidationError):
        TaskExecutionInput(prompt)
    with pytest.raises(ValidationError):
        GenerateRequest(prompt=prompt)


def test_input_is_immutable_and_normalizes_like_api():
    value = TaskExecutionInput("  Hello  ")
    assert value.prompt == GenerateRequest(prompt="  Hello  ").prompt
    assert len(TaskExecutionInput("x" * 32768).prompt) == 32768
    with pytest.raises(FrozenInstanceError):
        value.prompt = "changed"


def context_task():
    task = AgentTask.from_plan(
        task_id="x", plan=plan(("document", "vision", "reasoning")), timestamp=NOW
    ).start(updated_at=NOW)
    task = task.start_step("stage-1", updated_at=NOW)
    task = task.succeed_step("stage-1", StepResult(StepOutputType.TEXT, "prior"), updated_at=NOW)
    # A future result is deliberately present to prove position matters.
    task = task.start_step("stage-3", updated_at=NOW)
    task = task.succeed_step("stage-3", StepResult(StepOutputType.TEXT, "future"), updated_at=NOW)
    return task.start_step("stage-2", updated_at=NOW)


def test_context_excludes_future_results_and_preserves_snapshot():
    task = context_task()
    original = replace(task)
    value = TaskExecutionInput("question")
    context = build_stage_context(value, task, task.steps[1])
    assert "prior" in context and "future" not in context
    assert context == build_stage_context(value, task, task.steps[1])
    assert task == original


@pytest.mark.parametrize("extra", [0, 1])
def test_context_exact_limit_and_overflow(extra):
    task = context_task()
    value = TaskExecutionInput("question")
    initial_length = len(build_stage_context(value, task, task.steps[1]))
    content = "x" * (MAX_STAGE_CONTEXT_LENGTH - initial_length + len("prior") + extra)
    previous = replace(task.steps[0], result=StepResult(StepOutputType.TEXT, content))
    task = replace(task, steps=(previous, *task.steps[1:]))
    if extra:
        with pytest.raises(StageExecutionError, match="size limit"):
            build_stage_context(value, task, task.steps[1])
    else:
        assert len(build_stage_context(value, task, task.steps[1])) == MAX_STAGE_CONTEXT_LENGTH


def test_step_from_outside_snapshot_is_rejected_before_routing():
    task = context_task()
    forged = replace(task.steps[1], required_capabilities=("chat",))
    with pytest.raises(StageExecutionError, match="snapshot"):
        build_stage_context(TaskExecutionInput("Hello"), task, forged)


def test_context_overflow_does_not_invoke_provider():
    task = context_task()
    oversized = replace(task.steps[0], result=StepResult(
        StepOutputType.TEXT, "x" * MAX_STAGE_CONTEXT_LENGTH
    ))
    task = replace(task, steps=(oversized, *task.steps[1:]))
    provider = RecordingProvider()
    execute = executor(router(model_data("vision", capabilities=["vision"])), provider)
    with pytest.raises(StageExecutionError, match="size limit"):
        asyncio.run(execute.execute(task, task.steps[1]))
    assert not provider.requests
