from dataclasses import FrozenInstanceError, asdict, replace
from datetime import timedelta
import hashlib

import pytest

from conftest import model_data
from test_model_stage_execution import NOW, plan, router, run, RecordingProvider
from sovereign_api.agent_execution import StepStatus, TaskStatus
from sovereign_api.config import DeploymentEnvironment
from sovereign_api.errors import InvalidExecutionStateError, ProviderConnectionError
from sovereign_api.execution_provenance import result_digest
from sovereign_api.model_stage_execution import ModelStageExecutor, TaskExecutionInput


def execute(provider=None, route=None, capabilities=("chat",)):
    return run(ModelStageExecutor(
        execution_input=TaskExecutionInput("private original prompt"),
        router=route if route is not None else router(model_data(
            "approved-model", capabilities=list(capabilities)
        )),
        providers={"mock": provider if provider is not None else RecordingProvider()},
        clock=lambda: NOW,
    ), plan(capabilities))


def test_success_records_approved_facts_without_payloads():
    task = execute()
    step = task.steps[0]
    provenance = step.provenance
    assert task.status is TaskStatus.SUCCEEDED
    assert provenance.stage_id == step.stage_id
    assert provenance.model_id == "approved-model"
    assert provenance.provider == "mock"
    assert provenance.required_capabilities == ("chat",)
    assert provenance.routing_environment is DeploymentEnvironment.DEVELOPMENT
    assert provenance.started_at == provenance.completed_at == NOW
    assert provenance.success is True
    assert provenance.result_digest == result_digest("text", "output-1")
    serialized = repr(asdict(provenance))
    for excluded in ("private original prompt", "output-1", "http://", "credentials"):
        assert excluded not in serialized
    assert set(asdict(step.result)) == {"output_type", "content"}
    assert step.result.content == "output-1"


def test_multistage_provenance_is_preserved_and_deterministic():
    first = execute(capabilities=("vision", "reasoning"))
    second = execute(capabilities=("vision", "reasoning"))
    assert first == second
    assert [s.provenance.required_capabilities for s in first.steps] == [
        ("vision",), ("reasoning",),
    ]
    assert first.steps[0].provenance.result_digest != first.steps[1].provenance.result_digest


@pytest.mark.parametrize("failure", ["exception", "malformed", "missing"])
def test_failed_execution_has_provenance_without_digest_or_diagnostics(failure):
    class BrokenProvider:
        async def generate(self, request):
            if failure == "malformed":
                return None
            raise ProviderConnectionError("http://private.invalid credentials private response")

    executor = ModelStageExecutor(
        execution_input=TaskExecutionInput("private prompt"),
        router=router(model_data("approved", capabilities=["vision", "reasoning"])),
        providers={} if failure == "missing" else {"mock": BrokenProvider()},
        clock=lambda: NOW,
    )
    task = run(executor, plan(("vision", "reasoning")))
    assert task.status is TaskStatus.FAILED
    assert task.steps[0].provenance.success is False
    assert task.steps[0].provenance.result_digest is None
    assert task.steps[1].status is StepStatus.SKIPPED
    assert task.steps[1].provenance is None
    for value in ("http://private.invalid", "credentials", "private response", "private prompt"):
        assert value not in repr(task)


def test_routing_rejection_has_no_fabricated_provenance():
    task = execute(route=router(model_data("disabled", enabled=False)))
    assert task.status is TaskStatus.FAILED
    assert task.steps[0].provenance is None


def test_digest_has_explicit_canonical_encoding():
    expected = hashlib.sha256(
        '{"content":"héllo","output_type":"text"}'.encode("utf-8")
    ).hexdigest()
    assert result_digest("text", "héllo") == expected
    assert result_digest("text", "héllo") != result_digest("text", "hello")
    assert result_digest("text", "héllo") != result_digest("structured", "héllo")


def test_provenance_copies_collections_and_is_frozen():
    original = execute().steps[0].provenance
    capabilities = ["chat"]
    copied = replace(original, required_capabilities=capabilities)
    capabilities.append("coding")
    assert copied.required_capabilities == ("chat",)
    with pytest.raises(FrozenInstanceError):
        copied.model_id = "changed"
    with pytest.raises(AttributeError):
        copied.required_capabilities.append("coding")


@pytest.mark.parametrize("changes", [
    {"started_at": NOW.replace(tzinfo=None)},
    {"completed_at": NOW.replace(tzinfo=None)},
    {"completed_at": NOW - timedelta(seconds=1)},
    {"success": False},
    {"result_digest": None},
    {"result_digest": "invalid"},
])
def test_invalid_provenance_is_rejected(changes):
    with pytest.raises(InvalidExecutionStateError):
        replace(execute().steps[0].provenance, **changes)


def test_step_rejects_provenance_from_another_result():
    step = execute().steps[0]
    with pytest.raises(InvalidExecutionStateError):
        replace(step, provenance=replace(
            step.provenance, result_digest=result_digest("text", "different")
        ))


def test_clock_injection_captures_ordered_execution_times():
    times = iter([NOW, NOW + timedelta(seconds=2)])
    executor = ModelStageExecutor(
        execution_input=TaskExecutionInput("Hello"),
        router=router(model_data("approved")),
        providers={"mock": RecordingProvider()}, clock=lambda: next(times),
    )
    provenance = run(executor).steps[0].provenance
    assert provenance.started_at == NOW
    assert provenance.completed_at == NOW + timedelta(seconds=2)
