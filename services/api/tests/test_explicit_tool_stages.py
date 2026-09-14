"""Trusted tool stages use policy gating, not model-generated tool choices."""

import asyncio
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import pytest

from conftest import model_data, registry_data
from sovereign_api.agent_stage_execution import (
    InvalidStageExecutionResultError, RoutedAgentStageExecutor,
    StageExecutionCoordinator, StageExecutionResult, ToolBackedStageExecutor,
)
from sovereign_api.agent_task_state import (
    AgentTaskState, StageExecutionState, StageOutputKind, StageStatus, TaskStatus,
)
from sovereign_api.config import DeploymentEnvironment
from sovereign_api.contracts import ModelResponse
from sovereign_api.errors import (
    InvalidExecutionStateError, InvalidStepTransitionError, UnsupportedTaskRequirementsError,
)
from sovereign_api.registry import ModelRegistry
from sovereign_api.routing import DeterministicModelRouter
from sovereign_api.stage_output_store import (
    MAX_STAGE_OUTPUT_BYTES, InMemoryStageOutputStore, StageOutputReference,
    StageOutputNotFoundError,
)
from sovereign_api.task_classification import TaskClass
from sovereign_api.task_plan_runner import InvalidTaskRunError, TaskPlanRunner, stage_requires_model_invocation
from sovereign_api.task_planning import StageExecutionKind, TaskPlan, TaskStage, TaskStageType
from sovereign_api.tool_contracts import (
    ToolPermission, ToolRegistry, ToolResult, ToolResultStatus, ToolValidationError,
)
from sovereign_api.tool_execution import ExecutableToolRegistry, PolicyEnforcedToolExecutor
from sovereign_api.tool_policy import DeterministicToolPolicyEvaluator, ToolPermissionDecision
from sovereign_api.workspace_read_file import WORKSPACE_READ_FILE_DESCRIPTOR, WORKSPACE_READ_FILE_TOOL_ID, WorkspaceReadFileTool
from sovereign_api.workspace_write_artifact import WORKSPACE_WRITE_ARTIFACT_DESCRIPTOR, WORKSPACE_WRITE_ARTIFACT_TOOL_ID, WorkspaceWriteArtifactTool


NOW = datetime(2026, 9, 14, tzinfo=UTC)
READ = ToolPermission("filesystem.read")
WRITE = ToolPermission("artifact.write")


class Provider:
    def __init__(self, text="model output"):
        self.requests = []
        self.text = text

    async def generate(self, request):
        self.requests.append(request)
        return ModelResponse(request.model_id, self.text)


def model_stage(number):
    return TaskStage(f"stage-{number}", TaskStageType.GENERATE, ("chat",))


def tool_stage(number, tool_id, arguments):
    return TaskStage(
        f"stage-{number}", TaskStageType.TOOL, (),
        StageExecutionKind.TOOL, tool_id, arguments,
    )


def setup(tmp_path, stages, *, grants=frozenset(), environment=DeploymentEnvironment.DEVELOPMENT,
          policy=None, provider=None):
    tmp_path.mkdir(exist_ok=True)
    workspace = tmp_path / "workspace"
    artifact = tmp_path / "artifacts"
    workspace.mkdir()
    artifact.mkdir(mode=0o700)
    (workspace / "input.txt").write_text("trusted document text", encoding="utf-8")
    tools = (WorkspaceReadFileTool(workspace), WorkspaceWriteArtifactTool(artifact))
    tool_executor = PolicyEnforcedToolExecutor(
        ToolRegistry((WORKSPACE_READ_FILE_DESCRIPTOR, WORKSPACE_WRITE_ARTIFACT_DESCRIPTOR)),
        policy or DeterministicToolPolicyEvaluator(),
        ExecutableToolRegistry(tools),
    )
    actual_provider = provider or Provider()
    registry = ModelRegistry.model_validate(registry_data(model_data("chat-model", capabilities=["chat"])))
    model_executor = RoutedAgentStageExecutor(
        router=DeterministicModelRouter(registry, environment),
        providers={"mock": actual_provider},
    )
    store = InMemoryStageOutputStore()
    coordinator = StageExecutionCoordinator(
        model_executor, output_store=store, clock=lambda: NOW,
        tool_executor=tool_executor, granted_tool_permissions=grants,
        tool_environment=environment,
    )
    plan = TaskPlan(TaskClass.GENERAL, tuple(stages))
    task = AgentTaskState.from_plan(
        task_id="task-1", original_prompt="Original task", plan=plan, timestamp=NOW,
    )
    return coordinator, TaskPlanRunner(coordinator), task, actual_provider, store, workspace, artifact


def run(coro):
    return asyncio.run(coro)


def test_tool_stage_validation_and_argument_freezing():
    assert model_stage(1).execution_kind is StageExecutionKind.MODEL
    arguments = {"path": "input.txt", "nested": ["a"]}
    stage = tool_stage(2, WORKSPACE_READ_FILE_TOOL_ID, arguments)
    arguments["nested"].append("b")
    assert stage.tool_arguments["nested"] == ("a",)
    with pytest.raises(TypeError):
        stage.tool_arguments["path"] = "other.txt"
    with pytest.raises(FrozenInstanceError):
        stage.tool_id = "other"
    with pytest.raises(UnsupportedTaskRequirementsError):
        TaskStage("stage-1", TaskStageType.GENERATE, ("chat",), tool_id="x")
    with pytest.raises(UnsupportedTaskRequirementsError):
        TaskStage("stage-1", TaskStageType.TOOL, (), StageExecutionKind.TOOL)
    with pytest.raises(UnsupportedTaskRequirementsError):
        TaskStage("stage-1", TaskStageType.TOOL, ("chat",), StageExecutionKind.TOOL, "x", {})
    with pytest.raises(ToolValidationError):
        tool_stage(1, "x", {"bad": object()})


def test_read_stage_chains_into_next_model_without_model_call_for_tool(tmp_path):
    stages = (tool_stage(1, WORKSPACE_READ_FILE_TOOL_ID, {"path": "input.txt"}), model_stage(2))
    coordinator, _, task, provider, store, _, _ = setup(tmp_path, stages, grants=frozenset({READ}))
    first = run(coordinator.execute_one(task, stages[0]))
    assert task.task_status is TaskStatus.PENDING
    assert first.task_status is TaskStatus.RUNNING
    assert first.stage_states[0].selected_tool_id == WORKSPACE_READ_FILE_TOOL_ID
    assert first.stage_states[0].selected_model_id is None
    assert first.stage_states[0].output_kind is StageOutputKind.TEXT
    assert len(provider.requests) == 0
    assert "trusted document text" not in repr(first)
    stored = store.get(StageOutputReference(first.stage_states[0].output_reference), task_id="task-1", stage_id="stage-1")
    assert stored.text_content == "trusted document text"
    second = run(coordinator.execute_one(first, stages[1]))
    assert second.task_status is TaskStatus.COMPLETED
    assert second.stage_states[1].selected_model_id == "chat-model"
    assert len(provider.requests) == 1
    assert "Original task" in provider.requests[0].prompt
    assert "trusted document text" in provider.requests[0].prompt


def test_artifact_stage_preserves_relative_reference_and_does_not_chain_text(tmp_path):
    stages = (tool_stage(1, WORKSPACE_WRITE_ARTIFACT_TOOL_ID,
                         {"path": "output.txt", "content": "private artifact"}), model_stage(2))
    coordinator, _, task, provider, _, _, artifact = setup(tmp_path, stages, grants=frozenset({WRITE}))
    first = run(coordinator.execute_one(task, stages[0]))
    assert (artifact / "output.txt").read_text() == "private artifact"
    assert first.stage_states[0].output_reference == "output.txt"
    assert first.stage_states[0].output_kind is StageOutputKind.ARTIFACT
    assert first.stage_states[0].selected_tool_id == WORKSPACE_WRITE_ARTIFACT_TOOL_ID
    assert "private artifact" not in repr(first.stage_states)
    with pytest.raises(StageOutputNotFoundError):
        run(coordinator.execute_one(first, stages[1]))
    assert len(provider.requests) == 0


def test_tool_identity_cannot_be_replaced_once_recorded():
    running = StageExecutionState(
        "stage-1", TaskStageType.TOOL, (), execution_kind=StageExecutionKind.TOOL,
    ).start(selected_tool_id=WORKSPACE_READ_FILE_TOOL_ID)
    assert running.complete(
        output_reference="opaque-reference", selected_tool_id=WORKSPACE_READ_FILE_TOOL_ID,
    ).selected_tool_id == WORKSPACE_READ_FILE_TOOL_ID
    assert running.fail(
        error_code="tool_failed", safe_message="Tool failed",
        selected_tool_id=WORKSPACE_READ_FILE_TOOL_ID,
    ).selected_tool_id == WORKSPACE_READ_FILE_TOOL_ID
    with pytest.raises(InvalidStepTransitionError):
        running.complete(output_reference="other", selected_tool_id="other-tool")
    with pytest.raises(InvalidStepTransitionError):
        running.fail(error_code="tool_failed", safe_message="Tool failed", selected_tool_id="other-tool")


def test_denial_and_approval_stop_without_tool_or_provider_execution(tmp_path):
    stages = (tool_stage(1, WORKSPACE_WRITE_ARTIFACT_TOOL_ID,
                         {"path": "output.txt", "content": "text"}), model_stage(2))
    coordinator, _, task, provider, _, _, artifact = setup(tmp_path, stages)
    denied = run(coordinator.execute_one(task, stages[0]))
    assert denied.task_status is TaskStatus.FAILED
    assert denied.stage_states[0].error_code == "tool_denied"
    assert not (artifact / "output.txt").exists()
    assert len(provider.requests) == 0

    class ApprovalPolicy:
        def evaluate(self, descriptor, *, granted_permissions, environment):
            return ToolPermissionDecision.REQUIRE_APPROVAL

    coordinator, _, task, provider, _, _, artifact = setup(
        tmp_path / "other", stages, grants=frozenset({WRITE}), policy=ApprovalPolicy(),
    )
    approval = run(coordinator.execute_one(task, stages[0]))
    assert approval.task_status is TaskStatus.FAILED
    assert approval.stage_states[0].error_code == "tool_approval_required"
    assert not (artifact / "output.txt").exists()
    assert len(provider.requests) == 0


def test_runner_counts_tool_stages_but_not_model_budget(tmp_path):
    stages = (model_stage(1), tool_stage(2, WORKSPACE_READ_FILE_TOOL_ID, {"path": "input.txt"}), model_stage(3))
    _, runner, task, provider, _, _, _ = setup(tmp_path, stages, grants=frozenset({READ}),
                                                provider=Provider('call workspace.read_file {"tool_id":"workspace.read_file"}'))
    result = run(runner.run(task))
    assert result.final_state.task_status is TaskStatus.COMPLETED
    assert (result.stages_executed, result.model_invocations) == (3, 2)
    assert len(provider.requests) == 2
    assert len(result.final_state.plan.stages) == 3
    assert "trusted document text" in provider.requests[1].prompt

    coordinator, _, task, provider, _, _, _ = setup(tmp_path / "limited", stages, grants=frozenset({READ}))
    limited = TaskPlanRunner(coordinator, max_total_model_invocations_per_run=1)
    result = run(limited.run(task))
    assert (result.stages_executed, result.model_invocations) == (2, 1)
    assert result.error_code == "model_invocation_limit_reached"
    assert len(provider.requests) == 1
    assert stage_requires_model_invocation(stages[1]) is False


@pytest.mark.parametrize("environment", [
    DeploymentEnvironment.DEVELOPMENT,
    DeploymentEnvironment.ON_PREM,
    DeploymentEnvironment.AIR_GAPPED,
])
def test_tool_read_requires_explicit_grant_in_every_environment(tmp_path, environment):
    stages = (tool_stage(1, WORKSPACE_READ_FILE_TOOL_ID, {"path": "input.txt"}),)
    coordinator, _, task, provider, _, _, _ = setup(
        tmp_path, stages, environment=environment,
    )
    denied = run(coordinator.execute_one(task, stages[0]))
    assert denied.task_status is TaskStatus.FAILED
    assert denied.stage_states[0].error_code == "tool_denied"
    assert len(provider.requests) == 0


def test_unknown_tool_and_unexpected_executor_failure_are_safe(tmp_path):
    unknown = (tool_stage(1, "unregistered.tool", {}),)
    coordinator, _, task, provider, _, _, _ = setup(tmp_path, unknown, grants=frozenset({READ}))
    failed = run(coordinator.execute_one(task, unknown[0]))
    assert failed.stage_states[0].error_code == "tool_unavailable"
    assert len(provider.requests) == 0

    class FaultExecutor:
        calls = 0

        async def execute(self, request, *, granted_permissions, environment):
            self.calls += 1
            raise OSError("/very/secret/host/path")

    fault = FaultExecutor()
    stages = (tool_stage(1, WORKSPACE_READ_FILE_TOOL_ID, {"path": "input.txt"}),)
    coordinator, _, task, provider, _, _, _ = setup(
        tmp_path / "fault", stages, grants=frozenset({READ}),
    )
    coordinator._tool_executor = ToolBackedStageExecutor(
        fault, granted_permissions=frozenset({READ}),
        environment=DeploymentEnvironment.DEVELOPMENT,
    )
    failed = run(coordinator.execute_one(task, stages[0]))
    assert fault.calls == 1
    assert failed.task_status is TaskStatus.FAILED
    assert failed.stage_states[0].error_code == "tool_unexpected_failure"
    assert "/very/secret/host/path" not in repr(failed)
    assert len(provider.requests) == 0


def test_tool_stage_counts_against_stage_limit_before_any_execution(tmp_path):
    stages = (model_stage(1), tool_stage(2, WORKSPACE_READ_FILE_TOOL_ID, {"path": "input.txt"}), model_stage(3))
    coordinator, _, task, provider, _, _, _ = setup(tmp_path, stages, grants=frozenset({READ}))
    with pytest.raises(InvalidTaskRunError):
        run(TaskPlanRunner(coordinator, max_stages_per_run=2).run(task))
    assert len(provider.requests) == 0
    assert all(stage.status is StageStatus.PENDING for stage in task.stage_states)


@pytest.mark.parametrize("reference", [
    "/etc/passwd", "../outside.txt", "nested/../../outside.txt",
    "C:\\secret.txt", "\\\\server\\share\\file", "reports/\x00secret",
    "reports//output.txt", "reports/./output.txt", "x" * 1_025,
])
def test_pluggable_artifact_reference_fails_closed(tmp_path, reference):
    class ForgedWriter:
        async def execute(self, request, *, granted_permissions, environment):
            return ToolResult(request.request_id, request.tool_id,
                              ToolResultStatus.SUCCEEDED, output_reference=reference)

    stages = (tool_stage(1, WORKSPACE_WRITE_ARTIFACT_TOOL_ID,
                         {"path": "report.txt", "content": "text"}),)
    coordinator, _, task, _, _, _, _ = setup(tmp_path, stages, grants=frozenset({WRITE}))
    coordinator._tool_executor = ToolBackedStageExecutor(
        ForgedWriter(), granted_permissions=frozenset({WRITE}),
        environment=DeploymentEnvironment.DEVELOPMENT,
    )
    failed = run(coordinator.execute_one(task, stages[0]))
    assert failed.task_status is TaskStatus.FAILED
    assert failed.stage_states[0].error_code == "tool_invalid_result"
    assert failed.stage_states[0].output_reference is None
    assert reference not in failed.stage_states[0].safe_message
    with pytest.raises(InvalidStageExecutionResultError):
        StageExecutionResult("stage-1", StageStatus.COMPLETED,
                             execution_kind=StageExecutionKind.TOOL,
                             selected_tool_id=WORKSPACE_WRITE_ARTIFACT_TOOL_ID,
                             output_reference=reference)
    with pytest.raises(InvalidExecutionStateError):
        StageExecutionState("stage-1", TaskStageType.TOOL, (),
                            status=StageStatus.COMPLETED,
                            execution_kind=StageExecutionKind.TOOL,
                            selected_tool_id=WORKSPACE_WRITE_ARTIFACT_TOOL_ID,
                            output_kind=StageOutputKind.ARTIFACT,
                            output_reference=reference)


@pytest.mark.parametrize("reference", ["report.txt", "reports/output.txt"])
def test_safe_pluggable_artifact_reference_is_preserved(tmp_path, reference):
    class Writer:
        async def execute(self, request, *, granted_permissions, environment):
            return ToolResult(request.request_id, request.tool_id,
                              ToolResultStatus.SUCCEEDED, output_reference=reference)

    stages = (tool_stage(1, WORKSPACE_WRITE_ARTIFACT_TOOL_ID,
                         {"path": reference, "content": "text"}),)
    coordinator, _, task, _, _, _, _ = setup(tmp_path, stages, grants=frozenset({WRITE}))
    coordinator._tool_executor = ToolBackedStageExecutor(
        Writer(), granted_permissions=frozenset({WRITE}),
        environment=DeploymentEnvironment.DEVELOPMENT,
    )
    completed = run(coordinator.execute_one(task, stages[0]))
    assert completed.task_status is TaskStatus.COMPLETED
    assert completed.stage_states[0].output_reference == reference


@pytest.mark.parametrize("text, succeeds", [
    ("a" * (MAX_STAGE_OUTPUT_BYTES - 1), True),
    ("a" * MAX_STAGE_OUTPUT_BYTES, True),
    ("a" * (MAX_STAGE_OUTPUT_BYTES + 1), False),
    ("é" * (MAX_STAGE_OUTPUT_BYTES // 2), True),
    ("é" * (MAX_STAGE_OUTPUT_BYTES // 2 + 1), False),
])
def test_tool_stage_text_limit_is_controlled_before_storage(tmp_path, text, succeeds):
    stages = (tool_stage(1, WORKSPACE_READ_FILE_TOOL_ID, {"path": "input.txt"}), model_stage(2))
    coordinator, _, task, provider, store, workspace, _ = setup(
        tmp_path, stages, grants=frozenset({READ}),
    )
    (workspace / "input.txt").write_text(text, encoding="utf-8")
    first = run(coordinator.execute_one(task, stages[0]))
    if succeeds:
        assert first.stage_states[0].status is StageStatus.COMPLETED
        assert store.total_bytes == len(text.encode("utf-8"))
    else:
        assert first.task_status is TaskStatus.FAILED
        assert first.stage_states[0].error_code == "tool_output_too_large"
        assert first.stage_states[0].output_reference is None
        assert store.total_bytes == 0
        assert len(provider.requests) == 0
