"""Lease-gated model and tool execution against durable task snapshots."""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from conftest import model_data, registry_data
from sovereign_api.agent_stage_execution import (
    InvalidStageCoordinationError, RoutedAgentStageExecutor,
    StageExecutionCoordinator,
)
from sovereign_api.agent_task_state import AgentTaskState, StageStatus, TaskStatus
from sovereign_api.config import DeploymentEnvironment
from sovereign_api.contracts import ModelResponse
from sovereign_api.errors import ProviderConnectionError
from sovereign_api.registry import ModelRegistry
from sovereign_api.routing import DeterministicModelRouter
from sovereign_api.stage_output_store import InMemoryStageOutputStore
from sovereign_api.task_classification import TaskClass
from sovereign_api.task_planning import (
    StageExecutionKind, TaskPlan, TaskStage, TaskStageType,
)
from sovereign_api.task_state_repository import (
    SQLiteTaskStateRepository, StageClaimError, StageLeaseExpiredError,
    StageLeaseMismatchError,
    StaleTaskStateError,
)
from sovereign_api.tool_contracts import (
    ToolPermission, ToolRegistry, ToolResult, ToolResultStatus,
)
from sovereign_api.tool_execution import (
    ExecutableToolRegistry, PolicyEnforcedToolExecutor,
)
from sovereign_api.tool_policy import (
    DeterministicToolPolicyEvaluator, ToolPermissionDecision,
)
from sovereign_api.workspace_read_file import (
    WORKSPACE_READ_FILE_DESCRIPTOR, WORKSPACE_READ_FILE_TOOL_ID,
)


NOW = datetime(2026, 9, 16, tzinfo=UTC)
READ = ToolPermission("filesystem.read")


class MutableClock:
    def __init__(self, value=NOW + timedelta(seconds=1)):
        self.value = value

    def __call__(self):
        return self.value


class RecordingProvider:
    def __init__(self, *, before_response=None, failure=None):
        self.requests = []
        self.before_response = before_response
        self.failure = failure

    async def generate(self, request):
        self.requests.append(request)
        if self.before_response is not None:
            self.before_response()
        if self.failure is not None:
            raise self.failure
        return ModelResponse(request.model_id, "persisted output")


class CountingReadTool:
    tool_id = WORKSPACE_READ_FILE_TOOL_ID
    descriptor = WORKSPACE_READ_FILE_DESCRIPTOR

    def __init__(self):
        self.calls = 0

    async def execute(self, request):
        self.calls += 1
        return ToolResult(
            request.request_id, request.tool_id, ToolResultStatus.SUCCEEDED,
            text_content="tool output",
        )


class ApprovalPolicy:
    def evaluate(self, descriptor, *, granted_permissions, environment):
        return ToolPermissionDecision.REQUIRE_APPROVAL


def model_plan(stage_id="stage-1"):
    return TaskPlan(TaskClass.GENERAL, (
        TaskStage(stage_id, TaskStageType.GENERATE, ("chat",)),
    ))


def tool_plan():
    return TaskPlan(TaskClass.GENERAL, (
        TaskStage(
            "stage-1", TaskStageType.TOOL, (), StageExecutionKind.TOOL,
            WORKSPACE_READ_FILE_TOOL_ID, {"path": "input.txt"},
        ),
    ))


def task(task_id="task-1", plan=None):
    return AgentTaskState.from_plan(
        task_id=task_id, original_prompt="Persisted prompt",
        plan=plan or model_plan(), timestamp=NOW,
    )


def model_coordinator(provider, store=None):
    registry = ModelRegistry.model_validate(registry_data(model_data("model-chat")))
    executor = RoutedAgentStageExecutor(
        router=DeterministicModelRouter(
            registry, DeploymentEnvironment.DEVELOPMENT,
        ),
        providers={"mock": provider},
    )
    return StageExecutionCoordinator(
        executor, output_store=store or InMemoryStageOutputStore(),
        clock=lambda: NOW + timedelta(seconds=30),
    )


def tool_coordinator(tool, *, grants=frozenset(), policy=None):
    executor = PolicyEnforcedToolExecutor(
        ToolRegistry((WORKSPACE_READ_FILE_DESCRIPTOR,)),
        policy or DeterministicToolPolicyEvaluator(),
        ExecutableToolRegistry((tool,)),
    )

    class NoModelExecution:
        async def execute(self, task_state, stage, prompt):
            raise AssertionError("Tool stage must not invoke a model")

    return StageExecutionCoordinator(
        NoModelExecution(), output_store=InMemoryStageOutputStore(),
        clock=lambda: NOW + timedelta(seconds=30),
        tool_executor=executor, granted_tool_permissions=grants,
        tool_environment=DeploymentEnvironment.DEVELOPMENT,
    )


def run(coroutine):
    return asyncio.run(coroutine)


def execute(coordinator, repository, claim, authority=None):
    return run(coordinator.execute_claimed_stage_and_persist(
        repository, claim,
        claim.execution_authority if authority is None else authority,
    ))


def test_claimed_model_stage_executes_once_and_completes_persisted_state(tmp_path):
    path = tmp_path / "tasks.sqlite"
    clock = MutableClock()
    provider = RecordingProvider()
    coordinator = model_coordinator(provider)
    with SQLiteTaskStateRepository(path, clock=clock) as repository:
        repository.create(task())
        claim = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
        )
        result = execute(coordinator, repository, claim)
        assert len(provider.requests) == 1
        assert result.claimed_version == 2
        assert result.persisted.version == 3
        assert result.persisted.lease is None
        assert result.persisted.state.task_status is TaskStatus.COMPLETED
        assert result.persisted.state.stage_states[0].status is StageStatus.COMPLETED
        assert result.persisted.state.stage_states[0].selected_model_id == "model-chat"
        assert repository.get("task-1") == result.persisted


def test_claimed_provider_failure_is_persisted_once_and_clears_lease(tmp_path):
    path = tmp_path / "tasks.sqlite"
    provider = RecordingProvider(
        failure=ProviderConnectionError("private provider detail"),
    )
    coordinator = model_coordinator(provider)
    with SQLiteTaskStateRepository(path, clock=MutableClock()) as repository:
        repository.create(task())
        claim = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
        )
        result = execute(coordinator, repository, claim)
        assert len(provider.requests) == 1
        assert result.persisted.version == 3 and result.persisted.lease is None
        assert result.persisted.state.task_status is TaskStatus.FAILED
        assert result.persisted.state.stage_states[0].error_code == "provider_failed"
        assert "private provider detail" not in repr(result)


def test_missing_wrong_and_mutated_capabilities_make_zero_provider_calls(tmp_path):
    path = tmp_path / "tasks.sqlite"
    provider = RecordingProvider()
    coordinator = model_coordinator(provider)
    with SQLiteTaskStateRepository(path, clock=MutableClock()) as repository:
        repository.create(task())
        repository.create(task("task-2", model_plan("other-stage")))
        first = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
        )
        other = repository.claim_next_stage(
            "task-2", "other-stage", expected_version=1,
        )
        with pytest.raises(InvalidStageCoordinationError):
            run(coordinator.execute_claimed_stage_and_persist(
                repository, first, None,
            ))
        with pytest.raises(StageLeaseMismatchError):
            execute(coordinator, repository, first, other.execution_authority)
        for name in ("task_id", "stage_id", "lease_id", "claimed_version"):
            object.__setattr__(
                first.execution_authority, name,
                getattr(other.execution_authority, name),
            )
        with pytest.raises(StageLeaseMismatchError):
            execute(coordinator, repository, first, first.execution_authority)
        assert provider.requests == []


def test_tampered_claimed_snapshot_is_rejected_before_provider(tmp_path):
    path = tmp_path / "tasks.sqlite"
    provider = RecordingProvider()
    coordinator = model_coordinator(provider)
    with SQLiteTaskStateRepository(path, clock=MutableClock()) as repository:
        repository.create(task())
        claim = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
        )
        object.__setattr__(claim.persisted.state, "original_prompt", "tampered")
        with pytest.raises(StageLeaseMismatchError):
            execute(coordinator, repository, claim)
        assert provider.requests == []
        assert repository.get("task-1").state.original_prompt == "Persisted prompt"


def test_claimed_tool_stage_executes_once_and_persists_text_reference(tmp_path):
    path = tmp_path / "tasks.sqlite"
    tool = CountingReadTool()
    coordinator = tool_coordinator(tool, grants=frozenset({READ}))
    with SQLiteTaskStateRepository(path, clock=MutableClock()) as repository:
        repository.create(task(plan=tool_plan()))
        claim = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
        )
        result = execute(coordinator, repository, claim)
        stage = result.persisted.state.stage_states[0]
        assert tool.calls == 1
        assert result.persisted.version == 3 and result.persisted.lease is None
        assert stage.status is StageStatus.COMPLETED
        assert stage.selected_tool_id == WORKSPACE_READ_FILE_TOOL_ID
        assert stage.output_reference is not None


def test_valid_lease_does_not_override_permission_denial(tmp_path):
    path = tmp_path / "tasks.sqlite"
    tool = CountingReadTool()
    coordinator = tool_coordinator(tool)
    with SQLiteTaskStateRepository(path, clock=MutableClock()) as repository:
        repository.create(task(plan=tool_plan()))
        claim = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
        )
        result = execute(coordinator, repository, claim)
        assert tool.calls == 0
        assert result.persisted.state.task_status is TaskStatus.FAILED
        assert result.persisted.state.stage_states[0].error_code == "tool_denied"
        assert result.persisted.lease is None


def test_approval_required_suspends_and_persisted_resume_is_not_claimable(tmp_path):
    path = tmp_path / "tasks.sqlite"
    tool = CountingReadTool()
    coordinator = tool_coordinator(
        tool, grants=frozenset({READ}), policy=ApprovalPolicy(),
    )
    with SQLiteTaskStateRepository(path, clock=MutableClock()) as repository:
        repository.create(task(plan=tool_plan()))
        claim = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
        )
        result = execute(coordinator, repository, claim)
        assert tool.calls == 0
        assert result.persisted.state.task_status is TaskStatus.AWAITING_APPROVAL
        assert result.persisted.state.approval_request is not None
        assert result.persisted.lease is None
        with pytest.raises(StageClaimError):
            repository.claim_next_stage(
                "task-1", "stage-1", expected_version=3,
            )


def test_expiry_after_side_effect_rejects_completion_without_retry(tmp_path):
    path = tmp_path / "tasks.sqlite"
    clock = MutableClock()
    provider = RecordingProvider(
        before_response=lambda: setattr(clock, "value", NOW + timedelta(minutes=6)),
    )
    coordinator = model_coordinator(provider)
    with SQLiteTaskStateRepository(path, clock=clock) as repository:
        repository.create(task())
        claim = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
        )
        with pytest.raises(StageLeaseExpiredError):
            execute(coordinator, repository, claim)
        assert len(provider.requests) == 1
        persisted = repository.get("task-1")
        assert persisted.version == 2
        assert persisted.lease == claim.lease
        assert persisted.state.stage_states[0].status is StageStatus.RUNNING


def test_already_expired_claim_is_rejected_before_provider(tmp_path):
    path = tmp_path / "tasks.sqlite"
    clock = MutableClock()
    provider = RecordingProvider()
    coordinator = model_coordinator(provider)
    with SQLiteTaskStateRepository(path, clock=clock) as repository:
        repository.create(task())
        claim = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
            lease_duration=timedelta(seconds=5),
        )
        clock.value = NOW + timedelta(seconds=7)
        with pytest.raises(StageLeaseExpiredError):
            execute(coordinator, repository, claim)
        assert provider.requests == []


def test_superseded_worker_is_rejected_before_provider(tmp_path):
    path = tmp_path / "tasks.sqlite"
    clock = MutableClock()
    provider = RecordingProvider()
    coordinator = model_coordinator(provider)
    with SQLiteTaskStateRepository(path, clock=clock) as repository:
        repository.create(task())
        old = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
            lease_duration=timedelta(seconds=5),
        )
        clock.value = NOW + timedelta(seconds=6)
        replacement = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=2,
        )
        with pytest.raises(StaleTaskStateError):
            execute(coordinator, repository, old)
        assert provider.requests == []
        assert repository.get("task-1").lease == replacement.lease


def test_provider_runs_outside_sqlite_write_transaction(tmp_path):
    path = tmp_path / "tasks.sqlite"
    clock = MutableClock()
    observations = []
    with SQLiteTaskStateRepository(path, clock=clock) as repository:
        provider = RecordingProvider(
            before_response=lambda: observations.append(repository._db().in_transaction),
        )
        coordinator = model_coordinator(provider)
        repository.create(task())
        claim = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
        )
        execute(coordinator, repository, claim)
        assert observations == [False]
