"""Lease-gated model and tool execution against durable task snapshots."""

import asyncio
import copy
import pickle
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from conftest import model_data, registry_data
from sovereign_api.agent_stage_execution import (
    InvalidStageCoordinationError, RoutedAgentStageExecutor,
    StageExecutionCoordinator,
)
from sovereign_api.agent_task_state import (
    AgentTaskState, StageOutputKind, StageStatus, TaskStatus,
)
from sovereign_api.config import DeploymentEnvironment
from sovereign_api.contracts import ModelResponse
from sovereign_api.errors import ProviderConnectionError
from sovereign_api.registry import ModelRegistry
from sovereign_api.routing import DeterministicModelRouter
from sovereign_api.stage_output_store import InMemoryStageOutputStore, StageOutput
from sovereign_api.stage_execution_records import (
    ExecutionResultDurability, StageExecutionRecordError,
    StageExecutionRecordStatus,
    SuccessfulResultUnavailableError, derive_stage_idempotency_key,
    stage_text_digest,
)
from sovereign_api.task_classification import TaskClass
from sovereign_api.task_planning import (
    StageExecutionKind, TaskPlan, TaskStage, TaskStageType,
)
from sovereign_api.task_state_repository import (
    SQLiteTaskStateRepository, StageClaimError, StageLeaseExpiredError,
    StageLeaseMismatchError,
    StaleTaskStateError, TaskPersistenceError,
)
from sovereign_api.tool_contracts import (
    ToolDescriptor, ToolPermission, ToolRegistry, ToolRequest, ToolResult,
    ToolResultStatus, ToolRiskLevel, ToolSideEffectLevel,
)
from sovereign_api.tool_execution import (
    ExecutableToolRegistry, PolicyEnforcedToolExecutor,
    ToolExecutionIdentityError, ToolPolicyAuthorizationError,
    TrustedToolInvocationReceipt, _create_internal_tool_invocation_boundary,
)
from sovereign_api.tool_policy import (
    DeterministicToolPolicyEvaluator, ToolPermissionDecision,
)
from sovereign_api.workspace_read_file import (
    WORKSPACE_READ_FILE_DESCRIPTOR, WORKSPACE_READ_FILE_TOOL_ID,
)
from sovereign_api.workspace_write_artifact import (
    WORKSPACE_WRITE_ARTIFACT_DESCRIPTOR, WORKSPACE_WRITE_ARTIFACT_TOOL_ID,
)


NOW = datetime(2026, 9, 16, tzinfo=UTC)
READ = ToolPermission("filesystem.read")
WRITE = ToolPermission("artifact.write")


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

    def __init__(self, before_result=None):
        self.calls = 0
        self.before_result = before_result
        self.last_result = None

    async def execute(self, request):
        self.calls += 1
        if self.before_result is not None:
            self.before_result()
        self.last_result = ToolResult(
            request.request_id, request.tool_id, ToolResultStatus.SUCCEEDED,
            text_content="tool output",
        )
        return self.last_result


class ApprovalReadTool(CountingReadTool):
    descriptor = ToolDescriptor(
        WORKSPACE_READ_FILE_TOOL_ID, "Approval reader", "Approval reader",
        ("filesystem.read",), ToolRiskLevel.HIGH,
        ToolSideEffectLevel.DESTRUCTIVE, False, True, True, (READ,),
    )


class FailedReadTool(CountingReadTool):
    async def execute(self, request):
        self.calls += 1
        return ToolResult(
            request.request_id, request.tool_id, ToolResultStatus.FAILED,
            safe_message="Read failed", error_code="read_failed",
        )


class CountingWriteTool:
    tool_id = WORKSPACE_WRITE_ARTIFACT_TOOL_ID
    descriptor = WORKSPACE_WRITE_ARTIFACT_DESCRIPTOR

    def __init__(self):
        self.calls = 0

    async def execute(self, request):
        self.calls += 1
        return ToolResult(
            request.request_id, request.tool_id, ToolResultStatus.SUCCEEDED,
            output_reference=request.arguments["path"],
        )


class ApprovalPolicy:
    def __init__(self):
        self.calls = 0

    def evaluate(self, descriptor, *, granted_permissions, environment):
        self.calls += 1
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


def write_plan():
    return TaskPlan(TaskClass.GENERAL, (
        TaskStage(
            "stage-1", TaskStageType.TOOL, (), StageExecutionKind.TOOL,
            WORKSPACE_WRITE_ARTIFACT_TOOL_ID,
            {"path": "reports/output.txt", "content": "content"},
        ),
    ))


def task(task_id="task-1", plan=None):
    return AgentTaskState.from_plan(
        task_id=task_id, original_prompt="Persisted prompt",
        plan=plan or model_plan(), timestamp=NOW,
    )


def completed_tool_state(claim, reference, output_kind=StageOutputKind.TEXT):
    state = claim.persisted.state
    changed_at = state.updated_at + timedelta(microseconds=1)
    stage = state.stage_states[0].complete(
        output_reference=reference,
        selected_tool_id=state.plan.stages[0].tool_id,
        output_kind=output_kind,
    )
    state = state.update_stage(stage, updated_at=changed_at)
    return state.complete(updated_at=changed_at + timedelta(microseconds=1))


def failed_tool_state(claim, code="tool_failed"):
    state = claim.persisted.state
    return state.update_stage(
        state.stage_states[0].fail(
            error_code=code, safe_message="Stage execution failed",
            selected_tool_id=state.plan.stages[0].tool_id,
        ),
        updated_at=state.updated_at + timedelta(microseconds=1),
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


def tool_coordinator(tool, *, grants=frozenset(), policy=None, store=None):
    executor = PolicyEnforcedToolExecutor(
        ToolRegistry((tool.descriptor,)),
        policy or DeterministicToolPolicyEvaluator(),
        ExecutableToolRegistry((tool,)),
    )

    class NoModelExecution:
        async def execute(self, task_state, stage, prompt):
            raise AssertionError("Tool stage must not invoke a model")

    return StageExecutionCoordinator(
        NoModelExecution(), output_store=store or InMemoryStageOutputStore(),
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
    with SQLiteTaskStateRepository._for_test(path, clock=clock) as repository:
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
    with SQLiteTaskStateRepository._for_test(path, clock=MutableClock()) as repository:
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


def test_persisted_model_ignores_replaced_stage_executor_after_construction(tmp_path):
    provider = RecordingProvider()
    coordinator = model_coordinator(provider)

    class MaliciousExecutor:
        def __init__(self):
            self.calls = 0

        async def execute(self, task_state, stage, prompt):
            self.calls += 1
            raise AssertionError("replaceable executor entered persisted boundary")

    malicious = MaliciousExecutor()
    object.__setattr__(coordinator, "_executor", malicious)
    with SQLiteTaskStateRepository._for_test(
        tmp_path / "tasks.sqlite", clock=MutableClock(),
    ) as repository:
        repository.create(task())
        claim = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
        )
        result = execute(coordinator, repository, claim)

    assert malicious.calls == 0
    assert len(provider.requests) == 1
    assert result.persisted.state.task_status is TaskStatus.COMPLETED


def test_persisted_model_rejects_untrusted_output_store_before_provider(tmp_path):
    provider = RecordingProvider()

    class FakeStore:
        def put(self, output):
            raise AssertionError("untrusted store called")

        def get(self, reference, *, task_id, stage_id):
            raise AssertionError("untrusted store called")

    registry = ModelRegistry.model_validate(registry_data(model_data("model-chat")))
    coordinator = StageExecutionCoordinator(
        RoutedAgentStageExecutor(
            router=DeterministicModelRouter(
                registry, DeploymentEnvironment.DEVELOPMENT,
            ), providers={"mock": provider},
        ), output_store=FakeStore(),
        clock=lambda: NOW + timedelta(seconds=30),
    )
    with SQLiteTaskStateRepository._for_test(
        tmp_path / "tasks.sqlite", clock=MutableClock(),
    ) as repository:
        repository.create(task())
        claim = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
        )
        with pytest.raises(InvalidStageCoordinationError):
            execute(coordinator, repository, claim)

    assert provider.requests == []


def test_missing_wrong_and_mutated_capabilities_make_zero_provider_calls(tmp_path):
    path = tmp_path / "tasks.sqlite"
    provider = RecordingProvider()
    coordinator = model_coordinator(provider)
    with SQLiteTaskStateRepository._for_test(path, clock=MutableClock()) as repository:
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
    with SQLiteTaskStateRepository._for_test(path, clock=MutableClock()) as repository:
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
    attempt_rows_at_execution = []

    def observe_attempt_admission():
        with sqlite3.connect(path) as database:
            attempt_rows_at_execution.append(database.execute(
                "SELECT COUNT(*), COUNT(invocation_admitted_at) "
                "FROM stage_execution_attempts"
            ).fetchone())

    tool = CountingReadTool(before_result=observe_attempt_admission)
    coordinator = tool_coordinator(tool, grants=frozenset({READ}))
    with SQLiteTaskStateRepository._for_test(path, clock=MutableClock()) as repository:
        repository.create(task(plan=tool_plan()))
        claim = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
        )
        result = execute(coordinator, repository, claim)
        stage = result.persisted.state.stage_states[0]
        assert tool.calls == 1
        assert attempt_rows_at_execution == [(1, 1)]
        assert result.persisted.version == 3 and result.persisted.lease is None
        assert stage.status is StageStatus.COMPLETED
        assert stage.selected_tool_id == WORKSPACE_READ_FILE_TOOL_ID
        assert stage.output_reference is not None
        assert repository.get_stage_execution_record(
            "task-1", "stage-1",
        ).attempt_count == 1
        with sqlite3.connect(path) as database:
            assert database.execute(
                "SELECT COUNT(*) FROM stage_execution_attempts"
            ).fetchone()[0] == 1


@pytest.mark.parametrize("reuse_store", [True, False])
def test_known_read_success_requires_live_owned_ephemeral_output(tmp_path, reuse_store):
    path = tmp_path / f"tasks-{reuse_store}.sqlite"
    clock = MutableClock()
    store = InMemoryStageOutputStore()
    tool = CountingReadTool(
        before_result=lambda: setattr(clock, "value", NOW + timedelta(minutes=6)),
    )
    first_coordinator = tool_coordinator(
        tool, grants=frozenset({READ}), store=store,
    )
    with SQLiteTaskStateRepository._for_test(path, clock=clock) as repository:
        repository.create(task(plan=tool_plan()))
        first = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
        )
        with pytest.raises(StageLeaseExpiredError):
            execute(first_coordinator, repository, first)
    tool.before_result = None
    replay_store = store if reuse_store else InMemoryStageOutputStore()
    replay_coordinator = tool_coordinator(
        tool, grants=frozenset({READ}), store=replay_store,
    )
    with SQLiteTaskStateRepository._for_test(path, clock=clock) as repository:
        second = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=2,
        )
        if reuse_store:
            result = execute(replay_coordinator, repository, second)
            assert result.persisted.state.task_status is TaskStatus.COMPLETED
        else:
            with pytest.raises(SuccessfulResultUnavailableError):
                execute(replay_coordinator, repository, second)
            assert repository.get("task-1").state.task_status is TaskStatus.RUNNING
        assert tool.calls == 1
        record = repository.get_stage_execution_record("task-1", "stage-1")
        assert record.status is StageExecutionRecordStatus.SUCCEEDED
        assert record.result_durability is ExecutionResultDurability.EPHEMERAL
        assert record.result_content_digest is not None


def test_valid_lease_does_not_override_permission_denial(tmp_path):
    path = tmp_path / "tasks.sqlite"
    tool = CountingReadTool()
    coordinator = tool_coordinator(tool)
    with SQLiteTaskStateRepository._for_test(path, clock=MutableClock()) as repository:
        repository.create(task(plan=tool_plan()))
        claim = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
        )
        result = execute(coordinator, repository, claim)
        assert tool.calls == 0
        assert result.persisted.state.task_status is TaskStatus.FAILED
        assert result.persisted.state.stage_states[0].error_code == "tool_denied"
        assert result.persisted.lease is None
        with sqlite3.connect(path) as database:
            assert database.execute(
                "SELECT COUNT(*) FROM stage_execution_attempts"
            ).fetchone()[0] == 0
        with pytest.raises(StageExecutionRecordError):
            repository.get_stage_execution_record("task-1", "stage-1")


def test_approval_required_suspends_and_persisted_resume_is_not_claimable(tmp_path):
    path = tmp_path / "tasks.sqlite"
    tool = ApprovalReadTool()
    policy = DeterministicToolPolicyEvaluator()
    coordinator = tool_coordinator(
        tool, grants=frozenset({READ}), policy=policy,
    )
    with SQLiteTaskStateRepository._for_test(path, clock=MutableClock()) as repository:
        repository.create(task(plan=tool_plan()))
        claim = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
        )
        result = execute(coordinator, repository, claim)
        assert tool.calls == 0
        assert result.persisted.state.task_status is TaskStatus.AWAITING_APPROVAL
        assert result.persisted.state.approval_request is not None
        assert result.persisted.lease is None
        record = repository.get_stage_execution_record("task-1", "stage-1")
        assert record.status is StageExecutionRecordStatus.AWAITING_APPROVAL
        assert record.status is not StageExecutionRecordStatus.FAILED
        assert record.attempt_count == 0
        assert record.safe_result_reference is None
        assert result.persisted.version == 3
        with sqlite3.connect(path) as database:
            assert database.execute(
                "SELECT COUNT(*) FROM stage_execution_attempts"
            ).fetchone()[0] == 0
            assert database.execute(
                "SELECT attempt_count, side_effect_attempt_count "
                "FROM stage_execution_records"
            ).fetchone() == (0, 0)
        with pytest.raises(StageClaimError):
            repository.claim_next_stage(
                "task-1", "stage-1", expected_version=3,
            )
    with SQLiteTaskStateRepository._for_test(path, clock=MutableClock()) as reopened:
        record = reopened.get_stage_execution_record("task-1", "stage-1")
        assert record.status is StageExecutionRecordStatus.AWAITING_APPROVAL
        assert reopened.get("task-1").state.task_status is TaskStatus.AWAITING_APPROVAL
        with sqlite3.connect(path) as database:
            assert database.execute(
                "SELECT COUNT(*) FROM stage_execution_attempts"
            ).fetchone()[0] == 0
        with pytest.raises(StageClaimError):
            reopened.claim_next_stage(
                "task-1", "stage-1", expected_version=3,
            )


def test_crash_before_tool_policy_outcome_creates_no_attempt_or_record(tmp_path):
    path = tmp_path / "tasks.sqlite"
    clock = MutableClock()
    with SQLiteTaskStateRepository._for_test(path, clock=clock) as repository:
        repository.create(task(plan=tool_plan()))
        claim = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
            lease_duration=timedelta(seconds=5),
        )
        # Claim validation is the last durable action before policy admission.
        repository.validate_claimed_execution(claim, claim.execution_authority)
        with sqlite3.connect(path) as database:
            assert database.execute(
                "SELECT COUNT(*) FROM stage_execution_attempts"
            ).fetchone()[0] == 0
            assert database.execute(
                "SELECT COUNT(*) FROM stage_execution_records"
            ).fetchone()[0] == 0
        persisted = repository.get("task-1")
        assert persisted.state.task_status is TaskStatus.RUNNING
        assert persisted.lease == claim.lease
        clock.value += timedelta(seconds=6)
        reclaimed = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=2,
        )
        assert repository.prepare_stage_execution(
            reclaimed, reclaimed.execution_authority, admit_attempt=False,
        ) is None
        with sqlite3.connect(path) as database:
            assert database.execute(
                "SELECT COUNT(*) FROM stage_execution_attempts"
            ).fetchone()[0] == 0
            assert database.execute(
                "SELECT COUNT(*) FROM stage_execution_records"
            ).fetchone()[0] == 0


def test_approval_suspension_rolls_back_record_state_and_lease_together(tmp_path):
    path = tmp_path / "tasks.sqlite"
    tool = CountingReadTool()
    coordinator = tool_coordinator(
        tool, grants=frozenset({READ}), policy=ApprovalPolicy(),
    )
    with SQLiteTaskStateRepository._for_test(path, clock=MutableClock()) as repository:
        repository.create(task(plan=tool_plan()))
        claim = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
        )
        with sqlite3.connect(path) as database:
            database.execute("""
                CREATE TRIGGER reject_approval_suspend
                BEFORE UPDATE ON agent_task_states
                BEGIN
                    SELECT RAISE(ABORT, 'forced rollback');
                END
            """)
        with pytest.raises(TaskPersistenceError):
            execute(coordinator, repository, claim)
        assert tool.calls == 0
        persisted = repository.get("task-1")
        assert persisted.version == 2
        assert persisted.state.task_status is TaskStatus.RUNNING
        assert persisted.state.stage_states[0].status is StageStatus.RUNNING
        assert persisted.lease == claim.lease
        with sqlite3.connect(path) as database:
            assert database.execute(
                "SELECT COUNT(*) FROM stage_execution_attempts"
            ).fetchone()[0] == 0
            assert database.execute(
                "SELECT COUNT(*) FROM stage_execution_records"
            ).fetchone()[0] == 0


def test_unrelated_lease_owner_cannot_suspend_or_admit_tool_attempt(tmp_path):
    path = tmp_path / "tasks.sqlite"
    tool = CountingReadTool()
    coordinator = tool_coordinator(
        tool, grants=frozenset({READ}), policy=ApprovalPolicy(),
    )
    with SQLiteTaskStateRepository._for_test(path, clock=MutableClock()) as repository:
        repository.create(task(task_id="task-a", plan=tool_plan()))
        repository.create(task(task_id="task-b", plan=tool_plan()))
        claim_a = repository.claim_next_stage(
            "task-a", "stage-1", expected_version=1,
        )
        claim_b = repository.claim_next_stage(
            "task-b", "stage-1", expected_version=1,
        )
        with pytest.raises(StageLeaseMismatchError):
            execute(coordinator, repository, claim_a, claim_b.execution_authority)
        assert tool.calls == 0
        with sqlite3.connect(path) as database:
            assert database.execute(
                "SELECT COUNT(*) FROM stage_execution_attempts"
            ).fetchone()[0] == 0
            assert database.execute(
                "SELECT COUNT(*) FROM stage_execution_records"
            ).fetchone()[0] == 0


def test_replaceable_executor_cannot_return_success_without_admission(tmp_path):
    path = tmp_path / "tasks.sqlite"

    class SkipsAdmissionExecutor:
        calls = 0

        def describe(self, tool_id):
            return WORKSPACE_READ_FILE_DESCRIPTOR

        async def execute(self, request, **kwargs):
            self.calls += 1
            return ToolResult(
                request.request_id, request.tool_id, ToolResultStatus.SUCCEEDED,
                text_content="bypass",
            )

    executor = SkipsAdmissionExecutor()
    coordinator = StageExecutionCoordinator(
        type("NoModel", (), {"execute": lambda *args: None})(),
        output_store=InMemoryStageOutputStore(),
        clock=lambda: NOW + timedelta(seconds=30),
        tool_executor=executor,
        granted_tool_permissions=frozenset({READ}),
        tool_environment=DeploymentEnvironment.DEVELOPMENT,
    )
    with SQLiteTaskStateRepository._for_test(path, clock=MutableClock()) as repository:
        repository.create(task(plan=tool_plan()))
        claim = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
        )
        result = execute(coordinator, repository, claim)
        assert executor.calls == 0
        assert result.persisted.state.task_status is TaskStatus.FAILED
        with sqlite3.connect(path) as database:
            assert database.execute(
                "SELECT COUNT(*) FROM stage_execution_attempts"
            ).fetchone()[0] == 0


def test_persisted_path_rejects_replaceable_executor_components(tmp_path):
    path = tmp_path / "tasks.sqlite"
    base_tool = CountingReadTool()
    base = PolicyEnforcedToolExecutor(
        ToolRegistry((WORKSPACE_READ_FILE_DESCRIPTOR,)),
        DeterministicToolPolicyEvaluator(), ExecutableToolRegistry((base_tool,)),
    )

    class ReplaceableExecutor:
        evaluate_calls = 0
        execute_calls = 0

        def describe(self, tool_id):
            return base.describe(tool_id)

        def evaluate(self, request, **kwargs):
            self.evaluate_calls += 1
            base_tool.calls += 100
            raise AssertionError("replaceable policy path must not run")

        async def execute_authorized(self, request, authorization):
            self.execute_calls += 1
            base_tool.calls += 100
            raise AssertionError("replaceable invocation path must not run")

    replaceable = ReplaceableExecutor()
    coordinator = StageExecutionCoordinator(
        type("NoModel", (), {"execute": lambda *args: None})(),
        output_store=InMemoryStageOutputStore(),
        clock=lambda: NOW + timedelta(seconds=30),
        tool_executor=replaceable,
        granted_tool_permissions=frozenset({READ}),
        tool_environment=DeploymentEnvironment.DEVELOPMENT,
    )
    with SQLiteTaskStateRepository._for_test(path, clock=MutableClock()) as repository:
        repository.create(task(plan=tool_plan()))
        claim = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
        )
        result = execute(coordinator, repository, claim)
        assert replaceable.evaluate_calls == 0
        assert replaceable.execute_calls == 0
        assert base_tool.calls == 0
        assert result.persisted.state.task_status is TaskStatus.FAILED


@pytest.mark.parametrize("component", ["registry", "evaluator", "executables"])
def test_internal_boundary_rejects_malicious_component_without_calling_it(
    tmp_path, component,
):
    path = tmp_path / f"tasks-{component}.sqlite"
    tool = CountingReadTool()
    valid = PolicyEnforcedToolExecutor(
        ToolRegistry((tool.descriptor,)), DeterministicToolPolicyEvaluator(),
        ExecutableToolRegistry((tool,)),
    )

    class MaliciousComponent:
        calls = 0

        def get(self, *args, **kwargs):
            self.calls += 1
            tool.calls += 100
            raise AssertionError("malicious registry lookup ran")

        def evaluate(self, *args, **kwargs):
            self.calls += 1
            tool.calls += 100
            raise AssertionError("malicious policy evaluation ran")

    malicious = MaliciousComponent()
    forged = object.__new__(PolicyEnforcedToolExecutor)
    object.__setattr__(
        forged, "registry", malicious if component == "registry" else valid.registry,
    )
    object.__setattr__(
        forged, "policy_evaluator",
        malicious if component == "evaluator" else valid.policy_evaluator,
    )
    object.__setattr__(
        forged, "tools", malicious if component == "executables" else valid.tools,
    )
    coordinator = StageExecutionCoordinator(
        type("NoModel", (), {"execute": lambda *args: None})(),
        output_store=InMemoryStageOutputStore(),
        clock=lambda: NOW + timedelta(seconds=30), tool_executor=forged,
        granted_tool_permissions=frozenset({READ}),
        tool_environment=DeploymentEnvironment.DEVELOPMENT,
    )
    with SQLiteTaskStateRepository._for_test(path, clock=MutableClock()) as repository:
        repository.create(task(plan=tool_plan()))
        claim = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
        )
        result = execute(coordinator, repository, claim)
        assert result.persisted.state.task_status is TaskStatus.FAILED
    assert malicious.calls == 0
    assert tool.calls == 0


def test_internal_boundary_is_unchanged_when_original_executor_is_mutated(tmp_path):
    path = tmp_path / "tasks.sqlite"
    tool = CountingReadTool()
    executor = PolicyEnforcedToolExecutor(
        ToolRegistry((tool.descriptor,)), DeterministicToolPolicyEvaluator(),
        ExecutableToolRegistry((tool,)),
    )
    coordinator = StageExecutionCoordinator(
        type("NoModel", (), {"execute": lambda *args: None})(),
        output_store=InMemoryStageOutputStore(),
        clock=lambda: NOW + timedelta(seconds=30), tool_executor=executor,
        granted_tool_permissions=frozenset({READ}),
        tool_environment=DeploymentEnvironment.DEVELOPMENT,
    )

    class MutatedDependency:
        calls = 0

        def get(self, *args, **kwargs):
            self.calls += 1
            raise AssertionError("mutated dependency ran")

        def evaluate(self, *args, **kwargs):
            self.calls += 1
            raise AssertionError("mutated dependency ran")

    mutated = MutatedDependency()
    object.__setattr__(executor, "registry", mutated)
    object.__setattr__(executor, "policy_evaluator", mutated)
    object.__setattr__(executor, "tools", mutated)
    with SQLiteTaskStateRepository._for_test(path, clock=MutableClock()) as repository:
        repository.create(task(plan=tool_plan()))
        claim = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
        )
        result = execute(coordinator, repository, claim)
        assert result.persisted.state.task_status is TaskStatus.COMPLETED
    assert mutated.calls == 0
    assert tool.calls == 1


def test_persisted_boundary_and_store_attributes_cannot_redirect_execution(tmp_path):
    path = tmp_path / "tasks.sqlite"
    tool = CountingReadTool()
    coordinator = tool_coordinator(tool, grants=frozenset({READ}))

    class MaliciousProxy:
        calls = 0

        def __getattr__(self, name):
            self.calls += 1
            tool.calls += 100
            raise AssertionError("replacement authority was consulted")

    malicious = MaliciousProxy()
    object.__setattr__(coordinator, "_persisted_tool_boundary", malicious)
    object.__setattr__(coordinator, "_output_store", malicious)
    with SQLiteTaskStateRepository._for_test(path, clock=MutableClock()) as repository:
        repository.create(task(plan=tool_plan()))
        claim = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
        )
        result = execute(coordinator, repository, claim)
    assert result.persisted.state.task_status is TaskStatus.COMPLETED
    assert malicious.calls == 0
    assert tool.calls == 1


def test_persisted_read_rejects_untrusted_output_store_before_tool_call(tmp_path):
    path = tmp_path / "tasks.sqlite"
    tool = CountingReadTool()

    class ForgedStore:
        calls = 0

        def put(self, output):
            self.calls += 1
            return StageOutputReference("f" * 32)

        def get(self, reference, *, task_id, stage_id):
            self.calls += 1
            return StageOutput(
                "other-task", "other-stage", "text/plain", "tool output", NOW,
            )

    store = ForgedStore()
    coordinator = tool_coordinator(
        tool, grants=frozenset({READ}), store=store,
    )
    with SQLiteTaskStateRepository._for_test(path, clock=MutableClock()) as repository:
        repository.create(task(plan=tool_plan()))
        claim = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
        )
        result = execute(coordinator, repository, claim)
    assert result.persisted.state.task_status is TaskStatus.FAILED
    assert tool.calls == 0
    assert store.calls == 0


def test_persisted_policy_never_calls_replaceable_executor_before_admission(tmp_path):
    path = tmp_path / "tasks.sqlite"
    tool = CountingReadTool()
    policy = DeterministicToolPolicyEvaluator()
    base = PolicyEnforcedToolExecutor(
        ToolRegistry((WORKSPACE_READ_FILE_DESCRIPTOR,)), policy,
        ExecutableToolRegistry((tool,)),
    )

    class MaliciousExecutor:
        evaluate_calls = 0
        invocation_calls = 0

        def describe(self, tool_id):
            return WORKSPACE_READ_FILE_DESCRIPTOR

        def evaluate(self, request, **kwargs):
            self.evaluate_calls += 1
            tool.calls += 100
            return ToolPermissionDecision.REQUIRE_APPROVAL

        async def execute(self, request, **kwargs):
            self.invocation_calls += 1
            tool.calls += 100
            return ToolResult(
                request.request_id, request.tool_id,
                ToolResultStatus.SUCCEEDED, text_content="forged",
            )

    malicious = MaliciousExecutor()
    coordinator = StageExecutionCoordinator(
        type("NoModel", (), {"execute": lambda *args: None})(),
        output_store=InMemoryStageOutputStore(),
        clock=lambda: NOW + timedelta(seconds=30),
        tool_executor=malicious,
        granted_tool_permissions=frozenset({READ}),
        tool_environment=DeploymentEnvironment.DEVELOPMENT,
    )
    with SQLiteTaskStateRepository._for_test(path, clock=MutableClock()) as repository:
        repository.create(task(plan=tool_plan()))
        claim = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
        )
        result = execute(coordinator, repository, claim)
        assert malicious.evaluate_calls == 0
        assert malicious.invocation_calls == 0
        assert tool.calls == 0
        assert result.persisted.state.task_status is TaskStatus.FAILED
        with sqlite3.connect(path) as database:
            assert database.execute(
                "SELECT COUNT(*) FROM stage_execution_attempts"
            ).fetchone() == (0,)


def test_trusted_boundary_consumes_both_authorities_and_receipt_once(tmp_path):
    path = tmp_path / "tasks.sqlite"
    tool = CountingReadTool()
    executor = PolicyEnforcedToolExecutor(
        ToolRegistry((WORKSPACE_READ_FILE_DESCRIPTOR,)),
        DeterministicToolPolicyEvaluator(), ExecutableToolRegistry((tool,)),
    )
    boundary = _create_internal_tool_invocation_boundary(executor)
    assert boundary is not None
    request = ToolRequest(
        "request-1", WORKSPACE_READ_FILE_TOOL_ID, "read_file",
        {"path": "input.txt"}, "task-1", "stage-1",
    )
    with SQLiteTaskStateRepository._for_test(path, clock=MutableClock()) as repository:
        repository.create(task(plan=tool_plan()))
        claim = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
        )
        evaluation = boundary.evaluate(
            request, granted_permissions=frozenset({READ}),
            environment=DeploymentEnvironment.DEVELOPMENT,
        )
        prepared = repository.prepare_stage_execution(
            claim, claim.execution_authority,
        )
        permit = repository.authorize_attempt_invocation(
            claim, claim.execution_authority, prepared,
            prepared.execution_authority,
        )
        receipt = run(boundary.invoke(
            repository, request, evaluation.authorization, permit,
        ))
        assert tool.calls == 1
        with pytest.raises(TypeError):
            copy.copy(receipt)
        with pytest.raises(TypeError):
            copy.deepcopy(receipt)
        with pytest.raises(TypeError):
            pickle.dumps(receipt)
        with pytest.raises(ToolPolicyAuthorizationError):
            run(boundary.invoke(
                repository, request, evaluation.authorization, permit,
            ))
        assert tool.calls == 1
        outcome = boundary.inspect_receipt(
            receipt, task_id="task-1", stage_id="stage-1",
            tool_id=WORKSPACE_READ_FILE_TOOL_ID,
            attempt_id=prepared.execution_authority.attempt_id,
            idempotency_key=prepared.record.idempotency_key.value,
            lease_id=claim.lease.lease_id,
            claimed_version=claim.lease.claimed_version,
        )
        assert outcome.result.text_content == "tool output"
        with pytest.raises(ToolExecutionIdentityError):
            TrustedToolInvocationReceipt()

        repository.create(task(task_id="task-2", plan=tool_plan()))
        second_claim = repository.claim_next_stage(
            "task-2", "stage-1", expected_version=1,
        )
        second_prepared = repository.prepare_stage_execution(
            second_claim, second_claim.execution_authority,
        )
        object.__setattr__(second_prepared.execution_authority, "task_id", "task-1")
        with pytest.raises(StageExecutionRecordError):
            repository.authorize_attempt_invocation(
                second_claim, second_claim.execution_authority,
                second_prepared, second_prepared.execution_authority,
            )
        assert tool.calls == 1

        repository.create(task(task_id="task-3", plan=tool_plan()))
        third_claim = repository.claim_next_stage(
            "task-3", "stage-1", expected_version=1,
        )
        third_request = ToolRequest(
            "request-3", WORKSPACE_READ_FILE_TOOL_ID, "read_file",
            {"path": "input.txt"}, "task-3", "stage-1",
        )
        third_evaluation = boundary.evaluate(
            third_request, granted_permissions=frozenset({READ}),
            environment=DeploymentEnvironment.DEVELOPMENT,
        )
        third_prepared = repository.prepare_stage_execution(
            third_claim, third_claim.execution_authority,
        )
        third_permit = repository.authorize_attempt_invocation(
            third_claim, third_claim.execution_authority,
            third_prepared, third_prepared.execution_authority,
        )
        object.__setattr__(third_evaluation.authorization, "stage_id", "other-stage")
        with pytest.raises(ToolPolicyAuthorizationError):
            run(boundary.invoke(
                repository, third_request, third_evaluation.authorization,
                third_permit,
            ))
        assert tool.calls == 1


def test_stale_attempt_authority_and_permit_cannot_invoke_after_reconciliation(tmp_path):
    path = tmp_path / "tasks.sqlite"
    clock = MutableClock()
    tool = CountingReadTool()
    executor = PolicyEnforcedToolExecutor(
        ToolRegistry((tool.descriptor,)), DeterministicToolPolicyEvaluator(),
        ExecutableToolRegistry((tool,)),
    )
    boundary = _create_internal_tool_invocation_boundary(executor)
    assert boundary is not None
    request = ToolRequest(
        "request-stale", WORKSPACE_READ_FILE_TOOL_ID, "read_file",
        {"path": "input.txt"}, "task-1", "stage-1",
    )
    with SQLiteTaskStateRepository._for_test(path, clock=clock) as repository:
        repository.create(task(plan=tool_plan()))
        first = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
            lease_duration=timedelta(seconds=5),
        )
        prepared = repository.prepare_stage_execution(
            first, first.execution_authority,
        )
        permit = repository.authorize_attempt_invocation(
            first, first.execution_authority, prepared,
            prepared.execution_authority,
        )
        evaluation = boundary.evaluate(
            request, granted_permissions=frozenset({READ}),
            environment=DeploymentEnvironment.DEVELOPMENT,
        )
        clock.value += timedelta(seconds=6)
        replacement = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=2,
        )
        with pytest.raises(StageExecutionRecordError):
            repository.prepare_stage_execution(
                replacement, replacement.execution_authority,
            )
        with pytest.raises(StageExecutionRecordError):
            run(boundary.invoke(
                repository, request, evaluation.authorization, permit,
            ))
        assert tool.calls == 0
        with sqlite3.connect(path) as database:
            assert database.execute(
                "SELECT invocation_admitted_at FROM stage_execution_attempts"
            ).fetchone() == (None,)
        assert repository.get_stage_execution_record(
            "task-1", "stage-1",
        ).status is StageExecutionRecordStatus.UNKNOWN


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("attempt_id", "b" * 32),
        ("idempotency_key", "a" * 64),
        ("lease_id", "f" * 32),
        ("claimed_version", 999),
    ),
)
def test_mutated_invocation_permit_never_reaches_tool(tmp_path, field, replacement):
    path = tmp_path / f"tasks-{field}.sqlite"
    tool = CountingReadTool()
    executor = PolicyEnforcedToolExecutor(
        ToolRegistry((tool.descriptor,)), DeterministicToolPolicyEvaluator(),
        ExecutableToolRegistry((tool,)),
    )
    boundary = _create_internal_tool_invocation_boundary(executor)
    assert boundary is not None
    with SQLiteTaskStateRepository._for_test(path, clock=MutableClock()) as repository:
        repository.create(task(plan=tool_plan()))
        claim = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
        )
        prepared = repository.prepare_stage_execution(
            claim, claim.execution_authority,
        )
        permit = repository.authorize_attempt_invocation(
            claim, claim.execution_authority, prepared,
            prepared.execution_authority,
        )
        request = ToolRequest(
            "request-mutated", WORKSPACE_READ_FILE_TOOL_ID, "read_file",
            {"path": "input.txt"}, "task-1", "stage-1",
        )
        evaluation = boundary.evaluate(
            request, granted_permissions=frozenset({READ}),
            environment=DeploymentEnvironment.DEVELOPMENT,
        )
        object.__setattr__(permit, field, replacement)
        with pytest.raises(StageExecutionRecordError):
            run(boundary.invoke(
                repository, request, evaluation.authorization, permit,
            ))
        assert tool.calls == 0


def test_tool_outcome_recording_requires_matching_one_use_receipt(tmp_path):
    path = tmp_path / "tasks.sqlite"
    tool = CountingReadTool()
    executor = PolicyEnforcedToolExecutor(
        ToolRegistry((tool.descriptor,)), DeterministicToolPolicyEvaluator(),
        ExecutableToolRegistry((tool,)),
    )
    boundary = _create_internal_tool_invocation_boundary(executor)
    assert boundary is not None
    request = ToolRequest(
        "request-receipt", WORKSPACE_READ_FILE_TOOL_ID, "read_file",
        {"path": "input.txt"}, "task-1", "stage-1",
    )
    with SQLiteTaskStateRepository._for_test(path, clock=MutableClock()) as repository:
        repository.create(task(plan=tool_plan()))
        claim = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
        )
        prepared = repository.prepare_stage_execution(
            claim, claim.execution_authority,
        )
        fabricated = completed_tool_state(
            claim, "reports/fabricated.txt", StageOutputKind.ARTIFACT,
        )
        with pytest.raises(StageExecutionRecordError):
            repository.record_stage_execution_outcome(
                claim, claim.execution_authority, prepared,
                prepared.execution_authority, fabricated,
            )
        with pytest.raises(StageExecutionRecordError):
            repository.complete_claim(
                "task-1", "stage-1", claim.lease.lease_id,
                claim.execution_authority, fabricated,
                expected_version=claim.persisted.version,
            )
        assert tool.calls == 0

        permit = repository.authorize_attempt_invocation(
            claim, claim.execution_authority, prepared,
            prepared.execution_authority,
        )
        evaluation = boundary.evaluate(
            request, granted_permissions=frozenset({READ}),
            environment=DeploymentEnvironment.DEVELOPMENT,
        )
        receipt = run(boundary.invoke(
            repository, request, evaluation.authorization, permit,
        ))
        assert tool.calls == 1
        with pytest.raises(StageExecutionRecordError):
            repository.record_stage_execution_outcome(
                claim, claim.execution_authority, prepared,
                prepared.execution_authority, failed_tool_state(claim),
                invocation_receipt=receipt,
            )
        assert repository.get_stage_execution_record(
            "task-1", "stage-1",
        ).status is StageExecutionRecordStatus.IN_PROGRESS


def test_matching_receipt_finalizes_once_and_cannot_cross_task(tmp_path):
    path = tmp_path / "tasks.sqlite"
    tool = CountingReadTool()
    executor = PolicyEnforcedToolExecutor(
        ToolRegistry((tool.descriptor,)), DeterministicToolPolicyEvaluator(),
        ExecutableToolRegistry((tool,)),
    )
    store = InMemoryStageOutputStore()
    boundary = _create_internal_tool_invocation_boundary(
        executor, output_store=store,
    )
    assert boundary is not None
    with SQLiteTaskStateRepository._for_test(path, clock=MutableClock()) as repository:
        repository.create(task(plan=tool_plan()))
        claim = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
        )
        prepared = repository.prepare_stage_execution(
            claim, claim.execution_authority,
        )
        permit = repository.authorize_attempt_invocation(
            claim, claim.execution_authority, prepared,
            prepared.execution_authority,
        )
        request = ToolRequest(
            "request-once", WORKSPACE_READ_FILE_TOOL_ID, "read_file",
            {"path": "input.txt"}, "task-1", "stage-1",
        )
        evaluation = boundary.evaluate(
            request, granted_permissions=frozenset({READ}),
            environment=DeploymentEnvironment.DEVELOPMENT,
        )
        receipt = run(boundary.invoke(
            repository, request, evaluation.authorization, permit,
        ))
        inspected = boundary.inspect_receipt(
            receipt, task_id="task-1", stage_id="stage-1",
            tool_id=WORKSPACE_READ_FILE_TOOL_ID,
            attempt_id=prepared.execution_authority.attempt_id,
            idempotency_key=prepared.record.idempotency_key.value,
            lease_id=claim.lease.lease_id,
            claimed_version=claim.lease.claimed_version,
        )
        object.__setattr__(inspected.result, "text_content", "forged inspection")
        object.__setattr__(tool.last_result, "text_content", "forged original")
        provisional = completed_tool_state(claim, "d" * 32)
        for field, replacement in (
            ("stage_id", "other-stage"), ("tool_id", "other.tool"),
        ):
            original = getattr(receipt, field)
            object.__setattr__(receipt, field, replacement)
            with pytest.raises(StageExecutionRecordError):
                repository.record_stage_execution_outcome(
                    claim, claim.execution_authority, prepared,
                    prepared.execution_authority, provisional,
                    invocation_receipt=receipt,
                    result_content_digest=stage_text_digest("tool output"),
                )
            object.__setattr__(receipt, field, original)
        repository.create(task(task_id="task-2", plan=tool_plan()))
        other = repository.claim_next_stage(
            "task-2", "stage-1", expected_version=1,
        )
        other_prepared = repository.prepare_stage_execution(
            other, other.execution_authority,
        )
        repository.authorize_attempt_invocation(
            other, other.execution_authority, other_prepared,
            other_prepared.execution_authority,
        )
        with pytest.raises(StageExecutionRecordError):
            repository.record_stage_execution_outcome(
                other, other.execution_authority, other_prepared,
                other_prepared.execution_authority,
                completed_tool_state(other, "e" * 32),
                invocation_receipt=receipt,
                result_content_digest=stage_text_digest("tool output"),
            )
        reference = store.put(StageOutput(
            "task-1", "stage-1", "text/plain", "tool output",
            NOW + timedelta(seconds=2),
        ))
        boundary.bind_text_output(
            receipt, output_reference=reference.value,
        )
        terminal = completed_tool_state(claim, reference.value)
        repository.record_stage_execution_outcome(
            claim, claim.execution_authority, prepared,
            prepared.execution_authority, terminal,
            invocation_receipt=receipt,
            result_content_digest=stage_text_digest("tool output"),
        )
        with pytest.raises(StageExecutionRecordError):
            repository.record_stage_execution_outcome(
                claim, claim.execution_authority, prepared,
                prepared.execution_authority, terminal,
                invocation_receipt=receipt,
                result_content_digest=stage_text_digest("tool output"),
            )


def test_artifact_receipt_cannot_be_rewritten_to_fabricated_reference(tmp_path):
    path = tmp_path / "tasks.sqlite"
    tool = CountingWriteTool()
    executor = PolicyEnforcedToolExecutor(
        ToolRegistry((tool.descriptor,)), DeterministicToolPolicyEvaluator(),
        ExecutableToolRegistry((tool,)),
    )
    boundary = _create_internal_tool_invocation_boundary(executor)
    assert boundary is not None
    with SQLiteTaskStateRepository._for_test(path, clock=MutableClock()) as repository:
        repository.create(task(plan=write_plan()))
        claim = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
        )
        prepared = repository.prepare_stage_execution(
            claim, claim.execution_authority,
        )
        permit = repository.authorize_attempt_invocation(
            claim, claim.execution_authority, prepared,
            prepared.execution_authority,
        )
        request = ToolRequest(
            "request-write", WORKSPACE_WRITE_ARTIFACT_TOOL_ID,
            "write_artifact",
            {"path": "reports/output.txt", "content": "content"},
            "task-1", "stage-1",
        )
        evaluation = boundary.evaluate(
            request, granted_permissions=frozenset({WRITE}),
            environment=DeploymentEnvironment.DEVELOPMENT,
        )
        receipt = run(boundary.invoke(
            repository, request, evaluation.authorization, permit,
        ))
        exact = completed_tool_state(
            claim, "reports/output.txt", StageOutputKind.ARTIFACT,
        )
        # A low-level fabricated write result has no captured-root byte
        # verification binding, so it cannot become durable success.
        with pytest.raises(StageExecutionRecordError):
            repository.record_stage_execution_outcome(
                claim, claim.execution_authority, prepared,
                prepared.execution_authority, exact,
                invocation_receipt=receipt,
                result_content_digest=stage_text_digest("content"),
            )


def test_failed_tool_receipt_cannot_be_recorded_as_success(tmp_path):
    path = tmp_path / "tasks.sqlite"
    tool = FailedReadTool()
    executor = PolicyEnforcedToolExecutor(
        ToolRegistry((tool.descriptor,)), DeterministicToolPolicyEvaluator(),
        ExecutableToolRegistry((tool,)),
    )
    boundary = _create_internal_tool_invocation_boundary(executor)
    assert boundary is not None
    with SQLiteTaskStateRepository._for_test(path, clock=MutableClock()) as repository:
        repository.create(task(plan=tool_plan()))
        claim = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
        )
        prepared = repository.prepare_stage_execution(
            claim, claim.execution_authority,
        )
        permit = repository.authorize_attempt_invocation(
            claim, claim.execution_authority, prepared,
            prepared.execution_authority,
        )
        request = ToolRequest(
            "request-failed", WORKSPACE_READ_FILE_TOOL_ID, "read_file",
            {"path": "input.txt"}, "task-1", "stage-1",
        )
        evaluation = boundary.evaluate(
            request, granted_permissions=frozenset({READ}),
            environment=DeploymentEnvironment.DEVELOPMENT,
        )
        receipt = run(boundary.invoke(
            repository, request, evaluation.authorization, permit,
        ))
        with pytest.raises(StageExecutionRecordError):
            repository.record_stage_execution_outcome(
                claim, claim.execution_authority, prepared,
                prepared.execution_authority,
                completed_tool_state(claim, "c" * 32),
                invocation_receipt=receipt,
                result_content_digest=stage_text_digest("invented"),
            )
        assert tool.calls == 1


def test_failed_receipt_binds_complete_claim_to_exact_terminal_outcome(tmp_path):
    path = tmp_path / "tasks.sqlite"
    tool = FailedReadTool()
    executor = PolicyEnforcedToolExecutor(
        ToolRegistry((tool.descriptor,)), DeterministicToolPolicyEvaluator(),
        ExecutableToolRegistry((tool,)),
    )
    boundary = _create_internal_tool_invocation_boundary(executor)
    assert boundary is not None
    with SQLiteTaskStateRepository._for_test(path, clock=MutableClock()) as repository:
        repository.create(task(plan=tool_plan()))
        claim = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
        )
        prepared = repository.prepare_stage_execution(
            claim, claim.execution_authority,
        )
        permit = repository.authorize_attempt_invocation(
            claim, claim.execution_authority, prepared,
            prepared.execution_authority,
        )
        request = ToolRequest(
            "request-failure-binding", WORKSPACE_READ_FILE_TOOL_ID, "read_file",
            {"path": "input.txt"}, "task-1", "stage-1",
        )
        evaluation = boundary.evaluate(
            request, granted_permissions=frozenset({READ}),
            environment=DeploymentEnvironment.DEVELOPMENT,
        )
        receipt = run(boundary.invoke(
            repository, request, evaluation.authorization, permit,
        ))
        exact = failed_tool_state(claim)
        repository.record_stage_execution_outcome(
            claim, claim.execution_authority, prepared,
            prepared.execution_authority, exact, invocation_receipt=receipt,
        )
        state = claim.persisted.state
        changed_at = state.updated_at + timedelta(microseconds=1)
        cancelled = state.update_stage(
            state.stage_states[0].cancel(), updated_at=changed_at,
        )
        mismatched_code = failed_tool_state(claim, "other_failure")
        mismatched_tool = failed_tool_state(claim)
        object.__setattr__(
            mismatched_tool.stage_states[0], "selected_tool_id", "other.tool",
        )
        for proposed in (
            completed_tool_state(claim, "a" * 32), cancelled,
            mismatched_code, mismatched_tool,
        ):
            with pytest.raises((StageExecutionRecordError, StageClaimError)):
                repository.complete_claim(
                    "task-1", "stage-1", claim.lease.lease_id,
                    claim.execution_authority, proposed,
                    expected_version=claim.persisted.version,
                )
        saved = repository.complete_claim(
            "task-1", "stage-1", claim.lease.lease_id,
            claim.execution_authority, exact,
            expected_version=claim.persisted.version,
        )
        assert saved.state.task_status is TaskStatus.FAILED


def test_legacy_approval_migration_deletes_only_fake_attempt_and_is_idempotent(tmp_path):
    path = tmp_path / "tasks.sqlite"
    awaiting = run(tool_coordinator(
        CountingReadTool(), grants=frozenset({READ}), policy=ApprovalPolicy(),
    ).execute_one(task(plan=tool_plan()), tool_plan().stages[0]))
    ambiguous = run(tool_coordinator(
        CountingReadTool(), grants=frozenset({READ}), policy=ApprovalPolicy(),
    ).execute_one(
        task(task_id="task-2", plan=tool_plan()), tool_plan().stages[0],
    ))
    with SQLiteTaskStateRepository._for_test(path, clock=MutableClock()) as repository:
        repository.create(awaiting)
        repository.create(ambiguous)
        for number in range(3, 13):
            repository.create(run(tool_coordinator(
                CountingReadTool(), grants=frozenset({READ}),
                policy=ApprovalPolicy(),
            ).execute_one(
                task(task_id=f"task-{number}", plan=tool_plan()),
                tool_plan().stages[0],
            )))
    timestamp = NOW.isoformat()
    key = derive_stage_idempotency_key(
        awaiting, awaiting.plan.stages[0],
    ).value
    modified_plan = TaskPlan(TaskClass.GENERAL, (
        TaskStage(
            "stage-1", TaskStageType.TOOL, (), StageExecutionKind.TOOL,
            WORKSPACE_READ_FILE_TOOL_ID, {"path": "different.txt"},
        ),
    ))
    modified_key = derive_stage_idempotency_key(
        task(task_id="task-9", plan=modified_plan), modified_plan.stages[0],
    ).value
    other_task_state = task(task_id="different-task", plan=tool_plan())
    other_task_key = derive_stage_idempotency_key(
        other_task_state, other_task_state.plan.stages[0],
    ).value
    attempt_id = "b" * 32
    with sqlite3.connect(path) as database:
        database.execute(
            "INSERT INTO stage_execution_records "
            "(idempotency_key, task_id, stage_id, execution_kind, status, "
            "attempt_count, current_attempt_id, first_started_at, last_started_at, "
            "completed_at, result_durability, side_effect_attempt_count, "
            "legacy_approval_only, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, 1, ?, ?)",
            (key, "task-1", "stage-1", "tool", "failed", 1, attempt_id,
             timestamp, timestamp, timestamp, 0, timestamp, timestamp),
        )
        database.execute(
            "INSERT INTO stage_execution_attempts "
            "(attempt_id, idempotency_key, lease_id, claimed_version, status, "
            "started_at, completed_at, attempt_proof) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (attempt_id, key, "c" * 32, 2, "failed", timestamp, timestamp,
             "d" * 64),
        )
        database.execute(
            "INSERT INTO stage_execution_records "
            "(idempotency_key, task_id, stage_id, execution_kind, status, "
            "attempt_count, current_attempt_id, first_started_at, last_started_at, "
            "completed_at, result_durability, side_effect_attempt_count, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)",
            ("e" * 64, "task-2", "stage-1", "tool", "failed", 1, "f" * 32,
             timestamp, timestamp, timestamp, 1, timestamp, timestamp),
        )
        database.execute(
            "INSERT INTO stage_execution_attempts "
            "(attempt_id, idempotency_key, lease_id, claimed_version, status, "
            "started_at, completed_at, attempt_proof) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("f" * 32, "e" * 64, "1" * 32, 2, "failed", timestamp, timestamp,
             "2" * 64),
        )
        preserved = (
            ("3" * 64, "task-3", "wrong-stage", "tool", "failed", "3" * 32),
            ("4" * 64, "task-4", "stage-1", "model", "failed", "4" * 32),
            ("5" * 64, "task-5", "stage-1", "tool", "unknown", "5" * 32),
            ("6" * 64, "task-6", "stage-1", "tool", "succeeded", "6" * 32),
            ("7" * 64, "task-7", "stage-1", "tool", "failed", "7" * 32),
            (other_task_key, "task-8", "stage-1", "tool", "failed", "8" * 32),
            (modified_key, "task-9", "stage-1", "tool", "failed", "9" * 32),
        )
        for record_key, task_id, stage_id, kind, status, saved_attempt_id in preserved:
            database.execute(
                "INSERT INTO stage_execution_records "
                "(idempotency_key, task_id, stage_id, execution_kind, status, "
                "attempt_count, current_attempt_id, first_started_at, last_started_at, "
                "completed_at, result_durability, side_effect_attempt_count, "
                "legacy_approval_only, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, ?, NULL, 1, 1, ?, ?)",
                (record_key, task_id, stage_id, kind, status, saved_attempt_id,
                 timestamp, timestamp, timestamp, timestamp, timestamp),
            )
            database.execute(
                "INSERT INTO stage_execution_attempts "
                "(attempt_id, idempotency_key, lease_id, claimed_version, status, "
                "started_at, completed_at, attempt_proof) "
                "VALUES (?, ?, ?, 2, ?, ?, ?, ?)",
                (saved_attempt_id, record_key, "7" * 32, status,
                 timestamp, timestamp, "8" * 64),
            )
        admission_evidence = (
            ("task-10", "a" * 32, 0, timestamp),
            ("task-11", "c" * 32, 1, None),
            ("task-12", "d" * 32, 1, timestamp),
        )
        for task_id, saved_attempt_id, attempt_count, admitted_at in admission_evidence:
            logical = task(task_id=task_id, plan=tool_plan())
            record_key = derive_stage_idempotency_key(
                logical, logical.plan.stages[0],
            ).value
            database.execute(
                "INSERT INTO stage_execution_records "
                "(idempotency_key, task_id, stage_id, execution_kind, status, "
                "attempt_count, current_attempt_id, first_started_at, last_started_at, "
                "completed_at, result_durability, side_effect_attempt_count, "
                "legacy_approval_only, created_at, updated_at) "
                "VALUES (?, ?, 'stage-1', 'tool', 'failed', 1, ?, ?, ?, ?, "
                "NULL, ?, 1, ?, ?)",
                (record_key, task_id, saved_attempt_id, timestamp, timestamp,
                 timestamp, attempt_count, timestamp, timestamp),
            )
            database.execute(
                "INSERT INTO stage_execution_attempts "
                "(attempt_id, idempotency_key, lease_id, claimed_version, status, "
                "started_at, invocation_admitted_at, completed_at, attempt_proof) "
                "VALUES (?, ?, ?, 2, 'failed', ?, ?, ?, ?)",
                (saved_attempt_id, record_key, "7" * 32, timestamp,
                 admitted_at, timestamp, "8" * 64),
            )
    for _ in range(2):
        with SQLiteTaskStateRepository._for_test(path, clock=MutableClock()) as repository:
            record = repository.get_stage_execution_record("task-1", "stage-1")
            assert record.status is StageExecutionRecordStatus.AWAITING_APPROVAL
            assert record.current_attempt_id is None
            assert record.attempt_count == 0
            assert repository.get("task-1").state.task_status is TaskStatus.AWAITING_APPROVAL
        with sqlite3.connect(path) as database:
            assert database.execute(
                "SELECT attempt_id FROM stage_execution_attempts ORDER BY attempt_id"
            ).fetchall() == [
                ("3" * 32,), ("4" * 32,), ("5" * 32,),
                ("6" * 32,), ("7" * 32,), ("8" * 32,),
                ("9" * 32,), ("a" * 32,), ("c" * 32,),
                ("d" * 32,), ("f" * 32,),
            ]
            assert database.execute(
                "SELECT status, side_effect_attempt_count, legacy_approval_only "
                "FROM stage_execution_records WHERE task_id = 'task-2'"
            ).fetchone() == ("failed", 1, 0)
            assert database.execute(
                "SELECT COUNT(*) FROM stage_execution_records "
                "WHERE task_id IN ('task-7', 'task-8', 'task-9') "
                "AND status = 'failed' AND legacy_approval_only = 1"
            ).fetchone() == (3,)
            assert database.execute(
                "SELECT task_id, side_effect_attempt_count, legacy_approval_only "
                "FROM stage_execution_records "
                "WHERE task_id IN ('task-10', 'task-11', 'task-12') "
                "ORDER BY task_id"
            ).fetchall() == [
                ("task-10", 0, 1), ("task-11", 1, 1), ("task-12", 1, 1),
            ]


def test_expiry_after_side_effect_rejects_completion_without_retry(tmp_path):
    path = tmp_path / "tasks.sqlite"
    clock = MutableClock()
    provider = RecordingProvider(
        before_response=lambda: setattr(clock, "value", NOW + timedelta(minutes=6)),
    )
    coordinator = model_coordinator(provider)
    with SQLiteTaskStateRepository._for_test(path, clock=clock) as repository:
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
    with SQLiteTaskStateRepository._for_test(path, clock=clock) as repository:
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
    with SQLiteTaskStateRepository._for_test(path, clock=clock) as repository:
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
    with SQLiteTaskStateRepository._for_test(path, clock=clock) as repository:
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
