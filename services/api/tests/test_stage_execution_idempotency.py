"""Persistent logical idempotency and physical stage-attempt tests."""

import asyncio
import copy
import pickle
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from conftest import model_data, registry_data
from sovereign_api.agent_stage_execution import (
    RoutedAgentStageExecutor, StageExecutionCoordinator,
)
from sovereign_api.agent_task_state import AgentTaskState, StageOutputKind, TaskStatus
from sovereign_api.config import DeploymentEnvironment
from sovereign_api.contracts import ModelResponse
from sovereign_api.registry import ModelRegistry
from sovereign_api.routing import DeterministicModelRouter
from sovereign_api.stage_execution_records import (
    ExecutionResultDurability,
    StageExecutionAlreadyInProgressError, StageExecutionAttemptMismatchError,
    StageExecutionOutcomeUnknownError, StageExecutionPreviouslyFailedError,
    StageExecutionRecordStatus, SuccessfulResultUnavailableError,
    derive_stage_idempotency_key,
)
from sovereign_api.stage_output_store import InMemoryStageOutputStore
from sovereign_api.task_classification import TaskClass
from sovereign_api.task_planning import (
    StageExecutionKind, TaskPlan, TaskStage, TaskStageType,
)
from sovereign_api.task_state_repository import (
    CorruptTaskStateError, SQLiteTaskStateRepository, StageLeaseExpiredError,
    ValidatedStageExecutionAttempt,
)
import sovereign_api.task_state_repository as repository_module
from sovereign_api.tool_contracts import (
    ToolPermission, ToolRegistry, ToolResult, ToolResultStatus,
)
from sovereign_api.tool_execution import (
    ExecutableToolRegistry, PolicyEnforcedToolExecutor,
)
from sovereign_api.tool_policy import DeterministicToolPolicyEvaluator
from sovereign_api.workspace_write_artifact import (
    WORKSPACE_WRITE_ARTIFACT_DESCRIPTOR, WORKSPACE_WRITE_ARTIFACT_TOOL_ID,
    WorkspaceWriteArtifactTool,
)


NOW = datetime(2026, 9, 16, 12, tzinfo=UTC)


class MutableClock:
    def __init__(self, value=NOW + timedelta(seconds=1)):
        self.value = value

    def __call__(self):
        return self.value


class CountingProvider:
    def __init__(self, *, before_response=None, fail=False):
        self.calls = 0
        self.before_response = before_response
        self.fail = fail

    async def generate(self, request):
        self.calls += 1
        if self.before_response is not None:
            self.before_response()
        if self.fail:
            raise RuntimeError("private provider failure")
        return ModelResponse(request.model_id, "bounded model output")


class CountingWriteTool:
    tool_id = WORKSPACE_WRITE_ARTIFACT_TOOL_ID
    descriptor = WORKSPACE_WRITE_ARTIFACT_DESCRIPTOR

    def __init__(self, before_result=None):
        self.calls = 0
        self.before_result = before_result

    async def execute(self, request):
        self.calls += 1
        if self.before_result is not None:
            self.before_result()
        return ToolResult(
            request.request_id, request.tool_id, ToolResultStatus.SUCCEEDED,
            output_reference=request.arguments["path"],
        )


def model_plan():
    return TaskPlan(TaskClass.GENERAL, (
        TaskStage("stage-1", TaskStageType.GENERATE, ("chat",)),
    ))


def write_plan(path="reports/output.txt", content="content"):
    return TaskPlan(TaskClass.GENERAL, (
        TaskStage(
            "stage-1", TaskStageType.TOOL, (), StageExecutionKind.TOOL,
            WORKSPACE_WRITE_ARTIFACT_TOOL_ID, {"path": path, "content": content},
        ),
    ))


def task(*, prompt="Persist this prompt", plan=None):
    return AgentTaskState.from_plan(
        task_id="task-1", original_prompt=prompt, plan=plan or model_plan(),
        timestamp=NOW,
    )


def coordinator(provider, store=None):
    registry = ModelRegistry.model_validate(registry_data(model_data("model-chat")))
    return StageExecutionCoordinator(
        RoutedAgentStageExecutor(
            router=DeterministicModelRouter(
                registry, DeploymentEnvironment.DEVELOPMENT,
            ),
            providers={"mock": provider},
        ),
        output_store=store or InMemoryStageOutputStore(),
        clock=lambda: NOW + timedelta(minutes=20),
    )


def write_coordinator(tool):
    executor = PolicyEnforcedToolExecutor(
        ToolRegistry((WORKSPACE_WRITE_ARTIFACT_DESCRIPTOR,)),
        DeterministicToolPolicyEvaluator(), ExecutableToolRegistry((tool,)),
    )

    class NoModel:
        async def execute(self, task_state, stage, prompt):
            raise AssertionError("Tool execution must not invoke the model")

    return StageExecutionCoordinator(
        NoModel(), output_store=InMemoryStageOutputStore(),
        clock=lambda: NOW + timedelta(minutes=20), tool_executor=executor,
        granted_tool_permissions=frozenset({ToolPermission("artifact.write")}),
        tool_environment=DeploymentEnvironment.DEVELOPMENT,
    )


@pytest.mark.parametrize("mutation", ["intact", "modified", "missing", "symlink"])
def test_durable_artifact_replay_verifies_original_content(tmp_path, mutation):
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir(mode=0o700)
    (artifact_root / "reports").mkdir(mode=0o700)
    tool = WorkspaceWriteArtifactTool(artifact_root)
    execution = write_coordinator(tool)
    clock = MutableClock()
    path = tmp_path / "tasks.sqlite"
    with SQLiteTaskStateRepository._for_test(path, clock=clock) as repository:
        repository.create(task(plan=write_plan()))
        first = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
        )
        actual_complete = repository.complete_claim

        def fail_completion(*args, **kwargs):
            raise StageLeaseExpiredError("forced completion gap")

        repository.complete_claim = fail_completion
        with pytest.raises(StageLeaseExpiredError):
            asyncio.run(execution.execute_claimed_stage_and_persist(
                repository, first, first.execution_authority,
            ))
        repository.complete_claim = actual_complete
        artifact = artifact_root / "reports" / "output.txt"
        assert artifact.read_text() == "content"
        if mutation == "modified":
            artifact.write_text("changed")
        elif mutation == "missing":
            artifact.unlink()
        elif mutation == "symlink":
            artifact.unlink()
            outside = tmp_path / "outside.txt"
            outside.write_text("content")
            artifact.symlink_to(outside)
        clock.value += timedelta(minutes=6)
        second = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=2,
        )
        if mutation == "intact":
            result = asyncio.run(execution.execute_claimed_stage_and_persist(
                repository, second, second.execution_authority,
            ))
            assert result.persisted.state.task_status is TaskStatus.COMPLETED
        else:
            with pytest.raises(SuccessfulResultUnavailableError):
                asyncio.run(execution.execute_claimed_stage_and_persist(
                    repository, second, second.execution_authority,
                ))
        record = repository.get_stage_execution_record("task-1", "stage-1")
        assert record.status is StageExecutionRecordStatus.SUCCEEDED
        assert record.result_content_digest is not None

def completed_state(claim, reference="a" * 32):
    state = claim.persisted.state
    stage_at = state.updated_at + timedelta(microseconds=1)
    state = state.update_stage(
        state.stage_states[0].complete(
            output_reference=reference, selected_model_id="model-chat",
            output_kind=StageOutputKind.TEXT,
        ),
        updated_at=stage_at,
    )
    return state.complete(updated_at=stage_at + timedelta(microseconds=1))


def failed_state(claim):
    state = claim.persisted.state
    return state.update_stage(
        state.stage_states[0].fail(
            error_code="provider_failed", safe_message="Stage execution failed",
            selected_model_id="model-chat",
        ),
        updated_at=state.updated_at + timedelta(microseconds=1),
    )


def test_key_is_stable_across_claim_reopen_and_reclaim(tmp_path):
    path = tmp_path / "tasks.sqlite"
    clock = MutableClock()
    initial = task()
    expected = derive_stage_idempotency_key(initial, initial.plan.stages[0])
    with SQLiteTaskStateRepository._for_test(path, clock=clock) as repository:
        repository.create(initial)
        first = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
            lease_duration=timedelta(seconds=5),
        )
        assert derive_stage_idempotency_key(
            first.persisted.state, first.persisted.state.plan.stages[0],
        ) == expected
    clock.value += timedelta(seconds=6)
    with SQLiteTaskStateRepository._for_test(path, clock=clock) as repository:
        second = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=2,
        )
        assert second.lease.lease_id != first.lease.lease_id
        assert derive_stage_idempotency_key(
            second.persisted.state, second.persisted.state.plan.stages[0],
        ) == expected


def test_key_changes_with_prompt_tool_arguments_and_artifact_content():
    first = task(prompt="first")
    second = task(prompt="second")
    assert derive_stage_idempotency_key(first, first.plan.stages[0]) != (
        derive_stage_idempotency_key(second, second.plan.stages[0])
    )
    write_a = task(plan=write_plan(content="a"))
    write_b = task(plan=write_plan(content="b"))
    write_c = task(plan=write_plan(path="reports/other.txt", content="a"))
    key_a = derive_stage_idempotency_key(write_a, write_a.plan.stages[0])
    assert key_a != derive_stage_idempotency_key(write_b, write_b.plan.stages[0])
    assert key_a != derive_stage_idempotency_key(write_c, write_c.plan.stages[0])
    assert key_a == derive_stage_idempotency_key(
        task(plan=write_plan(content="a")), write_a.plan.stages[0],
    )


def test_first_prepare_creates_one_in_progress_attempt_and_blocks_second(tmp_path):
    path = tmp_path / "tasks.sqlite"
    with SQLiteTaskStateRepository._for_test(path, clock=MutableClock()) as repository:
        repository.create(task())
        claim = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
        )
        prepared = repository.prepare_stage_execution(
            claim, claim.execution_authority,
        )
        assert prepared.record.status is StageExecutionRecordStatus.IN_PROGRESS
        assert prepared.record.attempt_count == 1
        assert prepared.execution_authority is not None
        with pytest.raises(StageExecutionAlreadyInProgressError):
            repository.prepare_stage_execution(claim, claim.execution_authority)
        with sqlite3.connect(path) as database:
            assert database.execute(
                "SELECT COUNT(*) FROM stage_execution_attempts"
            ).fetchone()[0] == 1


def test_attempt_authority_is_opaque_noncopyable_and_secret_is_not_persisted(tmp_path):
    path = tmp_path / "tasks.sqlite"
    with SQLiteTaskStateRepository._for_test(path, clock=MutableClock()) as repository:
        repository.create(task())
        claim = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
        )
        prepared = repository.prepare_stage_execution(
            claim, claim.execution_authority,
        )
        authority = prepared.execution_authority
        secret = repository_module._attempt_authority_secret(authority)
        assert secret is not None and secret.hex() not in repr(authority)
        with pytest.raises(TypeError):
            copy.copy(authority)
        with pytest.raises(TypeError):
            copy.deepcopy(authority)
        with pytest.raises(TypeError):
            pickle.dumps(authority)
        forged = object.__new__(ValidatedStageExecutionAttempt)
        for name in ("attempt_id", "idempotency_key", "task_id", "stage_id"):
            object.__setattr__(forged, name, getattr(authority, name))
        with pytest.raises(StageExecutionAttemptMismatchError):
            repository.record_stage_execution_outcome(
                claim, claim.execution_authority, prepared, forged,
                completed_state(claim),
            )
    database_bytes = path.read_bytes()
    assert secret not in database_bytes
    assert secret.hex().encode() not in database_bytes


def test_failed_record_is_not_automatically_retried(tmp_path):
    path = tmp_path / "tasks.sqlite"
    clock = MutableClock()
    with SQLiteTaskStateRepository._for_test(path, clock=clock) as repository:
        repository.create(task())
        first = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
            lease_duration=timedelta(seconds=5),
        )
        prepared = repository.prepare_stage_execution(
            first, first.execution_authority,
        )
        with pytest.raises(StageExecutionAttemptMismatchError):
            repository.record_stage_execution_outcome(
                first, first.execution_authority, prepared,
                prepared.execution_authority, failed_state(first),
            )
        clock.value += timedelta(seconds=6)
        second = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=2,
        )
        with pytest.raises(StageExecutionOutcomeUnknownError):
            repository.prepare_stage_execution(second, second.execution_authority)
        assert repository.get_stage_execution_record(
            "task-1", "stage-1",
        ).attempt_count == 1


def test_stale_in_progress_becomes_unknown_and_is_not_reexecuted(tmp_path):
    path = tmp_path / "tasks.sqlite"
    clock = MutableClock()
    with SQLiteTaskStateRepository._for_test(path, clock=clock) as repository:
        repository.create(task())
        first = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
            lease_duration=timedelta(seconds=5),
        )
        repository.prepare_stage_execution(first, first.execution_authority)
    clock.value += timedelta(seconds=6)
    with SQLiteTaskStateRepository._for_test(path, clock=clock) as repository:
        second = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=2,
        )
        with pytest.raises(StageExecutionOutcomeUnknownError):
            repository.prepare_stage_execution(second, second.execution_authority)
        record = repository.get_stage_execution_record("task-1", "stage-1")
        assert record.status is StageExecutionRecordStatus.UNKNOWN
        assert record.attempt_count == 1
        with pytest.raises(StageExecutionOutcomeUnknownError):
            repository.prepare_stage_execution(second, second.execution_authority)


def test_known_success_suppresses_duplicate_after_completion_lease_expiry(tmp_path):
    path = tmp_path / "tasks.sqlite"
    clock = MutableClock()
    store = InMemoryStageOutputStore()
    provider = CountingProvider(
        before_response=lambda: setattr(clock, "value", NOW + timedelta(minutes=6)),
    )
    execution = coordinator(provider, store)
    with SQLiteTaskStateRepository._for_test(path, clock=clock) as repository:
        repository.create(task())
        first = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
        )
        with pytest.raises(StageLeaseExpiredError):
            asyncio.run(execution.execute_claimed_stage_and_persist(
                repository, first, first.execution_authority,
            ))
        assert provider.calls == 1
        assert repository.get_stage_execution_record(
            "task-1", "stage-1",
        ).result_durability is ExecutionResultDurability.EPHEMERAL
        second = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=2,
        )
        result = asyncio.run(execution.execute_claimed_stage_and_persist(
            repository, second, second.execution_authority,
        ))
        assert provider.calls == 1
        assert result.persisted.state.task_status is TaskStatus.COMPLETED
        assert result.stage_report.model_invocations == 0


@pytest.mark.parametrize("corruption", ["missing", "unadmitted", "failed"])
def test_known_success_requires_admitted_terminal_attempt(tmp_path, corruption):
    path = tmp_path / "tasks.sqlite"
    clock = MutableClock()
    provider = CountingProvider(
        before_response=lambda: setattr(clock, "value", NOW + timedelta(minutes=6)),
    )
    execution = coordinator(provider, InMemoryStageOutputStore())
    with SQLiteTaskStateRepository._for_test(path, clock=clock) as repository:
        repository.create(task())
        first = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
        )
        with pytest.raises(StageLeaseExpiredError):
            asyncio.run(execution.execute_claimed_stage_and_persist(
                repository, first, first.execution_authority,
            ))
        record = repository.get_stage_execution_record("task-1", "stage-1")
        if corruption == "missing":
            repository._db().execute(
                "DELETE FROM stage_execution_attempts WHERE attempt_id = ?",
                (record.current_attempt_id,),
            )
        elif corruption == "unadmitted":
            repository._db().execute(
                "UPDATE stage_execution_attempts SET invocation_admitted_at = NULL "
                "WHERE attempt_id = ?", (record.current_attempt_id,),
            )
        else:
            repository._db().execute(
                "UPDATE stage_execution_attempts SET status = ? WHERE attempt_id = ?",
                (StageExecutionRecordStatus.FAILED.value, record.current_attempt_id),
            )
        repository._db().commit()
        second = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=2,
        )
        with pytest.raises(CorruptTaskStateError):
            asyncio.run(execution.execute_claimed_stage_and_persist(
                repository, second, second.execution_authority,
            ))
        assert provider.calls == 1
        assert repository.get("task-1").state.task_status is TaskStatus.RUNNING


def test_known_model_success_with_missing_ephemeral_output_requires_reconciliation(
    tmp_path,
):
    path = tmp_path / "tasks.sqlite"
    clock = MutableClock()
    provider = CountingProvider(
        before_response=lambda: setattr(clock, "value", NOW + timedelta(minutes=6)),
    )
    with SQLiteTaskStateRepository._for_test(path, clock=clock) as repository:
        repository.create(task())
        first = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
        )
        with pytest.raises(StageLeaseExpiredError):
            asyncio.run(coordinator(provider).execute_claimed_stage_and_persist(
                repository, first, first.execution_authority,
            ))
    with SQLiteTaskStateRepository._for_test(path, clock=clock) as repository:
        second = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=2,
        )
        with pytest.raises(SuccessfulResultUnavailableError) as captured:
            asyncio.run(coordinator(provider).execute_claimed_stage_and_persist(
                repository, second, second.execution_authority,
            ))
        assert captured.value.code == "successful_result_unavailable"
        assert provider.calls == 1
        assert repository.get("task-1").state.task_status is TaskStatus.RUNNING
        record = repository.get_stage_execution_record("task-1", "stage-1")
        assert record.status is StageExecutionRecordStatus.SUCCEEDED
        assert record.result_durability is ExecutionResultDurability.EPHEMERAL


def test_known_model_success_with_digest_mismatch_requires_reconciliation(tmp_path):
    path = tmp_path / "tasks.sqlite"
    clock = MutableClock()
    store = InMemoryStageOutputStore()
    provider = CountingProvider(
        before_response=lambda: setattr(clock, "value", NOW + timedelta(minutes=6)),
    )
    execution = coordinator(provider, store)
    with SQLiteTaskStateRepository._for_test(path, clock=clock) as repository:
        repository.create(task())
        first = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
        )
        with pytest.raises(StageLeaseExpiredError):
            asyncio.run(execution.execute_claimed_stage_and_persist(
                repository, first, first.execution_authority,
            ))
    with sqlite3.connect(path) as database:
        database.execute(
            "UPDATE stage_execution_records SET result_content_digest = ?",
            ("d" * 64,),
        )
    with SQLiteTaskStateRepository._for_test(path, clock=clock) as repository:
        second = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=2,
        )
        with pytest.raises(SuccessfulResultUnavailableError):
            asyncio.run(execution.execute_claimed_stage_and_persist(
                repository, second, second.execution_authority,
            ))
        assert provider.calls == 1
        assert repository.get_stage_execution_record(
            "task-1", "stage-1",
        ).status is StageExecutionRecordStatus.SUCCEEDED


def test_unverified_artifact_write_result_fails_before_success_is_recorded(tmp_path):
    path = tmp_path / "tasks.sqlite"
    clock = MutableClock()
    tool = CountingWriteTool(
        before_result=lambda: setattr(clock, "value", NOW + timedelta(minutes=6)),
    )
    execution = write_coordinator(tool)
    with SQLiteTaskStateRepository._for_test(path, clock=clock) as repository:
        repository.create(task(plan=write_plan()))
        first = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
        )
        result = asyncio.run(execution.execute_claimed_stage_and_persist(
            repository, first, first.execution_authority,
        ))
        assert tool.calls == 1
        assert result.persisted.state.task_status is TaskStatus.FAILED
        record = repository.get_stage_execution_record("task-1", "stage-1")
        assert record.status is StageExecutionRecordStatus.FAILED
        assert record.result_durability is ExecutionResultDurability.NONE


def test_unverified_artifact_reference_fails_closed(tmp_path):
    path = tmp_path / "tasks.sqlite"
    clock = MutableClock()
    tool = CountingWriteTool(
        before_result=lambda: setattr(clock, "value", NOW + timedelta(minutes=6)),
    )
    execution = write_coordinator(tool)
    with SQLiteTaskStateRepository._for_test(path, clock=clock) as repository:
        repository.create(task(plan=write_plan()))
        first = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
        )
        result = asyncio.run(execution.execute_claimed_stage_and_persist(
            repository, first, first.execution_authority,
        ))
        assert result.persisted.state.task_status is TaskStatus.FAILED
        assert tool.calls == 1


def test_execution_records_store_digests_not_raw_sensitive_inputs(tmp_path):
    path = tmp_path / "tasks.sqlite"
    prompt = "private-prompt-value"
    with SQLiteTaskStateRepository._for_test(path, clock=MutableClock()) as repository:
        state = task(prompt=prompt)
        repository.create(state)
        claim = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
        )
        repository.prepare_stage_execution(claim, claim.execution_authority)
    with sqlite3.connect(path) as database:
        record_values = database.execute(
            "SELECT * FROM stage_execution_records"
        ).fetchone()
        attempt_values = database.execute(
            "SELECT * FROM stage_execution_attempts"
        ).fetchone()
    serialized_records = repr((record_values, attempt_values))
    assert prompt not in serialized_records
    assert "private-prompt-value" not in serialized_records


def test_legacy_success_without_durability_migrates_conservatively(tmp_path):
    path = tmp_path / "legacy.sqlite"
    timestamp = NOW.isoformat()
    with sqlite3.connect(path) as database:
        database.execute("""
            CREATE TABLE stage_execution_records (
                idempotency_key TEXT PRIMARY KEY, task_id TEXT NOT NULL,
                stage_id TEXT NOT NULL, execution_kind TEXT NOT NULL,
                status TEXT NOT NULL, attempt_count INTEGER NOT NULL,
                current_attempt_id TEXT, first_started_at TEXT NOT NULL,
                last_started_at TEXT NOT NULL, completed_at TEXT,
                safe_result_reference TEXT, output_kind TEXT,
                selected_model_id TEXT, selected_tool_id TEXT,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                UNIQUE (task_id, stage_id)
            )
        """)
        database.execute(
            "INSERT INTO stage_execution_records VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("a" * 64, "task-1", "stage-1", "model", "succeeded", 1,
             "b" * 32, timestamp, timestamp, timestamp, "c" * 32, "text",
             "model-chat", None, timestamp, timestamp),
        )
    with SQLiteTaskStateRepository._for_test(path, clock=MutableClock()) as repository:
        record = repository.get_stage_execution_record("task-1", "stage-1")
        assert record.status is StageExecutionRecordStatus.SUCCEEDED
        assert record.result_durability is ExecutionResultDurability.NONE
        assert record.result_content_digest is None


def test_model_failure_is_recorded_once_without_retry(tmp_path):
    path = tmp_path / "tasks.sqlite"
    provider = CountingProvider(fail=True)
    execution = coordinator(provider)
    with SQLiteTaskStateRepository._for_test(path, clock=MutableClock()) as repository:
        repository.create(task())
        claim = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
        )
        result = asyncio.run(execution.execute_claimed_stage_and_persist(
            repository, claim, claim.execution_authority,
        ))
        assert provider.calls == 1
        assert result.persisted.state.task_status is TaskStatus.FAILED
        assert repository.get_stage_execution_record(
            "task-1", "stage-1",
        ).status is StageExecutionRecordStatus.FAILED
