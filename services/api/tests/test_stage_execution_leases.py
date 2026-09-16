"""Atomic persisted stage claiming and lease-bound completion tests."""

import copy
import json
import pickle
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from threading import Barrier

import pytest

from test_task_state_repository import START, at, awaiting_task, mixed_task, model_task
from sovereign_api.agent_task_state import (
    AgentTaskState, StageOutputKind, StageStatus, TaskStatus,
)
import sovereign_api.task_state_repository as repository_module
from sovereign_api.task_state_repository import (
    DEFAULT_STAGE_LEASE_DURATION, MAX_STAGE_LEASE_DURATION,
    ClaimedTaskState, CorruptTaskStateError, InvalidLeaseConfigurationError,
    InvalidRepositoryClockError, SQLiteTaskStateRepository,
    StageAlreadyClaimedError, StageClaimError,
    StageLeaseExpiredError, StageLeaseMismatchError, StaleTaskStateError,
    TaskPersistenceError, ValidatedStageExecutionLease, serialize_task_state,
)


def repo(path, *, clock=None):
    return SQLiteTaskStateRepository(
        path, clock=clock if clock is not None else (lambda: at(10)),
    )


class MutableClock:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.value


def other_model_task():
    return AgentTaskState.from_plan(
        task_id="task-2", original_prompt="Other",
        plan=model_task().plan, timestamp=START,
    )


def legacy_running_state():
    state = model_task().start(updated_at=at(1))
    return state.update_stage(state.stage_states[0].start(), updated_at=at(2))


def write_pre_lease_database(path, state):
    with sqlite3.connect(path) as database:
        database.execute("""
            CREATE TABLE agent_task_states (
                task_id TEXT PRIMARY KEY, state_json TEXT NOT NULL,
                version INTEGER NOT NULL, created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL, persisted_at TEXT NOT NULL
            )
        """)
        database.execute(
            "INSERT INTO agent_task_states "
            "(task_id, state_json, version, created_at, updated_at, persisted_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (state.task_id, serialize_task_state(state), 1,
             state.created_at.isoformat(), state.updated_at.isoformat(), at(3).isoformat()),
        )


def completed_state(claim):
    state = claim.persisted.state
    active = next(item for item in state.stage_states
                  if item.stage_id == claim.lease.stage_id)
    state = state.update_stage(active.complete(
        output_reference="a" * 32, selected_model_id="model-1",
        output_kind=StageOutputKind.TEXT,
    ), updated_at=at(20))
    if all(item.status is StageStatus.COMPLETED for item in state.stage_states):
        state = state.complete(updated_at=at(21))
    return state


def failed_state(claim):
    state = claim.persisted.state
    active = next(item for item in state.stage_states
                  if item.stage_id == claim.lease.stage_id)
    return state.update_stage(active.fail(
        error_code="provider_failed", safe_message="Stage execution failed",
    ), updated_at=at(20))


def finish(repository, claim, state):
    return repository.complete_claim(
        claim.lease.task_id, claim.lease.stage_id, claim.lease.lease_id,
        claim.execution_authority, state,
        expected_version=claim.persisted.version,
    )


def test_claim_next_pending_stage_is_atomic_and_immutable(tmp_path):
    path = tmp_path / "tasks.sqlite"
    with repo(path) as repository:
        repository.create(model_task())
        claim = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
        )
        assert type(claim) is ClaimedTaskState
        assert claim.persisted.version == claim.lease.claimed_version == 2
        assert claim.persisted.state.task_status is TaskStatus.RUNNING
        assert claim.persisted.state.current_stage_id == "stage-1"
        assert claim.persisted.state.stage_states[0].status is StageStatus.RUNNING
        assert claim.lease.claimed_at == at(10)
        assert claim.lease.lease_expires_at == at(10) + DEFAULT_STAGE_LEASE_DURATION
        assert len(claim.lease.lease_id) == 32
        assert repository.get("task-1") == claim.persisted
        with pytest.raises(FrozenInstanceError):
            claim.lease.stage_id = "other"


def test_competing_stale_reader_loses_before_external_execution(tmp_path):
    path = tmp_path / "tasks.sqlite"
    calls = {"a": 0, "b": 0}
    with repo(path) as first, repo(path) as second:
        first.create(model_task())
        assert first.get("task-1").version == second.get("task-1").version == 1
        claim = first.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
        )
        calls["a"] += 1
        with pytest.raises(StaleTaskStateError):
            second.claim_next_stage(
                "task-1", "stage-1", expected_version=1,
            )
        assert calls == {"a": 1, "b": 0}
        assert second.get("task-1").lease == claim.lease


def test_concurrent_repository_instances_have_one_claim_winner(tmp_path):
    path = tmp_path / "tasks.sqlite"
    with repo(path) as repository:
        repository.create(model_task())
    barrier = Barrier(2)

    def worker(offset):
        with repo(path) as repository:
            loaded = repository.get("task-1")
            barrier.wait(timeout=5)
            try:
                return repository.claim_next_stage(
                    "task-1", "stage-1", expected_version=loaded.version,
                )
            except StaleTaskStateError:
                return "stale"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(worker, (0, 1)))
    assert sum(item == "stale" for item in results) == 1
    assert sum(type(item) is ClaimedTaskState for item in results) == 1


def test_wrong_or_non_executable_stage_cannot_be_claimed(tmp_path):
    path = tmp_path / "tasks.sqlite"
    with repo(path) as repository:
        repository.create(mixed_task())
        with pytest.raises(StageClaimError):
            repository.claim_next_stage(
                "task-1", "stage-2", expected_version=1,
            )
        assert repository.get("task-1").version == 1


def test_completed_stage_cannot_be_claimed_again(tmp_path):
    path = tmp_path / "tasks.sqlite"
    state = mixed_task().start(updated_at=at(1))
    state = state.update_stage(state.stage_states[0].start(), updated_at=at(2))
    state = state.update_stage(state.stage_states[0].complete(
        output_reference="a" * 32, selected_model_id="model-1",
    ), updated_at=at(3))
    with repo(path) as repository:
        repository.create(state)
        with pytest.raises(StageClaimError):
            repository.claim_next_stage(
                "task-1", "stage-1", expected_version=1,
            )
        claim = repository.claim_next_stage(
            "task-1", "stage-2", expected_version=1,
        )
        assert claim.lease.stage_id == "stage-2"


@pytest.mark.parametrize("terminal", [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED])
def test_terminal_task_cannot_be_claimed(tmp_path, terminal):
    state = model_task().start(updated_at=at(1))
    if terminal is TaskStatus.COMPLETED:
        state = state.update_stage(state.stage_states[0].start(), updated_at=at(2))
        state = state.update_stage(state.stage_states[0].complete(
            output_reference="a" * 32, selected_model_id="model-1",
        ), updated_at=at(3)).complete(updated_at=at(4))
    elif terminal is TaskStatus.FAILED:
        state = state.fail(updated_at=at(2))
    else:
        state = state.cancel(updated_at=at(2))
    path = tmp_path / "tasks.sqlite"
    with repo(path) as repository:
        repository.create(state)
        with pytest.raises(StageClaimError):
            repository.claim_next_stage(
                "task-1", "stage-1", expected_version=1,
            )


def test_awaiting_approval_cannot_be_normally_claimed(tmp_path):
    path = tmp_path / "tasks.sqlite"
    with repo(path) as repository:
        repository.create(awaiting_task())
        with pytest.raises(StageClaimError):
            repository.claim_next_stage(
                "task-1", "stage-2", expected_version=1,
            )
        loaded = repository.get("task-1")
        assert loaded.state.task_status is TaskStatus.AWAITING_APPROVAL
        assert loaded.lease is None


def test_valid_owner_completes_and_clears_lease(tmp_path):
    path = tmp_path / "tasks.sqlite"
    with repo(path) as repository:
        repository.create(model_task())
        claim = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
        )
        saved = finish(repository, claim, completed_state(claim))
        assert saved.version == 3 and saved.lease is None
        assert saved.state.task_status is TaskStatus.COMPLETED
        assert repository.get("task-1") == saved


def test_valid_owner_persists_failure_and_clears_lease(tmp_path):
    path = tmp_path / "tasks.sqlite"
    with repo(path) as repository:
        repository.create(model_task())
        claim = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
        )
        saved = finish(repository, claim, failed_state(claim))
        assert saved.version == 3 and saved.lease is None
        assert saved.state.task_status is TaskStatus.FAILED


def test_wrong_lease_identity_and_forged_capability_fail_closed(tmp_path):
    path = tmp_path / "tasks.sqlite"
    with repo(path) as repository:
        repository.create(model_task())
        claim = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
        )
        state = completed_state(claim)
        with pytest.raises(StageLeaseMismatchError):
            repository.complete_claim(
                "task-1", "stage-1", "0" * 32, claim.execution_authority,
                state, expected_version=2,
            )
        with pytest.raises(StageLeaseMismatchError):
            ValidatedStageExecutionLease(
                claim.lease.lease_id, "task-1", "stage-1", 2,
            )
        forged = object.__new__(ValidatedStageExecutionLease)
        for name, value in (
            ("lease_id", claim.lease.lease_id), ("task_id", "task-1"),
            ("stage_id", "stage-1"), ("claimed_version", 2),
        ):
            object.__setattr__(forged, name, value)
        with pytest.raises(StageLeaseMismatchError):
            repository.complete_claim(
                "task-1", "stage-1", claim.lease.lease_id, forged,
                state, expected_version=2,
            )
        assert repository.get("task-1").version == 2


def test_claim_authority_is_unique_bound_noncopyable_and_not_persisted(tmp_path):
    path = tmp_path / "tasks.sqlite"
    with repo(path) as repository:
        repository.create(model_task())
        repository.create(other_model_task())
        claim_a = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
        )
        claim_b = repository.claim_next_stage(
            "task-2", "stage-1", expected_version=1,
        )
        secret_a = repository_module._lease_authority_secret(claim_a.execution_authority)
        secret_b = repository_module._lease_authority_secret(claim_b.execution_authority)
        assert secret_a is not None and secret_b is not None and secret_a != secret_b
        assert secret_a.hex() not in repr(claim_a.execution_authority)
        with pytest.raises(TypeError):
            copy.copy(claim_a.execution_authority)
        with pytest.raises(TypeError):
            copy.deepcopy(claim_a.execution_authority)
        with pytest.raises(TypeError):
            pickle.dumps(claim_a.execution_authority)

        with sqlite3.connect(path) as database:
            rows = database.execute(
                "SELECT task_id, lease_proof FROM agent_task_states ORDER BY task_id"
            ).fetchall()
        assert rows[0][1] != rows[1][1]
        database_bytes = path.read_bytes()
        assert secret_a not in database_bytes
        assert secret_a.hex().encode() not in database_bytes


def test_authority_field_mutation_or_secret_reuse_cannot_cross_claims(tmp_path):
    path = tmp_path / "tasks.sqlite"
    with repo(path) as repository:
        repository.create(model_task())
        repository.create(other_model_task())
        claim_a = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
        )
        claim_b = repository.claim_next_stage(
            "task-2", "stage-1", expected_version=1,
        )
        for field in ("lease_id", "task_id", "stage_id", "claimed_version"):
            object.__setattr__(
                claim_a.execution_authority, field,
                getattr(claim_b.execution_authority, field),
            )
        with pytest.raises(StageLeaseMismatchError):
            repository.complete_claim(
                "task-2", "stage-1", claim_b.lease.lease_id,
                claim_a.execution_authority, completed_state(claim_b),
                expected_version=2,
            )

        secret_a = repository_module._lease_authority_secret(claim_a.execution_authority)
        assert secret_a is not None
        reused = repository_module._lease_authority(claim_b.lease, secret_a)
        with pytest.raises(StageLeaseMismatchError):
            repository.complete_claim(
                "task-2", "stage-1", claim_b.lease.lease_id, reused,
                completed_state(claim_b), expected_version=2,
            )
        assert repository.get("task-2").version == 2


@pytest.mark.parametrize("field,value", [
    ("task_id", "other-task"), ("stage_id", "other-stage"),
])
def test_wrong_task_or_stage_cannot_complete(tmp_path, field, value):
    path = tmp_path / "tasks.sqlite"
    with repo(path) as repository:
        repository.create(model_task())
        claim = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
        )
        arguments = {
            "task_id": "task-1", "stage_id": "stage-1",
            "lease_id": claim.lease.lease_id,
        }
        arguments[field] = value
        with pytest.raises((StageLeaseMismatchError, TaskPersistenceError)):
            repository.complete_claim(
                arguments["task_id"], arguments["stage_id"], arguments["lease_id"],
                claim.execution_authority, completed_state(claim),
                expected_version=2,
            )


def test_claim_survives_restart_and_unexpired_claim_blocks(tmp_path):
    path = tmp_path / "tasks.sqlite"
    with repo(path) as repository:
        repository.create(model_task())
        claim = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
        )
    with repo(path) as reopened:
        loaded = reopened.get("task-1")
        assert loaded.lease == claim.lease
        with pytest.raises(StageAlreadyClaimedError):
            reopened.claim_next_stage(
                "task-1", "stage-1", expected_version=2,
            )


def test_caller_cannot_future_date_premature_reclaim(tmp_path):
    path = tmp_path / "tasks.sqlite"
    owner_clock = MutableClock(at(10))
    contender_clock = MutableClock(at(11))
    with repo(path, clock=owner_clock) as owner, repo(
        path, clock=contender_clock,
    ) as contender:
        owner.create(model_task())
        claim = owner.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
        )
        with pytest.raises(TypeError):
            contender.claim_next_stage(
                "task-1", "stage-1", expected_version=2, now=at(10_000),
            )
        with pytest.raises(StageAlreadyClaimedError):
            contender.claim_next_stage(
                "task-1", "stage-1", expected_version=2,
            )
        assert owner.get("task-1").lease == claim.lease


def test_expired_claim_can_be_reclaimed_and_old_owner_is_stale(tmp_path):
    path = tmp_path / "tasks.sqlite"
    first_clock = MutableClock(at(10))
    with repo(path, clock=first_clock) as repository:
        repository.create(model_task())
        old = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
            lease_duration=timedelta(seconds=5),
        )
    with repo(path, clock=MutableClock(at(16))) as reopened:
        new = reopened.claim_next_stage(
            "task-1", "stage-1", expected_version=2,
        )
        assert new.persisted.version == 3
        assert new.lease.lease_id != old.lease.lease_id
        with pytest.raises(StaleTaskStateError):
            finish(reopened, old, completed_state(old))
        assert reopened.get("task-1").lease == new.lease


def test_expired_lease_cannot_complete_without_reclaim(tmp_path):
    path = tmp_path / "tasks.sqlite"
    clock = MutableClock(at(10))
    with repo(path, clock=clock) as repository:
        repository.create(model_task())
        claim = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
            lease_duration=timedelta(seconds=5),
        )
        clock.value = at(15)
        with pytest.raises(TypeError):
            repository.complete_claim(
                "task-1", "stage-1", claim.lease.lease_id,
                claim.execution_authority, completed_state(claim),
                expected_version=2, now=at(11),
            )
        with pytest.raises(StageLeaseExpiredError):
            finish(repository, claim, completed_state(claim))
        assert repository.get("task-1").lease == claim.lease


@pytest.mark.parametrize("bad_value", [datetime(2026, 9, 15), object()])
def test_invalid_repository_clock_fails_without_mutation(tmp_path, bad_value):
    path = tmp_path / "tasks.sqlite"
    with repo(path) as repository:
        repository.create(model_task())
    with repo(path, clock=lambda: bad_value) as repository:
        with pytest.raises(InvalidRepositoryClockError):
            repository.claim_next_stage(
                "task-1", "stage-1", expected_version=1,
            )
    with repo(path) as repository:
        assert repository.get("task-1").version == 1


def test_raising_repository_clock_fails_without_mutation(tmp_path):
    path = tmp_path / "tasks.sqlite"
    with repo(path) as repository:
        repository.create(model_task())

    def broken_clock():
        raise RuntimeError("untrusted clock failure")

    with repo(path, clock=broken_clock) as repository:
        with pytest.raises(InvalidRepositoryClockError):
            repository.claim_next_stage(
                "task-1", "stage-1", expected_version=1,
            )
    with repo(path) as repository:
        assert repository.get("task-1").version == 1


def test_authority_operations_read_repository_clock_once(tmp_path):
    path = tmp_path / "tasks.sqlite"

    class AdvancingClock:
        def __init__(self):
            self.calls = 0

        def __call__(self):
            self.calls += 1
            return at(self.calls)

    clock = AdvancingClock()
    with repo(path, clock=clock) as repository:
        repository.create(model_task())
        before_claim = clock.calls
        claim = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
        )
        assert clock.calls == before_claim + 1
        before_completion = clock.calls
        finish(repository, claim, completed_state(claim))
        assert clock.calls == before_completion + 1


@pytest.mark.parametrize("duration", [
    timedelta(0),
    timedelta(seconds=-1),
    MAX_STAGE_LEASE_DURATION + timedelta(microseconds=1),
    300,
])
def test_invalid_claim_duration_fails_before_change(tmp_path, duration):
    path = tmp_path / "tasks.sqlite"
    with repo(path) as repository:
        repository.create(model_task())
        with pytest.raises(InvalidLeaseConfigurationError):
            repository.claim_next_stage(
                "task-1", "stage-1", expected_version=1, lease_duration=duration,
            )
        assert repository.get("task-1").version == 1


def test_general_replace_cannot_bypass_or_clear_active_lease(tmp_path):
    path = tmp_path / "tasks.sqlite"
    with repo(path) as repository:
        repository.create(model_task())
        claim = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
        )
        with pytest.raises(TaskPersistenceError):
            repository.replace(
                "task-1", completed_state(claim), expected_version=2,
            )
        with pytest.raises(StaleTaskStateError):
            repository.replace(
                "task-1", model_task(), expected_version=1,
            )
        assert repository.get("task-1").lease == claim.lease


def test_old_schema_database_is_migrated_without_changing_json_schema(tmp_path):
    path = tmp_path / "tasks.sqlite"
    state = model_task()
    with sqlite3.connect(path) as database:
        database.execute("""
            CREATE TABLE agent_task_states (
                task_id TEXT PRIMARY KEY, state_json TEXT NOT NULL,
                version INTEGER NOT NULL, created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL, persisted_at TEXT NOT NULL
            )
        """)
        database.execute(
            "INSERT INTO agent_task_states "
            "(task_id, state_json, version, created_at, updated_at, persisted_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (state.task_id, serialize_task_state(state), 1,
             state.created_at.isoformat(), state.updated_at.isoformat(), at(1).isoformat()),
        )
    with repo(path) as repository:
        assert repository.get("task-1").state == state
        claim = repository.claim_next_stage(
            "task-1", "stage-1", expected_version=1,
        )
        assert claim.persisted.version == 2


def test_pre_lease_running_row_is_marked_recovered_and_completed(tmp_path):
    path = tmp_path / "tasks.sqlite"
    state = legacy_running_state()
    write_pre_lease_database(path, state)
    recovery_clock = MutableClock(at(10))

    with repo(path, clock=recovery_clock) as first, repo(
        path, clock=MutableClock(at(11)),
    ) as second:
        loaded = first.get("task-1")
        assert loaded.state == state
        assert loaded.lease is None
        assert loaded.legacy_unleased_running is True
        with pytest.raises(StageClaimError):
            first.claim_next_stage(
                "task-1", "stage-1", expected_version=1,
            )
        with pytest.raises(TypeError):
            first.claim_legacy_running_stage(
                "task-1", "stage-1", expected_version=1, now=at(10_000),
            )
        calls_before_recovery = recovery_clock.calls
        recovered = first.claim_legacy_running_stage(
            "task-1", "stage-1", expected_version=1,
        )
        assert recovery_clock.calls == calls_before_recovery + 1
        assert recovered.persisted.version == 2
        assert recovered.persisted.state == state
        assert recovered.persisted.legacy_unleased_running is False
        assert recovered.lease.claimed_at == at(10)
        assert recovered.lease.lease_expires_at == at(10) + DEFAULT_STAGE_LEASE_DURATION
        with pytest.raises(StaleTaskStateError):
            second.claim_legacy_running_stage(
                "task-1", "stage-1", expected_version=1,
            )

    with repo(path) as reopened:
        assert reopened.get("task-1").lease == recovered.lease
        saved = finish(reopened, recovered, completed_state(recovered))
        assert saved.version == 3
        assert saved.state.task_status is TaskStatus.COMPLETED


def test_unrelated_authority_cannot_complete_recovered_legacy_claim(tmp_path):
    path = tmp_path / "tasks.sqlite"
    other_path = tmp_path / "other.sqlite"
    write_pre_lease_database(path, legacy_running_state())
    with repo(path) as repository, repo(other_path) as other:
        recovered = repository.claim_legacy_running_stage(
            "task-1", "stage-1", expected_version=1,
        )
        other.create(other_model_task())
        unrelated = other.claim_next_stage(
            "task-2", "stage-1", expected_version=1,
        )
        for field in ("lease_id", "task_id", "stage_id", "claimed_version"):
            object.__setattr__(
                unrelated.execution_authority, field,
                getattr(recovered.execution_authority, field),
            )
        with pytest.raises(StageLeaseMismatchError):
            repository.complete_claim(
                "task-1", "stage-1", recovered.lease.lease_id,
                unrelated.execution_authority, completed_state(recovered),
                expected_version=2,
            )


def test_invalid_pre_lease_running_shape_is_not_marked_recoverable(tmp_path):
    path = tmp_path / "tasks.sqlite"
    state = legacy_running_state()
    write_pre_lease_database(path, state)
    with sqlite3.connect(path) as database:
        payload = json.loads(database.execute(
            "SELECT state_json FROM agent_task_states WHERE task_id = ?", ("task-1",),
        ).fetchone()[0])
        payload["state"]["current_stage_id"] = "wrong-stage"
        database.execute(
            "UPDATE agent_task_states SET state_json = ? WHERE task_id = ?",
            (json.dumps(payload), "task-1"),
        )
    with repo(path) as repository:
        with pytest.raises(CorruptTaskStateError):
            repository.get("task-1")
        with pytest.raises(CorruptTaskStateError):
            repository.claim_legacy_running_stage(
                "task-1", "stage-1", expected_version=1,
            )


def test_partial_or_inconsistent_lease_columns_are_corruption(tmp_path):
    path = tmp_path / "tasks.sqlite"
    with repo(path) as repository:
        repository.create(model_task())
    with sqlite3.connect(path) as database:
        database.execute(
            "UPDATE agent_task_states SET lease_id = ? WHERE task_id = ?",
            ("a" * 32, "task-1"),
        )
    with repo(path) as repository:
        with pytest.raises(CorruptTaskStateError):
            repository.get("task-1")
