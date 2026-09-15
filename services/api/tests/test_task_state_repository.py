"""Durable, explicit JSON task snapshots and SQLite compare-and-swap tests."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from threading import Barrier

import pytest

from sovereign_api.agent_task_state import AgentTaskState, StageOutputKind, StageStatus, TaskStatus
from sovereign_api.task_classification import TaskClass
from sovereign_api.task_planning import StageExecutionKind, TaskPlan, TaskStage, TaskStageType
from sovereign_api.task_state_repository import (
    CorruptTaskStateError, InvalidTaskDatabaseConfigurationError,
    SQLiteTaskStateRepository, StaleTaskStateError, TaskAlreadyExistsError,
    TaskNotFoundError, TaskPersistenceError, deserialize_task_state,
    serialize_task_state,
)
from sovereign_api.tool_approval import ApprovalRequest, request_fingerprint
from sovereign_api.tool_contracts import (
    ToolPermission, ToolRequest, ToolRiskLevel, ToolSideEffectLevel,
)


START = datetime(2026, 9, 14, 8, tzinfo=UTC)


def at(seconds):
    return START + timedelta(seconds=seconds)


def model_task():
    plan = TaskPlan(TaskClass.GENERAL, (
        TaskStage("stage-1", TaskStageType.GENERATE, ("chat",)),
    ))
    return AgentTaskState.from_plan(
        task_id="task-1", original_prompt="Hello", plan=plan, timestamp=START,
    )


def mixed_task():
    plan = TaskPlan(TaskClass.DOCUMENT, (
        TaskStage("stage-1", TaskStageType.DOCUMENT, ("document",)),
        TaskStage(
            "stage-2", TaskStageType.TOOL, (), StageExecutionKind.TOOL,
            "workspace.write_artifact", {"path": "reports/output.txt", "content": "safe"},
        ),
    ))
    return AgentTaskState.from_plan(
        task_id="task-1", original_prompt="Summarize report", plan=plan, timestamp=START,
    )


def first_completed(state):
    state = state.start(updated_at=at(1))
    state = state.update_stage(state.stage_states[0].start(), updated_at=at(2))
    return state.update_stage(
        state.stage_states[0].complete(
            output_reference="a" * 32, selected_model_id="model-1",
            output_kind=StageOutputKind.TEXT,
        ), updated_at=at(3),
    )


def awaiting_task():
    state = first_completed(mixed_task())
    stage = state.plan.stages[1]
    state = state.update_stage(
        state.stage_states[1].start(), updated_at=at(4),
    )
    permission = ToolPermission("artifact.write")
    request = ToolRequest(
        "request-1", stage.tool_id, "write_artifact", stage.tool_arguments,
        state.task_id, stage.stage_id,
    )
    approval = ApprovalRequest(
        approval_id="a" * 32, request_id=request.request_id,
        task_id=state.task_id, stage_id=stage.stage_id,
        tool_id=stage.tool_id, operation=request.operation,
        requested_permissions=(permission,), risk_level=ToolRiskLevel.MEDIUM,
        side_effect_level=ToolSideEffectLevel.WRITE,
        created_at=at(5), safe_summary="Approval required",
        request_fingerprint=request_fingerprint(request, (permission,)),
    )
    return state.update_stage(
        state.stage_states[1].await_approval(selected_tool_id=stage.tool_id),
        updated_at=at(5), approval_request=approval,
    )


def repository(tmp_path):
    return SQLiteTaskStateRepository(tmp_path / "tasks.sqlite", clock=lambda: at(10))


@pytest.mark.parametrize("bad", [None, "", "tasks.sqlite", "/bad\x00path", object()])
def test_database_path_must_be_explicit_and_valid(tmp_path, bad):
    with pytest.raises(InvalidTaskDatabaseConfigurationError):
        SQLiteTaskStateRepository(bad)
    assert not (tmp_path / "tasks.sqlite").exists()


def test_missing_parent_and_directory_path_are_rejected(tmp_path, monkeypatch):
    with pytest.raises(InvalidTaskDatabaseConfigurationError):
        SQLiteTaskStateRepository(tmp_path / "missing" / "tasks.sqlite")
    with pytest.raises(InvalidTaskDatabaseConfigurationError):
        SQLiteTaskStateRepository(tmp_path)
    monkeypatch.delenv("SOVEREIGN_TASK_DB_PATH", raising=False)
    with pytest.raises(InvalidTaskDatabaseConfigurationError):
        SQLiteTaskStateRepository.from_environment()
    monkeypatch.setenv("SOVEREIGN_TASK_DB_PATH", str(tmp_path / "configured.sqlite"))
    with SQLiteTaskStateRepository.from_environment() as repo:
        assert repo.create(model_task()).version == 1


def test_create_get_duplicate_unknown_and_immutability(tmp_path):
    state = model_task()
    with repository(tmp_path) as repo:
        created = repo.create(state)
        assert created.version == 1 and created.state == state
        assert created.persisted_at == at(10)
        loaded = repo.get(state.task_id)
        assert loaded == created and loaded.state is not state
        with pytest.raises(FrozenInstanceError):
            loaded.version = 2
        with pytest.raises(FrozenInstanceError):
            loaded.state.stage_states[0].status = StageStatus.COMPLETED
        with pytest.raises(TaskAlreadyExistsError):
            repo.create(state)
        with pytest.raises(TaskNotFoundError):
            repo.get("other-task")


def test_cas_stale_writer_cannot_overwrite_newer_state(tmp_path):
    state = model_task()
    with repository(tmp_path) as first, repository(tmp_path) as second:
        first.create(state)
        reader_a = first.get(state.task_id)
        reader_b = second.get(state.task_id)
        assert reader_a.version == reader_b.version == 1
        new_a = reader_a.state.start(updated_at=at(1))
        saved = first.replace(state.task_id, new_a, expected_version=1)
        assert saved.version == 2 and saved.state == new_a
        with pytest.raises(StaleTaskStateError):
            second.replace(state.task_id, reader_b.state, expected_version=1)
        assert second.get(state.task_id) == saved


def test_two_repository_instances_racing_have_one_cas_winner(tmp_path):
    path = tmp_path / "tasks.sqlite"
    with SQLiteTaskStateRepository(path) as repo:
        repo.create(model_task())
    ready = Barrier(2)

    def writer(second):
        with SQLiteTaskStateRepository(path) as repo:
            loaded = repo.get("task-1")
            ready.wait(timeout=5)
            try:
                return repo.replace(
                    "task-1", loaded.state.start(updated_at=at(second)),
                    expected_version=loaded.version,
                )
            except StaleTaskStateError:
                return "stale"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(writer, (1, 2)))
    assert sum(result == "stale" for result in results) == 1
    winners = [result for result in results if result != "stale"]
    assert len(winners) == 1 and winners[0].version == 2
    with SQLiteTaskStateRepository(path) as repo:
        assert repo.get("task-1") == winners[0]


def test_replace_identity_version_and_missing_task_fail_closed(tmp_path):
    state = model_task()
    with repository(tmp_path) as repo:
        repo.create(state)
        for bad in (0, -1, True, "1"):
            with pytest.raises(TaskPersistenceError):
                repo.replace(state.task_id, state, expected_version=bad)
        with pytest.raises(TaskPersistenceError):
            repo.replace("another-task", state, expected_version=1)
        changed_plan = TaskPlan(TaskClass.GENERAL, (
            TaskStage("different-stage", TaskStageType.GENERATE, ("chat",)),
        ))
        with pytest.raises(TaskPersistenceError):
            repo.replace(state.task_id, AgentTaskState.from_plan(
                task_id=state.task_id, original_prompt=state.original_prompt,
                plan=changed_plan, timestamp=START,
            ), expected_version=1)
        with pytest.raises(TaskNotFoundError):
            repo.replace("missing", AgentTaskState.from_plan(
                task_id="missing", original_prompt="Hello", plan=state.plan,
                timestamp=START,
            ), expected_version=1)
        assert repo.get(state.task_id).version == 1


def test_round_trip_model_tool_completed_pending_and_approval(tmp_path):
    for state in (model_task(), first_completed(mixed_task()), awaiting_task()):
        decoded = deserialize_task_state(serialize_task_state(state))
        assert decoded == state and decoded is not state
        assert decoded.plan.stages == state.plan.stages
    with repository(tmp_path) as repo:
        state = awaiting_task()
        repo.create(state)
        loaded = repo.get(state.task_id).state
        assert loaded == state
        assert loaded.approval_request == state.approval_request
        assert loaded.stage_states[0].selected_model_id == "model-1"
        assert loaded.stage_states[1].selected_tool_id == "workspace.write_artifact"
        assert loaded.plan.stages[1].tool_arguments["path"] == "reports/output.txt"
        with pytest.raises(TypeError):
            loaded.plan.stages[1].tool_arguments["path"] = "other.txt"
        assert "_capability" not in serialize_task_state(loaded)


def test_round_trip_completed_tool_with_artifact_reference():
    state = awaiting_task()
    state = state.update_stage(state.stage_states[1].resume_approval(), updated_at=at(6))
    state = state.update_stage(state.stage_states[1].complete(
        output_reference="reports/output.txt", selected_tool_id="workspace.write_artifact",
        output_kind=StageOutputKind.ARTIFACT,
    ), updated_at=at(7))
    state = state.complete(updated_at=at(8))
    assert deserialize_task_state(serialize_task_state(state)) == state


def test_repository_survives_reopen(tmp_path):
    path = tmp_path / "tasks.sqlite"
    with SQLiteTaskStateRepository(path, clock=lambda: at(10)) as repo:
        saved = repo.create(model_task())
    with SQLiteTaskStateRepository(path, clock=lambda: at(11)) as reopened:
        assert reopened.get("task-1") == saved
    with pytest.raises(TaskPersistenceError):
        reopened.get("task-1")


def test_canonical_json_and_no_raw_output_storage():
    state = first_completed(mixed_task())
    payload = serialize_task_state(state)
    assert payload == serialize_task_state(state)
    assert payload.startswith('{"schema_version":1,"state":')
    assert "a" * 32 in payload
    assert "stage output body" not in payload


@pytest.mark.parametrize("mutation", [
    lambda data: data.update(schema_version=2),
    lambda data: data["state"].update(task_status="unknown"),
    lambda data: data["state"]["stage_states"][0].update(selected_model_id="bad\x00id"),
    lambda data: data["state"]["stage_states"][0].update(stage_id="wrong"),
    lambda data: data["state"]["plan"]["stages"][0].update(execution_kind="other"),
    lambda data: data["state"]["plan"]["stages"][0].update(tool_arguments=["bad"]),
])
def test_invalid_persisted_nested_state_is_rejected(mutation):
    data = json.loads(serialize_task_state(model_task()))
    mutation(data)
    with pytest.raises(CorruptTaskStateError):
        deserialize_task_state(json.dumps(data))


def test_corrupt_stage_order_and_tool_arguments_fail_closed():
    data = json.loads(serialize_task_state(awaiting_task()))
    data["state"]["stage_states"][0]["status"] = "pending"
    with pytest.raises(CorruptTaskStateError):
        deserialize_task_state(json.dumps(data))
    data = json.loads(serialize_task_state(awaiting_task()))
    data["state"]["plan"]["stages"][1]["tool_arguments"] = ["not an object"]
    with pytest.raises(CorruptTaskStateError):
        deserialize_task_state(json.dumps(data))


@pytest.mark.parametrize("payload", ["{", "{}", '{"schema_version":1,"schema_version":1,"state":{}}', "null"])
def test_malformed_envelope_is_rejected(payload):
    with pytest.raises(CorruptTaskStateError):
        deserialize_task_state(payload)


def test_database_corruption_is_typed_and_safe(tmp_path):
    path = tmp_path / "tasks.sqlite"
    with repository(tmp_path) as repo:
        repo.create(model_task())
    with sqlite3.connect(path) as db:
        db.execute("UPDATE agent_task_states SET state_json = ? WHERE task_id = ?",
                   ("{malformed", "task-1"))
    with SQLiteTaskStateRepository(path) as repo:
        with pytest.raises(CorruptTaskStateError) as error:
            repo.get("task-1")
        assert str(path) not in str(error.value)
        with pytest.raises(CorruptTaskStateError):
            repo.replace("task-1", model_task(), expected_version=1)


def test_tampered_in_memory_nested_state_cannot_be_stored(tmp_path):
    state = model_task()
    object.__setattr__(state.stage_states[0], "stage_id", "other-stage")
    with repository(tmp_path) as repo:
        with pytest.raises(TaskPersistenceError):
            repo.create(state)
        with pytest.raises(TaskNotFoundError):
            repo.get(state.task_id)
