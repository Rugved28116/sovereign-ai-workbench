"""Bounded process-local output storage and task/stage ownership."""

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import pytest

from sovereign_api.stage_output_store import (
    MAX_STAGE_OUTPUT_BYTES, MAX_TOTAL_STAGE_OUTPUT_BYTES,
    InMemoryStageOutputStore, InvalidStageOutputError,
    InvalidStageOutputReferenceError, StageOutput, StageOutputCapacityError,
    StageOutputNotFoundError, StageOutputOwnershipError, StageOutputReference,
    StageOutputTooLargeError,
)


NOW = datetime(2026, 9, 13, tzinfo=UTC)


def _output(text: str = "hello", *, task_id: str = "task-1", stage_id: str = "stage-1"):
    return StageOutput(task_id, stage_id, "text/plain", text, NOW)


def test_put_get_roundtrip_is_opaque_immutable_and_defensively_owned():
    store = InMemoryStageOutputStore()
    supplied = _output("sensitive stage text")
    reference = store.put(supplied)
    assert isinstance(reference, StageOutputReference)
    assert len(reference.value) == 32
    assert "sensitive" not in reference.value
    assert "task-1" not in reference.value
    assert store.get(reference, task_id="task-1", stage_id="stage-1") == supplied
    assert store.total_bytes == len("sensitive stage text".encode("utf-8"))
    with pytest.raises(FrozenInstanceError):
        reference.value = "changed"
    with pytest.raises(FrozenInstanceError):
        supplied.text_content = "changed"
    object.__setattr__(supplied, "text_content", "caller changed")
    assert store.get(reference, task_id="task-1", stage_id="stage-1").text_content == "sensitive stage text"
    retrieved = store.get(reference, task_id="task-1", stage_id="stage-1")
    object.__setattr__(retrieved, "text_content", "retrieval changed")
    assert store.get(reference, task_id="task-1", stage_id="stage-1").text_content == "sensitive stage text"


def test_unknown_malformed_and_cross_identity_references_fail_closed():
    store = InMemoryStageOutputStore()
    reference = store.put(_output())
    with pytest.raises(InvalidStageOutputReferenceError):
        StageOutputReference("not-a-reference")
    with pytest.raises(InvalidStageOutputReferenceError):
        store.get("not-a-reference", task_id="task-1", stage_id="stage-1")
    with pytest.raises(StageOutputNotFoundError):
        store.get(StageOutputReference("0" * 32), task_id="task-1", stage_id="stage-1")
    with pytest.raises(StageOutputOwnershipError):
        store.get(reference, task_id="task-2", stage_id="stage-1")
    with pytest.raises(StageOutputOwnershipError):
        store.get(reference, task_id="task-1", stage_id="stage-2")
    assert store.get(reference, task_id="task-1", stage_id="stage-1").text_content == "hello"


@pytest.mark.parametrize("size", [MAX_STAGE_OUTPUT_BYTES - 1, MAX_STAGE_OUTPUT_BYTES])
def test_per_output_boundary_accepts_at_or_below_limit(size: int):
    text = "a" * size
    store = InMemoryStageOutputStore()
    reference = store.put(_output(text))
    assert len(store.get(reference, task_id="task-1", stage_id="stage-1").text_content) == size
    assert store.total_bytes == size


def test_per_output_limit_rejects_more_than_utf8_byte_limit():
    with pytest.raises(StageOutputTooLargeError):
        _output("a" * (MAX_STAGE_OUTPUT_BYTES + 1))
    with pytest.raises(StageOutputTooLargeError):
        _output("€" * (MAX_STAGE_OUTPUT_BYTES // 3 + 1))
    with pytest.raises(InvalidStageOutputError):
        _output("bad\ud800surrogate")


def test_total_capacity_is_exact_no_eviction_and_deterministically_accounted():
    store = InMemoryStageOutputStore(max_total_bytes=5)
    first = store.put(_output("é"))  # two encoded bytes
    second = store.put(_output("abc", stage_id="stage-2"))
    assert store.total_bytes == 5
    with pytest.raises(StageOutputCapacityError):
        store.put(_output("x", stage_id="stage-3"))
    assert store.total_bytes == 5
    assert store.get(first, task_id="task-1", stage_id="stage-1").text_content == "é"
    assert store.get(second, task_id="task-1", stage_id="stage-2").text_content == "abc"


def test_total_capacity_configuration_cannot_exceed_fixed_limit():
    with pytest.raises(InvalidStageOutputError):
        InMemoryStageOutputStore(max_total_bytes=MAX_TOTAL_STAGE_OUTPUT_BYTES + 1)


def test_entry_count_prevents_unbounded_tiny_output_metadata(monkeypatch):
    monkeypatch.setattr("sovereign_api.stage_output_store.MAX_STAGE_OUTPUT_ENTRIES", 2)
    store = InMemoryStageOutputStore()
    store.put(_output("x", stage_id="stage-1"))
    store.put(_output("y", stage_id="stage-2"))
    with pytest.raises(StageOutputCapacityError):
        store.put(_output("z", stage_id="stage-3"))
    assert store.total_bytes == 2


def test_output_contract_rejects_non_text_and_naive_timestamp():
    with pytest.raises(InvalidStageOutputError):
        StageOutput("task-1", "stage-1", "application/octet-stream", "x", NOW)
    with pytest.raises(InvalidStageOutputError):
        StageOutput("task-1", "stage-1", "text/plain", "x", datetime(2026, 9, 13))
