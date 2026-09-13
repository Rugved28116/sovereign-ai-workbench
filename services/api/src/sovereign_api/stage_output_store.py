"""Bounded provider-neutral outputs for process-local stage chaining."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from threading import Lock
from typing import Protocol
from uuid import uuid4

from sovereign_api.errors import OrchestrationError


MAX_STAGE_OUTPUT_BYTES = 262_144
MAX_TOTAL_STAGE_OUTPUT_BYTES = 8_388_608
MAX_STAGE_OUTPUT_ENTRIES = 1_024
_REFERENCE_PATTERN = re.compile(r"[0-9a-f]{32}\Z")


class StageOutputStoreError(OrchestrationError):
    code = "stage_output_store_error"


class InvalidStageOutputError(StageOutputStoreError):
    code = "invalid_stage_output"


class StageOutputTooLargeError(StageOutputStoreError):
    code = "stage_output_too_large"


class StageOutputCapacityError(StageOutputStoreError):
    code = "stage_output_capacity_exceeded"


class InvalidStageOutputReferenceError(StageOutputStoreError):
    code = "invalid_stage_output_reference"


class StageOutputNotFoundError(StageOutputStoreError):
    code = "stage_output_not_found"


class StageOutputOwnershipError(StageOutputStoreError):
    code = "stage_output_ownership_mismatch"


@dataclass(frozen=True, slots=True)
class StageOutput:
    task_id: str
    stage_id: str
    content_type: str
    text_content: str
    created_at: datetime

    def __post_init__(self) -> None:
        if any(
            type(value) is not str or not value.strip()
            for value in (self.task_id, self.stage_id)
        ):
            raise InvalidStageOutputError("Stage output identity is invalid")
        if self.content_type != "text/plain":
            raise InvalidStageOutputError("Stage output content type is unsupported")
        if type(self.text_content) is not str:
            raise InvalidStageOutputError("Stage output must be UTF-8 text")
        if (
            type(self.created_at) is not datetime
            or self.created_at.tzinfo is None
            or self.created_at.utcoffset() != timedelta(0)
        ):
            raise InvalidStageOutputError("Stage output timestamp must be UTC")
        if len(self.text_content) > MAX_STAGE_OUTPUT_BYTES:
            raise StageOutputTooLargeError("Stage output exceeds the size limit")
        try:
            size = len(self.text_content.encode("utf-8", errors="strict"))
        except UnicodeEncodeError:
            raise InvalidStageOutputError("Stage output must be UTF-8 text") from None
        if size > MAX_STAGE_OUTPUT_BYTES:
            raise StageOutputTooLargeError("Stage output exceeds the size limit")


@dataclass(frozen=True, slots=True)
class StageOutputReference:
    value: str

    def __post_init__(self) -> None:
        if type(self.value) is not str or _REFERENCE_PATTERN.fullmatch(self.value) is None:
            raise InvalidStageOutputReferenceError("Stage output reference is invalid")


class StageOutputStore(Protocol):
    def put(self, output: StageOutput) -> StageOutputReference: ...

    def get(
        self, reference: StageOutputReference, *, task_id: str, stage_id: str,
    ) -> StageOutput: ...


class InMemoryStageOutputStore:
    """Explicit process-local store; no eviction, durability, or cross-process use."""

    def __init__(self, *, max_total_bytes: int = MAX_TOTAL_STAGE_OUTPUT_BYTES) -> None:
        if type(max_total_bytes) is not int or not 1 <= max_total_bytes <= MAX_TOTAL_STAGE_OUTPUT_BYTES:
            raise InvalidStageOutputError("Stage output capacity is invalid")
        self._max_total_bytes = max_total_bytes
        self._outputs: dict[str, StageOutput] = {}
        self._total_bytes = 0
        self._lock = Lock()

    @property
    def total_bytes(self) -> int:
        with self._lock:
            return self._total_bytes

    def put(self, output: StageOutput) -> StageOutputReference:
        if type(output) is not StageOutput:
            raise InvalidStageOutputError("Stage output is invalid")
        # Reconstruct so the store owns its validated immutable value.
        stored = StageOutput(
            output.task_id, output.stage_id, output.content_type,
            output.text_content, output.created_at,
        )
        size = len(stored.text_content.encode("utf-8"))
        with self._lock:
            if (
                self._total_bytes + size > self._max_total_bytes
                or len(self._outputs) >= MAX_STAGE_OUTPUT_ENTRIES
            ):
                raise StageOutputCapacityError("Stage output store is full")
            reference = StageOutputReference(uuid4().hex)
            if reference.value in self._outputs:
                raise StageOutputCapacityError("Stage output reference is unavailable")
            self._outputs[reference.value] = stored
            self._total_bytes += size
            return reference

    def get(
        self, reference: StageOutputReference, *, task_id: str, stage_id: str,
    ) -> StageOutput:
        if type(reference) is not StageOutputReference:
            raise InvalidStageOutputReferenceError("Stage output reference is invalid")
        reference.__post_init__()
        if any(type(value) is not str or not value.strip() for value in (task_id, stage_id)):
            raise InvalidStageOutputError("Stage output identity is invalid")
        with self._lock:
            output = self._outputs.get(reference.value)
            if output is None:
                raise StageOutputNotFoundError("Stage output is unavailable")
            if output.task_id != task_id or output.stage_id != stage_id:
                raise StageOutputOwnershipError("Stage output is not available to this stage")
            # Do not hand out the stored object even to a caller that bypasses
            # frozen dataclass guards with object.__setattr__.
            return StageOutput(
                output.task_id, output.stage_id, output.content_type,
                output.text_content, output.created_at,
            )
