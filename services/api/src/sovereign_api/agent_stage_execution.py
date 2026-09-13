"""Execute one planned agent stage without introducing an autonomous task loop."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import PureWindowsPath
from types import MappingProxyType
from typing import Protocol

from sovereign_api.agent_task_state import AgentTaskState, StageStatus, TaskStatus
from sovereign_api.contracts import ModelRequest, ModelResponse
from sovereign_api.errors import (
    InvalidExecutionStateError, OrchestrationError, ProviderError, RoutingError,
)
from sovereign_api.providers.base import ModelProvider
from sovereign_api.registry.models import valid_model_id
from sovereign_api.routing import DeterministicModelRouter
from sovereign_api.task_planning import TaskStage, TaskStageType


MAX_STAGE_OUTPUT_BYTES = 1_048_576
_SAFE_FAILURE = "Stage execution failed"
_UNEXPECTED_FAILURE = "Stage execution failed unexpectedly"


class InvalidStageCoordinationError(OrchestrationError):
    code = "invalid_stage_coordination"


class InvalidStageExecutionResultError(OrchestrationError):
    code = "invalid_stage_execution_result"


@dataclass(frozen=True, slots=True)
class StageExecutionResult:
    """Small, provider-neutral outcome; output content is never stored in task state."""

    stage_id: str
    status: StageStatus
    output_reference: str | None = None
    selected_model_id: str | None = None
    safe_message: str | None = None
    error_code: str | None = None

    def __post_init__(self) -> None:
        if type(self.stage_id) is not str or not self.stage_id.strip():
            raise InvalidStageExecutionResultError("Stage result ID is invalid")
        if type(self.status) is not StageStatus or self.status not in (
            StageStatus.COMPLETED, StageStatus.FAILED,
        ):
            raise InvalidStageExecutionResultError("Stage result status is invalid")
        for value in (self.output_reference, self.safe_message, self.error_code):
            if value is not None and (
                type(value) is not str or not value.strip() or len(value) > 256
                or any(ord(character) < 32 or ord(character) == 127 for character in value)
            ):
                raise InvalidStageExecutionResultError("Stage result field is invalid")
        if self.selected_model_id is not None and not valid_model_id(self.selected_model_id):
            raise InvalidStageExecutionResultError("Selected model ID is invalid")
        if self.output_reference is not None and (
            self.output_reference.startswith("/")
            or "\\" in self.output_reference
            or PureWindowsPath(self.output_reference).drive
            or ".." in self.output_reference.split("/")
        ):
            raise InvalidStageExecutionResultError("Stage output reference is invalid")
        if self.status is StageStatus.COMPLETED:
            if self.output_reference is None or self.error_code is not None:
                raise InvalidStageExecutionResultError("Completed stage result is invalid")
        elif self.output_reference is not None or self.error_code is None or self.safe_message is None:
            raise InvalidStageExecutionResultError("Failed stage result is invalid")

    @classmethod
    def failed(
        cls, stage_id: str, *, error_code: str,
        selected_model_id: str | None = None,
    ) -> StageExecutionResult:
        return cls(
            stage_id=stage_id, status=StageStatus.FAILED,
            selected_model_id=selected_model_id,
            safe_message=_SAFE_FAILURE, error_code=error_code,
        )


class StageExecutor(Protocol):
    async def execute(
        self, task: AgentTaskState, stage: TaskStage,
    ) -> StageExecutionResult: ...


class RoutedAgentStageExecutor:
    """Route a single model stage through the existing sovereign router/provider stack."""

    def __init__(
        self, *, router: DeterministicModelRouter,
        providers: Mapping[str, ModelProvider],
    ) -> None:
        self._router = router
        self._providers = MappingProxyType(dict(providers))

    async def execute(
        self, task: AgentTaskState, stage: TaskStage,
    ) -> StageExecutionResult:
        if stage.stage_type not in (
            TaskStageType.GENERATE, TaskStageType.CODE, TaskStageType.DOCUMENT,
            TaskStageType.VISION, TaskStageType.REASON,
        ):
            return StageExecutionResult.failed(stage.stage_id, error_code="unsupported_stage")
        try:
            decision = self._router.route(frozenset(stage.required_capabilities))
        except RoutingError:
            return StageExecutionResult.failed(stage.stage_id, error_code="routing_failed")

        model_id = decision.model.id
        provider = self._providers.get(decision.model.provider)
        if provider is None:
            return StageExecutionResult.failed(
                stage.stage_id, error_code="provider_unavailable", selected_model_id=model_id,
            )
        try:
            response = await provider.generate(
                ModelRequest(model_id=model_id, prompt=task.original_prompt)
            )
        except ProviderError:
            return StageExecutionResult.failed(
                stage.stage_id, error_code="provider_failed", selected_model_id=model_id,
            )
        except Exception:
            # Provider implementations are an external-runtime boundary.
            return StageExecutionResult.failed(
                stage.stage_id, error_code="provider_failed", selected_model_id=model_id,
            )

        if (
            type(response) is not ModelResponse
            or response.model_id != model_id
            or type(response.content) is not str
            or not response.content.strip()
            or len(response.content) > MAX_STAGE_OUTPUT_BYTES
        ):
            return StageExecutionResult.failed(
                stage.stage_id, error_code="provider_response_invalid",
                selected_model_id=model_id,
            )
        try:
            encoded = response.content.encode("utf-8", errors="strict")
        except UnicodeEncodeError:
            return StageExecutionResult.failed(
                stage.stage_id, error_code="provider_response_invalid",
                selected_model_id=model_id,
            )
        if len(encoded) > MAX_STAGE_OUTPUT_BYTES:
            return StageExecutionResult.failed(
                stage.stage_id, error_code="provider_response_invalid",
                selected_model_id=model_id,
            )
        return StageExecutionResult(
            stage_id=stage.stage_id,
            status=StageStatus.COMPLETED,
            # Integrity identifier only: there is no output store to dereference yet.
            output_reference="sha256:" + hashlib.sha256(encoded).hexdigest(),
            selected_model_id=model_id,
        )


class StageExecutionCoordinator:
    """Advance exactly one next-pending stage through immutable state transitions."""

    def __init__(
        self, executor: StageExecutor,
        *, clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._executor = executor
        self._clock = clock if clock is not None else lambda: datetime.now(UTC)

    def _timestamp_after(self, previous: datetime) -> datetime:
        try:
            value = self._clock()
            if (
                type(value) is not datetime or value.tzinfo is None
                or value.utcoffset() != timedelta(0)
            ):
                raise InvalidExecutionStateError("Coordinator clock must return UTC")
            return max(value, previous + timedelta(microseconds=1))
        except Exception:
            raise InvalidExecutionStateError("Coordinator clock is unavailable") from None

    async def execute_one(
        self, task: AgentTaskState, stage: TaskStage,
    ) -> AgentTaskState:
        if type(task) is not AgentTaskState or type(stage) is not TaskStage:
            raise InvalidStageCoordinationError("Task or stage is invalid")
        if task.task_status not in (TaskStatus.PENDING, TaskStatus.RUNNING):
            raise InvalidStageCoordinationError("Terminal task cannot execute a stage")
        if task.current_stage_id is not None:
            raise InvalidStageCoordinationError("A stage is already running")
        next_index = next(
            (index for index, item in enumerate(task.stage_states)
             if item.status is StageStatus.PENDING), None,
        )
        if next_index is None or any(
            item.status is not StageStatus.COMPLETED
            for item in task.stage_states[:next_index]
        ):
            raise InvalidStageCoordinationError("Stage is not the next planned stage")
        expected = task.plan.stages[next_index]
        if (
            stage.stage_id != expected.stage_id
            or stage.stage_type is not expected.stage_type
            or stage.required_capabilities != expected.required_capabilities
        ):
            raise InvalidStageCoordinationError("Stage is not the next planned stage")

        # Acquire every possible transition timestamp before the executor can
        # invoke a provider. Clock failure therefore cannot discard a completed
        # provider call and invite a sequential retry of the old snapshot.
        transition_count = (
            (1 if task.task_status is TaskStatus.PENDING else 0)
            + 2  # stage start and terminal outcome
            + (1 if next_index == len(task.stage_states) - 1 else 0)
        )
        timestamps: list[datetime] = []
        previous = task.updated_at
        for _ in range(transition_count):
            previous = self._timestamp_after(previous)
            timestamps.append(previous)
        transition_times = iter(timestamps)

        current = task
        if current.task_status is TaskStatus.PENDING:
            current = current.start(updated_at=next(transition_times))
        current = current.update_stage(
            current.stage_states[next_index].start(),
            updated_at=next(transition_times),
        )

        try:
            result = await self._executor.execute(current, stage)
        except Exception:
            result = StageExecutionResult.failed(
                stage.stage_id, error_code="stage_unexpected_failure",
            )
        if type(result) is not StageExecutionResult or result.stage_id != stage.stage_id:
            result = StageExecutionResult.failed(
                stage.stage_id, error_code="stage_invalid_result",
            )

        active = current.stage_states[next_index]
        if result.status is StageStatus.COMPLETED:
            current = current.update_stage(
                active.complete(
                    output_reference=result.output_reference,
                    selected_model_id=result.selected_model_id,
                ),
                updated_at=next(transition_times),
            )
            if all(item.status is StageStatus.COMPLETED for item in current.stage_states):
                current = current.complete(updated_at=next(transition_times))
        else:
            # Error text from pluggable executors is never copied into task state.
            code = (
                result.error_code
                if result.error_code in {
                    "routing_failed", "provider_failed", "provider_unavailable",
                    "provider_response_invalid", "unsupported_stage",
                    "stage_unexpected_failure", "stage_invalid_result",
                }
                else "stage_execution_failed"
            )
            current = current.update_stage(
                active.fail(
                    error_code=code,
                    safe_message=(
                        _UNEXPECTED_FAILURE if code == "stage_unexpected_failure"
                        else _SAFE_FAILURE
                    ),
                    selected_model_id=result.selected_model_id,
                ),
                updated_at=next(transition_times),
            )
        return current
