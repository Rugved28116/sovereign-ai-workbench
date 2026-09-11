"""Minimal provider-neutral sequential task orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from sovereign_api.agent_execution import (
    AgentStep,
    AgentTask,
    StepOutputType,
    StepResult,
)
from sovereign_api.errors import StageExecutionError
from sovereign_api.task_planning import TaskPlan

Clock = Callable[[], datetime]

UNEXPECTED_STAGE_FAILURE = "Stage execution failed unexpectedly"


class StageExecutor(Protocol):
    """Execute one logical stage without exposing infrastructure details."""

    async def execute(self, task: AgentTask, step: AgentStep) -> StepResult:
        """Return a typed result or raise StageExecutionError."""
        ...


@dataclass(frozen=True, slots=True)
class MockStageExecutor:
    """Deterministic, network-free executor for development and tests only."""

    failing_stage_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "failing_stage_ids", frozenset(self.failing_stage_ids)
        )

    async def execute(self, task: AgentTask, step: AgentStep) -> StepResult:
        if step.stage_id in self.failing_stage_ids:
            raise StageExecutionError(
                f"Mock execution failed for stage {step.stage_id}"
            )
        return StepResult(
            output_type=StepOutputType.TEXT,
            content=f"Mock result for {task.task_id}/{step.stage_id}",
        )


class SequentialTaskOrchestrator:
    """Execute planned stages in order through immutable task transitions."""

    def __init__(
        self,
        executor: StageExecutor,
        *,
        clock: Clock | None = None,
    ) -> None:
        self._executor = executor
        self._clock = clock if clock is not None else _utc_now

    async def execute(self, plan: TaskPlan, *, task_id: str) -> AgentTask:
        task = AgentTask.from_plan(
            task_id=task_id,
            plan=plan,
            timestamp=self._clock(),
        )
        task = task.start(updated_at=self._clock())

        for step_index in range(len(task.steps)):
            current_step = task.steps[step_index]
            task = task.start_step(
                current_step.stage_id,
                updated_at=self._clock(),
            )
            running_step = task.steps[step_index]

            try:
                result = await self._executor.execute(task, running_step)
            except StageExecutionError as error:
                return task.fail_step(
                    running_step.stage_id,
                    str(error),
                    updated_at=self._clock(),
                )
            except Exception:
                return task.fail_step(
                    running_step.stage_id,
                    UNEXPECTED_STAGE_FAILURE,
                    updated_at=self._clock(),
                )

            task = task.succeed_step(
                running_step.stage_id,
                result,
                updated_at=self._clock(),
            )

        return task.succeed(updated_at=self._clock())


def _utc_now() -> datetime:
    return datetime.now(UTC)
