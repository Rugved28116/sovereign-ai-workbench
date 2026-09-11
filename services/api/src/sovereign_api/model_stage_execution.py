"""Routing-backed stage execution through the existing provider contract."""

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from dataclasses import dataclass
from types import MappingProxyType

from pydantic import TypeAdapter

from sovereign_api.agent_execution import (
    AgentStep, AgentTask, StepOutputType, StepResult, StepStatus, TaskStatus,
    StageExecutionRecord,
)
from sovereign_api.execution_provenance import (
    ExecutionProvenance, RoutedStageExecutionError, result_digest,
)
from sovereign_api.contracts import ModelRequest, ModelResponse
from sovereign_api.errors import ProviderError, RoutingError, StageExecutionError
from sovereign_api.prompt_validation import Prompt
from sovereign_api.providers.base import ModelProvider
from sovereign_api.routing import DeterministicModelRouter

MAX_STAGE_CONTEXT_LENGTH = 65_536
MODEL_STAGE_FAILURE = "Model stage execution failed"
_PROMPT_VALIDATOR = TypeAdapter(Prompt)


@dataclass(frozen=True, slots=True)
class TaskExecutionInput:
    """Validated original prompt, bound to an executor for one task run."""

    prompt: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt", _PROMPT_VALIDATOR.validate_python(self.prompt))


def build_stage_context(
    execution_input: TaskExecutionInput, task: AgentTask, step: AgentStep
) -> str:
    """Compose bounded context from the prompt and earlier successful results."""
    if task.status is not TaskStatus.RUNNING or step.status is not StepStatus.RUNNING:
        raise StageExecutionError("Model stage requires a running task and step")
    index = next(
        (index for index, candidate in enumerate(task.steps) if candidate == step),
        None,
    )
    if index is None:
        raise StageExecutionError("Model stage does not match the task snapshot")

    parts: list[str] = []
    length = 0

    def append(value: str) -> None:
        nonlocal length
        length += len(value)
        if length > MAX_STAGE_CONTEXT_LENGTH:
            raise StageExecutionError("Model stage context exceeds the size limit")
        parts.append(value)

    append("Original task:\n")
    append(execution_input.prompt)
    append("\n\nPrevious stage results:\n")
    for previous in task.steps[:index]:
        if previous.status is StepStatus.SUCCEEDED and previous.result is not None:
            append("[")
            append(previous.stage_id)
            append(" / ")
            append(previous.stage_type.value)
            append("]\n")
            append(previous.result.content)
            append("\n\n")
    append("Current stage:\n")
    append(step.stage_type.value)
    return "".join(parts)


class ModelStageExecutor:
    """Route each stage independently; composition supplies router and providers."""

    def __init__(
        self,
        *,
        execution_input: TaskExecutionInput,
        router: DeterministicModelRouter,
        providers: Mapping[str, ModelProvider],
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._input = execution_input
        self._router = router
        self._providers = MappingProxyType(dict(providers))
        self._clock = clock if clock is not None else lambda: datetime.now(UTC)

    async def execute(self, task: AgentTask, step: AgentStep) -> StageExecutionRecord:
        context = build_stage_context(self._input, task, step)
        try:
            decision = self._router.route(frozenset(step.required_capabilities))
        except RoutingError as error:
            raise StageExecutionError(MODEL_STAGE_FAILURE) from error

        # Capture approved routing facts before invoking any provider code.
        stage_id = step.stage_id
        capabilities = tuple(step.required_capabilities)
        model_id = decision.model.id
        provider_key = decision.model.provider
        environment = decision.environment
        started_at = self._clock()

        def provenance(result: StepResult | None = None) -> ExecutionProvenance:
            return ExecutionProvenance(
                stage_id=stage_id, model_id=model_id, provider=provider_key,
                required_capabilities=capabilities, routing_environment=environment,
                started_at=started_at, completed_at=self._clock(),
                success=result is not None,
                result_digest=None if result is None else result_digest(
                    result.output_type.value, result.content
                ),
            )

        # Same approved-provider lookup used by the generation composition root.
        provider = self._providers.get(provider_key)
        if provider is None:
            raise RoutedStageExecutionError(MODEL_STAGE_FAILURE, provenance())

        try:
            response = await provider.generate(
                ModelRequest(model_id=model_id, prompt=context)
            )
        except ProviderError as error:
            raise RoutedStageExecutionError(MODEL_STAGE_FAILURE, provenance()) from error
        except Exception as error:
            # Provider messages and infrastructure details are never task errors.
            raise RoutedStageExecutionError(MODEL_STAGE_FAILURE, provenance()) from error

        if (
            type(response) is not ModelResponse
            or response.model_id != model_id
            or type(response.content) is not str
            or not response.content.strip()
        ):
            raise RoutedStageExecutionError(MODEL_STAGE_FAILURE, provenance())
        result = StepResult(StepOutputType.TEXT, response.content)
        return StageExecutionRecord(result, provenance(result))
