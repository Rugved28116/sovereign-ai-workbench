"""Routing that keeps mandatory eligibility separate from optimization."""

from dataclasses import dataclass

from sovereign_api.config import DeploymentEnvironment
from sovereign_api.errors import InvalidOptimizerSelectionError, NoEligibleModelError
from sovereign_api.registry import ModelDefinition, ModelRegistry
from sovereign_api.routing.eligibility import SovereignEligibilityFilter
from sovereign_api.routing.optimizer import (
    DeterministicModelOptimizer,
    ModelOptimizer,
    RoutingCandidate,
)


@dataclass(frozen=True, slots=True)
class RouteDecision:
    model: ModelDefinition


def _is_approved_selection(
    selected: object, approved: dict[str, RoutingCandidate]
) -> bool:
    if type(selected) is not RoutingCandidate:
        return False

    try:
        model_id = selected.model_id
        priority = selected.priority
        capabilities = selected.capabilities
        context_length = selected.context_length
    except AttributeError:
        return False

    if (
        type(model_id) is not str
        or type(priority) is not int
        or type(context_length) is not int
        or type(capabilities) is not frozenset
        or any(type(capability) is not str for capability in capabilities)
    ):
        return False

    expected = approved.get(model_id)
    return expected is not None and (
        model_id == expected.model_id
        and priority == expected.priority
        and capabilities == expected.capabilities
        and context_length == expected.context_length
    )


class DeterministicModelRouter:
    def __init__(
        self,
        registry: ModelRegistry,
        environment: DeploymentEnvironment,
        eligibility_filter: SovereignEligibilityFilter | None = None,
        optimizer: ModelOptimizer | None = None,
    ) -> None:
        self._registry = registry
        self._environment = environment
        self._eligibility_filter = eligibility_filter or SovereignEligibilityFilter()
        self._optimizer = (
            optimizer if optimizer is not None else DeterministicModelOptimizer()
        )

    def route(self, required_capabilities: frozenset[str]) -> RouteDecision:
        # Stage 1 is authoritative: excluded models never reach ranking.
        eligible_models = self._eligibility_filter.filter(
            self._registry.models,
            environment=self._environment,
            required_capabilities=required_capabilities,
        )
        if not eligible_models:
            raise NoEligibleModelError(
                "No model is eligible for the active environment and required capabilities"
            )

        # Preserve private copies of the exact model states approved by Stage 1.
        # Stage 2 receives separate immutable, provider-neutral projections only.
        approved_models = {
            model.id: model.model_copy(deep=True) for model in eligible_models
        }
        approved_candidates = {
            model_id: RoutingCandidate.from_model(model)
            for model_id, model in approved_models.items()
        }
        optimizer_candidates = tuple(
            RoutingCandidate.from_model(model) for model in approved_models.values()
        )

        selected_candidate = self._optimizer.select(optimizer_candidates)

        # Treat optimizer output as untrusted. Both the model ID and complete
        # exposed state must correspond to the private Stage 1-approved snapshot.
        if not _is_approved_selection(selected_candidate, approved_candidates):
            raise InvalidOptimizerSelectionError(
                "Stage 2 optimizer returned an unknown or altered candidate"
            )

        return RouteDecision(model=approved_models[selected_candidate.model_id])
