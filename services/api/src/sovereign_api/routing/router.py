"""Routing that keeps mandatory eligibility separate from optimization."""

from dataclasses import dataclass

from sovereign_api.config import DeploymentEnvironment
from sovereign_api.errors import NoEligibleModelError
from sovereign_api.registry import ModelDefinition, ModelRegistry
from sovereign_api.routing.eligibility import SovereignEligibilityFilter


@dataclass(frozen=True, slots=True)
class RouteDecision:
    model: ModelDefinition


class DeterministicModelRouter:
    def __init__(
        self,
        registry: ModelRegistry,
        environment: DeploymentEnvironment,
        eligibility_filter: SovereignEligibilityFilter | None = None,
    ) -> None:
        self._registry = registry
        self._environment = environment
        self._eligibility_filter = eligibility_filter or SovereignEligibilityFilter()

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

        # Stage 2 currently has no adaptive optimizer. This deterministic ordering
        # is the baseline and cannot add to the Stage 1 candidate set.
        selected_model = min(
            eligible_models, key=lambda model: (model.priority, model.id)
        )
        return RouteDecision(model=selected_model)
