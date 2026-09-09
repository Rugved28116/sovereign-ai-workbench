"""Stage 1: mandatory sovereign eligibility filtering."""

from collections.abc import Iterable

from sovereign_api.config import DeploymentEnvironment
from sovereign_api.registry import ModelDefinition


class SovereignEligibilityFilter:
    def filter(
        self,
        models: Iterable[ModelDefinition],
        *,
        environment: DeploymentEnvironment,
        required_capabilities: frozenset[str],
    ) -> tuple[ModelDefinition, ...]:
        return tuple(
            model
            for model in models
            if model.enabled
            and environment in model.environments
            and not (
                model.provider == "mock"
                and environment is not DeploymentEnvironment.DEVELOPMENT
            )
            and required_capabilities.issubset(model.capabilities)
        )
