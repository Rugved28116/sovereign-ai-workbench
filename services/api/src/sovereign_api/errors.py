"""Typed failures at configuration, routing, and provider boundaries."""


class SovereignAPIError(Exception):
    """Base class for expected service failures."""

    code = "sovereign_api_error"


class InvalidEnvironmentError(SovereignAPIError):
    """Raised when the active deployment environment is not recognized."""

    code = "invalid_environment"


class RegistryValidationError(SovereignAPIError):
    """Raised when the model registry cannot be loaded or validated."""

    code = "invalid_registry"


class RoutingError(SovereignAPIError):
    """Base class for failures at the two-stage routing boundary."""

    code = "routing_error"


class NoEligibleModelError(RoutingError):
    """Raised when Stage 1 rejects every registered model."""

    code = "no_eligible_model"


class InvalidOptimizerSelectionError(RoutingError):
    """Raised when Stage 2 returns a model outside the Stage 1 candidate set."""

    code = "invalid_optimizer_selection"


class PlanningError(SovereignAPIError):
    """Base class for deterministic task planning failures."""

    code = "planning_error"


class UnsupportedTaskRequirementsError(PlanningError):
    """Raised when task requirements have no explicit planning rule."""

    code = "unsupported_task_requirements"


class UnsupportedProviderError(SovereignAPIError):
    """Raised when a routed model has no configured provider adapter."""

    code = "unsupported_provider"


class ProviderError(SovereignAPIError):
    """Base class for provider adapter failures."""

    code = "provider_error"


class ProviderConfigurationError(ProviderError):
    """Raised when provider-private configuration is missing or invalid."""

    code = "provider_configuration_error"


class ProviderConnectionError(ProviderError):
    """Raised when a configured local provider endpoint cannot be reached."""

    code = "provider_connection_error"


class ProviderResponseError(ProviderError):
    """Raised when a local provider returns an unsuccessful or invalid response."""

    code = "provider_response_error"
