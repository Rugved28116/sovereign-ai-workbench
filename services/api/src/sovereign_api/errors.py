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


class NoEligibleModelError(SovereignAPIError):
    """Raised when Stage 1 rejects every registered model."""

    code = "no_eligible_model"


class UnsupportedProviderError(SovereignAPIError):
    """Raised when a routed model has no configured provider adapter."""

    code = "unsupported_provider"
