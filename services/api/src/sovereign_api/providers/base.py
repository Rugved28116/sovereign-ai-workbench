"""Provider-neutral model interaction boundary."""

from typing import Protocol, runtime_checkable

from sovereign_api.contracts import ModelRequest, ModelResponse


@runtime_checkable
class ModelProvider(Protocol):
    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate a response without exposing provider-specific types."""
        ...
