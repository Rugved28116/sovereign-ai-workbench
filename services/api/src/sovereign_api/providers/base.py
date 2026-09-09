"""Provider-neutral model interaction boundary."""

from typing import Protocol

from sovereign_api.contracts import ModelRequest, ModelResponse


class ModelProvider(Protocol):
    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate a response without exposing provider-specific types."""
        ...
