"""Deterministic, network-free provider used only for development and tests."""

from sovereign_api.contracts import ModelRequest, ModelResponse


class MockProvider:
    async def generate(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            model_id=request.model_id,
            content=f"Mock response from {request.model_id}",
        )
