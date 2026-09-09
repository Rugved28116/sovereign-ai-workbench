"""Provider adapter for a sovereign vLLM OpenAI-compatible endpoint."""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx

from sovereign_api.contracts import ModelRequest, ModelResponse
from sovereign_api.errors import ProviderConfigurationError
from sovereign_api.providers._openai_compatible import (
    OpenAICompatibleTransport,
    validate_sovereign_base_url,
)

PROVIDER_KEY = "vllm"
BASE_URL_ENVIRONMENT_VARIABLE = "SOVEREIGN_VLLM_BASE_URL"


@dataclass(frozen=True, slots=True)
class VLLMConfig:
    base_url: str

    @classmethod
    def from_environment(cls) -> "VLLMConfig":
        raw_base_url = os.environ.get(BASE_URL_ENVIRONMENT_VARIABLE)
        if raw_base_url is None:
            raise ProviderConfigurationError(
                f"{BASE_URL_ENVIRONMENT_VARIABLE} must be set when {PROVIDER_KEY} is enabled"
            )
        return cls(base_url=validate_sovereign_base_url(raw_base_url))


class VLLMProvider:
    """Basic text-generation adapter for a separately operated vLLM server."""

    provider_key = PROVIDER_KEY

    def __init__(
        self,
        base_url: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._compatible_transport = OpenAICompatibleTransport(
            base_url, transport=transport
        )

    @classmethod
    def from_environment(cls) -> "VLLMProvider":
        config = VLLMConfig.from_environment()
        return cls(config.base_url)

    async def generate(self, request: ModelRequest) -> ModelResponse:
        return await self._compatible_transport.complete(request)
