import asyncio
import json
from pathlib import Path

import httpx
import pytest

from sovereign_api.config import DeploymentEnvironment
from sovereign_api.contracts import ModelRequest
from sovereign_api.errors import ProviderConfigurationError, ProviderResponseError
from sovereign_api.main import configure_providers, create_app
from sovereign_api.providers import ModelProvider, VLLMProvider
from sovereign_api.providers.vllm import BASE_URL_ENVIRONMENT_VARIABLE
from sovereign_api.registry.models import ModelRegistry
from sovereign_api.routing import DeterministicModelRouter

from conftest import model_data, registry_data, request_app, start_app, write_registry


def vllm_registry(
    tmp_path: Path, *, environments: list[str] | None = None
) -> Path:
    return write_registry(
        tmp_path,
        registry_data(
            model_data(
                "test-vllm-model",
                provider="vllm",
                environments=environments,
            )
        ),
    )


def test_vllm_provider_satisfies_contract_and_preserves_identity() -> None:
    provider = VLLMProvider(
        "http://127.0.0.1:8000/v1",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                json={"choices": [{"message": {"content": "vLLM response"}}]},
            )
        ),
    )

    assert isinstance(provider, ModelProvider)
    assert provider.provider_key == "vllm"


def test_vllm_provider_accepts_local_endpoint_and_serializes_request() -> None:
    captured_request: httpx.Request | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_request
        captured_request = request
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Generated locally"}}]},
        )

    provider = VLLMProvider(
        "http://localhost:8000/v1", transport=httpx.MockTransport(handler)
    )
    response = asyncio.run(
        provider.generate(ModelRequest(model_id="served-model", prompt="Hello vLLM"))
    )

    assert captured_request is not None
    assert captured_request.url.host == "127.0.0.1"
    assert json.loads(captured_request.content) == {
        "model": "served-model",
        "messages": [{"role": "user", "content": "Hello vLLM"}],
        "stream": False,
    }
    assert response.model_id == "served-model"
    assert response.content == "Generated locally"


def test_vllm_provider_rejects_public_endpoint() -> None:
    with pytest.raises(ProviderConfigurationError, match="outside approved"):
        VLLMProvider("https://8.8.8.8/v1")


def test_vllm_provider_rejects_oversized_response() -> None:
    provider = VLLMProvider(
        "http://127.0.0.1:8000/v1",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                headers={"content-length": str(4 * 1024 * 1024 + 1)},
                content=b"not read",
            )
        ),
    )

    with pytest.raises(ProviderResponseError, match="exceeds"):
        asyncio.run(
            provider.generate(ModelRequest(model_id="served-model", prompt="Hello"))
        )


def test_vllm_provider_returns_typed_error_for_malformed_response() -> None:
    provider = VLLMProvider(
        "http://127.0.0.1:8000/v1",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json={"choices": []})
        ),
    )

    with pytest.raises(ProviderResponseError, match="invalid response"):
        asyncio.run(
            provider.generate(ModelRequest(model_id="served-model", prompt="Hello"))
        )


def test_selected_vllm_provider_fails_closed_when_config_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SOVEREIGN_ENV", "development")
    monkeypatch.delenv(BASE_URL_ENVIRONMENT_VARIABLE, raising=False)

    with pytest.raises(ProviderConfigurationError, match="must be set"):
        start_app(create_app(registry_path=vllm_registry(tmp_path)))


def test_unused_vllm_config_does_not_block_unrelated_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(BASE_URL_ENVIRONMENT_VARIABLE, "https://8.8.8.8/v1")
    registry = ModelRegistry.model_validate(
        registry_data(
            model_data("development-mock"),
            model_data(
                "future-vllm",
                provider="vllm",
                environments=["on-prem"],
            ),
        )
    )

    providers = configure_providers(registry, DeploymentEnvironment.DEVELOPMENT)

    assert set(providers) == {"mock"}


def test_generate_response_preserves_selected_vllm_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = VLLMProvider(
        "http://127.0.0.1:8000/v1",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                json={"choices": [{"message": {"content": "From vLLM"}}]},
            )
        ),
    )
    monkeypatch.setenv("SOVEREIGN_ENV", "development")
    monkeypatch.setattr(
        VLLMProvider,
        "from_environment",
        classmethod(lambda _cls: provider),
    )

    response = request_app(
        create_app(registry_path=vllm_registry(tmp_path)),
        "POST",
        "/v1/generate",
        json={"prompt": "Hello", "required_capabilities": ["chat"]},
    )

    assert response.status_code == 200
    assert response.json()["provider"] == "vllm"
    assert response.json()["content"] == "From vLLM"


def test_router_preserves_vllm_provider_identity_without_runtime_logic() -> None:
    registry = ModelRegistry.model_validate(
        registry_data(model_data("served-model", provider="vllm"))
    )
    decision = DeterministicModelRouter(
        registry, DeploymentEnvironment.DEVELOPMENT
    ).route(frozenset({"chat"}))

    assert decision.model.id == "served-model"
    assert decision.model.provider == "vllm"
