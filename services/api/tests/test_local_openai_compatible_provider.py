import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from sovereign_api.contracts import ModelRequest
from sovereign_api.errors import (
    ProviderConfigurationError,
    ProviderConnectionError,
    ProviderResponseError,
)
from sovereign_api.main import create_app
from sovereign_api.providers.local_openai_compatible import (
    BASE_URL_ENVIRONMENT_VARIABLE,
    MAX_RESPONSE_BYTES,
    LocalOpenAICompatibleProvider,
    validate_sovereign_base_url,
)

from conftest import model_data, registry_data, start_app, write_registry


class ChunkedResponseStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


class NeverReadResponseStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.read_started = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self.read_started = True
        raise AssertionError("response body should not have been read")
        yield b""  # pragma: no cover


@pytest.mark.parametrize(
    ("endpoint", "expected"),
    [
        ("http://localhost:8000/v1/", "http://127.0.0.1:8000/v1"),
        ("http://127.0.0.1:8000/v1", "http://127.0.0.1:8000/v1"),
        ("http://127.42.0.1:8000/v1", "http://127.42.0.1:8000/v1"),
        ("http://10.20.30.40:8000/v1", "http://10.20.30.40:8000/v1"),
        ("http://172.16.5.4:8000/v1", "http://172.16.5.4:8000/v1"),
        ("http://192.168.10.20:8000/v1", "http://192.168.10.20:8000/v1"),
        ("http://[::1]:8000/v1", "http://[::1]:8000/v1"),
    ],
)
def test_sovereign_endpoint_is_accepted(endpoint: str, expected: str) -> None:
    assert validate_sovereign_base_url(endpoint) == expected


def test_public_ip_is_rejected() -> None:
    with pytest.raises(ProviderConfigurationError, match="outside approved"):
        validate_sovereign_base_url("https://8.8.8.8/v1")


def test_public_hostname_is_rejected() -> None:
    with pytest.raises(ProviderConfigurationError, match="approved IP literal"):
        validate_sovereign_base_url("https://example.com/v1")


def test_localhost_request_target_contains_only_loopback_literal() -> None:
    captured_url: httpx.URL | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_url
        captured_url = request.url
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Local response"}}]},
        )

    provider = LocalOpenAICompatibleProvider(
        "http://localhost:8123/custom/v1",
        transport=httpx.MockTransport(handler),
    )
    asyncio.run(provider.generate(ModelRequest(model_id="local", prompt="Hello")))

    assert captured_url is not None
    assert captured_url.host == "127.0.0.1"
    assert str(captured_url) == "http://127.0.0.1:8123/custom/v1/chat/completions"


@pytest.mark.parametrize(
    "endpoint",
    [
        "not-a-url",
        "ftp://127.0.0.1/v1",
        "http://",
        "http://127.0.0.1:invalid/v1",
        "http://user:password@127.0.0.1/v1",
        "http://127.0.0.1/v1?target=external",
        "http://127.0.0.1/v1\x00",
        "http://127.0.0.1/v1\r\nforwarded-host:example.com",
        "http://127.0.0.1/v1\x1f",
        "http://127.0.0.1/v1\x7f",
        "http://127.0.0.1/v1/%ZZ",
        "http://[::1",
        "http://[not::ipv6]/v1",
    ],
)
def test_malformed_or_unsafe_endpoint_is_rejected(endpoint: str) -> None:
    with pytest.raises(ProviderConfigurationError):
        LocalOpenAICompatibleProvider(endpoint)


def test_provider_serializes_compatible_request() -> None:
    captured_request: httpx.Request | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_request
        captured_request = request
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Local response"}}]},
        )

    provider = LocalOpenAICompatibleProvider(
        "http://127.0.0.1:8000/v1",
        transport=httpx.MockTransport(handler),
    )
    asyncio.run(
        provider.generate(ModelRequest(model_id="local-model", prompt="Hello local"))
    )

    assert captured_request is not None
    assert captured_request.method == "POST"
    assert str(captured_request.url) == "http://127.0.0.1:8000/v1/chat/completions"
    assert json.loads(captured_request.content) == {
        "model": "local-model",
        "messages": [{"role": "user", "content": "Hello local"}],
        "stream": False,
    }


def test_provider_parses_compatible_response() -> None:
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            200,
            json={
                "id": "ignored-provider-id",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Parsed locally"},
                    }
                ],
            },
        )
    )
    provider = LocalOpenAICompatibleProvider(
        "http://192.168.1.10:8000/v1", transport=transport
    )

    response = asyncio.run(
        provider.generate(ModelRequest(model_id="registry-model", prompt="Hello"))
    )

    assert response.model_id == "registry-model"
    assert response.content == "Parsed locally"


def test_connection_failure_returns_typed_provider_error() -> None:
    def fail_connection(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    provider = LocalOpenAICompatibleProvider(
        "http://127.0.0.1:8000/v1",
        transport=httpx.MockTransport(fail_connection),
    )

    with pytest.raises(ProviderConnectionError, match="Unable to reach"):
        asyncio.run(
            provider.generate(ModelRequest(model_id="local-model", prompt="Hello"))
        )


def test_invalid_response_returns_typed_provider_error() -> None:
    provider = LocalOpenAICompatibleProvider(
        "http://127.0.0.1:8000/v1",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json={"choices": []})
        ),
    )

    with pytest.raises(ProviderResponseError, match="invalid response"):
        asyncio.run(
            provider.generate(ModelRequest(model_id="local-model", prompt="Hello"))
        )


def test_response_just_below_size_limit_succeeds() -> None:
    prefix = b'{"choices":[{"message":{"content":"'
    suffix = b'"}}]}'
    content_size = MAX_RESPONSE_BYTES - 1 - len(prefix) - len(suffix)
    response_body = prefix + (b"x" * content_size) + suffix
    assert len(response_body) == MAX_RESPONSE_BYTES - 1

    provider = LocalOpenAICompatibleProvider(
        "http://127.0.0.1:8000/v1",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, content=response_body)
        ),
    )

    response = asyncio.run(
        provider.generate(ModelRequest(model_id="local-model", prompt="Hello"))
    )

    assert len(response.content) == content_size


def test_declared_oversized_response_fails_before_body_read() -> None:
    response_stream = NeverReadResponseStream()
    provider = LocalOpenAICompatibleProvider(
        "http://127.0.0.1:8000/v1",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                headers={"content-length": str(MAX_RESPONSE_BYTES + 1)},
                stream=response_stream,
            )
        ),
    )

    with pytest.raises(ProviderResponseError, match="exceeds"):
        asyncio.run(
            provider.generate(ModelRequest(model_id="local-model", prompt="Hello"))
        )

    assert not response_stream.read_started


def test_chunked_response_exceeding_size_limit_fails() -> None:
    provider = LocalOpenAICompatibleProvider(
        "http://127.0.0.1:8000/v1",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                stream=ChunkedResponseStream(
                    [b"x" * MAX_RESPONSE_BYTES, b"x"]
                ),
            )
        ),
    )

    with pytest.raises(ProviderResponseError, match="exceeds"):
        asyncio.run(
            provider.generate(ModelRequest(model_id="local-model", prompt="Hello"))
        )


def test_malformed_oversized_error_response_fails_safely() -> None:
    provider = LocalOpenAICompatibleProvider(
        "http://127.0.0.1:8000/v1",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                500,
                stream=ChunkedResponseStream(
                    [b"not-json" * (MAX_RESPONSE_BYTES // 8 + 1)]
                ),
            )
        ),
    )

    with pytest.raises(ProviderResponseError, match="exceeds"):
        asyncio.run(
            provider.generate(ModelRequest(model_id="local-model", prompt="Hello"))
        )


def local_provider_registry(tmp_path: Path) -> Path:
    return write_registry(
        tmp_path,
        registry_data(
            model_data(
                "local-model",
                provider="local-openai-compatible",
                environments=["development"],
            )
        ),
    )


def test_selected_provider_fails_closed_when_endpoint_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SOVEREIGN_ENV", "development")
    monkeypatch.delenv(BASE_URL_ENVIRONMENT_VARIABLE, raising=False)

    with pytest.raises(ProviderConfigurationError, match="must be set"):
        start_app(create_app(registry_path=local_provider_registry(tmp_path)))


def test_selected_provider_fails_closed_when_endpoint_is_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SOVEREIGN_ENV", "development")
    monkeypatch.setenv(BASE_URL_ENVIRONMENT_VARIABLE, "https://example.com/v1")

    with pytest.raises(ProviderConfigurationError, match="approved IP literal"):
        start_app(create_app(registry_path=local_provider_registry(tmp_path)))
