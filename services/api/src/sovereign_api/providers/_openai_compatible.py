"""Internal HTTP and codec mechanics for OpenAI-compatible local runtimes."""

from __future__ import annotations

import ipaddress
import json
import unicodedata
from urllib.parse import urlsplit, urlunsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from sovereign_api.contracts import ModelRequest, ModelResponse
from sovereign_api.errors import (
    ProviderConfigurationError,
    ProviderConnectionError,
    ProviderResponseError,
)

MAX_RESPONSE_BYTES = 4 * 1024 * 1024

_ALLOWED_IPV4_NETWORKS = (
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
)


def validate_sovereign_base_url(value: str) -> str:
    if value != value.strip():
        raise ProviderConfigurationError(
            "Local provider base URL must not contain surrounding whitespace"
        )
    if any(unicodedata.category(character) == "Cc" for character in value):
        raise ProviderConfigurationError(
            "Local provider base URL must not contain Unicode control characters"
        )
    _reject_malformed_percent_encoding(value)

    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as error:
        raise ProviderConfigurationError("Malformed local provider base URL") from error

    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        raise ProviderConfigurationError(
            "Local provider base URL must be an absolute HTTP or HTTPS URL"
        )
    if parsed.username is not None or parsed.password is not None:
        raise ProviderConfigurationError(
            "Local provider base URL must not contain credentials"
        )
    if parsed.query or parsed.fragment:
        raise ProviderConfigurationError(
            "Local provider base URL must not contain a query or fragment"
        )
    if port == 0:
        raise ProviderConfigurationError("Local provider base URL port must be non-zero")

    hostname = parsed.hostname.lower()
    if hostname == "localhost":
        canonical_host = "127.0.0.1"
    else:
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError as error:
            raise ProviderConfigurationError(
                "Local provider host must be localhost or an approved IP literal"
            ) from error

        if isinstance(address, ipaddress.IPv4Address):
            if not any(address in network for network in _ALLOWED_IPV4_NETWORKS):
                raise ProviderConfigurationError(
                    "Local provider IPv4 address is outside approved local/private ranges"
                )
            canonical_host = str(address)
        elif address == ipaddress.ip_address("::1"):
            canonical_host = f"[{address.compressed}]"
        else:
            raise ProviderConfigurationError(
                "Only the IPv6 loopback address is approved for local providers"
            )

    netloc = canonical_host if port is None else f"{canonical_host}:{port}"
    path = parsed.path.rstrip("/")
    canonical_base_url = urlunsplit((parsed.scheme, netloc, path, "", ""))
    _validated_request_url(canonical_base_url)
    return canonical_base_url


def _reject_malformed_percent_encoding(value: str) -> None:
    hexadecimal = frozenset("0123456789abcdefABCDEF")
    for index, character in enumerate(value):
        if character == "%" and (
            index + 2 >= len(value)
            or value[index + 1] not in hexadecimal
            or value[index + 2] not in hexadecimal
        ):
            raise ProviderConfigurationError(
                "Local provider base URL contains malformed percent encoding"
            )


def _validated_request_url(base_url: str) -> httpx.URL:
    try:
        return httpx.URL(f"{base_url}/chat/completions")
    except (httpx.InvalidURL, ValueError) as error:
        raise ProviderConfigurationError(
            "Local provider base URL cannot form a valid request URL"
        ) from error


class _CompatibleMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    content: str = Field(min_length=1)


class _CompatibleChoice(BaseModel):
    model_config = ConfigDict(extra="ignore")

    message: _CompatibleMessage


class _CompatibleResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    choices: list[_CompatibleChoice] = Field(min_length=1)


class OpenAICompatibleTransport:
    """Internal transport; provider adapters retain runtime-facing identity."""

    def __init__(
        self,
        base_url: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        canonical_base_url = validate_sovereign_base_url(base_url)
        self._request_url = _validated_request_url(canonical_base_url)
        self._transport = transport

    async def complete(self, request: ModelRequest) -> ModelResponse:
        payload = {
            "model": request.model_id,
            "messages": [{"role": "user", "content": request.prompt}],
            "stream": False,
        }
        try:
            async with httpx.AsyncClient(
                follow_redirects=False,
                timeout=30.0,
                transport=self._transport,
                trust_env=False,
            ) as client:
                async with client.stream(
                    "POST", self._request_url, json=payload
                ) as response:
                    response_body = await _read_bounded_response(response)
                    response.raise_for_status()
        except httpx.RequestError as error:
            raise ProviderConnectionError(
                "Unable to reach the configured local model provider"
            ) from error
        except httpx.HTTPStatusError as error:
            raise ProviderResponseError(
                "Local model provider returned an unsuccessful status"
            ) from error

        try:
            compatible_response = _CompatibleResponse.model_validate(
                json.loads(response_body)
            )
        except (ValueError, ValidationError) as error:
            raise ProviderResponseError(
                "Local model provider returned an invalid response"
            ) from error

        return ModelResponse(
            model_id=request.model_id,
            content=compatible_response.choices[0].message.content,
        )


async def _read_bounded_response(response: httpx.Response) -> bytes:
    declared_length = response.headers.get("content-length")
    if declared_length is not None:
        try:
            declared_bytes = int(declared_length)
        except ValueError:
            declared_bytes = None
        if declared_bytes is not None and declared_bytes > MAX_RESPONSE_BYTES:
            raise ProviderResponseError(
                f"Local model provider response exceeds {MAX_RESPONSE_BYTES} bytes"
            )

    response_body = bytearray()
    async for chunk in response.aiter_bytes():
        if len(response_body) + len(chunk) > MAX_RESPONSE_BYTES:
            raise ProviderResponseError(
                f"Local model provider response exceeds {MAX_RESPONSE_BYTES} bytes"
            )
        response_body.extend(chunk)
    return bytes(response_body)
