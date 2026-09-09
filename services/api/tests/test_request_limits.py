import asyncio
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from sovereign_api.api_models import (
    MAX_CAPABILITY_NAME_LENGTH,
    MAX_PROMPT_LENGTH,
    MAX_REQUIRED_CAPABILITIES,
    GenerateRequest,
)
from sovereign_api.body_limit import MAX_GENERATE_BODY_BYTES
from sovereign_api.main import create_app

from conftest import request_app


def valid_request(**overrides: Any) -> dict[str, Any]:
    request = {"prompt": "Test", "required_capabilities": ["chat"]}
    request.update(overrides)
    return request


def test_request_field_limits_accept_exact_boundaries() -> None:
    request = GenerateRequest.model_validate(
        valid_request(
            prompt="p" * MAX_PROMPT_LENGTH,
            required_capabilities=[
                f"capability-{index:02d}".ljust(MAX_CAPABILITY_NAME_LENGTH, "x")
                for index in range(MAX_REQUIRED_CAPABILITIES)
            ],
        )
    )

    assert len(request.prompt) == MAX_PROMPT_LENGTH
    assert len(request.required_capabilities) == MAX_REQUIRED_CAPABILITIES
    assert all(
        len(capability) == MAX_CAPABILITY_NAME_LENGTH
        for capability in request.required_capabilities
    )


@pytest.mark.parametrize(
    "payload",
    [
        valid_request(prompt=""),
        valid_request(prompt="p" * (MAX_PROMPT_LENGTH + 1)),
        valid_request(required_capabilities=[""]),
        valid_request(
            required_capabilities=["c" * (MAX_CAPABILITY_NAME_LENGTH + 1)]
        ),
        valid_request(
            required_capabilities=[
                f"capability-{index}" for index in range(MAX_REQUIRED_CAPABILITIES + 1)
            ]
        ),
        valid_request(required_capabilities=["chat", "chat"]),
    ],
)
def test_invalid_request_field_boundary_is_rejected(payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        GenerateRequest.model_validate(payload)


def test_generate_endpoint_rejects_oversized_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SOVEREIGN_ENV", "development")

    response = request_app(
        create_app(),
        "POST",
        "/v1/generate",
        json=valid_request(prompt="p" * (MAX_PROMPT_LENGTH + 1)),
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "malformed_request"


def test_generate_endpoint_rejects_oversized_request_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SOVEREIGN_ENV", "development")

    response = request_app(
        create_app(),
        "POST",
        "/v1/generate",
        json=valid_request(prompt="p" * (MAX_GENERATE_BODY_BYTES + 1)),
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_body_too_large"


def test_generate_endpoint_rejects_oversized_streamed_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SOVEREIGN_ENV", "development")
    app = create_app()

    async def oversized_chunks() -> AsyncIterator[bytes]:
        yield b"x" * (MAX_GENERATE_BODY_BYTES // 2)
        yield b"x" * (MAX_GENERATE_BODY_BYTES // 2 + 1)

    async def send_request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                return await client.post(
                    "/v1/generate",
                    content=oversized_chunks(),
                    headers={"content-type": "application/json"},
                )

    response = asyncio.run(send_request())

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_body_too_large"
