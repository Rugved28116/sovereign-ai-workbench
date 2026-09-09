from pathlib import Path

import pytest

from sovereign_api.main import create_app

from conftest import model_data, registry_data, request_app, write_registry


def test_development_coding_request_selects_mock_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SOVEREIGN_ENV", "development")

    response = request_app(
        create_app(),
        "POST",
        "/v1/generate",
        json={
            "prompt": "Write a Python function",
            "required_capabilities": ["coding"],
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "model_id": "mock-code",
        "provider": "mock",
        "content": "Mock response from mock-code",
        "routing": {
            "environment": "development",
            "required_capabilities": ["coding"],
        },
    }


def test_on_prem_cannot_route_development_only_mock_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SOVEREIGN_ENV", "on-prem")

    response = request_app(
        create_app(),
        "POST",
        "/v1/generate",
        json={"prompt": "Test", "required_capabilities": ["coding"]},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "no_eligible_model"


def test_air_gapped_cannot_route_development_only_mock_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SOVEREIGN_ENV", "air-gapped")

    response = request_app(
        create_app(),
        "POST",
        "/v1/generate",
        json={"prompt": "Test", "required_capabilities": ["coding"]},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "no_eligible_model"


def test_unsupported_provider_returns_typed_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SOVEREIGN_ENV", "development")
    registry_path = write_registry(
        tmp_path,
        registry_data(model_data("unsupported", provider="not-configured")),
    )

    response = request_app(
        create_app(registry_path=registry_path),
        "POST",
        "/v1/generate",
        json={"prompt": "Test", "required_capabilities": ["chat"]},
    )

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "unsupported_provider"


def test_malformed_request_returns_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SOVEREIGN_ENV", "development")

    response = request_app(
        create_app(),
        "POST",
        "/v1/generate",
        json={"prompt": "", "required_capabilities": []},
    )

    assert response.status_code == 422
    assert response.json() == {
        "error": {
            "code": "malformed_request",
            "message": "Request body failed validation",
        }
    }
