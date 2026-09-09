import pytest

from sovereign_api.main import create_app

from conftest import request_app


def test_health_returns_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOVEREIGN_ENV", "development")
    response = request_app(create_app(), "GET", "/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "sovereign-api"}
