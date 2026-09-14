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
            "capability_source": "explicit",
        },
    }


def test_missing_capabilities_are_inferred_before_routing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SOVEREIGN_ENV", "development")

    response = request_app(
        create_app(),
        "POST",
        "/v1/generate",
        json={"prompt": "Fix this Python bug"},
    )

    assert response.status_code == 200
    assert response.json()["model_id"] == "mock-code"
    assert response.json()["routing"] == {
        "environment": "development",
        "required_capabilities": ["coding"],
        "capability_source": "inferred",
        "task_class": "coding",
        "plan": {
            "task_class": "coding",
            "stages": [
                {
                    "stage_id": "stage-1",
                    "stage_type": "code",
                    "required_capabilities": ["coding"],
                }
            ],
        },
    }


def test_public_plan_openapi_schema_remains_typed_and_model_only(monkeypatch):
    monkeypatch.setenv("SOVEREIGN_ENV", "development")
    app = create_app()
    schemas = app.openapi()["components"]["schemas"]
    plan_property = schemas["RoutingInformation"]["properties"]["plan"]
    plan_ref = next(item["$ref"] for item in plan_property["anyOf"] if "$ref" in item)
    plan_schema = schemas[plan_ref.rsplit("/", 1)[-1]]
    stage_ref = plan_schema["properties"]["stages"]["items"]["$ref"]
    stage_schema = schemas[stage_ref.rsplit("/", 1)[-1]]
    assert set(plan_schema["properties"]) == {"task_class", "stages"}
    assert set(stage_schema["properties"]) == {
        "stage_id", "stage_type", "required_capabilities",
    }
    assert set(stage_schema["required"]) == {
        "stage_id", "stage_type", "required_capabilities",
    }
    assert stage_schema["properties"]["required_capabilities"]["items"]["type"] == "string"
    assert set(stage_schema["properties"]["stage_type"]["enum"]) == {
        "generate", "code", "document", "vision", "reason",
    }
    response = request_app(app, "POST", "/v1/generate", json={"prompt": "Fix this Python bug"})
    assert response.status_code == 200
    assert set(response.json()["routing"]["plan"]["stages"][0]) == {
        "stage_id", "stage_type", "required_capabilities",
    }


def test_explicit_capabilities_bypass_classifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnexpectedClassifier:
        def classify(self, prompt: str) -> None:
            pytest.fail("classifier must not run for explicit capabilities")

    class UnexpectedPlanner:
        def plan(self, requirements: object) -> None:
            pytest.fail("planner must not run for explicit capabilities")

    monkeypatch.setenv("SOVEREIGN_ENV", "development")
    response = request_app(
        create_app(
            task_classifier=UnexpectedClassifier(),
            task_planner=UnexpectedPlanner(),
        ),
        "POST",
        "/v1/generate",
        json={"prompt": "Inspect this image", "required_capabilities": ["coding"]},
    )

    assert response.status_code == 200
    assert response.json()["model_id"] == "mock-code"
    assert response.json()["routing"]["capability_source"] == "explicit"
    assert "task_class" not in response.json()["routing"]
    assert "plan" not in response.json()["routing"]


def test_inferred_unavailable_capability_still_fails_stage_1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SOVEREIGN_ENV", "development")

    response = request_app(
        create_app(),
        "POST",
        "/v1/generate",
        json={"prompt": "Inspect this image"},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "no_eligible_model"


def test_inferred_multi_capability_requires_all_capabilities_in_stage_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SOVEREIGN_ENV", "development")
    registry_path = write_registry(
        tmp_path,
        registry_data(
            model_data(
                "vision-only",
                capabilities=["vision"],
                priority=1,
            ),
            model_data(
                "vision-reasoning",
                capabilities=["vision", "reasoning"],
                priority=20,
            ),
        ),
    )

    response = request_app(
        create_app(registry_path=registry_path),
        "POST",
        "/v1/generate",
        json={"prompt": "compare these two diagrams"},
    )

    assert response.status_code == 200
    assert response.json()["model_id"] == "vision-reasoning"
    assert response.json()["routing"]["required_capabilities"] == [
        "vision",
        "reasoning",
    ]
    assert response.json()["routing"]["plan"] == {
        "task_class": "vision",
        "stages": [
            {
                "stage_id": "stage-1",
                "stage_type": "vision",
                "required_capabilities": ["vision"],
            },
            {
                "stage_id": "stage-2",
                "stage_type": "reason",
                "required_capabilities": ["reasoning"],
            },
        ],
    }


def test_no_eligible_multi_capability_model_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SOVEREIGN_ENV", "development")
    registry_path = write_registry(
        tmp_path,
        registry_data(model_data("vision-only", capabilities=["vision"])),
    )

    response = request_app(
        create_app(registry_path=registry_path),
        "POST",
        "/v1/generate",
        json={"prompt": "compare these two diagrams"},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "no_eligible_model"


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


@pytest.mark.parametrize("environment", ["on-prem", "air-gapped"])
def test_inferred_capability_cannot_bypass_mock_environment_restriction(
    environment: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SOVEREIGN_ENV", environment)

    response = request_app(
        create_app(),
        "POST",
        "/v1/generate",
        json={"prompt": "Fix this Python bug"},
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


def test_empty_prompt_without_capabilities_remains_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SOVEREIGN_ENV", "development")

    response = request_app(
        create_app(),
        "POST",
        "/v1/generate",
        json={"prompt": ""},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "malformed_request"
