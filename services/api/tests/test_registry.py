from pathlib import Path

import pytest

from sovereign_api.errors import RegistryValidationError
from sovereign_api.registry import load_registry

from conftest import model_data, registry_data, start_app, write_registry


def test_valid_registry_loads(tmp_path: Path) -> None:
    path = write_registry(tmp_path, registry_data(model_data("valid-model")))

    registry = load_registry(path)

    assert registry.schema_version == 1
    assert registry.models[0].id == "valid-model"
    assert registry.models[0].metadata.description == "Test model"


def test_malformed_yaml_fails(tmp_path: Path) -> None:
    path = tmp_path / "registry.yaml"
    path.write_text("models: [unterminated", encoding="utf-8")

    with pytest.raises(RegistryValidationError, match="Malformed YAML"):
        load_registry(path)


def test_missing_required_model_field_fails(tmp_path: Path) -> None:
    incomplete_model = model_data("incomplete")
    del incomplete_model["provider"]
    path = write_registry(tmp_path, registry_data(incomplete_model))

    with pytest.raises(RegistryValidationError, match="Invalid model registry"):
        load_registry(path)


def test_unknown_registry_environment_fails(tmp_path: Path) -> None:
    path = write_registry(
        tmp_path,
        registry_data(model_data("bad-environment", environments=["production"])),
    )

    with pytest.raises(RegistryValidationError, match="Invalid model registry"):
        load_registry(path)


def test_application_startup_fails_for_invalid_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sovereign_api.main import create_app

    monkeypatch.setenv("SOVEREIGN_ENV", "development")
    path = tmp_path / "registry.yaml"
    path.write_text("schema_version: wrong", encoding="utf-8")

    with pytest.raises(RegistryValidationError):
        start_app(create_app(registry_path=path))


@pytest.mark.parametrize(
    "provider_specific_field",
    [
        "endpoint",
        "api_url",
        "credential",
        "token",
        "runtime_flags",
        "provider_options",
        "command_line_arguments",
    ],
)
def test_provider_specific_metadata_field_is_rejected(
    tmp_path: Path, provider_specific_field: str
) -> None:
    model = model_data("invalid-metadata")
    model["metadata"][provider_specific_field] = "not-allowed"
    path = write_registry(tmp_path, registry_data(model))

    with pytest.raises(RegistryValidationError, match="Invalid model registry"):
        load_registry(path)
