"""Safe loading and validation for the shared model registry."""

from pathlib import Path

import yaml
from pydantic import ValidationError

from sovereign_api.errors import RegistryValidationError
from sovereign_api.registry.models import ModelRegistry


def load_registry(path: Path) -> ModelRegistry:
    try:
        with path.open(encoding="utf-8") as registry_file:
            raw_registry = yaml.safe_load(registry_file)
    except OSError as error:
        raise RegistryValidationError(f"Unable to read model registry at {path}") from error
    except yaml.YAMLError as error:
        raise RegistryValidationError(f"Malformed YAML in model registry at {path}") from error

    if not isinstance(raw_registry, dict):
        raise RegistryValidationError("Model registry root must be a mapping")

    try:
        return ModelRegistry.model_validate(raw_registry)
    except ValidationError as error:
        raise RegistryValidationError(f"Invalid model registry: {error}") from error
