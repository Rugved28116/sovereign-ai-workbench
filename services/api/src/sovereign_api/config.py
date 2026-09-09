"""Application configuration with fail-closed environment parsing."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from sovereign_api.errors import InvalidEnvironmentError


class DeploymentEnvironment(StrEnum):
    DEVELOPMENT = "development"
    ON_PREM = "on-prem"
    AIR_GAPPED = "air-gapped"


@dataclass(frozen=True, slots=True)
class Settings:
    environment: DeploymentEnvironment
    registry_path: Path


def default_registry_path() -> Path:
    return Path(__file__).resolve().parents[4] / "models" / "registry.yaml"


def parse_environment(value: str) -> DeploymentEnvironment:
    try:
        return DeploymentEnvironment(value)
    except ValueError as error:
        allowed = ", ".join(environment.value for environment in DeploymentEnvironment)
        raise InvalidEnvironmentError(
            f"Invalid SOVEREIGN_ENV {value!r}; expected one of: {allowed}"
        ) from error


def load_settings(*, registry_path: Path | None = None) -> Settings:
    raw_environment = os.environ.get("SOVEREIGN_ENV")
    if raw_environment is None:
        raise InvalidEnvironmentError("SOVEREIGN_ENV must be explicitly set")
    environment = parse_environment(raw_environment)
    return Settings(
        environment=environment,
        registry_path=registry_path or default_registry_path(),
    )
