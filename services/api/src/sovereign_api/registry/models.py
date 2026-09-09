"""Typed, provider-neutral model registry definitions."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StringConstraints,
    field_validator,
    model_validator,
)

from sovereign_api.config import DeploymentEnvironment

MetadataLabel = Annotated[str, StringConstraints(min_length=1, max_length=128)]
MetadataDescription = Annotated[str, StringConstraints(min_length=1, max_length=1024)]
MetadataTag = Annotated[str, StringConstraints(min_length=1, max_length=64)]


class RegistryMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    display_name: MetadataLabel | None = None
    description: MetadataDescription | None = None
    family: MetadataLabel | None = None
    size_class: MetadataLabel | None = None
    tags: tuple[MetadataTag, ...] = Field(default=(), max_length=32)

    @field_validator("display_name", "description", "family", "size_class")
    @classmethod
    def reject_surrounding_whitespace(cls, value: str | None) -> str | None:
        if value is not None and value != value.strip():
            raise ValueError("must not contain surrounding whitespace")
        return value

    @field_validator("tags", mode="before")
    @classmethod
    def require_tags_list(cls, value: object) -> object:
        if not isinstance(value, list):
            raise ValueError("must be a YAML list")
        return value

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(value != value.strip() for value in values):
            raise ValueError("tags must not contain surrounding whitespace")
        if len(values) != len(set(values)):
            raise ValueError("tags must not contain duplicates")
        return values


class ModelDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    provider: str
    enabled: StrictBool
    environments: tuple[DeploymentEnvironment, ...]
    capabilities: tuple[str, ...]
    context_length: StrictInt = Field(gt=0)
    priority: StrictInt
    metadata: RegistryMetadata = Field(default_factory=RegistryMetadata)

    @field_validator("id", "provider")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("must be a non-empty string without surrounding whitespace")
        return value

    @field_validator("environments", "capabilities", mode="before")
    @classmethod
    def require_yaml_list(cls, value: object) -> object:
        if not isinstance(value, list):
            raise ValueError("must be a YAML list")
        return value

    @field_validator("capabilities")
    @classmethod
    def validate_capabilities(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values:
            raise ValueError("must contain at least one capability")
        if any(not value or value != value.strip() for value in values):
            raise ValueError(
                "capabilities must be non-empty strings without surrounding whitespace"
            )
        if len(values) != len(set(values)):
            raise ValueError("capabilities must not contain duplicates")
        return values

    @field_validator("environments")
    @classmethod
    def validate_environments(
        cls, values: tuple[DeploymentEnvironment, ...]
    ) -> tuple[DeploymentEnvironment, ...]:
        if not values:
            raise ValueError("must contain at least one environment")
        if len(values) != len(set(values)):
            raise ValueError("environments must not contain duplicates")
        return values


class ModelRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: StrictInt
    models: tuple[ModelDefinition, ...]

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: int) -> int:
        if value != 1:
            raise ValueError("unsupported schema_version; expected 1")
        return value

    @field_validator("models", mode="before")
    @classmethod
    def require_models_list(cls, value: object) -> object:
        if not isinstance(value, list):
            raise ValueError("must be a YAML list")
        return value

    @model_validator(mode="after")
    def validate_unique_model_ids(self) -> Self:
        model_ids = [model.id for model in self.models]
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("model IDs must be unique")
        return self
