"""Typed public API request and response models."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from sovereign_api.config import DeploymentEnvironment

MAX_PROMPT_LENGTH = 32_768
MAX_REQUIRED_CAPABILITIES = 16
MAX_CAPABILITY_NAME_LENGTH = 64

Prompt = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_PROMPT_LENGTH),
]
CapabilityName = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, min_length=1, max_length=MAX_CAPABILITY_NAME_LENGTH
    ),
]


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: Prompt
    required_capabilities: list[CapabilityName] = Field(
        min_length=1, max_length=MAX_REQUIRED_CAPABILITIES
    )

    @field_validator("required_capabilities")
    @classmethod
    def reject_duplicate_capabilities(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("required_capabilities must not contain duplicates")
        return values


class RoutingInformation(BaseModel):
    environment: DeploymentEnvironment
    required_capabilities: list[str]


class GenerateResponse(BaseModel):
    model_id: str
    provider: str
    content: str
    routing: RoutingInformation


class HealthResponse(BaseModel):
    status: str
    service: str


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
