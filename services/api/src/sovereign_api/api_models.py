"""Typed public API request and response models."""

from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from sovereign_api.config import DeploymentEnvironment
from sovereign_api.prompt_validation import MAX_PROMPT_LENGTH, Prompt
from sovereign_api.task_classification import TaskClass
from sovereign_api.task_planning import TaskPlan

MAX_REQUIRED_CAPABILITIES = 16
MAX_CAPABILITY_NAME_LENGTH = 64

CapabilityName = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, min_length=1, max_length=MAX_CAPABILITY_NAME_LENGTH
    ),
]


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: Prompt
    required_capabilities: list[CapabilityName] | None = Field(
        default=None, min_length=1, max_length=MAX_REQUIRED_CAPABILITIES
    )

    @field_validator("required_capabilities")
    @classmethod
    def reject_duplicate_capabilities(
        cls, values: list[str] | None
    ) -> list[str] | None:
        if values is None:
            return None
        if len(values) != len(set(values)):
            raise ValueError("required_capabilities must not contain duplicates")
        return values

    @model_validator(mode="after")
    def reject_explicit_null_capabilities(self) -> Self:
        if (
            "required_capabilities" in self.model_fields_set
            and self.required_capabilities is None
        ):
            raise ValueError("required_capabilities must be omitted or contain values")
        return self


class RoutingInformation(BaseModel):
    environment: DeploymentEnvironment
    required_capabilities: list[str]
    capability_source: Literal["explicit", "inferred"] | None = None
    task_class: TaskClass | None = None
    plan: TaskPlan | None = None


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
