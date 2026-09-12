"""Immutable tool contracts only; no executable tools or resource access."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping, Protocol

from sovereign_api.errors import SovereignAPIError


class ToolValidationError(SovereignAPIError):
    code = "invalid_tool_contract"


class SafeToolError(SovereignAPIError):
    """A tool's typed, caller-safe operational failure."""

    code = "tool_operational_error"


class UnknownToolError(SovereignAPIError):
    code = "unknown_tool"


def _text(value: object) -> None:
    if type(value) is not str or not value.strip():
        raise ToolValidationError("Expected a non-empty string")


@dataclass(frozen=True, slots=True)
class ToolPermission:
    identifier: str

    def __post_init__(self) -> None:
        if type(self.identifier) is not str or re.fullmatch(
            r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+", self.identifier
        ) is None:
            raise ToolValidationError("Expected an exact dotted permission identifier")


class ToolRiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ToolSideEffectLevel(StrEnum):
    NONE = "none"
    READ = "read"
    WRITE = "write"
    DESTRUCTIVE = "destructive"


@dataclass(frozen=True, slots=True)
class ToolDescriptor:
    tool_id: str
    display_name: str
    description: str
    capabilities: tuple[str, ...]
    risk_level: ToolRiskLevel
    side_effect_level: ToolSideEffectLevel
    requires_network: bool
    supports_read: bool
    supports_write: bool
    required_permissions: tuple[ToolPermission, ...]

    def __post_init__(self) -> None:
        for name in ("capabilities", "required_permissions"):
            values = getattr(self, name)
            if type(values) not in (tuple, list):
                raise ToolValidationError("Descriptor collections must be lists or tuples")
            object.__setattr__(self, name, tuple(values))
        self.validate()

    def validate(self) -> None:
        for value in (self.tool_id, self.display_name, self.description):
            _text(value)
        if type(self.capabilities) is not tuple or not self.capabilities:
            raise ToolValidationError("Capabilities must be a non-empty tuple")
        for capability in self.capabilities:
            _text(capability)
        if type(self.required_permissions) is not tuple:
            raise ToolValidationError("Permissions must be a tuple")
        for permission in self.required_permissions:
            if type(permission) is not ToolPermission:
                raise ToolValidationError("Expected ToolPermission values")
            permission.__post_init__()
        if type(self.risk_level) is not ToolRiskLevel:
            raise ToolValidationError("Unknown tool risk")
        if type(self.side_effect_level) is not ToolSideEffectLevel:
            raise ToolValidationError("Unknown tool side effects")
        if any(type(value) is not bool for value in (
            self.requires_network, self.supports_read, self.supports_write
        )):
            raise ToolValidationError("Tool flags must be booleans")
        if self.side_effect_level is ToolSideEffectLevel.READ and not self.supports_read:
            raise ToolValidationError("Read effects require read support")
        if self.side_effect_level in (
            ToolSideEffectLevel.WRITE, ToolSideEffectLevel.DESTRUCTIVE
        ) and not self.supports_write:
            raise ToolValidationError("Write effects require write support")
        if self.supports_write and self.side_effect_level not in (
            ToolSideEffectLevel.WRITE, ToolSideEffectLevel.DESTRUCTIVE
        ):
            raise ToolValidationError("Write support must declare write effects")


type FrozenJSON = (
    str | int | float | bool | None | tuple[FrozenJSON, ...]
    | Mapping[str, FrozenJSON]
)
MAX_ARGUMENT_DEPTH = 32
MAX_ARGUMENT_NODES = 10_000


def _freeze_arguments(value: object, depth: int, budget: list[int]) -> FrozenJSON:
    budget[0] -= 1
    if depth > MAX_ARGUMENT_DEPTH or budget[0] < 0:
        raise ToolValidationError("Tool arguments exceed structural limits")
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) in (list, tuple):
        return tuple(_freeze_arguments(item, depth + 1, budget) for item in value)
    if type(value) in (dict, MappingProxyType):
        copied = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ToolValidationError("Argument object keys must be strings")
            copied[key] = _freeze_arguments(item, depth + 1, budget)
        return MappingProxyType(copied)
    raise ToolValidationError("Tool arguments must contain only JSON values")


@dataclass(frozen=True, slots=True)
class ToolRequest:
    request_id: str
    tool_id: str
    operation: str
    arguments: Mapping[str, FrozenJSON]
    task_id: str
    stage_id: str

    def __post_init__(self) -> None:
        for value in (
            self.request_id, self.tool_id, self.operation, self.task_id, self.stage_id
        ):
            _text(value)
        if type(self.arguments) not in (dict, MappingProxyType):
            raise ToolValidationError("Arguments must be a JSON object")
        object.__setattr__(self, "arguments", _freeze_arguments(
            self.arguments, 0, [MAX_ARGUMENT_NODES]
        ))


class ToolResultStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


MAX_INLINE_TEXT_BYTES = 1_048_576


@dataclass(frozen=True, slots=True)
class ToolResult:
    request_id: str
    tool_id: str
    status: ToolResultStatus
    output_reference: str | None = None
    safe_message: str | None = None
    error_code: str | None = None
    text_content: str | None = None

    def __post_init__(self) -> None:
        _text(self.request_id)
        _text(self.tool_id)
        if type(self.status) is not ToolResultStatus:
            raise ToolValidationError("Unknown tool result status")
        for value in (self.output_reference, self.safe_message, self.error_code):
            if value is not None:
                _text(value)
        if self.text_content is not None and type(self.text_content) is not str:
            raise ToolValidationError("Inline text must be a string")
        if self.text_content is not None:
            try:
                text_size = len(self.text_content.encode("utf-8", errors="strict"))
            except UnicodeEncodeError as error:
                raise ToolValidationError("Inline text must be valid UTF-8") from error
            if text_size > MAX_INLINE_TEXT_BYTES:
                raise ToolValidationError("Inline text exceeds the result limit")
        if self.status is ToolResultStatus.FAILED:
            if self.error_code is None or self.safe_message is None:
                raise ToolValidationError("Failure requires safe code and message")
            if self.output_reference is not None or self.text_content is not None:
                raise ToolValidationError("Failure cannot claim successful output")
        else:
            if self.error_code is not None:
                raise ToolValidationError("Success cannot contain an error code")
            if self.output_reference is not None and self.text_content is not None:
                raise ToolValidationError(
                    "Success must use exactly one output channel"
                )


class Tool(Protocol):
    """Provider-neutral contract for an explicitly registered implementation."""

    @property
    def tool_id(self) -> str: ...

    @property
    def descriptor(self) -> ToolDescriptor: ...

    async def execute(self, request: ToolRequest) -> ToolResult: ...


@dataclass(frozen=True, slots=True)
class ToolRegistry:
    descriptors: tuple[ToolDescriptor, ...]

    def __post_init__(self) -> None:
        if type(self.descriptors) not in (tuple, list):
            raise ToolValidationError("Registry requires descriptor values")
        descriptors = tuple(self.descriptors)
        ids = set()
        for descriptor in descriptors:
            if type(descriptor) is not ToolDescriptor:
                raise ToolValidationError("Registry stores only descriptors")
            descriptor.validate()
            if descriptor.tool_id in ids:
                raise ToolValidationError("Duplicate tool ID")
            ids.add(descriptor.tool_id)
        object.__setattr__(self, "descriptors", descriptors)

    def get(self, tool_id: str) -> ToolDescriptor:
        _text(tool_id)
        for descriptor in self.descriptors:
            if descriptor.tool_id == tool_id:
                return descriptor
        raise UnknownToolError("Tool is not registered")
