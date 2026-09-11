"""Policy-gated execution boundary for explicitly registered local tools."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Protocol

from sovereign_api.config import DeploymentEnvironment
from sovereign_api.errors import SovereignAPIError
from sovereign_api.tool_contracts import (
    Tool,
    ToolPermission,
    ToolRegistry,
    ToolRequest,
    ToolResult,
    ToolValidationError,
)
from sovereign_api.tool_policy import ToolPermissionDecision, ToolPolicyEvaluator


class ToolExecutionError(SovereignAPIError):
    code = "tool_execution_error"


class ToolPermissionDeniedError(ToolExecutionError):
    code = "tool_permission_denied"


class ToolApprovalRequiredError(ToolExecutionError):
    code = "tool_approval_required"


class ToolImplementationUnavailableError(ToolExecutionError):
    code = "tool_implementation_unavailable"


class ToolExecutor(Protocol):
    async def execute(
        self,
        request: ToolRequest,
        *,
        granted_permissions: frozenset[ToolPermission],
        environment: DeploymentEnvironment,
    ) -> ToolResult: ...


@dataclass(frozen=True, slots=True)
class PolicyEnforcedToolExecutor:
    """Resolve a trusted descriptor, evaluate policy, then invoke its tool."""

    registry: ToolRegistry
    policy_evaluator: ToolPolicyEvaluator
    tools: Mapping[str, Tool]

    def __post_init__(self) -> None:
        implementations = dict(self.tools)
        for tool_id, tool in implementations.items():
            if type(tool_id) is not str:
                raise ToolValidationError("Tool implementation IDs must be strings")
            descriptor = self.registry.get(tool_id)
            if tool.descriptor != descriptor:
                raise ToolValidationError(
                    "Tool implementation descriptor must match the registry"
                )
        object.__setattr__(self, "tools", MappingProxyType(implementations))

    async def execute(
        self,
        request: ToolRequest,
        *,
        granted_permissions: frozenset[ToolPermission],
        environment: DeploymentEnvironment,
    ) -> ToolResult:
        descriptor = self.registry.get(request.tool_id)
        decision = self.policy_evaluator.evaluate(
            descriptor,
            granted_permissions=granted_permissions,
            environment=environment,
        )
        if decision is ToolPermissionDecision.REQUIRE_APPROVAL:
            raise ToolApprovalRequiredError("Tool execution requires approval")
        if decision is not ToolPermissionDecision.ALLOW:
            raise ToolPermissionDeniedError("Tool execution denied")

        tool = self.tools.get(descriptor.tool_id)
        if tool is None:
            raise ToolImplementationUnavailableError(
                "Tool implementation is unavailable"
            )
        return await tool.execute(request)
