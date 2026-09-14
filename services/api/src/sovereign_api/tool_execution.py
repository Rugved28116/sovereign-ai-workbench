"""Policy-gated execution boundary for explicitly registered local tools."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Mapping, Protocol

from sovereign_api.config import DeploymentEnvironment
from sovereign_api.errors import SovereignAPIError
from sovereign_api.tool_contracts import (
    SafeToolError,
    Tool,
    ToolDescriptor,
    ToolPermission,
    ToolRegistry,
    ToolRequest,
    ToolResult,
    UnknownToolError,
)
from sovereign_api.tool_policy import ToolPermissionDecision, ToolPolicyEvaluator
from sovereign_api.tool_approval import (
    ValidatedApprovalAuthorization, _is_validated_authorization,
    request_fingerprint,
)


class ToolExecutionError(SovereignAPIError):
    code = "tool_execution_error"


class ToolPermissionDeniedError(ToolExecutionError):
    code = "tool_permission_denied"


class ToolApprovalRequiredError(ToolExecutionError):
    code = "tool_approval_required"


class ToolImplementationUnavailableError(ToolExecutionError):
    code = "tool_implementation_unavailable"


class ToolExecutionIdentityError(ToolExecutionError):
    code = "tool_execution_identity_mismatch"


class ToolInvocationError(ToolExecutionError):
    code = "tool_invocation_failed"


@dataclass(frozen=True, slots=True)
class ExecutableRegistration:
    tool_id: str
    execute_callable: Callable[[ToolRequest], Awaitable[ToolResult]]


@dataclass(frozen=True, slots=True, init=False)
class ExecutableToolRegistry:
    """Capture execution authority once; tools must freeze security-relevant state."""

    registrations: tuple[ExecutableRegistration, ...]
    descriptors: tuple[ToolDescriptor, ...]

    def __init__(self, implementations: Iterable[Tool]) -> None:
        ids: set[str] = set()
        registrations: list[ExecutableRegistration] = []
        descriptors: list[ToolDescriptor] = []
        for implementation in implementations:
            try:
                tool_id = implementation.tool_id
                descriptor = implementation.descriptor
                captured_execute = implementation.execute
            except Exception:
                raise ToolExecutionIdentityError(
                    "Tool implementation registration is invalid"
                ) from None
            if type(tool_id) is not str or not tool_id.strip():
                raise ToolExecutionIdentityError("Tool implementation ID is invalid")
            if tool_id in ids:
                raise ToolExecutionIdentityError("Duplicate tool implementation ID")
            if type(descriptor) is not ToolDescriptor or descriptor.tool_id != tool_id:
                raise ToolExecutionIdentityError("Tool implementation descriptor is invalid")
            if not callable(captured_execute):
                raise ToolExecutionIdentityError("Tool implementation is not executable")
            ids.add(tool_id)
            registrations.append(ExecutableRegistration(tool_id, captured_execute))
            descriptors.append(descriptor)
        object.__setattr__(self, "registrations", tuple(registrations))
        object.__setattr__(self, "descriptors", tuple(descriptors))

    def get(self, tool_id: str) -> ExecutableRegistration:
        for registration in self.registrations:
            if registration.tool_id == tool_id:
                return registration
        raise ToolImplementationUnavailableError("Tool implementation is unavailable")


class ToolExecutor(Protocol):
    def describe(self, tool_id: str) -> ToolDescriptor: ...

    async def execute(
        self,
        request: ToolRequest,
        *,
        granted_permissions: frozenset[ToolPermission],
        environment: DeploymentEnvironment,
        approval_authorization: ValidatedApprovalAuthorization | None = None,
    ) -> ToolResult: ...


@dataclass(frozen=True, slots=True)
class PolicyEnforcedToolExecutor:
    """Resolve a trusted descriptor, evaluate policy, then invoke its tool."""

    registry: ToolRegistry
    policy_evaluator: ToolPolicyEvaluator
    tools: Mapping[str, Tool] | ExecutableToolRegistry

    def __post_init__(self) -> None:
        if isinstance(self.tools, ExecutableToolRegistry):
            implementations = self.tools
        else:
            copied = dict(self.tools)
            for tool_id, tool in copied.items():
                try:
                    matches = type(tool_id) is str and tool_id == tool.tool_id
                except Exception:
                    raise ToolExecutionIdentityError(
                        "Tool implementation ID is invalid"
                    ) from None
                if not matches:
                    raise ToolExecutionIdentityError(
                        "Tool implementation ID does not match registration"
                    )
            implementations = ExecutableToolRegistry(tuple(copied.values()))
        for captured_descriptor in implementations.descriptors:
            try:
                descriptor = self.registry.get(captured_descriptor.tool_id)
            except UnknownToolError:
                raise ToolExecutionIdentityError(
                    "Tool implementation has no registered descriptor"
                ) from None
            if captured_descriptor != descriptor:
                raise ToolExecutionIdentityError(
                    "Tool implementation descriptor does not match registration"
                )
        object.__setattr__(self, "tools", implementations)

    def describe(self, tool_id: str) -> ToolDescriptor:
        return self.registry.get(tool_id)

    async def execute(
        self,
        request: ToolRequest,
        *,
        granted_permissions: frozenset[ToolPermission],
        environment: DeploymentEnvironment,
        approval_authorization: ValidatedApprovalAuthorization | None = None,
    ) -> ToolResult:
        descriptor = self.registry.get(request.tool_id)
        try:
            decision = self.policy_evaluator.evaluate(
                descriptor,
                granted_permissions=granted_permissions,
                environment=environment,
            )
        except Exception:
            raise ToolPermissionDeniedError("Tool execution denied") from None
        if decision is ToolPermissionDecision.DENY or decision not in (
            ToolPermissionDecision.ALLOW, ToolPermissionDecision.REQUIRE_APPROVAL,
        ):
            raise ToolPermissionDeniedError("Tool execution denied")
        if approval_authorization is not None:
            if (
                not _is_validated_authorization(approval_authorization)
                or approval_authorization.task_id != request.task_id
                or approval_authorization.stage_id != request.stage_id
                or approval_authorization.tool_id != request.tool_id
                or type(granted_permissions) is not frozenset
                or not frozenset(descriptor.required_permissions) <= granted_permissions
                or type(environment) is not DeploymentEnvironment
                or (descriptor.requires_network and (
                    environment is DeploymentEnvironment.AIR_GAPPED
                    or ToolPermission("network.access") not in granted_permissions
                ))
                or approval_authorization.request_fingerprint
                != request_fingerprint(request, descriptor.required_permissions)
            ):
                raise ToolPermissionDeniedError("Tool approval does not match request")
        if (
            decision is ToolPermissionDecision.REQUIRE_APPROVAL
            and approval_authorization is None
        ):
            raise ToolApprovalRequiredError("Tool execution requires approval")

        registration = self.tools.get(descriptor.tool_id)
        if registration.tool_id != descriptor.tool_id or request.tool_id != descriptor.tool_id:
            raise ToolExecutionIdentityError("Tool execution identity mismatch")
        try:
            result = await registration.execute_callable(request)
        except SafeToolError:
            raise
        except Exception:
            pass
        else:
            if (
                type(result) is not ToolResult
                or result.tool_id != descriptor.tool_id
                or result.request_id != request.request_id
            ):
                raise ToolExecutionIdentityError("Tool result identity mismatch")
            return result
        # Outside the handler: do not retain the tool exception as context either.
        raise ToolInvocationError("Tool execution failed") from None
