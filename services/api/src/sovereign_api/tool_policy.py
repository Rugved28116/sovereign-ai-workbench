"""Pure default-deny tool permission evaluation; approval grants no permission."""

from enum import StrEnum
from typing import Protocol

from sovereign_api.config import DeploymentEnvironment
from sovereign_api.tool_contracts import (
    ToolDescriptor, ToolPermission, ToolRiskLevel, ToolSideEffectLevel,
    ToolValidationError,
)


class ToolPermissionDecision(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


class ToolPolicyEvaluator(Protocol):
    def evaluate(
        self, descriptor: ToolDescriptor, *,
        granted_permissions: frozenset[ToolPermission],
        environment: DeploymentEnvironment,
    ) -> ToolPermissionDecision: ...


class DeterministicToolPolicyEvaluator:
    def evaluate(
        self, descriptor: ToolDescriptor, *,
        granted_permissions: frozenset[ToolPermission],
        environment: DeploymentEnvironment,
    ) -> ToolPermissionDecision:
        deny = ToolPermissionDecision.DENY
        if type(environment) is not DeploymentEnvironment:
            return deny
        if type(descriptor) is not ToolDescriptor:
            return deny
        try:
            descriptor.validate()
        except (ToolValidationError, AttributeError):
            return deny
        if type(granted_permissions) is not frozenset:
            return deny
        for permission in granted_permissions:
            if type(permission) is not ToolPermission:
                return deny
            try:
                permission.__post_init__()
            except (ToolValidationError, AttributeError):
                return deny
        # Empty declarations do not make a tool permission-free.
        if not descriptor.required_permissions:
            return deny
        if not frozenset(descriptor.required_permissions) <= granted_permissions:
            return deny
        if descriptor.requires_network:
            # No destination scope exists yet: all network tools are prohibited.
            if environment is DeploymentEnvironment.AIR_GAPPED:
                return deny
            if ToolPermission("network.access") not in granted_permissions:
                return deny
        if descriptor.side_effect_level is ToolSideEffectLevel.DESTRUCTIVE:
            return ToolPermissionDecision.REQUIRE_APPROVAL
        if descriptor.supports_write and descriptor.risk_level in (
            ToolRiskLevel.HIGH, ToolRiskLevel.CRITICAL
        ):
            return ToolPermissionDecision.REQUIRE_APPROVAL
        return ToolPermissionDecision.ALLOW
