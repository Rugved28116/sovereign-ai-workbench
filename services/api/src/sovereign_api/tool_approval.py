"""Immutable, request-bound approval values; no user identity or persistence."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from types import MappingProxyType

from sovereign_api.errors import SovereignAPIError
from sovereign_api.tool_contracts import (
    FrozenJSON, ToolPermission, ToolRequest, ToolRiskLevel, ToolSideEffectLevel,
)


class InvalidToolApprovalError(SovereignAPIError):
    code = "invalid_tool_approval"


class ApprovalChoice(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"


def _utc(value: datetime) -> bool:
    try:
        return type(value) is datetime and value.tzinfo is not None and value.utcoffset() == timedelta(0)
    except Exception:
        return False


def _plain(value: FrozenJSON) -> object:
    if type(value) in (dict, MappingProxyType):
        return {key: _plain(item) for key, item in value.items()}
    if type(value) is tuple:
        return [_plain(item) for item in value]
    return value


def request_fingerprint(
    request: ToolRequest, permissions: tuple[ToolPermission, ...],
) -> str:
    """Hash canonical request identity, arguments, and exact permission declaration."""
    if type(request) is not ToolRequest or type(permissions) is not tuple:
        raise InvalidToolApprovalError("Approval request is invalid")
    try:
        request = ToolRequest(
            request_id=request.request_id, task_id=request.task_id,
            stage_id=request.stage_id, tool_id=request.tool_id,
            operation=request.operation, arguments=request.arguments,
        )
        identifiers = tuple(sorted(permission.identifier for permission in permissions))
        if any(type(permission) is not ToolPermission for permission in permissions):
            raise ValueError
        payload = {
            "request_id": request.request_id,
            "task_id": request.task_id,
            "stage_id": request.stage_id,
            "tool_id": request.tool_id,
            "operation": request.operation,
            "arguments": _plain(request.arguments),
            "permissions": identifiers,
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8", errors="strict")
    except Exception:
        raise InvalidToolApprovalError("Approval request is invalid") from None
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    approval_id: str
    request_id: str
    task_id: str
    stage_id: str
    tool_id: str
    operation: str
    requested_permissions: tuple[ToolPermission, ...]
    risk_level: ToolRiskLevel
    side_effect_level: ToolSideEffectLevel
    created_at: datetime
    safe_summary: str
    request_fingerprint: str

    def __post_init__(self) -> None:
        if type(self.requested_permissions) not in (tuple, list):
            raise InvalidToolApprovalError("Approval request is invalid")
        object.__setattr__(self, "requested_permissions", tuple(self.requested_permissions))
        try:
            for permission in self.requested_permissions:
                if type(permission) is not ToolPermission:
                    raise ValueError
                permission.__post_init__()
        except Exception:
            raise InvalidToolApprovalError("Approval request is invalid") from None
        if (
            type(self.approval_id) is not str or re.fullmatch(r"[0-9a-f]{32}", self.approval_id) is None
            or type(self.request_fingerprint) is not str
            or re.fullmatch(r"[0-9a-f]{64}", self.request_fingerprint) is None
            or any(type(value) is not str or not value.strip() for value in (
                self.request_id, self.task_id, self.stage_id, self.tool_id, self.operation,
            ))
            or not self.requested_permissions
            or any(type(item) is not ToolPermission for item in self.requested_permissions)
            or type(self.risk_level) is not ToolRiskLevel
            or type(self.side_effect_level) is not ToolSideEffectLevel
            or not _utc(self.created_at)
            or type(self.safe_summary) is not str or not self.safe_summary.strip()
            or len(self.safe_summary) > 256
            or any(ord(character) < 32 or ord(character) == 127 for character in self.safe_summary)
        ):
            raise InvalidToolApprovalError("Approval request is invalid")


@dataclass(frozen=True, slots=True)
class ApprovalDecision:
    approval_id: str
    decision: ApprovalChoice
    decided_at: datetime
    safe_reason: str | None = None

    def __post_init__(self) -> None:
        if (
            type(self.approval_id) is not str or re.fullmatch(r"[0-9a-f]{32}", self.approval_id) is None
            or type(self.decision) is not ApprovalChoice
            or not _utc(self.decided_at)
            or (self.safe_reason is not None and (
                type(self.safe_reason) is not str or not self.safe_reason.strip()
                or len(self.safe_reason) > 256
                or any(ord(character) < 32 or ord(character) == 127 for character in self.safe_reason)
            ))
        ):
            raise InvalidToolApprovalError("Approval decision is invalid")


@dataclass(frozen=True, slots=True)
class ApprovalAuthorization:
    """Legacy request identity data; this is not execution authority."""

    approval_id: str
    request_fingerprint: str

    def __post_init__(self) -> None:
        if (
            type(self.approval_id) is not str or re.fullmatch(r"[0-9a-f]{32}", self.approval_id) is None
            or type(self.request_fingerprint) is not str
            or re.fullmatch(r"[0-9a-f]{64}", self.request_fingerprint) is None
        ):
            raise InvalidToolApprovalError("Approval authorization is invalid")


_APPROVAL_CAPABILITY = object()


@dataclass(frozen=True, slots=True, init=False)
class ValidatedApprovalAuthorization:
    """In-process capability issued only after a validated APPROVE decision."""

    approval_id: str
    request_fingerprint: str
    task_id: str
    stage_id: str
    tool_id: str
    _capability: object = field(repr=False, compare=False)

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise InvalidToolApprovalError("Approval authorization is invalid")


def _issue_validated_authorization(
    state: AgentTaskState,
    stage: TaskStage,
    approval: ApprovalRequest,
    decision: ApprovalDecision,
    request: ToolRequest,
    permissions: tuple[ToolPermission, ...],
) -> ValidatedApprovalAuthorization:
    """Coordinator-only issuance after suspended-state validation."""
    # Local imports keep approval value types independent from execution state.
    from sovereign_api.agent_task_state import AgentTaskState, StageStatus, TaskStatus
    from sovereign_api.task_planning import TaskStage

    if (
        type(state) is not AgentTaskState
        or type(stage) is not TaskStage
        or state.task_status is not TaskStatus.AWAITING_APPROVAL
        or state.current_stage_id != stage.stage_id
        or state.approval_request != approval
        or not any(
            planned == stage and current.stage_id == stage.stage_id
            and current.status is StageStatus.AWAITING_APPROVAL
            for planned, current in zip(state.plan.stages, state.stage_states)
        )
        or type(approval) is not ApprovalRequest
        or type(decision) is not ApprovalDecision
        or type(request) is not ToolRequest
        or decision.decision is not ApprovalChoice.APPROVE
        or decision.approval_id != approval.approval_id
        or decision.decided_at < approval.created_at
        or (request.request_id, request.task_id, request.stage_id,
            request.tool_id, request.operation)
        != (approval.request_id, approval.task_id, approval.stage_id,
            approval.tool_id, approval.operation)
        or permissions != approval.requested_permissions
        or request_fingerprint(request, permissions) != approval.request_fingerprint
    ):
        raise InvalidToolApprovalError("Approval authorization is invalid")
    authorization = object.__new__(ValidatedApprovalAuthorization)
    for name, value in (
        ("approval_id", approval.approval_id),
        ("request_fingerprint", approval.request_fingerprint),
        ("task_id", approval.task_id),
        ("stage_id", approval.stage_id),
        ("tool_id", approval.tool_id),
        ("_capability", _APPROVAL_CAPABILITY),
    ):
        object.__setattr__(authorization, name, value)
    return authorization


def _is_validated_authorization(value: object) -> bool:
    return (
        type(value) is ValidatedApprovalAuthorization
        and getattr(value, "_capability", None) is _APPROVAL_CAPABILITY
    )
