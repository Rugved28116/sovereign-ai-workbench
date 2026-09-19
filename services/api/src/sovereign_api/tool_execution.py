"""Policy-gated execution boundary for explicitly registered local tools."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
import hashlib
import hmac
import json
from typing import Mapping, Protocol
import weakref

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
    ToolResultStatus,
    UnknownToolError,
)
from sovereign_api.tool_policy import (
    DeterministicToolPolicyEvaluator, ToolPermissionDecision, ToolPolicyEvaluator,
)
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


class ToolPolicyAuthorizationError(ToolExecutionError):
    code = "tool_policy_authorization_invalid"


class ValidatedToolExecutionAuthorization:
    """One-use in-process authority issued only for an explicit ALLOW decision."""

    __slots__ = (
        "request_id", "tool_id", "task_id", "stage_id", "request_fingerprint",
        "__weakref__",
    )

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise ToolPolicyAuthorizationError("Tool policy authorization is invalid")

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Tool policy authorization is immutable")

    def __copy__(self) -> object:
        raise TypeError("Tool policy authorization cannot be copied")

    def __deepcopy__(self, memo: object) -> object:
        raise TypeError("Tool policy authorization cannot be copied")

    def __reduce__(self) -> object:
        raise TypeError("Tool policy authorization cannot be serialized")

    def __reduce_ex__(self, protocol: int) -> object:
        raise TypeError("Tool policy authorization cannot be serialized")


_TOOL_POLICY_AUTHORIZATIONS: dict[
    int, tuple[
        weakref.ReferenceType[ValidatedToolExecutionAuthorization],
        str, str, str, str, str,
    ]
] = {}


def _issue_tool_execution_authorization(
    request: ToolRequest, descriptor: ToolDescriptor,
) -> ValidatedToolExecutionAuthorization:
    authorization = object.__new__(ValidatedToolExecutionAuthorization)
    for name, value in (
        ("request_id", request.request_id), ("tool_id", request.tool_id),
        ("task_id", request.task_id), ("stage_id", request.stage_id),
        ("request_fingerprint", request_fingerprint(
            request, descriptor.required_permissions,
        )),
    ):
        object.__setattr__(authorization, name, value)
    identity = id(authorization)

    def discard(reference: object) -> None:
        current = _TOOL_POLICY_AUTHORIZATIONS.get(identity)
        if current is not None and current[0] is reference:
            _TOOL_POLICY_AUTHORIZATIONS.pop(identity, None)

    reference = weakref.ref(authorization, discard)
    _TOOL_POLICY_AUTHORIZATIONS[identity] = (
        reference, request.request_id, request.tool_id, request.task_id,
        request.stage_id, authorization.request_fingerprint,
    )
    return authorization


def _consume_tool_execution_authorization(
    authorization: object, request: ToolRequest, descriptor: ToolDescriptor,
) -> bool:
    if type(authorization) is not ValidatedToolExecutionAuthorization:
        return False
    registered = _TOOL_POLICY_AUTHORIZATIONS.pop(id(authorization), None)
    expected = (
        request.request_id, request.tool_id, request.task_id, request.stage_id,
        request_fingerprint(request, descriptor.required_permissions),
    )
    return bool(
        registered is not None and registered[0]() is authorization
        and registered[1:] == expected
        and (
            authorization.request_id, authorization.tool_id,
            authorization.task_id, authorization.stage_id,
            authorization.request_fingerprint,
        ) == expected
    )


@dataclass(frozen=True, slots=True)
class ToolPolicyEvaluation:
    decision: ToolPermissionDecision
    authorization: ValidatedToolExecutionAuthorization | None = None

    def __post_init__(self) -> None:
        if (
            type(self.decision) is not ToolPermissionDecision
            or (self.decision is ToolPermissionDecision.ALLOW) != (
                type(self.authorization) is ValidatedToolExecutionAuthorization
            )
        ):
            raise ToolPolicyAuthorizationError("Tool policy evaluation is invalid")


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


class TrustedToolInvocationReceipt:
    """Opaque one-use proof of one trusted captured-callable invocation."""

    __slots__ = (
        "task_id", "stage_id", "tool_id", "attempt_id", "idempotency_key",
        "lease_id", "claimed_version", "__weakref__",
    )

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise ToolExecutionIdentityError("Tool invocation receipt is invalid")

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Tool invocation receipt is immutable")

    def __copy__(self) -> object:
        raise TypeError("Tool invocation receipt cannot be copied")

    def __deepcopy__(self, memo: object) -> object:
        raise TypeError("Tool invocation receipt cannot be copied")

    def __reduce__(self) -> object:
        raise TypeError("Tool invocation receipt cannot be serialized")

    def __reduce_ex__(self, protocol: int) -> object:
        raise TypeError("Tool invocation receipt cannot be serialized")


@dataclass(frozen=True, slots=True)
class TrustedToolInvocationOutcome:
    result: ToolResult | None
    error_code: str | None
    stage_output_reference: str | None
    result_content_digest: str | None


_ToolResultSnapshot = tuple[
    str, str, str, str | None, str | None, str | None, str | None,
]


_TRUSTED_TOOL_RECEIPTS: dict[
    int, tuple[
        weakref.ReferenceType[TrustedToolInvocationReceipt],
        str, str, str, str, str, str, int, _ToolResultSnapshot | None, str,
        str | None, str | None, str | None,
    ]
] = {}


class TrustedToolInvocationBoundary:
    """Trusted persisted policy and exactly-one captured-callable boundary."""

    __slots__ = ("__weakref__",)

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise ToolExecutionIdentityError("Persisted tool boundary is internal")

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Persisted tool boundary is immutable")

    def descriptor(self, tool_id: str) -> ToolDescriptor:
        descriptors, _, _, _ = _trusted_boundary_components(self)
        for descriptor in descriptors:
            if descriptor.tool_id == tool_id:
                return _copy_tool_descriptor(descriptor)
        raise ToolImplementationUnavailableError("Tool implementation is unavailable")

    def evaluate(
        self, request: ToolRequest, *,
        granted_permissions: frozenset[ToolPermission],
        environment: DeploymentEnvironment,
    ) -> ToolPolicyEvaluation:
        descriptors, _, evaluator, _ = _trusted_boundary_components(self)
        descriptor = next(
            (item for item in descriptors if item.tool_id == request.tool_id), None,
        )
        if descriptor is None:
            return ToolPolicyEvaluation(ToolPermissionDecision.DENY, None)
        try:
            decision = evaluator(
                descriptor,
                granted_permissions=granted_permissions,
                environment=environment,
            )
        except Exception:
            decision = ToolPermissionDecision.DENY
        authorization = (
            _issue_tool_execution_authorization(request, descriptor)
            if decision is ToolPermissionDecision.ALLOW else None
        )
        return ToolPolicyEvaluation(decision, authorization)

    async def invoke(
        self, repository: object, request: ToolRequest,
        policy_authorization: ValidatedToolExecutionAuthorization,
        invocation_permit: object,
    ) -> TrustedToolInvocationReceipt:
        descriptors, registrations, _, _ = _trusted_boundary_components(self)
        descriptor = next(
            (item for item in descriptors if item.tool_id == request.tool_id), None,
        )
        if descriptor is None:
            raise ToolImplementationUnavailableError(
                "Tool implementation is unavailable"
            )
        if not _consume_tool_execution_authorization(
            policy_authorization, request, descriptor,
        ):
            raise ToolPolicyAuthorizationError(
                "Tool policy authorization is invalid"
            )
        from sovereign_api.task_state_repository import SQLiteTaskStateRepository
        if type(repository) is not SQLiteTaskStateRepository:
            raise ToolExecutionIdentityError("Tool invocation repository is invalid")
        permit_identity = repository.consume_stage_invocation_permit(
            invocation_permit,
        )
        (
            attempt_id, idempotency_key, task_id, stage_id, tool_id,
            lease_id, claimed_version,
        ) = permit_identity
        if (task_id, stage_id, tool_id) != (
            request.task_id, request.stage_id, request.tool_id,
        ):
            raise ToolExecutionIdentityError("Tool invocation permit is invalid")
        registration = next(
            (item for item in registrations if item.tool_id == descriptor.tool_id),
            None,
        )
        if registration is None:
            raise ToolImplementationUnavailableError(
                "Tool implementation is unavailable"
            )
        if registration.tool_id != request.tool_id:
            raise ToolExecutionIdentityError("Tool execution identity mismatch")
        result_snapshot = None
        error_code = None
        try:
            candidate = await registration.execute_callable(request)
        except SafeToolError:
            error_code = "tool_failed"
        except Exception:
            error_code = "tool_unexpected_failure"
        else:
            if (
                type(candidate) is ToolResult
                and candidate.request_id == request.request_id
                and candidate.tool_id == request.tool_id
            ):
                checked_result = ToolResult(
                    candidate.request_id, candidate.tool_id, candidate.status,
                    output_reference=candidate.output_reference,
                    safe_message=candidate.safe_message,
                    error_code=candidate.error_code,
                    text_content=candidate.text_content,
                )
                result_snapshot = _snapshot_tool_result(checked_result)
            else:
                error_code = "tool_invalid_result"
        receipt = object.__new__(TrustedToolInvocationReceipt)
        for name, value in (
            ("task_id", request.task_id), ("stage_id", request.stage_id),
            ("tool_id", request.tool_id), ("attempt_id", attempt_id),
            ("idempotency_key", idempotency_key), ("lease_id", lease_id),
            ("claimed_version", claimed_version),
        ):
            object.__setattr__(receipt, name, value)
        identity = id(receipt)

        def discard(reference: object) -> None:
            current = _TRUSTED_TOOL_RECEIPTS.get(identity)
            if current is not None and current[0] is reference:
                _TRUSTED_TOOL_RECEIPTS.pop(identity, None)

        reference = weakref.ref(receipt, discard)
        _TRUSTED_TOOL_RECEIPTS[identity] = (
            reference, request.task_id, request.stage_id, request.tool_id,
            attempt_id, idempotency_key, lease_id, claimed_version,
            result_snapshot, _snapshot_digest(result_snapshot, error_code),
            error_code,
            result_snapshot[3] if result_snapshot is not None else None, None,
        )
        return receipt

    def inspect_receipt(
        self, receipt: TrustedToolInvocationReceipt, *,
        task_id: str, stage_id: str, tool_id: str, attempt_id: str,
        idempotency_key: str, lease_id: str, claimed_version: int,
    ) -> TrustedToolInvocationOutcome:
        return _trusted_tool_invocation_outcome(
            receipt, task_id=task_id, stage_id=stage_id, tool_id=tool_id,
            attempt_id=attempt_id, idempotency_key=idempotency_key,
            lease_id=lease_id, claimed_version=claimed_version, consume=False,
        )

    def bind_text_output(
        self, receipt: TrustedToolInvocationReceipt, *, output_reference: str,
    ) -> None:
        """Bind a process-local text reference to the exact received text."""
        from sovereign_api.stage_output_store import (
            InMemoryStageOutputStore, StageOutput, StageOutputReference,
        )
        _, _, _, output_store = _trusted_boundary_components(self)
        stored = _TRUSTED_TOOL_RECEIPTS.get(id(receipt))
        if (
            stored is None or stored[0]() is not receipt
            or stored[11] is not None or stored[12] is not None
            or stored[8] is None or stored[10] is not None
            or stored[8][4] is None or stored[8][3] is not None
            or not hmac.compare_digest(
                stored[9], _snapshot_digest(stored[8], stored[10]),
            )
            or type(output_store) is not InMemoryStageOutputStore
        ):
            raise ToolExecutionIdentityError("Tool invocation receipt is invalid")
        try:
            output = output_store.get(
                StageOutputReference(output_reference),
                task_id=stored[1], stage_id=stored[2],
            )
            if (
                type(output) is not StageOutput
                or output.task_id != stored[1]
                or output.stage_id != stored[2]
                or output.text_content != stored[8][4]
            ):
                raise ValueError
            digest = hashlib.sha256(
                output.text_content.encode("utf-8", errors="strict")
            ).hexdigest()
        except Exception:
            raise ToolExecutionIdentityError(
                "Tool stage output binding is invalid"
            ) from None
        _TRUSTED_TOOL_RECEIPTS[id(receipt)] = (
            *stored[:11], output_reference, digest,
        )

    def bind_verified_artifact(
        self, receipt: TrustedToolInvocationReceipt, *, expected_digest: str,
    ) -> None:
        """Bind a write result only after its captured-root bytes are verified.

        A tool result is merely a claim.  The captured artifact-root verifier is
        the authority for a durable write result, so an unsuccessful verification
        is converted into the controlled receipt failure used for finalization.
        """
        stored = _TRUSTED_TOOL_RECEIPTS.get(id(receipt))
        if (
            stored is None or stored[0]() is not receipt
            or stored[11] != (stored[8][3] if stored[8] is not None else None)
            or stored[12] is not None
            or stored[8] is None or stored[10] is not None
            or stored[8][3] is None or stored[8][4] is not None
            or not hmac.compare_digest(
                stored[9], _snapshot_digest(stored[8], stored[10]),
            )
        ):
            raise ToolExecutionIdentityError("Tool invocation receipt is invalid")
        verified = self.verify_artifact(
            tool_id=stored[3], reference=stored[8][3],
            content_digest=expected_digest,
        )
        if verified:
            _TRUSTED_TOOL_RECEIPTS[id(receipt)] = (
                *stored[:11], stored[8][3], expected_digest,
            )
            return
        # Keep the admitted attempt auditable, but never let an unverified
        # implementation result become a durable success.
        error_code = "tool_artifact_verification_failed"
        _TRUSTED_TOOL_RECEIPTS[id(receipt)] = (
            *stored[:8], None, _snapshot_digest(None, error_code), error_code,
            None, None,
        )

    def verify_artifact(
        self, *, tool_id: str, reference: str, content_digest: str,
    ) -> bool:
        """Verify a built-in artifact using its captured root and secure reader."""
        _trusted_boundary_components(self)
        captured = _TRUSTED_ARTIFACT_ROOTS.get(id(self))
        if captured is None or captured[0]() is not self or captured[1] != tool_id:
            return False
        try:
            from sovereign_api.workspace_write_artifact import (
                verify_artifact_content,
            )
            return verify_artifact_content(
                captured[2], reference, content_digest,
            )
        except Exception:
            return False


def _copy_tool_descriptor(descriptor: ToolDescriptor) -> ToolDescriptor:
    if type(descriptor) is not ToolDescriptor:
        raise ToolExecutionIdentityError("Tool descriptor is invalid")
    return ToolDescriptor(
        descriptor.tool_id, descriptor.display_name, descriptor.description,
        tuple(descriptor.capabilities), descriptor.risk_level,
        descriptor.side_effect_level, descriptor.requires_network,
        descriptor.supports_read, descriptor.supports_write,
        tuple(ToolPermission(item.identifier) for item in descriptor.required_permissions),
    )


_TRUSTED_TOOL_BOUNDARIES: dict[
    int, tuple[
        weakref.ReferenceType[TrustedToolInvocationBoundary],
        tuple[ToolDescriptor, ...], tuple[ExecutableRegistration, ...],
        Callable[..., ToolPermissionDecision], object,
    ]
] = {}
_TRUSTED_ARTIFACT_ROOTS: dict[int, tuple[object, str, object]] = {}


def _trusted_boundary_components(
    boundary: TrustedToolInvocationBoundary,
) -> tuple[
    tuple[ToolDescriptor, ...], tuple[ExecutableRegistration, ...],
    Callable[..., ToolPermissionDecision], object,
]:
    if type(boundary) is not TrustedToolInvocationBoundary:
        raise ToolExecutionIdentityError("Persisted tool boundary is invalid")
    stored = _TRUSTED_TOOL_BOUNDARIES.get(id(boundary))
    if stored is None or stored[0]() is not boundary:
        raise ToolExecutionIdentityError("Persisted tool boundary is invalid")
    return stored[1], stored[2], stored[3], stored[4]


def _snapshot_tool_result(result: ToolResult) -> _ToolResultSnapshot:
    return (
        result.request_id, result.tool_id, result.status.value,
        result.output_reference, result.text_content,
        result.safe_message, result.error_code,
    )


def _snapshot_digest(
    snapshot: _ToolResultSnapshot | None, error_code: str | None,
) -> str:
    encoded = json.dumps(
        {"error_code": error_code, "result": snapshot},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8", errors="strict")
    return hashlib.sha256(encoded).hexdigest()


def _result_from_snapshot(snapshot: _ToolResultSnapshot | None) -> ToolResult | None:
    if snapshot is None:
        return None
    return ToolResult(
        snapshot[0], snapshot[1], ToolResultStatus(snapshot[2]),
        output_reference=snapshot[3], text_content=snapshot[4],
        safe_message=snapshot[5], error_code=snapshot[6],
    )


def _create_internal_tool_invocation_boundary(
    executor: object, *, output_store: object | None = None,
) -> TrustedToolInvocationBoundary | None:
    """Snapshot only the exact trusted concrete tool composition."""
    if type(executor) is not PolicyEnforcedToolExecutor:
        return None
    if (
        type(executor.registry) is not ToolRegistry
        or type(executor.policy_evaluator) is not DeterministicToolPolicyEvaluator
        or type(executor.tools) is not ExecutableToolRegistry
        or type(executor.registry.descriptors) is not tuple
        or type(executor.tools.registrations) is not tuple
        or type(executor.tools.descriptors) is not tuple
    ):
        return None
    try:
        descriptors = tuple(
            _copy_tool_descriptor(item) for item in executor.registry.descriptors
        )
        registry = ToolRegistry(descriptors)
        registrations = tuple(
            ExecutableRegistration(item.tool_id, item.execute_callable)
            for item in executor.tools.registrations
            if type(item) is ExecutableRegistration
            and type(item.tool_id) is str and callable(item.execute_callable)
        )
        captured_descriptors = tuple(
            _copy_tool_descriptor(item) for item in executor.tools.descriptors
        )
        if (
            len(registrations) != len(executor.tools.registrations)
            or tuple(item.tool_id for item in registrations)
            != tuple(item.tool_id for item in captured_descriptors)
            or captured_descriptors != descriptors
        ):
            return None
        boundary = object.__new__(TrustedToolInvocationBoundary)
        identity = id(boundary)

        def discard(reference: object) -> None:
            current = _TRUSTED_TOOL_BOUNDARIES.get(identity)
            if current is not None and current[0] is reference:
                _TRUSTED_TOOL_BOUNDARIES.pop(identity, None)
                _TRUSTED_ARTIFACT_ROOTS.pop(identity, None)

        evaluator = DeterministicToolPolicyEvaluator()
        _TRUSTED_TOOL_BOUNDARIES[identity] = (
            weakref.ref(boundary, discard), descriptors, registrations,
            evaluator.evaluate, output_store,
        )
        try:
            from sovereign_api.workspace_write_artifact import (
                WORKSPACE_WRITE_ARTIFACT_TOOL_ID, WorkspaceWriteArtifactTool,
            )
            write_registration = next(
                (item for item in registrations
                 if item.tool_id == WORKSPACE_WRITE_ARTIFACT_TOOL_ID), None,
            )
            implementation = (
                write_registration.execute_callable.__self__
                if write_registration is not None else None
            )
            if type(implementation) is WorkspaceWriteArtifactTool:
                _TRUSTED_ARTIFACT_ROOTS[identity] = (
                    weakref.ref(boundary, discard),
                    WORKSPACE_WRITE_ARTIFACT_TOOL_ID,
                    implementation.artifact_root,
                )
        except Exception:
            pass
        return boundary
    except Exception:
        return None


def _trusted_tool_invocation_outcome(
    receipt: object, *, task_id: str, stage_id: str, tool_id: str,
    attempt_id: str, idempotency_key: str, lease_id: str,
    claimed_version: int, consume: bool,
) -> TrustedToolInvocationOutcome:
    if type(receipt) is not TrustedToolInvocationReceipt or type(consume) is not bool:
        raise ToolExecutionIdentityError("Tool invocation receipt is invalid")
    stored = _TRUSTED_TOOL_RECEIPTS.get(id(receipt))
    expected = (
        task_id, stage_id, tool_id, attempt_id, idempotency_key,
        lease_id, claimed_version,
    )
    visible = (
        receipt.task_id, receipt.stage_id, receipt.tool_id, receipt.attempt_id,
        receipt.idempotency_key, receipt.lease_id, receipt.claimed_version,
    )
    if (
        stored is None or stored[0]() is not receipt
        or stored[1:8] != expected or visible != expected
        or not hmac.compare_digest(
            stored[9], _snapshot_digest(stored[8], stored[10]),
        )
    ):
        raise ToolExecutionIdentityError("Tool invocation receipt is invalid")
    if consume:
        _TRUSTED_TOOL_RECEIPTS.pop(id(receipt), None)
    return TrustedToolInvocationOutcome(
        _result_from_snapshot(stored[8]), stored[10], stored[11], stored[12],
    )


def _consume_trusted_tool_invocation_receipt(
    receipt: object, *, task_id: str, stage_id: str, tool_id: str,
    attempt_id: str, idempotency_key: str, lease_id: str,
    claimed_version: int,
) -> TrustedToolInvocationOutcome:
    return _trusted_tool_invocation_outcome(
        receipt, task_id=task_id, stage_id=stage_id, tool_id=tool_id,
        attempt_id=attempt_id, idempotency_key=idempotency_key,
        lease_id=lease_id, claimed_version=claimed_version, consume=True,
    )


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
        return await self._invoke(request, descriptor)

    async def _invoke(
        self, request: ToolRequest, descriptor: ToolDescriptor,
    ) -> ToolResult:
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
