"""Trusted registration, policy-first execution, and fail-closed tool outcomes."""

import asyncio
from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path
import traceback

import pytest

from sovereign_api.config import DeploymentEnvironment
from sovereign_api.tool_contracts import (
    ToolDescriptor,
    ToolPermission,
    ToolRegistry,
    ToolRequest,
    ToolResult,
    ToolResultStatus,
    ToolRiskLevel,
    ToolSideEffectLevel,
    UnknownToolError,
)
from sovereign_api.tool_execution import (
    ExecutableRegistration,
    ExecutableToolRegistry,
    PolicyEnforcedToolExecutor,
    ToolApprovalRequiredError,
    ToolExecutionIdentityError,
    ToolImplementationUnavailableError,
    ToolInvocationError,
    ToolPermissionDeniedError,
)
from sovereign_api.tool_policy import DeterministicToolPolicyEvaluator
from sovereign_api.workspace_read_file import (
    FILESYSTEM_READ_PERMISSION,
    WORKSPACE_READ_FILE_DESCRIPTOR,
    WORKSPACE_READ_FILE_TOOL_ID,
    WORKSPACE_TOOL_REGISTRY,
    WorkspaceFileNotFoundError,
    WorkspaceReadFileTool,
    create_workspace_tool_executor,
)


def _request(tool_id: str = WORKSPACE_READ_FILE_TOOL_ID, path: str = "safe.txt") -> ToolRequest:
    return ToolRequest(
        "request-1", tool_id, "read_file", {"path": path}, "task-1", "stage-1"
    )


def _run(
    executor: PolicyEnforcedToolExecutor,
    request: ToolRequest | None = None,
    *,
    permissions: frozenset[ToolPermission] = frozenset({FILESYSTEM_READ_PERMISSION}),
    environment: DeploymentEnvironment = DeploymentEnvironment.DEVELOPMENT,
) -> ToolResult:
    return asyncio.run(
        executor.execute(
            request or _request(),
            granted_permissions=permissions,
            environment=environment,
        )
    )


@dataclass
class RecordingTool:
    descriptor: ToolDescriptor = WORKSPACE_READ_FILE_DESCRIPTOR
    calls: int = 0
    override_id: str | None = None
    raw_failure: bool = False
    wrong_result_id: bool = False

    @property
    def tool_id(self) -> str:
        return self.override_id or self.descriptor.tool_id

    async def execute(self, request: ToolRequest) -> ToolResult:
        self.calls += 1
        if self.raw_failure:
            raise OSError("/very/secret/host/path")
        return ToolResult(
            request.request_id,
            "fixture.wrong" if self.wrong_result_id else request.tool_id,
            ToolResultStatus.SUCCEEDED,
            text_content="fixture result",
        )


def _recording_executor(tool: RecordingTool) -> PolicyEnforcedToolExecutor:
    return PolicyEnforcedToolExecutor(
        WORKSPACE_TOOL_REGISTRY,
        DeterministicToolPolicyEvaluator(),
        ExecutableToolRegistry((tool,)),
    )


def test_explicit_workspace_registration_reads_only_with_permission(tmp_path: Path) -> None:
    (tmp_path / "safe.txt").write_text("safe", encoding="utf-8")
    executor = create_workspace_tool_executor(tmp_path)

    assert executor.registry.descriptors == (WORKSPACE_READ_FILE_DESCRIPTOR,)
    assert tuple(tool.tool_id for tool in executor.tools.registrations) == (
        WORKSPACE_READ_FILE_TOOL_ID,
    )
    assert _run(executor).text_content == "safe"
    assert _run(executor).text_content == "safe"


@pytest.mark.parametrize("environment", list(DeploymentEnvironment))
def test_missing_permission_denies_before_tool_invocation(
    environment: DeploymentEnvironment,
) -> None:
    tool = RecordingTool()
    with pytest.raises(ToolPermissionDeniedError):
        _run(_recording_executor(tool), permissions=frozenset(), environment=environment)
    assert tool.calls == 0


@pytest.mark.parametrize(
    "environment",
    [DeploymentEnvironment.ON_PREM, DeploymentEnvironment.AIR_GAPPED],
)
def test_explicit_permission_allows_local_read_in_sovereign_environments(
    tmp_path: Path, environment: DeploymentEnvironment,
) -> None:
    (tmp_path / "safe.txt").write_text("local", encoding="utf-8")
    assert _run(create_workspace_tool_executor(tmp_path), environment=environment).text_content == "local"


def test_approval_required_never_invokes_implementation() -> None:
    permission = ToolPermission("filesystem.write")
    descriptor = ToolDescriptor(
        "fixture.destructive", "Fixture", "Approval fixture", ("filesystem.write",),
        ToolRiskLevel.HIGH, ToolSideEffectLevel.DESTRUCTIVE,
        False, False, True, (permission,),
    )
    tool = RecordingTool(descriptor)
    executor = PolicyEnforcedToolExecutor(
        ToolRegistry((descriptor,)), DeterministicToolPolicyEvaluator(),
        ExecutableToolRegistry((tool,)),
    )
    request = _request(descriptor.tool_id)

    with pytest.raises(ToolApprovalRequiredError):
        _run(executor, request, permissions=frozenset({permission}))
    with pytest.raises(ToolPermissionDeniedError):
        _run(executor, request, permissions=frozenset())
    assert tool.calls == 0


def test_unknown_descriptor_and_missing_implementation_fail_closed() -> None:
    executor = PolicyEnforcedToolExecutor(
        WORKSPACE_TOOL_REGISTRY, DeterministicToolPolicyEvaluator(),
        ExecutableToolRegistry(()),
    )
    with pytest.raises(UnknownToolError):
        _run(executor, _request("fixture.unknown"))
    with pytest.raises(ToolImplementationUnavailableError):
        _run(executor)


def test_implementation_without_descriptor_is_rejected() -> None:
    with pytest.raises(ToolExecutionIdentityError):
        PolicyEnforcedToolExecutor(
            ToolRegistry(()), DeterministicToolPolicyEvaluator(),
            ExecutableToolRegistry((RecordingTool(),)),
        )


def test_duplicate_implementations_are_rejected() -> None:
    with pytest.raises(ToolExecutionIdentityError):
        ExecutableToolRegistry((RecordingTool(), RecordingTool()))


def test_implementation_registry_copies_caller_collection() -> None:
    tool = RecordingTool()
    supplied = [tool]
    registry = ExecutableToolRegistry(supplied)
    supplied.clear()
    assert len(registry.registrations) == 1
    assert type(registry.registrations[0]) is ExecutableRegistration
    assert registry.registrations[0].tool_id == tool.tool_id


def test_registration_and_descriptor_identity_mismatches_are_rejected() -> None:
    tool = RecordingTool()
    with pytest.raises(ToolExecutionIdentityError):
        PolicyEnforcedToolExecutor(
            WORKSPACE_TOOL_REGISTRY, DeterministicToolPolicyEvaluator(),
            {"fixture.wrong": tool},
        )
    tool.override_id = "fixture.wrong"
    with pytest.raises(ToolExecutionIdentityError):
        _recording_executor(tool)


def test_captured_registration_identity_survives_instance_change() -> None:
    tool = RecordingTool()
    executor = _recording_executor(tool)
    tool.override_id = "fixture.changed"
    assert _run(executor).tool_id == WORKSPACE_READ_FILE_TOOL_ID
    assert executor.tools.registrations[0].tool_id == WORKSPACE_READ_FILE_TOOL_ID
    assert tool.calls == 1


def test_captured_descriptor_survives_instance_change() -> None:
    tool = RecordingTool()
    executor = _recording_executor(tool)
    tool.descriptor = ToolDescriptor(
        "fixture.other", "Fixture", "Different descriptor", ("filesystem.read",),
        ToolRiskLevel.LOW, ToolSideEffectLevel.READ,
        False, True, False, (FILESYSTEM_READ_PERMISSION,),
    )
    assert _run(executor).tool_id == WORKSPACE_READ_FILE_TOOL_ID
    assert executor.tools.descriptors == (WORKSPACE_READ_FILE_DESCRIPTOR,)
    assert tool.calls == 1


def test_replacing_execute_after_registration_uses_original_callable() -> None:
    tool = RecordingTool()
    executor = _recording_executor(tool)
    expected = ToolResult(
        "request-1", WORKSPACE_READ_FILE_TOOL_ID, ToolResultStatus.SUCCEEDED,
        text_content="fixture result",
    )
    replacement_called = False

    async def replacement(request: ToolRequest) -> ToolResult:
        nonlocal replacement_called
        replacement_called = True
        return ToolResult(
            request.request_id, request.tool_id, ToolResultStatus.SUCCEEDED,
            text_content="replacement result",
        )

    tool.execute = replacement
    assert _run(executor) == expected
    assert tool.calls == 1
    assert not replacement_called
    with pytest.raises(FrozenInstanceError):
        executor.tools.registrations[0].execute_callable = replacement


def test_workspace_security_configuration_is_immutable(tmp_path: Path) -> None:
    (tmp_path / "safe.txt").write_text("safe", encoding="utf-8")
    tool = WorkspaceReadFileTool(tmp_path)
    executor = PolicyEnforcedToolExecutor(
        WORKSPACE_TOOL_REGISTRY, DeterministicToolPolicyEvaluator(),
        ExecutableToolRegistry((tool,)),
    )
    for name, value in (
        ("workspace_root", Path("/")),
        ("descriptor", None),
        ("tool_id", "fixture.other"),
        ("size_limit", 100_000_000),
    ):
        with pytest.raises((FrozenInstanceError, TypeError, AttributeError)):
            setattr(tool, name, value)
    assert not hasattr(tool, "__dict__")
    assert tool.workspace_root == tmp_path
    assert _run(executor).text_content == "safe"


def test_mismatched_result_identity_is_rejected() -> None:
    tool = RecordingTool(wrong_result_id=True)
    with pytest.raises(ToolExecutionIdentityError):
        _run(_recording_executor(tool))
    assert tool.calls == 1


def test_request_arguments_cannot_redirect_tool_identity() -> None:
    tool = RecordingTool()
    request = ToolRequest(
        "request-1", WORKSPACE_READ_FILE_TOOL_ID, "read_file",
        {"path": "safe.txt", "tool_id": "fixture.other"}, "task-1", "stage-1",
    )
    assert _run(_recording_executor(tool), request).text_content == "fixture result"
    assert tool.calls == 1


def test_workspace_error_remains_typed_and_safe(tmp_path: Path) -> None:
    with pytest.raises(WorkspaceFileNotFoundError) as captured:
        _run(create_workspace_tool_executor(tmp_path))
    assert str(tmp_path) not in str(captured.value)


def test_unexpected_tool_failure_is_sanitized() -> None:
    tool = RecordingTool(raw_failure=True)
    with pytest.raises(ToolInvocationError) as captured:
        _run(_recording_executor(tool))
    assert "/very/secret/host/path" not in str(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert "/very/secret/host/path" not in "".join(
        traceback.format_exception(captured.value)
    )
    assert tool.calls == 1


def test_unexpected_policy_failure_denies_without_invocation() -> None:
    class BrokenPolicy:
        def evaluate(self, *args: object, **kwargs: object) -> object:
            raise RuntimeError("raw private policy diagnostic")

    tool = RecordingTool()
    executor = PolicyEnforcedToolExecutor(
        WORKSPACE_TOOL_REGISTRY, BrokenPolicy(), ExecutableToolRegistry((tool,))
    )
    with pytest.raises(ToolPermissionDeniedError) as captured:
        _run(executor)
    assert "private" not in str(captured.value)
    assert tool.calls == 0


def test_legacy_tool_result_without_output_remains_valid() -> None:
    assert ToolResult("request-1", "tool-1", ToolResultStatus.SUCCEEDED).status is ToolResultStatus.SUCCEEDED
