"""Artifact writes remain bounded, confined, atomic, and permission-gated."""

import asyncio
import errno
import os
from pathlib import Path
import socket

import pytest

from sovereign_api.config import DeploymentEnvironment
from sovereign_api.tool_contracts import ToolRequest, ToolResultStatus
from sovereign_api.tool_execution import (
    ExecutableToolRegistry,
    PolicyEnforcedToolExecutor,
    ToolApprovalRequiredError,
    ToolPermissionDeniedError,
)
from sovereign_api.tool_policy import DeterministicToolPolicyEvaluator, ToolPermissionDecision
from sovereign_api.workspace_read_file import (
    FILESYSTEM_READ_PERMISSION,
    WORKSPACE_READ_FILE_TOOL_ID,
)
from sovereign_api.workspace_write_artifact import (
    ARTIFACT_ROOT_ENV,
    ARTIFACT_WRITE_PERMISSION,
    MAX_ARTIFACT_BYTES,
    WORKSPACE_ARTIFACT_TOOL_REGISTRY,
    WORKSPACE_WRITE_ARTIFACT_DESCRIPTOR,
    WORKSPACE_WRITE_ARTIFACT_TOOL_ID,
    ArtifactAbsolutePathError,
    ArtifactDirectorySecurityError,
    ArtifactFileAccessError,
    ArtifactInvalidTargetError,
    ArtifactParentNotFoundError,
    ArtifactPathOutsideRootError,
    ArtifactRootNotConfiguredError,
    ArtifactSafeOpenUnsupportedError,
    ArtifactSymlinkError,
    ArtifactTooLargeError,
    InvalidArtifactArgumentsError,
    InvalidArtifactRootError,
    WorkspaceWriteArtifactTool,
    create_workspace_artifact_tool_executor,
)


def _request(path: object = "output.txt", content: object = "artifact", **extra: object) -> ToolRequest:
    return ToolRequest(
        "request-1", WORKSPACE_WRITE_ARTIFACT_TOOL_ID, "write_artifact",
        {"path": path, "content": content, **extra}, "task-1", "stage-1",
    )


def _execute(
    root: Path,
    request: ToolRequest | None = None,
    *,
    permissions: frozenset = frozenset({ARTIFACT_WRITE_PERMISSION}),
    environment: DeploymentEnvironment = DeploymentEnvironment.DEVELOPMENT,
):
    executor = create_workspace_artifact_tool_executor(root, root)
    return asyncio.run(
        executor.execute(
            request if request is not None else _request(),
            granted_permissions=permissions,
            environment=environment,
        )
    )


def test_valid_explicit_root_and_read_write_registration(tmp_path: Path) -> None:
    tool = WorkspaceWriteArtifactTool.from_environment({ARTIFACT_ROOT_ENV: str(tmp_path)})
    executor = create_workspace_artifact_tool_executor(tmp_path, tmp_path)
    assert tool.artifact_root == tmp_path
    assert executor.registry.descriptors == WORKSPACE_ARTIFACT_TOOL_REGISTRY.descriptors
    assert {registration.tool_id for registration in executor.tools.registrations} == {
        WORKSPACE_READ_FILE_TOOL_ID, WORKSPACE_WRITE_ARTIFACT_TOOL_ID,
    }
    assert tool.descriptor == WORKSPACE_WRITE_ARTIFACT_DESCRIPTOR


def test_missing_and_invalid_roots_are_typed(tmp_path: Path) -> None:
    with pytest.raises(ArtifactRootNotConfiguredError):
        WorkspaceWriteArtifactTool.from_environment({})
    with pytest.raises(InvalidArtifactRootError):
        WorkspaceWriteArtifactTool(tmp_path / "missing")
    file_root = tmp_path / "file.txt"
    file_root.write_text("not a directory", encoding="utf-8")
    with pytest.raises(InvalidArtifactRootError):
        WorkspaceWriteArtifactTool(file_root)


@pytest.mark.parametrize("mode", [0o777, 0o770])
def test_shared_writable_artifact_root_is_rejected(tmp_path: Path, mode: int) -> None:
    original_mode = tmp_path.stat().st_mode & 0o777
    try:
        tmp_path.chmod(mode)
        with pytest.raises(ArtifactDirectorySecurityError) as captured:
            WorkspaceWriteArtifactTool(tmp_path)
        assert str(tmp_path) not in str(captured.value)
        assert not (tmp_path / "output.txt").exists()
    finally:
        tmp_path.chmod(original_mode)


@pytest.mark.parametrize("mode", [0o777, 0o770])
def test_shared_writable_nested_parent_is_rejected(tmp_path: Path, mode: int) -> None:
    parent = tmp_path / "reports"
    parent.mkdir()
    try:
        parent.chmod(mode)
        with pytest.raises(ArtifactDirectorySecurityError) as captured:
            _execute(tmp_path, _request("reports/output.txt"))
        assert str(parent) not in str(captured.value)
        assert tuple(parent.iterdir()) == ()
    finally:
        parent.chmod(0o700)


def test_private_same_user_root_and_nested_parent_are_accepted(tmp_path: Path) -> None:
    parent = tmp_path / "reports"
    parent.mkdir(mode=0o700)
    assert tmp_path.stat().st_uid == os.geteuid()
    assert parent.stat().st_uid == os.geteuid()
    assert _execute(tmp_path, _request("reports/output.txt")).status is ToolResultStatus.SUCCEEDED
    assert (parent / "output.txt").read_text(encoding="utf-8") == "artifact"


@pytest.mark.skipif(os.name != "posix" or not Path("/tmp").is_dir(), reason="POSIX /tmp unavailable")
def test_shared_tmp_cannot_be_artifact_root() -> None:
    if not (Path("/tmp").stat().st_mode & 0o002):
        pytest.skip("/tmp is not world-writable on this host")
    with pytest.raises(ArtifactDirectorySecurityError):
        WorkspaceWriteArtifactTool("/tmp")


def test_wrong_owner_artifact_root_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    actual_uid = os.geteuid()
    monkeypatch.setattr("sovereign_api.workspace_write_artifact.os.geteuid", lambda: actual_uid + 1)
    with pytest.raises(ArtifactDirectorySecurityError):
        WorkspaceWriteArtifactTool(tmp_path)


def test_root_permission_change_after_construction_is_rechecked(tmp_path: Path) -> None:
    tool = WorkspaceWriteArtifactTool(tmp_path)
    original_mode = tmp_path.stat().st_mode & 0o777
    try:
        tmp_path.chmod(0o770)
        with pytest.raises(ArtifactDirectorySecurityError):
            asyncio.run(tool.execute(_request()))
        assert not (tmp_path / "output.txt").exists()
    finally:
        tmp_path.chmod(original_mode)


@pytest.mark.parametrize("suffix", ["\x00bad", "\nbad", "\rbad"])
def test_malformed_root_is_typed(tmp_path: Path, suffix: str) -> None:
    with pytest.raises(InvalidArtifactRootError):
        WorkspaceWriteArtifactTool(str(tmp_path) + suffix)


def test_root_symlink_is_rejected(tmp_path: Path) -> None:
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(InvalidArtifactRootError):
        WorkspaceWriteArtifactTool(alias)


def test_create_nested_and_replace_return_relative_reference(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    request = _request("reports/output.txt", "Résumé ✓")
    first = _execute(tmp_path, request)
    second = _execute(tmp_path, request)
    assert first == second
    assert first.status is ToolResultStatus.SUCCEEDED
    assert first.output_reference == "reports/output.txt"
    assert first.text_content is None
    assert str(tmp_path) not in first.safe_message
    target = reports / "output.txt"
    assert target.read_text(encoding="utf-8") == "Résumé ✓"
    old_inode = target.stat().st_ino
    _execute(tmp_path, _request("reports/output.txt", "replacement"))
    assert target.read_text(encoding="utf-8") == "replacement"
    assert target.stat().st_ino != old_inode
    assert {item.name for item in reports.iterdir()} == {"output.txt"}


@pytest.mark.parametrize("path", ["../outside.txt", "nested/../../outside.txt", "./a", "sub//a", "a/../b"])
def test_traversal_and_aliases_are_rejected(tmp_path: Path, path: str) -> None:
    with pytest.raises(ArtifactPathOutsideRootError):
        _execute(tmp_path, _request(path))


@pytest.mark.parametrize("path", ["/tmp/outside.txt", "C:\\outside.txt", "C:outside.txt"])
def test_absolute_and_drive_paths_are_rejected(tmp_path: Path, path: str) -> None:
    with pytest.raises(ArtifactAbsolutePathError):
        _execute(tmp_path, _request(path))


def test_missing_parent_is_not_created(tmp_path: Path) -> None:
    with pytest.raises(ArtifactParentNotFoundError):
        _execute(tmp_path, _request("missing/output.txt"))
    assert not (tmp_path / "missing").exists()


@pytest.mark.parametrize("external", [False, True])
def test_final_symlink_is_rejected(tmp_path: Path, external: bool) -> None:
    target = (tmp_path.parent if external else tmp_path) / "target.txt"
    target.write_text("keep", encoding="utf-8")
    (tmp_path / "output.txt").symlink_to(target)
    with pytest.raises(ArtifactSymlinkError):
        _execute(tmp_path)
    assert target.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("external", [False, True])
def test_intermediate_symlink_is_rejected(tmp_path: Path, external: bool) -> None:
    actual = (tmp_path.parent if external else tmp_path) / "real-dir"
    actual.mkdir(exist_ok=True)
    (tmp_path / "alias").symlink_to(actual, target_is_directory=True)
    with pytest.raises(ArtifactSymlinkError):
        _execute(tmp_path, _request("alias/output.txt"))
    assert not (actual / "output.txt").exists()


def test_parent_replacement_cannot_redirect_writer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parent = tmp_path / "reports"
    parent.mkdir()
    outside = tmp_path.parent / "outside-writer"
    outside.mkdir(exist_ok=True)
    executor = create_workspace_artifact_tool_executor(tmp_path, tmp_path)
    real_open = os.open
    replaced = False

    def replace_then_open(path: object, flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
        nonlocal replaced
        if path == "reports" and dir_fd is not None and not replaced:
            parent.rmdir()
            parent.symlink_to(outside, target_is_directory=True)
            replaced = True
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr("sovereign_api.workspace_write_artifact.os.open", replace_then_open)
    monkeypatch.setattr(
        "sovereign_api.workspace_write_artifact.os.supports_dir_fd",
        set(os.supports_dir_fd) | {replace_then_open},
    )
    with pytest.raises(ArtifactSymlinkError):
        asyncio.run(executor.execute(
            _request("reports/output.txt"),
            granted_permissions=frozenset({ARTIFACT_WRITE_PERMISSION}),
            environment=DeploymentEnvironment.DEVELOPMENT,
        ))
    assert replaced
    assert not (outside / "output.txt").exists()


@pytest.mark.parametrize("kind", ["directory", "fifo", "socket"])
def test_special_targets_are_rejected(tmp_path: Path, kind: str) -> None:
    target = tmp_path / "output.txt"
    if kind == "directory":
        target.mkdir()
    elif kind == "fifo":
        if not hasattr(os, "mkfifo"):
            pytest.skip("FIFO unavailable")
        os.mkfifo(target)
    else:
        if not hasattr(socket, "AF_UNIX"):
            pytest.skip("Unix sockets unavailable")
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            server.bind(str(target))
        except PermissionError:
            server.close()
            pytest.skip("Unix socket creation is not permitted")
    try:
        with pytest.raises(ArtifactInvalidTargetError):
            _execute(tmp_path)
    finally:
        if kind == "socket":
            server.close()


@pytest.mark.parametrize("size,accepted", [
    (MAX_ARTIFACT_BYTES - 1, True),
    (MAX_ARTIFACT_BYTES, True),
    (MAX_ARTIFACT_BYTES + 1, False),
])
def test_encoded_size_limit(tmp_path: Path, size: int, accepted: bool) -> None:
    request = _request(content="a" * size)
    if accepted:
        assert _execute(tmp_path, request).status is ToolResultStatus.SUCCEEDED
        assert (tmp_path / "output.txt").stat().st_size == size
    else:
        with pytest.raises(ArtifactTooLargeError):
            _execute(tmp_path, request)
        assert not (tmp_path / "output.txt").exists()


def test_multibyte_size_is_measured_after_encoding(tmp_path: Path) -> None:
    with pytest.raises(ArtifactTooLargeError):
        _execute(tmp_path, _request(content="€" * (MAX_ARTIFACT_BYTES // 3 + 1)))
    assert not (tmp_path / "output.txt").exists()


@pytest.mark.parametrize("path,content,extra", [
    (None, "text", {}), ("", "text", {}), ("output.txt", None, {}),
    ("output.txt", 123, {}), ("output.txt", "text", {"mode": "append"}),
])
def test_arguments_are_closed_and_typed(tmp_path: Path, path: object, content: object, extra: dict) -> None:
    with pytest.raises(InvalidArtifactArgumentsError):
        _execute(tmp_path, _request(path, content, **extra))


def test_missing_required_argument_is_rejected(tmp_path: Path) -> None:
    request = ToolRequest(
        "request-1", WORKSPACE_WRITE_ARTIFACT_TOOL_ID, "write_artifact",
        {"path": "output.txt"}, "task-1", "stage-1",
    )
    with pytest.raises(InvalidArtifactArgumentsError):
        _execute(tmp_path, request)


@pytest.mark.parametrize("environment", list(DeploymentEnvironment))
def test_environment_does_not_grant_permission(tmp_path: Path, environment: DeploymentEnvironment) -> None:
    before = tuple(tmp_path.iterdir())
    with pytest.raises(ToolPermissionDeniedError):
        _execute(tmp_path, permissions=frozenset(), environment=environment)
    assert tuple(tmp_path.iterdir()) == before


@pytest.mark.parametrize("environment", list(DeploymentEnvironment))
def test_explicit_permission_allows_write(tmp_path: Path, environment: DeploymentEnvironment) -> None:
    assert _execute(tmp_path, environment=environment).output_reference == "output.txt"
    assert (tmp_path / "output.txt").read_text(encoding="utf-8") == "artifact"


def test_approval_required_does_not_create_target_or_temporary_file(tmp_path: Path) -> None:
    class ApprovalPolicy:
        def evaluate(self, *args: object, **kwargs: object) -> ToolPermissionDecision:
            return ToolPermissionDecision.REQUIRE_APPROVAL

    executor = PolicyEnforcedToolExecutor(
        WORKSPACE_ARTIFACT_TOOL_REGISTRY,
        ApprovalPolicy(),
        ExecutableToolRegistry((WorkspaceWriteArtifactTool(tmp_path),)),
    )
    before = tuple(tmp_path.iterdir())
    with pytest.raises(ToolApprovalRequiredError):
        asyncio.run(executor.execute(
            _request(), granted_permissions=frozenset({ARTIFACT_WRITE_PERMISSION}),
            environment=DeploymentEnvironment.DEVELOPMENT,
        ))
    assert tuple(tmp_path.iterdir()) == before


def test_failed_write_preserves_old_target_and_cleans_temp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "output.txt"
    target.write_text("old", encoding="utf-8")

    def fail_write(file_descriptor: int, content: bytes) -> int:
        raise OSError(errno.EIO, "/private/host/path")

    monkeypatch.setattr("sovereign_api.workspace_write_artifact.os.write", fail_write)
    with pytest.raises(ArtifactFileAccessError) as captured:
        _execute(tmp_path)
    assert "/private/host/path" not in str(captured.value)
    assert target.read_text(encoding="utf-8") == "old"
    assert {item.name for item in tmp_path.iterdir()} == {"output.txt"}


def test_target_stays_old_until_atomic_rename(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "output.txt"
    target.write_text("old", encoding="utf-8")
    real_write = os.write
    observed = False

    def observe_write(file_descriptor: int, content: bytes) -> int:
        nonlocal observed
        observed = True
        assert target.read_text(encoding="utf-8") == "old"
        return real_write(file_descriptor, content)

    monkeypatch.setattr("sovereign_api.workspace_write_artifact.os.write", observe_write)
    _execute(tmp_path, _request(content="new"))
    assert observed
    assert target.read_text(encoding="utf-8") == "new"


def test_target_symlink_swap_during_write_fails_without_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "output.txt"
    target.write_text("old", encoding="utf-8")
    outside = tmp_path.parent / "outside-target.txt"
    outside.write_text("outside", encoding="utf-8")
    real_write = os.write
    swapped = False

    def swap_target(file_descriptor: int, content: bytes) -> int:
        nonlocal swapped
        if not swapped:
            target.unlink()
            target.symlink_to(outside)
            swapped = True
        return real_write(file_descriptor, content)

    monkeypatch.setattr("sovereign_api.workspace_write_artifact.os.write", swap_target)
    with pytest.raises(ArtifactSymlinkError):
        _execute(tmp_path, _request(content="new"))
    assert swapped
    assert target.is_symlink()
    assert outside.read_text(encoding="utf-8") == "outside"
    assert {item.name for item in tmp_path.iterdir()} == {"output.txt"}


def test_temp_name_collision_never_overwrites_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    colliding = tmp_path / (".artifact-tmp-" + "a" * 32)
    colliding.write_text("untouched", encoding="utf-8")
    names = iter(("a" * 32, "b" * 32))
    monkeypatch.setattr(
        "sovereign_api.workspace_write_artifact.secrets.token_hex",
        lambda size: next(names),
    )
    assert _execute(tmp_path).output_reference == "output.txt"
    assert colliding.read_text(encoding="utf-8") == "untouched"
    assert {item.name for item in tmp_path.iterdir()} == {
        colliding.name, "output.txt",
    }


def test_reader_and_writer_remain_separate_roots(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    artifact = tmp_path / "artifacts"
    workspace.mkdir()
    artifact.mkdir()
    (workspace / "input.txt").write_text("read me", encoding="utf-8")
    executor = create_workspace_artifact_tool_executor(workspace, artifact)
    write_result = asyncio.run(executor.execute(
        _request("output.txt", "written"),
        granted_permissions=frozenset({ARTIFACT_WRITE_PERMISSION}),
        environment=DeploymentEnvironment.DEVELOPMENT,
    ))
    read_request = ToolRequest(
        "request-2", WORKSPACE_READ_FILE_TOOL_ID, "read_file",
        {"path": "input.txt"}, "task-1", "stage-1",
    )
    read_result = asyncio.run(executor.execute(
        read_request,
        granted_permissions=frozenset({FILESYSTEM_READ_PERMISSION}),
        environment=DeploymentEnvironment.DEVELOPMENT,
    ))
    assert write_result.output_reference == "output.txt"
    assert read_result.text_content == "read me"
    assert (artifact / "output.txt").read_text(encoding="utf-8") == "written"
    assert not (workspace / "output.txt").exists()


def test_missing_safe_platform_primitive_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sovereign_api.workspace_write_artifact.os.supports_dir_fd",
        set(os.supports_dir_fd) - {os.rename},
    )
    with pytest.raises(ArtifactSafeOpenUnsupportedError):
        _execute(tmp_path)
    assert not (tmp_path / "output.txt").exists()
