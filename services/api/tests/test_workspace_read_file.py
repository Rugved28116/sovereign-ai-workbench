import asyncio
from dataclasses import dataclass
import errno
import os
from pathlib import Path
import socket
from types import SimpleNamespace

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
    ToolValidationError,
)
from sovereign_api.tool_execution import (
    PolicyEnforcedToolExecutor,
    ToolApprovalRequiredError,
    ToolPermissionDeniedError,
)
from sovereign_api.tool_policy import DeterministicToolPolicyEvaluator
from sovereign_api.workspace_read_file import (
    FILESYSTEM_READ_PERMISSION,
    MAX_WORKSPACE_FILE_BYTES,
    WORKSPACE_READ_FILE_DESCRIPTOR,
    WORKSPACE_READ_FILE_TOOL_ID,
    WORKSPACE_READ_OPERATION,
    WORKSPACE_ROOT_ENV,
    WORKSPACE_TOOL_REGISTRY,
    AbsoluteWorkspacePathError,
    InvalidWorkspaceReadArgumentsError,
    InvalidWorkspaceRootError,
    WorkspaceFileAccessError,
    WorkspaceFileNotFoundError,
    WorkspaceFileTooLargeError,
    WorkspaceInvalidUTF8Error,
    WorkspaceNotRegularFileError,
    WorkspacePathOutsideRootError,
    WorkspaceReadFileTool,
    WorkspaceRootNotConfiguredError,
    WorkspaceSafeOpenUnsupportedError,
)


def _request(path: object, **extra: object) -> ToolRequest:
    return ToolRequest(
        request_id="request-1",
        tool_id=WORKSPACE_READ_FILE_TOOL_ID,
        operation=WORKSPACE_READ_OPERATION,
        arguments={"path": path, **extra},
        task_id="task-1",
        stage_id="stage-1",
    )


def _executor(root: Path) -> PolicyEnforcedToolExecutor:
    tool = WorkspaceReadFileTool(root)
    return PolicyEnforcedToolExecutor(
        registry=WORKSPACE_TOOL_REGISTRY,
        policy_evaluator=DeterministicToolPolicyEvaluator(),
        tools={WORKSPACE_READ_FILE_TOOL_ID: tool},
    )


def _execute(
    root: Path,
    path: object,
    *,
    permissions: frozenset[ToolPermission] = frozenset(
        {FILESYSTEM_READ_PERMISSION}
    ),
    environment: DeploymentEnvironment = DeploymentEnvironment.DEVELOPMENT,
) -> ToolResult:
    return asyncio.run(
        _executor(root).execute(
            _request(path),
            granted_permissions=permissions,
            environment=environment,
        )
    )


def _execute_with_tool(tool: WorkspaceReadFileTool, path: object) -> ToolResult:
    executor = PolicyEnforcedToolExecutor(
        registry=WORKSPACE_TOOL_REGISTRY,
        policy_evaluator=DeterministicToolPolicyEvaluator(),
        tools={WORKSPACE_READ_FILE_TOOL_ID: tool},
    )
    return asyncio.run(
        executor.execute(
            _request(path),
            granted_permissions=frozenset({FILESYSTEM_READ_PERMISSION}),
            environment=DeploymentEnvironment.DEVELOPMENT,
        )
    )


def test_explicit_existing_directory_is_canonicalized(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()

    tool = WorkspaceReadFileTool.from_environment(
        {WORKSPACE_ROOT_ENV: str(root / ".")}
    )

    assert tool.workspace_root == root.resolve()


def test_string_and_path_workspace_roots_are_accepted(tmp_path: Path) -> None:
    assert WorkspaceReadFileTool(str(tmp_path)).workspace_root == tmp_path
    assert WorkspaceReadFileTool(tmp_path).workspace_root == tmp_path


def test_missing_workspace_root_is_rejected() -> None:
    with pytest.raises(WorkspaceRootNotConfiguredError):
        WorkspaceReadFileTool.from_environment({})


@pytest.mark.parametrize("kind", ["missing", "file", "relative"])
def test_invalid_workspace_root_is_rejected(tmp_path: Path, kind: str) -> None:
    if kind == "missing":
        root = tmp_path / "missing"
    elif kind == "file":
        root = tmp_path / "file.txt"
        root.write_text("text", encoding="utf-8")
    else:
        root = Path("relative-workspace")

    with pytest.raises(InvalidWorkspaceRootError):
        WorkspaceReadFileTool(root)


def test_reads_utf8_file_and_nested_file_deterministically(tmp_path: Path) -> None:
    nested = tmp_path / "reports"
    nested.mkdir()
    target = nested / "inspection.txt"
    target.write_text("Résumé ✓\n", encoding="utf-8")

    first = _execute(tmp_path, "reports/inspection.txt")
    second = _execute(tmp_path, "reports/inspection.txt")

    assert first == second
    assert first.status is ToolResultStatus.SUCCEEDED
    assert first.text_content == "Résumé ✓\n"
    assert first.output_reference is None
    assert str(tmp_path) not in first.safe_message


def test_empty_utf8_file_is_supported(tmp_path: Path) -> None:
    (tmp_path / "empty.txt").write_bytes(b"")

    assert _execute(tmp_path, "empty.txt").text_content == ""


def test_inline_text_uses_utf8_byte_limit_and_one_output_channel() -> None:
    with pytest.raises(ToolValidationError):
        ToolResult(
            "request-1",
            WORKSPACE_READ_FILE_TOOL_ID,
            ToolResultStatus.SUCCEEDED,
            text_content="€" * (MAX_WORKSPACE_FILE_BYTES // 3 + 1),
        )
    with pytest.raises(ToolValidationError):
        ToolResult(
            "request-1",
            WORKSPACE_READ_FILE_TOOL_ID,
            ToolResultStatus.SUCCEEDED,
            output_reference="output-1",
            text_content="text",
        )


@pytest.mark.parametrize(
    "path",
    [
        "../secret.txt",
        "subdir/../../secret.txt",
        "./safe.txt",
        "subdir/../safe.txt",
        "subdir//safe.txt",
    ],
)
def test_traversal_and_ambiguous_paths_are_rejected(
    tmp_path: Path,
    path: str,
) -> None:
    with pytest.raises(WorkspacePathOutsideRootError):
        _execute(tmp_path, path)


def test_backslash_path_is_rejected_as_malformed(tmp_path: Path) -> None:
    with pytest.raises(InvalidWorkspaceReadArgumentsError):
        _execute(tmp_path, "subdir\\safe.txt")


@pytest.mark.parametrize(
    "path",
    ["/etc/passwd", "C:\\Windows\\system.ini", "C:relative.txt"],
)
def test_absolute_and_drive_paths_are_rejected(tmp_path: Path, path: str) -> None:
    with pytest.raises(AbsoluteWorkspacePathError):
        _execute(tmp_path, path)


def test_external_symlink_is_rejected(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    (workspace / "escape").symlink_to(outside)

    with pytest.raises(WorkspacePathOutsideRootError):
        _execute(workspace, "escape")


def test_nested_symlink_escape_is_rejected(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    (workspace / "nested").symlink_to(outside, target_is_directory=True)

    with pytest.raises(WorkspacePathOutsideRootError):
        _execute(workspace, "nested/secret.txt")


def test_final_symlink_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "actual.txt"
    target.write_text("inside", encoding="utf-8")
    (tmp_path / "alias.txt").symlink_to(target)

    with pytest.raises(WorkspacePathOutsideRootError):
        _execute(tmp_path, "alias.txt")


def test_workspace_root_final_component_symlink_is_rejected(tmp_path: Path) -> None:
    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "workspace"
    alias.symlink_to(actual, target_is_directory=True)

    with pytest.raises(InvalidWorkspaceRootError):
        WorkspaceReadFileTool(alias)


def test_workspace_root_ancestor_symlink_is_rejected(tmp_path: Path) -> None:
    actual_parent = tmp_path / "actual-parent"
    workspace = actual_parent / "workspace"
    workspace.mkdir(parents=True)
    alias_parent = tmp_path / "alias-parent"
    alias_parent.symlink_to(actual_parent, target_is_directory=True)

    with pytest.raises(InvalidWorkspaceRootError):
        WorkspaceReadFileTool(alias_parent / "workspace")


def test_nested_workspace_root_ancestor_symlink_is_rejected(tmp_path: Path) -> None:
    actual = tmp_path / "actual"
    workspace = actual / "nested" / "workspace"
    workspace.mkdir(parents=True)
    first = tmp_path / "first"
    first.mkdir()
    (first / "nested").symlink_to(actual / "nested", target_is_directory=True)

    with pytest.raises(InvalidWorkspaceRootError):
        WorkspaceReadFileTool(first / "nested" / "workspace")


def test_missing_file_is_a_typed_safe_error(tmp_path: Path) -> None:
    with pytest.raises(WorkspaceFileNotFoundError) as captured:
        _execute(tmp_path, "missing.txt")

    assert str(tmp_path) not in str(captured.value)


def test_directory_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "directory").mkdir()

    with pytest.raises(WorkspaceNotRegularFileError):
        _execute(tmp_path, "directory")


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO unavailable")
def test_fifo_is_rejected_without_blocking(tmp_path: Path) -> None:
    os.mkfifo(tmp_path / "pipe")

    with pytest.raises(WorkspaceNotRegularFileError):
        _execute(tmp_path, "pipe")


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="Unix sockets unavailable")
def test_socket_is_rejected(tmp_path: Path) -> None:
    socket_path = tmp_path / "socket"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(socket_path))
        with pytest.raises(WorkspaceNotRegularFileError):
            _execute(tmp_path, "socket")
    finally:
        server.close()


@pytest.mark.parametrize(
    ("size", "succeeds"),
    [
        (MAX_WORKSPACE_FILE_BYTES - 1, True),
        (MAX_WORKSPACE_FILE_BYTES, True),
        (MAX_WORKSPACE_FILE_BYTES + 1, False),
    ],
)
def test_file_size_boundary(tmp_path: Path, size: int, succeeds: bool) -> None:
    (tmp_path / "sized.txt").write_bytes(b"a" * size)

    if succeeds:
        assert len(_execute(tmp_path, "sized.txt").text_content or "") == size
    else:
        with pytest.raises(WorkspaceFileTooLargeError):
            _execute(tmp_path, "sized.txt")


def test_actual_read_is_bounded_when_metadata_understates_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "growing.txt"
    target.write_bytes(b"a" * (MAX_WORKSPACE_FILE_BYTES + 10))
    real_fstat = os.fstat
    requested_bytes = 0
    real_read = os.read

    def understated_fstat(file_descriptor: int) -> SimpleNamespace:
        result = real_fstat(file_descriptor)
        return SimpleNamespace(
            st_mode=result.st_mode,
            st_dev=result.st_dev,
            st_ino=result.st_ino,
            st_size=0,
        )

    def tracked_read(file_descriptor: int, amount: int) -> bytes:
        nonlocal requested_bytes
        requested_bytes += amount
        return real_read(file_descriptor, amount)

    monkeypatch.setattr(
        "sovereign_api.workspace_read_file.os.fstat",
        understated_fstat,
    )
    monkeypatch.setattr(
        "sovereign_api.workspace_read_file.os.read",
        tracked_read,
    )

    with pytest.raises(WorkspaceFileTooLargeError):
        _execute(tmp_path, "growing.txt")

    assert requested_bytes <= MAX_WORKSPACE_FILE_BYTES + 1


def test_invalid_utf8_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "binary.txt").write_bytes(b"valid\xffinvalid")

    with pytest.raises(WorkspaceInvalidUTF8Error):
        _execute(tmp_path, "binary.txt")


@pytest.mark.parametrize("path", [None, "", 7, True])
def test_missing_empty_or_non_string_path_is_rejected(
    tmp_path: Path,
    path: object,
) -> None:
    arguments = {} if path is None else {"path": path}
    request = ToolRequest(
        "request-1",
        WORKSPACE_READ_FILE_TOOL_ID,
        WORKSPACE_READ_OPERATION,
        arguments,
        "task-1",
        "stage-1",
    )

    with pytest.raises(InvalidWorkspaceReadArgumentsError):
        asyncio.run(WorkspaceReadFileTool(tmp_path).execute(request))


def test_unexpected_arguments_and_operation_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(InvalidWorkspaceReadArgumentsError):
        asyncio.run(
            WorkspaceReadFileTool(tmp_path).execute(_request("file.txt", mode="raw"))
        )
    invalid_operation = ToolRequest(
        "request-1",
        WORKSPACE_READ_FILE_TOOL_ID,
        "other",
        {"path": "file.txt"},
        "task-1",
        "stage-1",
    )
    with pytest.raises(InvalidWorkspaceReadArgumentsError):
        asyncio.run(WorkspaceReadFileTool(tmp_path).execute(invalid_operation))


def test_request_arguments_are_copied_before_execution(tmp_path: Path) -> None:
    (tmp_path / "safe.txt").write_text("safe", encoding="utf-8")
    arguments = {"path": "safe.txt"}
    request = ToolRequest(
        "request-1",
        WORKSPACE_READ_FILE_TOOL_ID,
        WORKSPACE_READ_OPERATION,
        arguments,
        "task-1",
        "stage-1",
    )
    arguments["path"] = "other.txt"

    result = asyncio.run(
        _executor(tmp_path).execute(
            request,
            granted_permissions=frozenset({FILESYSTEM_READ_PERMISSION}),
            environment=DeploymentEnvironment.DEVELOPMENT,
        )
    )

    assert result.text_content == "safe"


def test_permission_denial_precedes_any_file_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "safe.txt").write_text("safe", encoding="utf-8")
    accessed = False

    def unexpected_access(*args: object) -> None:
        nonlocal accessed
        accessed = True
        raise AssertionError("filesystem was accessed")

    monkeypatch.setattr(
        "sovereign_api.workspace_read_file._open_relative_components",
        unexpected_access,
    )

    with pytest.raises(ToolPermissionDeniedError):
        _execute(tmp_path, "safe.txt", permissions=frozenset())

    assert not accessed


def test_ancestor_replacement_cannot_redirect_descriptor_relative_walk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    nested = workspace / "nested"
    nested.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    real_open = os.open
    replaced = False

    def replace_ancestor_then_open(
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal replaced
        if dir_fd is not None and path == "nested" and not replaced:
            nested.rmdir()
            nested.symlink_to(outside, target_is_directory=True)
            replaced = True
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr("sovereign_api.workspace_read_file.os.open", replace_ancestor_then_open)
    monkeypatch.setattr(
        "sovereign_api.workspace_read_file.os.supports_dir_fd",
        set(os.supports_dir_fd) | {replace_ancestor_then_open},
    )

    with pytest.raises(WorkspacePathOutsideRootError):
        _execute(workspace, "nested/secret.txt")
    assert replaced


def test_workspace_root_ancestor_replacement_cannot_redirect_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ancestor = tmp_path / "workspace-parent"
    workspace = ancestor / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "secret.txt").write_text("inside", encoding="utf-8")
    outside_parent = tmp_path / "outside-parent"
    outside_workspace = outside_parent / "workspace"
    outside_workspace.mkdir(parents=True)
    (outside_workspace / "secret.txt").write_text("outside", encoding="utf-8")
    saved_ancestor = tmp_path / "saved-workspace-parent"
    tool = WorkspaceReadFileTool(workspace)
    real_open = os.open
    replaced = False

    def replace_root_ancestor_then_open(
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal replaced
        if dir_fd is not None and path == ancestor.name and not replaced:
            ancestor.rename(saved_ancestor)
            ancestor.symlink_to(outside_parent, target_is_directory=True)
            replaced = True
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(
        "sovereign_api.workspace_read_file.os.open",
        replace_root_ancestor_then_open,
    )
    monkeypatch.setattr(
        "sovereign_api.workspace_read_file.os.supports_dir_fd",
        set(os.supports_dir_fd) | {replace_root_ancestor_then_open},
    )

    with pytest.raises(InvalidWorkspaceRootError):
        _execute_with_tool(tool, "secret.txt")
    assert replaced


def test_all_file_components_are_opened_relative_to_verified_directory_handles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "safe.txt").write_text("safe", encoding="utf-8")
    real_open = os.open
    relative_paths: list[object] = []

    def inspect_open(
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if dir_fd is not None:
            relative_paths.append(path)
            assert not Path(path).is_absolute()
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr("sovereign_api.workspace_read_file.os.open", inspect_open)
    monkeypatch.setattr(
        "sovereign_api.workspace_read_file.os.supports_dir_fd",
        set(os.supports_dir_fd) | {inspect_open},
    )

    assert _execute(tmp_path, "nested/safe.txt").text_content == "safe"
    assert relative_paths[-2:] == ["nested", "safe.txt"]
    assert all(not Path(path).is_absolute() for path in relative_paths)


def test_missing_safe_descriptor_capability_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr("sovereign_api.workspace_read_file.os.O_NOFOLLOW", raising=False)
    opened = False
    real_open = os.open

    def unexpected_open(*args: object, **kwargs: object) -> int:
        nonlocal opened
        opened = True
        return real_open(*args, **kwargs)

    monkeypatch.setattr("sovereign_api.workspace_read_file.os.open", unexpected_open)
    (tmp_path / "safe.txt").write_text("safe", encoding="utf-8")

    with pytest.raises(WorkspaceSafeOpenUnsupportedError):
        _execute(tmp_path, "safe.txt")
    assert not opened


def test_missing_directory_flag_fails_closed_before_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr("sovereign_api.workspace_read_file.os.O_DIRECTORY", raising=False)
    opened = False

    def unexpected_open(*args: object, **kwargs: object) -> int:
        nonlocal opened
        opened = True
        raise AssertionError("descriptor traversal must not start")

    monkeypatch.setattr("sovereign_api.workspace_read_file.os.open", unexpected_open)

    with pytest.raises(WorkspaceSafeOpenUnsupportedError):
        WorkspaceReadFileTool(tmp_path)
    assert not opened


@pytest.mark.parametrize("unsupported", [os.open, os.stat])
def test_missing_required_dir_fd_capability_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsupported: object,
) -> None:
    monkeypatch.setattr(
        "sovereign_api.workspace_read_file.os.supports_dir_fd",
        set(os.supports_dir_fd) - {unsupported},
    )
    opened = False

    def unexpected_root_open(*args: object, **kwargs: object) -> int:
        nonlocal opened
        opened = True
        raise AssertionError("descriptor traversal must not start")

    monkeypatch.setattr(
        "sovereign_api.workspace_read_file._open_workspace_root",
        unexpected_root_open,
    )

    with pytest.raises(WorkspaceSafeOpenUnsupportedError):
        WorkspaceReadFileTool(tmp_path)
    assert not opened


def test_missing_stat_follow_symlinks_capability_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sovereign_api.workspace_read_file.os.supports_follow_symlinks",
        set(os.supports_follow_symlinks) - {os.stat},
    )
    opened = False

    def unexpected_root_open(*args: object, **kwargs: object) -> int:
        nonlocal opened
        opened = True
        raise AssertionError("descriptor traversal must not start")

    monkeypatch.setattr(
        "sovereign_api.workspace_read_file._open_workspace_root",
        unexpected_root_open,
    )

    with pytest.raises(WorkspaceSafeOpenUnsupportedError):
        WorkspaceReadFileTool(tmp_path)
    assert not opened


def test_unsupported_stat_during_traversal_closes_open_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "safe.txt").write_text("safe", encoding="utf-8")
    tool = WorkspaceReadFileTool(tmp_path)
    real_open = os.open
    real_close = os.close
    real_stat = os.stat
    opened: list[int] = []
    closed: list[int] = []

    def tracked_open(
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        opened.append(descriptor)
        return descriptor

    def unsupported_stat(
        path: object,
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> os.stat_result:
        if dir_fd is not None:
            raise NotImplementedError("fixture detail must remain internal")
        return real_stat(path, follow_symlinks=follow_symlinks)

    def tracked_close(descriptor: int) -> None:
        closed.append(descriptor)
        real_close(descriptor)

    with monkeypatch.context() as patch:
        patch.setattr("sovereign_api.workspace_read_file.os.open", tracked_open)
        patch.setattr("sovereign_api.workspace_read_file.os.stat", unsupported_stat)
        patch.setattr("sovereign_api.workspace_read_file.os.close", tracked_close)
        patch.setattr(
            "sovereign_api.workspace_read_file.os.supports_dir_fd",
            set(os.supports_dir_fd) | {tracked_open, unsupported_stat},
        )
        patch.setattr(
            "sovereign_api.workspace_read_file.os.supports_follow_symlinks",
            set(os.supports_follow_symlinks) | {unsupported_stat},
        )

        with pytest.raises(WorkspaceSafeOpenUnsupportedError):
            _execute_with_tool(tool, "safe.txt")

    assert sorted(opened) == sorted(closed)


@pytest.mark.parametrize(
    "unsupported_errno_name",
    [name for name in ("ENOTSUP", "EOPNOTSUPP", "ENOSYS") if hasattr(errno, name)],
)
def test_unsupported_open_errors_are_typed_and_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsupported_errno_name: str,
) -> None:
    tool = WorkspaceReadFileTool(tmp_path)
    diagnostic = "raw operating system diagnostic"
    unsupported_errno = getattr(errno, unsupported_errno_name)
    real_open = os.open

    def unsupported_open(
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if path == "/":
            raise OSError(unsupported_errno, diagnostic)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr("sovereign_api.workspace_read_file.os.open", unsupported_open)
    monkeypatch.setattr(
        "sovereign_api.workspace_read_file.os.supports_dir_fd",
        set(os.supports_dir_fd) | {unsupported_open},
    )

    with pytest.raises(WorkspaceSafeOpenUnsupportedError) as captured:
        _execute_with_tool(tool, "safe.txt")
    assert diagnostic not in str(captured.value)


def test_unsupported_stat_oserror_is_typed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = WorkspaceReadFileTool(tmp_path)
    real_stat = os.stat
    unsupported_errno = getattr(errno, "ENOTSUP", errno.ENOSYS)

    def unsupported_stat(
        path: object,
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> os.stat_result:
        if dir_fd is not None:
            raise OSError(unsupported_errno, "raw stat diagnostic")
        return real_stat(path, follow_symlinks=follow_symlinks)

    monkeypatch.setattr("sovereign_api.workspace_read_file.os.stat", unsupported_stat)
    monkeypatch.setattr(
        "sovereign_api.workspace_read_file.os.supports_dir_fd",
        set(os.supports_dir_fd) | {unsupported_stat},
    )
    monkeypatch.setattr(
        "sovereign_api.workspace_read_file.os.supports_follow_symlinks",
        set(os.supports_follow_symlinks) | {unsupported_stat},
    )

    with pytest.raises(WorkspaceSafeOpenUnsupportedError):
        _execute_with_tool(tool, "safe.txt")


def test_unsupported_fstat_is_typed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = WorkspaceReadFileTool(tmp_path)

    def unsupported_fstat(file_descriptor: int) -> os.stat_result:
        raise NotImplementedError("raw fstat diagnostic")

    monkeypatch.setattr("sovereign_api.workspace_read_file.os.fstat", unsupported_fstat)

    with pytest.raises(WorkspaceSafeOpenUnsupportedError) as captured:
        _execute_with_tool(tool, "safe.txt")
    assert "raw fstat diagnostic" not in str(captured.value)


def test_unsupported_read_is_typed_and_closes_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "safe.txt"
    target.write_text("safe", encoding="utf-8")
    tool = WorkspaceReadFileTool(tmp_path)
    real_open = os.open
    real_close = os.close
    opened: list[int] = []
    closed: list[int] = []

    def tracked_open(
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        opened.append(descriptor)
        return descriptor

    def tracked_close(descriptor: int) -> None:
        closed.append(descriptor)
        real_close(descriptor)

    def unsupported_read(file_descriptor: int, amount: int) -> bytes:
        raise NotImplementedError("raw read diagnostic")

    with monkeypatch.context() as patch:
        patch.setattr("sovereign_api.workspace_read_file.os.open", tracked_open)
        patch.setattr("sovereign_api.workspace_read_file.os.close", tracked_close)
        patch.setattr("sovereign_api.workspace_read_file.os.read", unsupported_read)
        patch.setattr(
            "sovereign_api.workspace_read_file.os.supports_dir_fd",
            set(os.supports_dir_fd) | {tracked_open},
        )

        with pytest.raises(WorkspaceSafeOpenUnsupportedError) as captured:
            _execute_with_tool(tool, "safe.txt")
        assert "raw read diagnostic" not in str(captured.value)

    assert sorted(opened) == sorted(closed)


def test_ordinary_access_error_retains_file_access_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "safe.txt"
    target.write_text("safe", encoding="utf-8")
    tool = WorkspaceReadFileTool(tmp_path)
    real_open = os.open

    def denied_final_open(
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if path == "safe.txt":
            raise OSError(errno.EACCES, "raw access diagnostic")
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr("sovereign_api.workspace_read_file.os.open", denied_final_open)
    monkeypatch.setattr(
        "sovereign_api.workspace_read_file.os.supports_dir_fd",
        set(os.supports_dir_fd) | {denied_final_open},
    )

    with pytest.raises(WorkspaceFileAccessError):
        _execute_with_tool(tool, "safe.txt")


@pytest.mark.parametrize("suffix", ["\x00bad", "\nbad", "\rbad"])
def test_malformed_workspace_root_is_typed_error(
    tmp_path: Path,
    suffix: str,
) -> None:
    with pytest.raises(InvalidWorkspaceRootError):
        WorkspaceReadFileTool.from_environment(
            {WORKSPACE_ROOT_ENV: str(tmp_path) + suffix}
        )


def test_arbitrary_workspace_root_object_is_rejected_without_string_conversion() -> None:
    class HostileRoot:
        string_conversion_called = False

        def __str__(self) -> str:
            self.string_conversion_called = True
            raise AssertionError("__str__ must not be invoked")

    root = HostileRoot()

    with pytest.raises(InvalidWorkspaceRootError):
        WorkspaceReadFileTool(root)  # type: ignore[arg-type]
    assert not root.string_conversion_called


def test_broken_supported_path_conversion_is_typed_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_fspath(value: object) -> str:
        raise ValueError("fixture detail must remain internal")

    monkeypatch.setattr("sovereign_api.workspace_read_file.os.fspath", broken_fspath)

    with pytest.raises(InvalidWorkspaceRootError) as captured:
        WorkspaceReadFileTool(tmp_path)
    assert "fixture detail" not in str(captured.value)


@pytest.mark.parametrize("environment", list(DeploymentEnvironment))
def test_explicit_read_permission_allows_every_environment(
    tmp_path: Path,
    environment: DeploymentEnvironment,
) -> None:
    (tmp_path / "safe.txt").write_text("safe", encoding="utf-8")

    assert _execute(
        tmp_path,
        "safe.txt",
        environment=environment,
    ).text_content == "safe"


def test_require_approval_does_not_execute() -> None:
    permission = ToolPermission("filesystem.write")
    descriptor = ToolDescriptor(
        tool_id="test.destructive",
        display_name="Destructive fixture",
        description="Test descriptor only",
        capabilities=("filesystem.write",),
        risk_level=ToolRiskLevel.HIGH,
        side_effect_level=ToolSideEffectLevel.DESTRUCTIVE,
        requires_network=False,
        supports_read=False,
        supports_write=True,
        required_permissions=(permission,),
    )

    @dataclass
    class RecordingTool:
        descriptor: ToolDescriptor
        called: bool = False

        async def execute(self, request: ToolRequest) -> ToolResult:
            self.called = True
            raise AssertionError("tool must not execute")

    tool = RecordingTool(descriptor)
    executor = PolicyEnforcedToolExecutor(
        ToolRegistry((descriptor,)),
        DeterministicToolPolicyEvaluator(),
        {descriptor.tool_id: tool},
    )

    with pytest.raises(ToolApprovalRequiredError):
        asyncio.run(
            executor.execute(
                ToolRequest(
                    "request-1",
                    descriptor.tool_id,
                    "delete",
                    {},
                    "task-1",
                    "stage-1",
                ),
                granted_permissions=frozenset({permission}),
                environment=DeploymentEnvironment.DEVELOPMENT,
            )
        )

    assert not tool.called


def test_production_registry_contains_only_workspace_reader() -> None:
    assert WORKSPACE_TOOL_REGISTRY.descriptors == (
        WORKSPACE_READ_FILE_DESCRIPTOR,
    )


def test_read_does_not_change_content_mtime_or_create_files(tmp_path: Path) -> None:
    target = tmp_path / "safe.txt"
    target.write_text("unchanged", encoding="utf-8")
    before_names = set(tmp_path.iterdir())
    before_mtime = target.stat().st_mtime_ns

    _execute(tmp_path, "safe.txt")

    assert target.read_text(encoding="utf-8") == "unchanged"
    assert target.stat().st_mtime_ns == before_mtime
    assert set(tmp_path.iterdir()) == before_names
