"""Bounded UTF-8 reads within one explicitly configured workspace root."""

from __future__ import annotations

import errno
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Mapping

from sovereign_api.tool_execution import (
    ExecutableToolRegistry,
    PolicyEnforcedToolExecutor,
)
from sovereign_api.tool_policy import DeterministicToolPolicyEvaluator
from sovereign_api.tool_contracts import (
    SafeToolError,
    ToolDescriptor,
    ToolPermission,
    ToolRegistry,
    ToolRequest,
    ToolResult,
    ToolResultStatus,
    ToolRiskLevel,
    ToolSideEffectLevel,
)


WORKSPACE_ROOT_ENV = "SOVEREIGN_WORKSPACE_ROOT"
WORKSPACE_READ_FILE_TOOL_ID = "workspace.read_file"
WORKSPACE_READ_OPERATION = "read_file"
MAX_WORKSPACE_FILE_BYTES = 1_048_576
_READ_CHUNK_BYTES = 65_536

FILESYSTEM_READ_PERMISSION = ToolPermission("filesystem.read")
WORKSPACE_READ_FILE_DESCRIPTOR = ToolDescriptor(
    tool_id=WORKSPACE_READ_FILE_TOOL_ID,
    display_name="Workspace file reader",
    description="Read one bounded UTF-8 file within the configured workspace",
    capabilities=("filesystem.read",),
    risk_level=ToolRiskLevel.LOW,
    side_effect_level=ToolSideEffectLevel.READ,
    requires_network=False,
    supports_read=True,
    supports_write=False,
    required_permissions=(FILESYSTEM_READ_PERMISSION,),
)
WORKSPACE_TOOL_REGISTRY = ToolRegistry((WORKSPACE_READ_FILE_DESCRIPTOR,))


class WorkspaceReadFileError(SafeToolError):
    code = "workspace_read_file_error"


class WorkspaceRootNotConfiguredError(WorkspaceReadFileError):
    code = "workspace_root_not_configured"


class InvalidWorkspaceRootError(WorkspaceReadFileError):
    code = "invalid_workspace_root"


class InvalidWorkspaceReadArgumentsError(WorkspaceReadFileError):
    code = "invalid_workspace_read_arguments"


class AbsoluteWorkspacePathError(WorkspaceReadFileError):
    code = "absolute_workspace_path"


class WorkspacePathOutsideRootError(WorkspaceReadFileError):
    code = "workspace_path_outside_root"


class WorkspaceFileNotFoundError(WorkspaceReadFileError):
    code = "workspace_file_not_found"


class WorkspaceNotRegularFileError(WorkspaceReadFileError):
    code = "workspace_not_regular_file"


class WorkspaceFileTooLargeError(WorkspaceReadFileError):
    code = "workspace_file_too_large"


class WorkspaceInvalidUTF8Error(WorkspaceReadFileError):
    code = "workspace_invalid_utf8"


class WorkspaceFileAccessError(WorkspaceReadFileError):
    code = "workspace_file_access_error"


class WorkspaceSafeOpenUnsupportedError(WorkspaceReadFileError):
    code = "workspace_safe_open_unsupported"


_UNSUPPORTED_OPERATION_ERRNOS = frozenset(
    value
    for name in ("ENOTSUP", "EOPNOTSUPP", "ENOSYS")
    if (value := getattr(errno, name, None)) is not None
)


def _raise_if_unsupported(error: BaseException) -> None:
    if isinstance(error, NotImplementedError) or (
        isinstance(error, OSError) and error.errno in _UNSUPPORTED_OPERATION_ERRNOS
    ):
        raise WorkspaceSafeOpenUnsupportedError(
            "Safe descriptor-relative file access is unavailable"
        ) from error


def _configured_root(value: object) -> Path:
    if value is None or (type(value) is str and not value.strip()):
        raise WorkspaceRootNotConfiguredError("Workspace root is not configured")
    try:
        if type(value) is str:
            raw_root = value
        elif isinstance(value, Path):
            raw_root = os.fspath(value)
        else:
            raise TypeError("unsupported root type")
        if type(raw_root) is not str:
            raise TypeError("root path must be text")
        if any(
            ord(character) < 32 or ord(character) == 127
            for character in raw_root
        ):
            raise ValueError("control character")
        if "\\" in raw_root:
            raise ValueError("platform separator")
        root_parts = tuple(raw_root.split("/"))
        if (
            not raw_root.startswith("/")
            or not root_parts[1:]
            or any(part in ("", ".", "..") for part in root_parts[1:])
        ):
            raise ValueError("root must be an unambiguous absolute path")
        root = Path(raw_root)
        if not root.is_absolute():
            raise ValueError("relative root")
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise InvalidWorkspaceRootError("Workspace root is invalid") from error

    common_flags, directory_flag = _require_safe_open_support()
    root_fd = _open_workspace_root(root, common_flags, directory_flag)
    try:
        root_status = os.fstat(root_fd)
        if not stat.S_ISDIR(root_status.st_mode):
            raise InvalidWorkspaceRootError("Workspace root must be a directory")
    except (NotImplementedError, OSError) as error:
        _raise_if_unsupported(error)
        raise InvalidWorkspaceRootError("Workspace root is invalid") from error
    finally:
        os.close(root_fd)
    return root


def _relative_parts(value: object) -> tuple[str, ...]:
    if type(value) is not str or not value:
        raise InvalidWorkspaceReadArgumentsError(
            "Tool argument 'path' must be a non-empty string"
        )
    windows_path = PureWindowsPath(value)
    if value.startswith("/") or windows_path.is_absolute() or windows_path.drive:
        raise AbsoluteWorkspacePathError("Workspace path must be relative")
    if "\x00" in value or "\\" in value:
        raise InvalidWorkspaceReadArgumentsError("Workspace path is malformed")
    parts = tuple(value.split("/"))
    if any(part in ("", ".", "..") for part in parts):
        raise WorkspacePathOutsideRootError("Workspace path is not allowed")
    return parts


def _require_safe_open_support() -> tuple[int, int]:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    supports_dir_fd = getattr(os, "supports_dir_fd", ())
    supports_follow_symlinks = getattr(os, "supports_follow_symlinks", ())
    if (
        os.name != "posix"
        or nofollow is None
        or directory is None
        or os.open not in supports_dir_fd
        or os.stat not in supports_dir_fd
        or os.stat not in supports_follow_symlinks
    ):
        raise WorkspaceSafeOpenUnsupportedError(
            "Safe descriptor-relative file access is unavailable"
        )
    common = os.O_RDONLY | nofollow
    common |= getattr(os, "O_CLOEXEC", 0)
    common |= getattr(os, "O_NONBLOCK", 0)
    return common, directory


def _open_workspace_root(
    root: Path,
    common_flags: int,
    directory_flag: int,
) -> int:
    current_fd: int | None = None
    try:
        current_fd = os.open("/", common_flags | directory_flag)
        for component in root.parts[1:]:
            try:
                expected = os.stat(
                    component,
                    dir_fd=current_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError as error:
                raise InvalidWorkspaceRootError("Workspace root is invalid") from error
            if stat.S_ISLNK(expected.st_mode) or not stat.S_ISDIR(expected.st_mode):
                raise InvalidWorkspaceRootError(
                    "Workspace root components must be directories"
                )
            next_fd = os.open(
                component,
                common_flags | directory_flag,
                dir_fd=current_fd,
            )
            try:
                opened = os.fstat(next_fd)
                if (
                    not stat.S_ISDIR(opened.st_mode)
                    or (opened.st_dev, opened.st_ino)
                    != (expected.st_dev, expected.st_ino)
                ):
                    raise InvalidWorkspaceRootError(
                        "Workspace root changed during access"
                    )
            except Exception:
                os.close(next_fd)
                raise
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except WorkspaceReadFileError:
        if current_fd is not None:
            os.close(current_fd)
        raise
    except (NotImplementedError, OSError) as error:
        if current_fd is not None:
            os.close(current_fd)
        _raise_if_unsupported(error)
        raise InvalidWorkspaceRootError("Workspace root is invalid") from error


def _open_relative_components(
    root: Path,
    parts: tuple[str, ...],
) -> tuple[int, list[int]]:
    common_flags, directory_flag = _require_safe_open_support()
    root_fd = _open_workspace_root(root, common_flags, directory_flag)
    descriptors = [root_fd]
    parent_fd = root_fd
    try:
        for component in parts[:-1]:
            try:
                component_status = os.stat(
                    component,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError as error:
                raise WorkspaceFileNotFoundError(
                    "Workspace file was not found"
                ) from error
            except OSError as error:
                _raise_if_unsupported(error)
                raise WorkspaceFileAccessError(
                    "Workspace path cannot be accessed"
                ) from error
            if stat.S_ISLNK(component_status.st_mode):
                raise WorkspacePathOutsideRootError(
                    "Workspace path contains a symlink"
                )
            if not stat.S_ISDIR(component_status.st_mode):
                raise WorkspaceNotRegularFileError(
                    "Workspace path is not a regular file"
                )
            try:
                child_fd = os.open(
                    component,
                    common_flags | directory_flag,
                    dir_fd=parent_fd,
                )
            except FileNotFoundError as error:
                raise WorkspaceFileNotFoundError(
                    "Workspace file was not found"
                ) from error
            except OSError as error:
                _raise_if_unsupported(error)
                if error.errno == errno.ELOOP:
                    raise WorkspacePathOutsideRootError(
                        "Workspace path contains a symlink"
                    ) from error
                if error.errno == errno.ENOTDIR:
                    try:
                        changed_status = os.stat(
                            component,
                            dir_fd=parent_fd,
                            follow_symlinks=False,
                        )
                    except OSError as stat_error:
                        _raise_if_unsupported(stat_error)
                        raise WorkspaceFileAccessError(
                            "Workspace path changed during access"
                        ) from stat_error
                    if stat.S_ISLNK(changed_status.st_mode):
                        raise WorkspacePathOutsideRootError(
                            "Workspace path contains a symlink"
                        ) from error
                    raise WorkspaceFileAccessError(
                        "Workspace path changed during access"
                    ) from error
                raise WorkspaceFileAccessError(
                    "Workspace path cannot be accessed"
                ) from error
            descriptors.append(child_fd)
            parent_fd = child_fd

        try:
            final_status = os.stat(
                parts[-1],
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError as error:
            raise WorkspaceFileNotFoundError("Workspace file was not found") from error
        except OSError as error:
            _raise_if_unsupported(error)
            raise WorkspaceFileAccessError(
                "Workspace file cannot be accessed"
            ) from error
        if stat.S_ISLNK(final_status.st_mode):
            raise WorkspacePathOutsideRootError(
                "Workspace path contains a symlink"
            )
        if not stat.S_ISREG(final_status.st_mode):
            raise WorkspaceNotRegularFileError(
                "Workspace path is not a regular file"
            )
        if final_status.st_size > MAX_WORKSPACE_FILE_BYTES:
            raise WorkspaceFileTooLargeError("Workspace file exceeds the size limit")
        try:
            file_descriptor = os.open(
                parts[-1],
                common_flags,
                dir_fd=parent_fd,
            )
        except FileNotFoundError as error:
            raise WorkspaceFileNotFoundError("Workspace file was not found") from error
        except OSError as error:
            _raise_if_unsupported(error)
            if error.errno == errno.ELOOP:
                raise WorkspacePathOutsideRootError(
                    "Workspace path contains a symlink"
                ) from error
            if error.errno in (errno.EISDIR, errno.ENXIO):
                raise WorkspaceNotRegularFileError(
                    "Workspace path is not a regular file"
                ) from error
            raise WorkspaceFileAccessError(
                "Workspace file cannot be accessed"
            ) from error
        descriptors.append(file_descriptor)
        return file_descriptor, descriptors, final_status
    except (NotImplementedError, OSError) as error:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        _raise_if_unsupported(error)
        raise WorkspaceFileAccessError(
            "Workspace file cannot be accessed"
        ) from error
    except WorkspaceReadFileError:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise


def _read_opened_file(file_descriptor: int, expected: os.stat_result) -> bytes:
    try:
        opened = os.fstat(file_descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise WorkspaceNotRegularFileError(
                "Workspace path is not a regular file"
            )
        if (opened.st_dev, opened.st_ino) != (expected.st_dev, expected.st_ino):
            raise WorkspaceFileAccessError("Workspace file changed during access")
        if opened.st_size > MAX_WORKSPACE_FILE_BYTES:
            raise WorkspaceFileTooLargeError("Workspace file exceeds the size limit")

        chunks = []
        remaining = MAX_WORKSPACE_FILE_BYTES + 1
        while remaining:
            chunk = os.read(file_descriptor, min(_READ_CHUNK_BYTES, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        content = b"".join(chunks)
        if len(content) > MAX_WORKSPACE_FILE_BYTES:
            raise WorkspaceFileTooLargeError("Workspace file exceeds the size limit")
        return content
    except WorkspaceReadFileError:
        raise
    except (NotImplementedError, OSError) as error:
        _raise_if_unsupported(error)
        raise WorkspaceFileAccessError("Workspace file cannot be read") from error


@dataclass(frozen=True, slots=True)
class WorkspaceReadFileTool:
    workspace_root: Path | str
    descriptor: ToolDescriptor = WORKSPACE_READ_FILE_DESCRIPTOR

    @property
    def tool_id(self) -> str:
        return WORKSPACE_READ_FILE_TOOL_ID

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "workspace_root",
            _configured_root(self.workspace_root),
        )
        if self.descriptor != WORKSPACE_READ_FILE_DESCRIPTOR:
            raise InvalidWorkspaceRootError("Workspace tool descriptor is invalid")

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> WorkspaceReadFileTool:
        source = os.environ if environment is None else environment
        return cls(source.get(WORKSPACE_ROOT_ENV))

    async def execute(self, request: ToolRequest) -> ToolResult:
        if request.tool_id != self.descriptor.tool_id:
            raise InvalidWorkspaceReadArgumentsError("Tool request ID does not match")
        if request.operation != WORKSPACE_READ_OPERATION:
            raise InvalidWorkspaceReadArgumentsError("Tool operation is not supported")
        if set(request.arguments) != {"path"}:
            raise InvalidWorkspaceReadArgumentsError(
                "Tool arguments must contain only 'path'"
            )

        parts = _relative_parts(request.arguments["path"])
        file_descriptor, descriptors, expected = _open_relative_components(
            self.workspace_root, parts
        )
        try:
            encoded = _read_opened_file(file_descriptor, expected)
        finally:
            for descriptor in reversed(descriptors):
                os.close(descriptor)
        try:
            content = encoded.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise WorkspaceInvalidUTF8Error(
                "Workspace file is not valid UTF-8 text"
            ) from error
        return ToolResult(
            request_id=request.request_id,
            tool_id=request.tool_id,
            status=ToolResultStatus.SUCCEEDED,
            text_content=content,
            safe_message="Workspace file read successfully",
        )


def create_workspace_tool_executor(
    workspace_root: Path | str,
) -> PolicyEnforcedToolExecutor:
    """Explicitly compose the sole production executable tool."""

    return PolicyEnforcedToolExecutor(
        registry=WORKSPACE_TOOL_REGISTRY,
        policy_evaluator=DeterministicToolPolicyEvaluator(),
        tools=ExecutableToolRegistry((WorkspaceReadFileTool(workspace_root),)),
    )
