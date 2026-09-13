"""Bounded, policy-gated UTF-8 artifact writes within one explicit output root."""

from __future__ import annotations

import errno
import os
import secrets
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, NoReturn

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
from sovereign_api.tool_execution import ExecutableToolRegistry, PolicyEnforcedToolExecutor
from sovereign_api.tool_policy import DeterministicToolPolicyEvaluator
from sovereign_api.workspace_read_file import (
    AbsoluteWorkspacePathError,
    WORKSPACE_READ_FILE_DESCRIPTOR,
    WorkspaceRootNotConfiguredError,
    WorkspacePathOutsideRootError,
    WorkspaceReadFileError,
    WorkspaceSafeOpenUnsupportedError,
    _configured_root,
    _open_workspace_root,
    _relative_parts,
    _require_safe_open_support,
    WorkspaceReadFileTool,
)


ARTIFACT_ROOT_ENV = "SOVEREIGN_ARTIFACT_ROOT"
WORKSPACE_WRITE_ARTIFACT_TOOL_ID = "workspace.write_artifact"
WORKSPACE_WRITE_OPERATION = "write_artifact"
MAX_ARTIFACT_BYTES = 1_048_576
_TEMP_PREFIX = ".artifact-tmp-"

ARTIFACT_WRITE_PERMISSION = ToolPermission("artifact.write")
WORKSPACE_WRITE_ARTIFACT_DESCRIPTOR = ToolDescriptor(
    tool_id=WORKSPACE_WRITE_ARTIFACT_TOOL_ID,
    display_name="Workspace artifact writer",
    description="Write one bounded UTF-8 artifact inside the configured artifact root",
    capabilities=("artifact.write",),
    risk_level=ToolRiskLevel.MEDIUM,
    side_effect_level=ToolSideEffectLevel.WRITE,
    requires_network=False,
    supports_read=False,
    supports_write=True,
    required_permissions=(ARTIFACT_WRITE_PERMISSION,),
)
WORKSPACE_ARTIFACT_TOOL_REGISTRY = ToolRegistry(
    (WORKSPACE_READ_FILE_DESCRIPTOR, WORKSPACE_WRITE_ARTIFACT_DESCRIPTOR)
)


class ArtifactWriteError(SafeToolError):
    code = "artifact_write_error"


class ArtifactRootNotConfiguredError(ArtifactWriteError):
    code = "artifact_root_not_configured"


class InvalidArtifactRootError(ArtifactWriteError):
    code = "invalid_artifact_root"


class ArtifactDirectorySecurityError(ArtifactWriteError):
    code = "artifact_directory_unsafe"


class InvalidArtifactArgumentsError(ArtifactWriteError):
    code = "invalid_artifact_arguments"


class ArtifactAbsolutePathError(ArtifactWriteError):
    code = "artifact_absolute_path"


class ArtifactPathOutsideRootError(ArtifactWriteError):
    code = "artifact_path_outside_root"


class ArtifactParentNotFoundError(ArtifactWriteError):
    code = "artifact_parent_not_found"


class ArtifactSymlinkError(ArtifactWriteError):
    code = "artifact_symlink_rejected"


class ArtifactInvalidTargetError(ArtifactWriteError):
    code = "artifact_invalid_target"


class ArtifactTooLargeError(ArtifactWriteError):
    code = "artifact_too_large"


class ArtifactSafeOpenUnsupportedError(ArtifactWriteError):
    code = "artifact_safe_open_unsupported"


class ArtifactFileAccessError(ArtifactWriteError):
    code = "artifact_file_access_error"


def _translate_reader_error(error: WorkspaceReadFileError, *, root: bool) -> NoReturn:
    if isinstance(error, WorkspaceRootNotConfiguredError):
        raise ArtifactRootNotConfiguredError("Artifact root is not configured") from None
    if isinstance(error, WorkspaceSafeOpenUnsupportedError):
        raise ArtifactSafeOpenUnsupportedError("Safe artifact writing is unavailable") from None
    if root:
        raise InvalidArtifactRootError("Artifact root is invalid") from None
    if isinstance(error, AbsoluteWorkspacePathError):
        raise ArtifactAbsolutePathError("Artifact path must be relative") from None
    if isinstance(error, WorkspacePathOutsideRootError):
        raise ArtifactPathOutsideRootError("Artifact path is not allowed") from None
    raise InvalidArtifactArgumentsError("Artifact path is malformed") from None


def _root(value: object) -> Path:
    try:
        root = _configured_root(value)
    except WorkspaceReadFileError as error:
        _translate_reader_error(error, root=True)
    common, directory = _safe_flags()
    try:
        root_fd = _open_workspace_root(root, common, directory)
    except WorkspaceReadFileError as error:
        _translate_reader_error(error, root=True)
    try:
        _require_private_directory(root_fd)
    finally:
        _close_all([root_fd])
    return root


def _parts(value: object) -> tuple[str, ...]:
    try:
        parts = _relative_parts(value)
    except WorkspaceReadFileError as error:
        _translate_reader_error(error, root=False)
    else:
        # The writer never exposes its private temporary-name namespace.
        if parts[-1].startswith(_TEMP_PREFIX):
            raise InvalidArtifactArgumentsError("Artifact path is reserved")
        return parts


def _safe_flags() -> tuple[int, int]:
    try:
        common, directory = _require_safe_open_support()
    except WorkspaceReadFileError as error:
        _translate_reader_error(error, root=True)
    supports = getattr(os, "supports_dir_fd", ())
    if (
        os.name != "posix"
        or getattr(os, "geteuid", None) is None
        or getattr(os, "rename", None) not in supports
        or getattr(os, "unlink", None) not in supports
        or any(getattr(os, name, None) is None for name in ("O_CREAT", "O_EXCL", "O_WRONLY"))
    ):
        raise ArtifactSafeOpenUnsupportedError("Safe artifact writing is unavailable")
    return common, directory


def _unsupported(error: BaseException) -> bool:
    return isinstance(error, NotImplementedError) or (
        isinstance(error, OSError)
        and error.errno in {
            getattr(errno, name)
            for name in ("ENOTSUP", "EOPNOTSUPP", "ENOSYS")
            if hasattr(errno, name)
        }
    )


def _require_private_directory(descriptor: int) -> None:
    """Only the service identity may rename entries in artifact directories."""

    try:
        metadata = os.fstat(descriptor)
        owner = os.geteuid()
    except (NotImplementedError, OSError) as error:
        if _unsupported(error):
            raise ArtifactSafeOpenUnsupportedError("Safe artifact writing is unavailable") from None
        raise ArtifactFileAccessError("Artifact directory cannot be inspected") from None
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != owner
        or metadata.st_mode & 0o022
    ):
        raise ArtifactDirectorySecurityError("Artifact directory is not private")


def _opened_parent(root: Path, parts: tuple[str, ...]) -> tuple[int, list[int]]:
    common, directory = _safe_flags()
    try:
        root_fd = _open_workspace_root(root, common, directory)
    except WorkspaceReadFileError as error:
        _translate_reader_error(error, root=True)
    descriptors = [root_fd]
    try:
        _require_private_directory(root_fd)
        for component in parts[:-1]:
            try:
                expected = os.stat(component, dir_fd=descriptors[-1], follow_symlinks=False)
            except FileNotFoundError:
                raise ArtifactParentNotFoundError("Artifact parent does not exist") from None
            if stat.S_ISLNK(expected.st_mode):
                raise ArtifactSymlinkError("Artifact path contains a symlink")
            if not stat.S_ISDIR(expected.st_mode):
                raise ArtifactInvalidTargetError("Artifact parent is not a directory")
            child = os.open(component, common | directory, dir_fd=descriptors[-1])
            try:
                opened = os.fstat(child)
                if not stat.S_ISDIR(opened.st_mode) or (
                    opened.st_dev, opened.st_ino
                ) != (expected.st_dev, expected.st_ino):
                    raise ArtifactFileAccessError("Artifact parent changed during access")
                _require_private_directory(child)
            except Exception:
                _close_all([child])
                raise
            descriptors.append(child)
        return descriptors[-1], descriptors
    except (NotImplementedError, OSError) as error:
        _close_all(descriptors)
        if _unsupported(error):
            raise ArtifactSafeOpenUnsupportedError("Safe artifact writing is unavailable") from None
        if isinstance(error, FileNotFoundError):
            raise ArtifactParentNotFoundError("Artifact parent does not exist") from None
        if isinstance(error, OSError) and error.errno in (errno.ELOOP, errno.ENOTDIR):
            raise ArtifactSymlinkError("Artifact path contains a symlink") from None
        raise ArtifactFileAccessError("Artifact parent cannot be accessed") from None
    except ArtifactWriteError:
        _close_all(descriptors)
        raise


def _close_all(descriptors: list[int]) -> None:
    close_error: BaseException | None = None
    for descriptor in reversed(descriptors):
        try:
            os.close(descriptor)
        except (NotImplementedError, OSError) as error:
            if close_error is None:
                close_error = error
    if close_error is not None:
        if _unsupported(close_error):
            raise ArtifactSafeOpenUnsupportedError("Safe artifact writing is unavailable") from None
        raise ArtifactFileAccessError("Artifact descriptor could not be closed") from None


def _target_identity(parent_fd: int, name: str) -> tuple[int, int] | None:
    try:
        target = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(target.st_mode):
        raise ArtifactSymlinkError("Artifact target is a symlink")
    if not stat.S_ISREG(target.st_mode):
        raise ArtifactInvalidTargetError("Artifact target is not a regular file")
    return target.st_dev, target.st_ino


def _write_bytes(parent_fd: int, name: str, content: bytes) -> None:
    temp_name: str | None = None
    temp_fd: int | None = None
    try:
        initial = _target_identity(parent_fd, name)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        flags |= getattr(os, "O_CLOEXEC", 0)
        for _ in range(3):
            candidate = _TEMP_PREFIX + secrets.token_hex(16)
            try:
                temp_fd = os.open(candidate, flags, 0o600, dir_fd=parent_fd)
            except FileExistsError:
                continue
            temp_name = candidate
            break
        if temp_fd is None or temp_name is None:
            raise ArtifactFileAccessError("Artifact temporary file is unavailable")

        position = 0
        while position < len(content):
            written = os.write(temp_fd, content[position:])
            if written <= 0:
                raise ArtifactFileAccessError("Artifact cannot be written")
            position += written
        os.fsync(temp_fd)
        opened_temp = os.fstat(temp_fd)
        named_temp = os.stat(temp_name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISREG(named_temp.st_mode) or (
            named_temp.st_dev, named_temp.st_ino
        ) != (opened_temp.st_dev, opened_temp.st_ino):
            raise ArtifactFileAccessError("Artifact temporary file changed")
        if _target_identity(parent_fd, name) != initial:
            raise ArtifactFileAccessError("Artifact target changed during access")
        # POSIX rename in one opened directory is atomic and replaces a regular target.
        os.rename(temp_name, name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        temp_name = None
    except (NotImplementedError, OSError) as error:
        if _unsupported(error):
            raise ArtifactSafeOpenUnsupportedError("Safe artifact writing is unavailable") from None
        raise ArtifactFileAccessError("Artifact cannot be written") from None
    finally:
        close_error: BaseException | None = None
        if temp_fd is not None:
            try:
                os.close(temp_fd)
            except (NotImplementedError, OSError) as error:
                close_error = error
        if temp_name is not None:
            try:
                os.unlink(temp_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
            except (NotImplementedError, OSError) as error:
                if _unsupported(error):
                    raise ArtifactSafeOpenUnsupportedError(
                        "Safe artifact writing is unavailable"
                    ) from None
                raise ArtifactFileAccessError("Artifact temporary file cannot be removed") from None
        if close_error is not None:
            if _unsupported(close_error):
                raise ArtifactSafeOpenUnsupportedError("Safe artifact writing is unavailable") from None
            raise ArtifactFileAccessError("Artifact descriptor could not be closed") from None


@dataclass(frozen=True, slots=True)
class WorkspaceWriteArtifactTool:
    artifact_root: Path | str
    descriptor: ToolDescriptor = WORKSPACE_WRITE_ARTIFACT_DESCRIPTOR

    @property
    def tool_id(self) -> str:
        return WORKSPACE_WRITE_ARTIFACT_TOOL_ID

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_root", _root(self.artifact_root))
        if self.descriptor != WORKSPACE_WRITE_ARTIFACT_DESCRIPTOR:
            raise InvalidArtifactRootError("Artifact tool descriptor is invalid")

    @classmethod
    def from_environment(
        cls, environment: Mapping[str, str] | None = None,
    ) -> WorkspaceWriteArtifactTool:
        source = os.environ if environment is None else environment
        return cls(source.get(ARTIFACT_ROOT_ENV))

    async def execute(self, request: ToolRequest) -> ToolResult:
        if request.tool_id != self.tool_id or request.operation != WORKSPACE_WRITE_OPERATION:
            raise InvalidArtifactArgumentsError("Artifact request is invalid")
        if set(request.arguments) != {"path", "content"}:
            raise InvalidArtifactArgumentsError("Artifact arguments are invalid")
        parts = _parts(request.arguments["path"])
        content = request.arguments["content"]
        if type(content) is not str:
            raise InvalidArtifactArgumentsError("Artifact content must be text")
        if len(content) > MAX_ARTIFACT_BYTES:
            raise ArtifactTooLargeError("Artifact exceeds the size limit")
        try:
            encoded = content.encode("utf-8", errors="strict")
        except UnicodeEncodeError:
            raise InvalidArtifactArgumentsError("Artifact content is not valid UTF-8") from None
        if len(encoded) > MAX_ARTIFACT_BYTES:
            raise ArtifactTooLargeError("Artifact exceeds the size limit")
        parent_fd, descriptors = _opened_parent(self.artifact_root, parts)
        try:
            _write_bytes(parent_fd, parts[-1], encoded)
        finally:
            _close_all(descriptors)
        return ToolResult(
            request_id=request.request_id,
            tool_id=request.tool_id,
            status=ToolResultStatus.SUCCEEDED,
            output_reference="/".join(parts),
            safe_message="Artifact written successfully",
        )


def create_workspace_artifact_tool_executor(
    workspace_root: Path | str, artifact_root: Path | str,
) -> PolicyEnforcedToolExecutor:
    """Explicitly register the read-only workspace tool and artifact writer."""

    return PolicyEnforcedToolExecutor(
        registry=WORKSPACE_ARTIFACT_TOOL_REGISTRY,
        policy_evaluator=DeterministicToolPolicyEvaluator(),
        tools=ExecutableToolRegistry(
            (WorkspaceReadFileTool(workspace_root), WorkspaceWriteArtifactTool(artifact_root))
        ),
    )
