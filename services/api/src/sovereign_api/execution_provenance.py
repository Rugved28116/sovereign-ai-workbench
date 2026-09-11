"""Internal immutable facts about an approved routed stage execution."""

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import re

from sovereign_api.config import DeploymentEnvironment
from sovereign_api.errors import InvalidExecutionStateError, StageExecutionError


def result_digest(output_type: str, content: str) -> str:
    """SHA-256 of compact, sorted-key JSON encoded as UTF-8 without ASCII escaping."""
    canonical = json.dumps(
        {"output_type": output_type, "content": content},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ExecutionProvenance:
    stage_id: str
    model_id: str
    provider: str
    required_capabilities: tuple[str, ...]
    routing_environment: DeploymentEnvironment
    started_at: datetime
    completed_at: datetime
    success: bool
    result_digest: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "required_capabilities", tuple(self.required_capabilities))
        if any(type(value) is not str or not value for value in (
            self.stage_id, self.model_id, self.provider, *self.required_capabilities,
        )) or not self.required_capabilities:
            raise InvalidExecutionStateError("Provenance identities and capabilities must be non-empty strings")
        if type(self.routing_environment) is not DeploymentEnvironment or type(self.success) is not bool:
            raise InvalidExecutionStateError("Invalid provenance environment or success flag")
        for timestamp in (self.started_at, self.completed_at):
            if type(timestamp) is not datetime or timestamp.utcoffset() is None:
                raise InvalidExecutionStateError("Provenance timestamps must be timezone-aware")
        if self.completed_at < self.started_at:
            raise InvalidExecutionStateError("Provenance completion precedes start")
        if self.success:
            if type(self.result_digest) is not str or re.fullmatch(r"[0-9a-f]{64}", self.result_digest) is None:
                raise InvalidExecutionStateError("Successful provenance requires a SHA-256 digest")
        elif self.result_digest is not None:
            raise InvalidExecutionStateError("Failed provenance cannot contain a result digest")


class RoutedStageExecutionError(StageExecutionError):
    """Safe execution failure carrying internal provenance separately."""

    def __init__(self, message: str, provenance: ExecutionProvenance) -> None:
        super().__init__(message)
        self.provenance = provenance
