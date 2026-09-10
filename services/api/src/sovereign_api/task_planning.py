"""Provider-neutral deterministic planning of logical task stages."""

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final, Mapping, Protocol

from sovereign_api.errors import UnsupportedTaskRequirementsError
from sovereign_api.task_classification import TaskClass, TaskRequirements


class TaskStageType(StrEnum):
    GENERATE = "generate"
    CODE = "code"
    DOCUMENT = "document"
    VISION = "vision"
    REASON = "reason"


@dataclass(frozen=True, slots=True)
class TaskStage:
    """One immutable logical stage; it carries no execution assignment."""

    stage_id: str
    stage_type: TaskStageType
    required_capabilities: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "required_capabilities", tuple(self.required_capabilities)
        )


@dataclass(frozen=True, slots=True)
class TaskPlan:
    """An immutable ordered description of possible future execution stages."""

    task_class: TaskClass
    stages: tuple[TaskStage, ...]

    def __post_init__(self) -> None:
        stages = tuple(self.stages)
        if any(type(stage) is not TaskStage for stage in stages):
            raise TypeError("stages must contain only TaskStage values")
        object.__setattr__(self, "stages", stages)


class TaskRequirementPlanner(Protocol):
    """Convert task requirements into a provider-neutral logical plan."""

    def plan(self, requirements: TaskRequirements) -> TaskPlan:
        """Return an immutable plan without executing or routing its stages."""
        ...


class DeterministicTaskRequirementPlanner:
    """Plan exact supported capability sequences with explicit stage mappings."""

    _STAGE_TYPE_BY_CAPABILITY: Final[Mapping[str, TaskStageType]] = MappingProxyType(
        {
            "chat": TaskStageType.GENERATE,
            "coding": TaskStageType.CODE,
            "document": TaskStageType.DOCUMENT,
            "vision": TaskStageType.VISION,
            "reasoning": TaskStageType.REASON,
        }
    )
    _SUPPORTED_CAPABILITY_SEQUENCES: Final[frozenset[tuple[str, ...]]] = frozenset(
        {
            ("chat",),
            ("coding",),
            ("document",),
            ("vision",),
            ("reasoning",),
            ("vision", "coding"),
            ("document", "coding"),
            ("vision", "reasoning"),
            ("document", "reasoning"),
            ("document", "vision"),
            ("coding", "reasoning"),
            ("document", "vision", "reasoning"),
        }
    )

    def plan(self, requirements: TaskRequirements) -> TaskPlan:
        capabilities = requirements.required_capabilities
        if capabilities not in self._SUPPORTED_CAPABILITY_SEQUENCES:
            raise UnsupportedTaskRequirementsError(
                "Task requirements do not have a deterministic planning rule"
            )

        stages = tuple(
            TaskStage(
                stage_id=f"stage-{index}",
                stage_type=self._STAGE_TYPE_BY_CAPABILITY[capability],
                required_capabilities=(capability,),
            )
            for index, capability in enumerate(capabilities, start=1)
        )
        return TaskPlan(task_class=requirements.task_class, stages=stages)
