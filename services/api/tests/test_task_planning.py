from dataclasses import FrozenInstanceError

import pytest

from sovereign_api.errors import UnsupportedTaskRequirementsError
from sovereign_api.task_classification import TaskClass, TaskRequirements
from sovereign_api.task_planning import (
    DeterministicTaskRequirementPlanner,
    TaskPlan,
    TaskStage,
    TaskStageType,
)


PLAN_CASES = [
    (TaskClass.GENERAL, ("chat",), (TaskStageType.GENERATE,)),
    (TaskClass.CODING, ("coding",), (TaskStageType.CODE,)),
    (TaskClass.DOCUMENT, ("document",), (TaskStageType.DOCUMENT,)),
    (TaskClass.VISION, ("vision",), (TaskStageType.VISION,)),
    (TaskClass.REASONING, ("reasoning",), (TaskStageType.REASON,)),
    (
        TaskClass.VISION,
        ("vision", "coding"),
        (TaskStageType.VISION, TaskStageType.CODE),
    ),
    (
        TaskClass.CODING,
        ("document", "coding"),
        (TaskStageType.DOCUMENT, TaskStageType.CODE),
    ),
    (
        TaskClass.VISION,
        ("vision", "reasoning"),
        (TaskStageType.VISION, TaskStageType.REASON),
    ),
    (
        TaskClass.DOCUMENT,
        ("document", "reasoning"),
        (TaskStageType.DOCUMENT, TaskStageType.REASON),
    ),
    (
        TaskClass.VISION,
        ("document", "vision"),
        (TaskStageType.DOCUMENT, TaskStageType.VISION),
    ),
    (
        TaskClass.CODING,
        ("coding", "reasoning"),
        (TaskStageType.CODE, TaskStageType.REASON),
    ),
    (
        TaskClass.VISION,
        ("document", "vision", "reasoning"),
        (TaskStageType.DOCUMENT, TaskStageType.VISION, TaskStageType.REASON),
    ),
]


@pytest.mark.parametrize(("task_class", "capabilities", "stage_types"), PLAN_CASES)
def test_deterministic_planning_rules(
    task_class: TaskClass,
    capabilities: tuple[str, ...],
    stage_types: tuple[TaskStageType, ...],
) -> None:
    requirements = TaskRequirements(task_class, capabilities)

    plan = DeterministicTaskRequirementPlanner().plan(requirements)

    assert plan.task_class is task_class
    assert tuple(stage.stage_type for stage in plan.stages) == stage_types
    assert tuple(stage.stage_id for stage in plan.stages) == tuple(
        f"stage-{index}" for index in range(1, len(stage_types) + 1)
    )
    assert tuple(
        stage.required_capabilities for stage in plan.stages
    ) == tuple((capability,) for capability in capabilities)


def test_planning_is_structurally_deterministic() -> None:
    requirements = TaskRequirements(
        TaskClass.VISION, ("document", "vision", "reasoning")
    )
    planner = DeterministicTaskRequirementPlanner()

    plans = [planner.plan(requirements) for _ in range(5)]

    assert plans == [plans[0]] * 5


def test_task_stage_defensively_copies_capabilities_and_is_immutable() -> None:
    capabilities = ["vision"]
    stage = TaskStage(
        "stage-1",
        TaskStageType.VISION,
        capabilities,  # type: ignore[arg-type]
    )

    capabilities.append("reasoning")

    assert stage.required_capabilities == ("vision",)
    with pytest.raises(FrozenInstanceError):
        stage.stage_id = "stage-2"
    with pytest.raises(AttributeError):
        stage.required_capabilities.append(  # type: ignore[attr-defined]
            "reasoning"
        )


def test_task_plan_defensively_copies_stages_and_is_immutable() -> None:
    stage = TaskStage("stage-1", TaskStageType.GENERATE, ("chat",))
    stages = [stage]
    plan = TaskPlan(
        TaskClass.GENERAL,
        stages,  # type: ignore[arg-type]
    )

    stages.append(TaskStage("stage-2", TaskStageType.REASON, ("reasoning",)))

    assert plan.stages == (stage,)
    with pytest.raises(FrozenInstanceError):
        plan.stages = ()
    with pytest.raises(AttributeError):
        plan.stages.append(stage)  # type: ignore[attr-defined]


def test_planner_does_not_mutate_original_requirements() -> None:
    requirements = TaskRequirements(TaskClass.VISION, ("vision", "reasoning"))
    original_requirements = TaskRequirements(
        requirements.task_class, requirements.required_capabilities
    )

    DeterministicTaskRequirementPlanner().plan(requirements)

    assert requirements == original_requirements


@pytest.mark.parametrize(
    "capabilities",
    [
        ("audio",),
        ("reasoning", "vision"),
        ("document", "vision", "coding"),
        (),
    ],
)
def test_unknown_or_unsupported_capability_sequences_fail_closed(
    capabilities: tuple[str, ...],
) -> None:
    requirements = TaskRequirements(TaskClass.GENERAL, capabilities)

    with pytest.raises(UnsupportedTaskRequirementsError):
        DeterministicTaskRequirementPlanner().plan(requirements)


def test_plan_contains_no_model_or_provider_identity() -> None:
    plan = DeterministicTaskRequirementPlanner().plan(
        TaskRequirements(TaskClass.VISION, ("vision", "coding"))
    )

    assert not hasattr(plan, "model_id")
    assert not hasattr(plan, "provider")
    assert all(not hasattr(stage, "model_id") for stage in plan.stages)
    assert all(not hasattr(stage, "provider") for stage in plan.stages)
