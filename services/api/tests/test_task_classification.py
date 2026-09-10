from dataclasses import FrozenInstanceError

import pytest

from sovereign_api.task_classification import (
    DeterministicTaskClassifier,
    TaskClass,
    TaskRequirements,
)


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("Fix this Python bug", TaskClass.CODING),
        ("Write a function", TaskClass.CODING),
        ("Run the unit tests", TaskClass.CODING),
        ("Summarize this inspection report", TaskClass.DOCUMENT),
        ("Extract findings from this PDF", TaskClass.DOCUMENT),
        ("Inspect this image", TaskClass.VISION),
        ("Analyze this engineering drawing", TaskClass.VISION),
        ("Compare these two approaches", TaskClass.REASONING),
        ("Calculate the result", TaskClass.REASONING),
        ("Hello", TaskClass.GENERAL),
        ("What is this system?", TaskClass.GENERAL),
    ],
)
def test_deterministic_task_examples(prompt: str, expected: TaskClass) -> None:
    assert DeterministicTaskClassifier().classify(prompt).task_class is expected


def test_precedence_is_vision_then_coding_then_document_then_reasoning() -> None:
    classifier = DeterministicTaskClassifier()

    assert (
        classifier.classify("Debug code that analyzes an image").task_class
        is TaskClass.VISION
    )
    assert (
        classifier.classify("Debug code in this report").task_class
        is TaskClass.CODING
    )
    assert (
        classifier.classify("Analyze this report").task_class
        is TaskClass.DOCUMENT
    )


def test_ambiguous_text_defaults_to_general() -> None:
    assert DeterministicTaskClassifier().classify(
        "Please help with this"
    ) == TaskRequirements(
        task_class=TaskClass.GENERAL,
        required_capabilities=("chat",),
    )


def test_identical_normalized_input_is_deterministic() -> None:
    classifier = DeterministicTaskClassifier()

    results = [
        classifier.classify("  COMPARE\tTHESE two DIAGRAMS  ") for _ in range(5)
    ]

    assert results == [
        TaskRequirements(
            task_class=TaskClass.VISION,
            required_capabilities=("vision", "reasoning"),
        )
    ] * 5


def test_punctuation_does_not_hide_a_signal_or_create_a_partial_phrase() -> None:
    classifier = DeterministicTaskClassifier()

    assert (
        classifier.classify("Fix this Python bug.").task_class
        is TaskClass.CODING
    )
    assert (
        classifier.classify("Discuss the inspection reporter").task_class
        is TaskClass.GENERAL
    )


@pytest.mark.parametrize(
    ("plural_prompt", "singular_prompt"),
    [
        ("compare these two diagrams", "compare this diagram"),
        ("inspect these images", "inspect this image"),
        ("analyze these screenshots", "analyze this screenshot"),
        ("review the photographs", "review the photograph"),
    ],
)
def test_plural_and_singular_vision_signals_take_precedence(
    plural_prompt: str, singular_prompt: str
) -> None:
    classifier = DeterministicTaskClassifier()

    assert classifier.classify(plural_prompt).task_class is TaskClass.VISION
    assert classifier.classify(singular_prompt).task_class is TaskClass.VISION


@pytest.mark.parametrize(
    ("prompt", "task_class", "capabilities"),
    [
        ("Hello", TaskClass.GENERAL, ("chat",)),
        ("Write a Python function", TaskClass.CODING, ("coding",)),
        ("Calculate the result", TaskClass.REASONING, ("reasoning",)),
        ("Summarize this report", TaskClass.DOCUMENT, ("document",)),
        ("Inspect this image", TaskClass.VISION, ("vision",)),
    ],
)
def test_single_signal_tasks_have_one_required_capability(
    prompt: str, task_class: TaskClass, capabilities: tuple[str, ...]
) -> None:
    requirements = DeterministicTaskClassifier().classify(prompt)

    assert requirements == TaskRequirements(task_class, capabilities)
    assert not hasattr(requirements, "provider")
    assert not hasattr(requirements, "model_id")


@pytest.mark.parametrize(
    ("prompt", "capabilities"),
    [
        ("Analyze this engineering drawing", ("vision", "reasoning")),
        ("analyze this Python screenshot", ("vision", "coding")),
        ("debug the script in this PDF", ("document", "coding")),
        ("compare these two diagrams", ("vision", "reasoning")),
        ("analyze this inspection report", ("document", "reasoning")),
        ("extract text from this scanned report", ("document", "vision")),
        ("explain this code", ("coding", "reasoning")),
        (
            "compare findings in this scanned report",
            ("document", "vision", "reasoning"),
        ),
    ],
)
def test_explicit_combination_rules(
    prompt: str, capabilities: tuple[str, ...]
) -> None:
    requirements = DeterministicTaskClassifier().classify(prompt)

    assert requirements.required_capabilities == capabilities


def test_task_requirements_copy_mutable_capability_input() -> None:
    capabilities = ["vision", "reasoning"]
    requirements = TaskRequirements(
        TaskClass.VISION,
        capabilities,  # type: ignore[arg-type]
    )

    assert requirements.required_capabilities == ("vision", "reasoning")
    assert isinstance(requirements.required_capabilities, tuple)

    capabilities.append("coding")

    assert requirements.required_capabilities == ("vision", "reasoning")


def test_task_requirements_capabilities_cannot_be_mutated_directly() -> None:
    requirements = DeterministicTaskClassifier().classify(
        "compare these two diagrams"
    )

    assert isinstance(requirements.required_capabilities, tuple)
    with pytest.raises(FrozenInstanceError):
        requirements.required_capabilities = ("chat",)
    with pytest.raises(AttributeError):
        requirements.required_capabilities.append(  # type: ignore[attr-defined]
            "coding"
        )
