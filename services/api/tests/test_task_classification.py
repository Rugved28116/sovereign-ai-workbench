import pytest

from sovereign_api.task_classification import (
    DeterministicTaskClassifier,
    TaskClass,
    required_capabilities_for,
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
    assert DeterministicTaskClassifier().classify(prompt) is expected


def test_precedence_is_vision_then_coding_then_document_then_reasoning() -> None:
    classifier = DeterministicTaskClassifier()

    assert classifier.classify("Debug code that analyzes an image") is TaskClass.VISION
    assert classifier.classify("Debug code in this report") is TaskClass.CODING
    assert classifier.classify("Analyze this report") is TaskClass.DOCUMENT


def test_ambiguous_text_defaults_to_general() -> None:
    assert (
        DeterministicTaskClassifier().classify("Please help with this")
        is TaskClass.GENERAL
    )


def test_identical_normalized_input_is_deterministic() -> None:
    classifier = DeterministicTaskClassifier()

    results = [classifier.classify("  FIX\tTHIS Python BUG  ") for _ in range(5)]

    assert results == [TaskClass.CODING] * 5


def test_punctuation_does_not_hide_a_signal_or_create_a_partial_phrase() -> None:
    classifier = DeterministicTaskClassifier()

    assert classifier.classify("Fix this Python bug.") is TaskClass.CODING
    assert classifier.classify("Discuss the inspection reporter") is TaskClass.GENERAL


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

    assert classifier.classify(plural_prompt) is TaskClass.VISION
    assert classifier.classify(singular_prompt) is TaskClass.VISION


@pytest.mark.parametrize(
    ("task_class", "capabilities"),
    [
        (TaskClass.GENERAL, ("chat",)),
        (TaskClass.CODING, ("coding",)),
        (TaskClass.REASONING, ("reasoning",)),
        (TaskClass.DOCUMENT, ("document",)),
        (TaskClass.VISION, ("vision",)),
    ],
)
def test_task_classes_map_to_provider_neutral_capabilities(
    task_class: TaskClass, capabilities: tuple[str, ...]
) -> None:
    assert required_capabilities_for(task_class) == capabilities
    assert not hasattr(task_class, "provider")
    assert not hasattr(task_class, "model_id")
