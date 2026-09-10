"""Provider-neutral deterministic task classification."""

import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final, Mapping, Protocol


class TaskClass(StrEnum):
    GENERAL = "general"
    CODING = "coding"
    REASONING = "reasoning"
    DOCUMENT = "document"
    VISION = "vision"


@dataclass(frozen=True, slots=True)
class TaskRequirements:
    """Immutable provider-neutral requirements inferred for one task."""

    task_class: TaskClass
    required_capabilities: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "required_capabilities", tuple(self.required_capabilities)
        )


TASK_CLASS_CAPABILITIES: Final[Mapping[TaskClass, tuple[str, ...]]] = MappingProxyType(
    {
        TaskClass.GENERAL: ("chat",),
        TaskClass.CODING: ("coding",),
        TaskClass.REASONING: ("reasoning",),
        TaskClass.DOCUMENT: ("document",),
        TaskClass.VISION: ("vision",),
    }
)


class TaskClassifier(Protocol):
    """Infer a provider-neutral task class from validated request text."""

    def classify(self, prompt: str) -> TaskRequirements:
        """Return requirements without selecting a model or provider."""
        ...


class DeterministicTaskClassifier:
    """Classify by conservative first-match rules over normalized text."""

    _PRECEDENCE: Final[tuple[TaskClass, ...]] = (
        TaskClass.VISION,
        TaskClass.CODING,
        TaskClass.DOCUMENT,
        TaskClass.REASONING,
    )
    _COMBINATION_RULES: Final[
        tuple[tuple[frozenset[TaskClass], tuple[str, ...]], ...]
    ] = (
        (
            frozenset(
                {TaskClass.DOCUMENT, TaskClass.VISION, TaskClass.REASONING}
            ),
            ("document", "vision", "reasoning"),
        ),
        (
            frozenset({TaskClass.VISION, TaskClass.CODING}),
            ("vision", "coding"),
        ),
        (
            frozenset({TaskClass.DOCUMENT, TaskClass.CODING}),
            ("document", "coding"),
        ),
        (
            frozenset({TaskClass.VISION, TaskClass.REASONING}),
            ("vision", "reasoning"),
        ),
        (
            frozenset({TaskClass.DOCUMENT, TaskClass.REASONING}),
            ("document", "reasoning"),
        ),
        (
            frozenset({TaskClass.DOCUMENT, TaskClass.VISION}),
            ("document", "vision"),
        ),
        (
            frozenset({TaskClass.CODING, TaskClass.REASONING}),
            ("coding", "reasoning"),
        ),
    )
    _TOKEN_SIGNALS: Final[Mapping[TaskClass, frozenset[str]]] = MappingProxyType(
        {
            TaskClass.VISION: frozenset(
                {
                    "image",
                    "photograph",
                    "drawing",
                    "diagram",
                    "screenshot",
                    "scan",
                    "scanned",
                }
            ),
            TaskClass.CODING: frozenset(
                {
                    "code",
                    "python",
                    "javascript",
                    "typescript",
                    "function",
                    "bug",
                    "debug",
                    "compile",
                    "test",
                    "repository",
                    "script",
                }
            ),
            TaskClass.DOCUMENT: frozenset(
                {"document", "report", "pdf", "summarize", "extract"}
            ),
            TaskClass.REASONING: frozenset(
                {"calculate", "analyze", "compare", "reason", "derive", "explain"}
            ),
        }
    )
    _KNOWN_TOKEN_SIGNALS: Final[frozenset[str]] = frozenset().union(
        *_TOKEN_SIGNALS.values()
    )
    _PHRASE_SIGNALS: Final[Mapping[TaskClass, tuple[str, ...]]] = MappingProxyType(
        {
            TaskClass.DOCUMENT: ("inspection report", "approval note"),
            TaskClass.REASONING: ("explain why",),
        }
    )

    def classify(self, prompt: str) -> TaskRequirements:
        normalized = " ".join(
            unicodedata.normalize("NFKC", prompt).casefold().split()
        )
        token_sequence = tuple(re.findall(r"[^\W_]+", normalized))
        tokens = frozenset(
            token[:-1]
            if token.endswith("s") and token[:-1] in self._KNOWN_TOKEN_SIGNALS
            else token
            for token in token_sequence
        )
        normalized_tokens = f" {' '.join(token_sequence)} "

        detected_classes = frozenset(
            task_class
            for task_class in self._PRECEDENCE
            if tokens & self._TOKEN_SIGNALS.get(task_class, frozenset())
            or any(
                f" {phrase} " in normalized_tokens
                for phrase in self._PHRASE_SIGNALS.get(task_class, ())
            )
        )
        if not detected_classes:
            return TaskRequirements(
                task_class=TaskClass.GENERAL,
                required_capabilities=TASK_CLASS_CAPABILITIES[TaskClass.GENERAL],
            )

        task_class = next(
            candidate
            for candidate in self._PRECEDENCE
            if candidate in detected_classes
        )
        for required_classes, required_capabilities in self._COMBINATION_RULES:
            if required_classes.issubset(detected_classes):
                return TaskRequirements(
                    task_class=task_class,
                    required_capabilities=required_capabilities,
                )

        return TaskRequirements(
            task_class=task_class,
            required_capabilities=TASK_CLASS_CAPABILITIES[task_class],
        )
