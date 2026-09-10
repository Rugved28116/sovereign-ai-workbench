"""Provider-neutral deterministic task classification."""

import re
import unicodedata
from enum import StrEnum
from types import MappingProxyType
from typing import Final, Mapping, Protocol


class TaskClass(StrEnum):
    GENERAL = "general"
    CODING = "coding"
    REASONING = "reasoning"
    DOCUMENT = "document"
    VISION = "vision"


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

    def classify(self, prompt: str) -> TaskClass:
        """Return one supported task class without selecting a model or provider."""
        ...


class DeterministicTaskClassifier:
    """Classify by conservative first-match rules over normalized text."""

    _PRECEDENCE: Final[tuple[TaskClass, ...]] = (
        TaskClass.VISION,
        TaskClass.CODING,
        TaskClass.DOCUMENT,
        TaskClass.REASONING,
    )
    _TOKEN_SIGNALS: Final[Mapping[TaskClass, frozenset[str]]] = MappingProxyType(
        {
            TaskClass.VISION: frozenset(
                {"image", "photograph", "drawing", "diagram", "screenshot", "scan"}
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
                {"calculate", "analyze", "compare", "reason", "derive"}
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

    def classify(self, prompt: str) -> TaskClass:
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

        for task_class in self._PRECEDENCE:
            if tokens & self._TOKEN_SIGNALS.get(task_class, frozenset()):
                return task_class
            if any(
                f" {phrase} " in normalized_tokens
                for phrase in self._PHRASE_SIGNALS.get(task_class, ())
            ):
                return task_class

        return TaskClass.GENERAL


def required_capabilities_for(task_class: TaskClass) -> tuple[str, ...]:
    """Map a supported task class to its provider-neutral capability requirement."""
    return TASK_CLASS_CAPABILITIES[task_class]
