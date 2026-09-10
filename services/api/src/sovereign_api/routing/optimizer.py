"""Stage 2: provider-neutral advisory model optimization."""

from dataclasses import dataclass
from typing import Protocol

from sovereign_api.registry import ModelDefinition


@dataclass(frozen=True, slots=True)
class RoutingCandidate:
    """Immutable provider-neutral model state exposed to Stage 2."""

    model_id: str
    priority: int
    capabilities: frozenset[str]
    context_length: int

    @classmethod
    def from_model(cls, model: ModelDefinition) -> "RoutingCandidate":
        return cls(
            model_id=model.id,
            priority=model.priority,
            capabilities=frozenset(model.capabilities),
            context_length=model.context_length,
        )


class ModelOptimizer(Protocol):
    """Select one model from the Stage 1-approved candidate set."""

    def select(
        self, eligible_candidates: tuple[RoutingCandidate, ...]
    ) -> RoutingCandidate:
        """Return one of the supplied immutable candidates."""
        ...


class DeterministicModelOptimizer:
    """Apply the provider-neutral deterministic baseline ordering."""

    def select(
        self, eligible_candidates: tuple[RoutingCandidate, ...]
    ) -> RoutingCandidate:
        return min(
            eligible_candidates,
            key=lambda candidate: (candidate.priority, candidate.model_id),
        )
