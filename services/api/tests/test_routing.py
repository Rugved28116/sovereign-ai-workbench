from collections.abc import Iterable
from dataclasses import FrozenInstanceError, replace

import pytest

from sovereign_api.config import DeploymentEnvironment
from sovereign_api.errors import InvalidOptimizerSelectionError, NoEligibleModelError
from sovereign_api.registry.models import ModelDefinition, ModelRegistry
from sovereign_api.routing import (
    DeterministicModelOptimizer,
    DeterministicModelRouter,
    RoutingCandidate,
    SovereignEligibilityFilter,
)

from conftest import model_data, registry_data


def build_router(
    *models: dict[str, object],
    environment: DeploymentEnvironment = DeploymentEnvironment.DEVELOPMENT,
) -> DeterministicModelRouter:
    registry = ModelRegistry.model_validate(registry_data(*models))
    return DeterministicModelRouter(registry, environment)


def test_lower_priority_number_wins() -> None:
    router = build_router(
        model_data("priority-20", priority=20),
        model_data("priority-10", priority=10),
    )

    assert router.route(frozenset({"chat"})).model.id == "priority-10"


def test_priority_tie_uses_lexicographically_smaller_model_id() -> None:
    router = build_router(
        model_data("model-z", priority=10),
        model_data("model-a", priority=10),
    )

    assert router.route(frozenset({"chat"})).model.id == "model-a"


def test_deterministic_selection_is_stable_across_repeated_calls() -> None:
    router = build_router(
        model_data("model-z", priority=10),
        model_data("model-a", priority=10),
        model_data("priority-20", priority=20),
    )

    selections = [
        router.route(frozenset({"chat"})).model.id for _ in range(5)
    ]

    assert selections == ["model-a"] * 5


def test_falsey_optimizer_is_preserved_and_invoked() -> None:
    class FalseyOptimizer:
        def __init__(self) -> None:
            self.invoked = False

        def __bool__(self) -> bool:
            return False

        def select(
            self, eligible_candidates: tuple[RoutingCandidate, ...]
        ) -> RoutingCandidate:
            self.invoked = True
            return next(
                candidate
                for candidate in eligible_candidates
                if candidate.model_id == "priority-20"
            )

    registry = ModelRegistry.model_validate(
        registry_data(
            model_data("priority-10", priority=10),
            model_data("priority-20", priority=20),
            model_data("disabled", enabled=False, priority=1),
        )
    )
    optimizer = FalseyOptimizer()
    router = DeterministicModelRouter(
        registry,
        DeploymentEnvironment.DEVELOPMENT,
        optimizer=optimizer,
    )

    decision = router.route(frozenset({"chat"}))

    assert optimizer.invoked
    assert decision.model.id == "priority-20"


def test_no_eligible_model_returns_typed_error() -> None:
    router = build_router(model_data("chat-only", capabilities=["chat"]))

    with pytest.raises(NoEligibleModelError):
        router.route(frozenset({"coding"}))


@pytest.mark.parametrize(
    "environment",
    [DeploymentEnvironment.ON_PREM, DeploymentEnvironment.AIR_GAPPED],
)
def test_misconfigured_mock_model_never_reaches_selection(
    environment: DeploymentEnvironment,
) -> None:
    router = build_router(
        model_data("misconfigured-mock", environments=[environment.value]),
        environment=environment,
    )

    with pytest.raises(NoEligibleModelError):
        router.route(frozenset({"chat"}))


def test_provider_identity_does_not_change_deterministic_ordering() -> None:
    router = build_router(
        model_data("model-z", provider="mock", priority=10),
        model_data(
            "model-a", provider="local-openai-compatible", priority=10
        ),
    )

    decision = router.route(frozenset({"chat"}))

    assert decision.model.id == "model-a"
    assert decision.model.provider == "local-openai-compatible"


def test_stage_1_runs_before_stage_2_and_only_passes_eligible_models() -> None:
    events: list[str] = []
    received_candidates: tuple[RoutingCandidate, ...] = ()

    class RecordingEligibilityFilter(SovereignEligibilityFilter):
        def filter(
            self,
            models: Iterable[ModelDefinition],
            *,
            environment: DeploymentEnvironment,
            required_capabilities: frozenset[str],
        ) -> tuple[ModelDefinition, ...]:
            events.append("stage-1")
            return super().filter(
                models,
                environment=environment,
                required_capabilities=required_capabilities,
            )

    class RecordingOptimizer:
        def select(
            self, eligible_candidates: tuple[RoutingCandidate, ...]
        ) -> RoutingCandidate:
            nonlocal received_candidates
            events.append("stage-2")
            received_candidates = eligible_candidates
            return DeterministicModelOptimizer().select(eligible_candidates)

    registry = ModelRegistry.model_validate(
        registry_data(
            model_data("eligible", priority=20),
            model_data("disabled", enabled=False, priority=1),
            model_data("missing-capability", capabilities=["coding"], priority=1),
        )
    )
    router = DeterministicModelRouter(
        registry,
        DeploymentEnvironment.DEVELOPMENT,
        eligibility_filter=RecordingEligibilityFilter(),
        optimizer=RecordingOptimizer(),
    )

    decision = router.route(frozenset({"chat"}))

    assert events == ["stage-1", "stage-2"]
    assert tuple(candidate.model_id for candidate in received_candidates) == (
        "eligible",
    )
    assert decision.model.id == "eligible"


def test_stage_2_is_not_called_when_stage_1_rejects_every_model() -> None:
    class UnexpectedOptimizer:
        def select(
            self, eligible_candidates: tuple[RoutingCandidate, ...]
        ) -> RoutingCandidate:
            pytest.fail("Stage 2 must not run without an eligible candidate")

    registry = ModelRegistry.model_validate(
        registry_data(model_data("disabled", enabled=False))
    )
    router = DeterministicModelRouter(
        registry,
        DeploymentEnvironment.DEVELOPMENT,
        optimizer=UnexpectedOptimizer(),
    )

    with pytest.raises(NoEligibleModelError):
        router.route(frozenset({"chat"}))


@pytest.mark.parametrize("selection", ["ineligible", "unknown"])
def test_optimizer_cannot_select_outside_stage_1_candidates(selection: str) -> None:
    eligible = ModelDefinition.model_validate(model_data("eligible"))
    ineligible = ModelDefinition.model_validate(
        model_data("ineligible", enabled=False, priority=1)
    )
    unknown = ModelDefinition.model_validate(model_data("unknown", priority=1))

    class InvalidOptimizer:
        def select(
            self, eligible_candidates: tuple[RoutingCandidate, ...]
        ) -> RoutingCandidate:
            assert tuple(
                candidate.model_id for candidate in eligible_candidates
            ) == ("eligible",)
            selected_model = ineligible if selection == "ineligible" else unknown
            return RoutingCandidate.from_model(selected_model)

    registry = ModelRegistry.model_validate(
        registry_data(
            eligible.model_dump(mode="json"),
            ineligible.model_dump(mode="json"),
        )
    )
    router = DeterministicModelRouter(
        registry,
        DeploymentEnvironment.DEVELOPMENT,
        optimizer=InvalidOptimizer(),
    )

    with pytest.raises(InvalidOptimizerSelectionError):
        router.route(frozenset({"chat"}))


def test_invalid_optimizer_selection_has_no_fallback_bypass() -> None:
    class RejectedModelOptimizer:
        def __init__(self, rejected_model: ModelDefinition) -> None:
            self._rejected_candidate = RoutingCandidate.from_model(rejected_model)

        def select(
            self, eligible_candidates: tuple[RoutingCandidate, ...]
        ) -> RoutingCandidate:
            return self._rejected_candidate

    registry = ModelRegistry.model_validate(
        registry_data(
            model_data("eligible", priority=10),
            model_data("rejected", enabled=False, priority=1),
        )
    )
    rejected_model = next(model for model in registry.models if not model.enabled)
    router = DeterministicModelRouter(
        registry,
        DeploymentEnvironment.DEVELOPMENT,
        optimizer=RejectedModelOptimizer(rejected_model),
    )

    with pytest.raises(InvalidOptimizerSelectionError):
        router.route(frozenset({"chat"}))


@pytest.mark.parametrize("field", ["model_id", "priority", "context_length"])
def test_optimizer_cannot_mutate_candidate_fields(field: str) -> None:
    class MutationAttemptOptimizer:
        def select(
            self, eligible_candidates: tuple[RoutingCandidate, ...]
        ) -> RoutingCandidate:
            candidate = eligible_candidates[0]
            with pytest.raises(FrozenInstanceError):
                setattr(candidate, field, "forged")
            return candidate

    registry = ModelRegistry.model_validate(registry_data(model_data("eligible")))
    router = DeterministicModelRouter(
        registry,
        DeploymentEnvironment.DEVELOPMENT,
        optimizer=MutationAttemptOptimizer(),
    )

    assert router.route(frozenset({"chat"})).model.id == "eligible"


def test_optimizer_cannot_mutate_candidate_capabilities() -> None:
    class CapabilityMutationOptimizer:
        def select(
            self, eligible_candidates: tuple[RoutingCandidate, ...]
        ) -> RoutingCandidate:
            candidate = eligible_candidates[0]
            assert isinstance(candidate.capabilities, frozenset)
            with pytest.raises(AttributeError):
                candidate.capabilities.add("forged")  # type: ignore[attr-defined]
            return candidate

    registry = ModelRegistry.model_validate(registry_data(model_data("eligible")))
    router = DeterministicModelRouter(
        registry,
        DeploymentEnvironment.DEVELOPMENT,
        optimizer=CapabilityMutationOptimizer(),
    )

    decision = router.route(frozenset({"chat"}))

    assert decision.model.capabilities == ("chat",)


@pytest.mark.parametrize(
    "hidden_field", ["enabled", "provider", "environments", "metadata"]
)
def test_optimizer_has_no_reference_to_hidden_model_state(hidden_field: str) -> None:
    class HiddenStateMutationOptimizer:
        def select(
            self, eligible_candidates: tuple[RoutingCandidate, ...]
        ) -> RoutingCandidate:
            candidate = eligible_candidates[0]
            assert not hasattr(candidate, hidden_field)
            with pytest.raises(AttributeError):
                object.__setattr__(candidate, hidden_field, ("forged",))
            return candidate

    registry = ModelRegistry.model_validate(registry_data(model_data("eligible")))
    approved_state = registry.models[0].model_dump(mode="python")
    router = DeterministicModelRouter(
        registry,
        DeploymentEnvironment.DEVELOPMENT,
        optimizer=HiddenStateMutationOptimizer(),
    )

    router.route(frozenset({"chat"}))

    assert registry.models[0].model_dump(mode="python") == approved_state


def test_forged_candidate_state_fails_closed() -> None:
    class ForgedCandidateOptimizer:
        def select(
            self, eligible_candidates: tuple[RoutingCandidate, ...]
        ) -> RoutingCandidate:
            return replace(eligible_candidates[0], priority=-1)

    registry = ModelRegistry.model_validate(registry_data(model_data("eligible")))
    router = DeterministicModelRouter(
        registry,
        DeploymentEnvironment.DEVELOPMENT,
        optimizer=ForgedCandidateOptimizer(),
    )

    with pytest.raises(InvalidOptimizerSelectionError):
        router.route(frozenset({"chat"}))


def test_malicious_in_place_mutation_cannot_change_stage_1_approval() -> None:
    class MaliciousOptimizer:
        def select(
            self, eligible_candidates: tuple[RoutingCandidate, ...]
        ) -> RoutingCandidate:
            candidate = eligible_candidates[0]
            object.__setattr__(candidate, "priority", -1)
            object.__setattr__(candidate, "capabilities", frozenset({"coding"}))
            return candidate

    registry = ModelRegistry.model_validate(
        registry_data(
            model_data(
                "eligible",
                provider="mock",
                environments=["development"],
                capabilities=["chat"],
                priority=10,
            )
        )
    )
    approved_state = registry.models[0].model_dump(mode="python")
    router = DeterministicModelRouter(
        registry,
        DeploymentEnvironment.DEVELOPMENT,
        optimizer=MaliciousOptimizer(),
    )

    with pytest.raises(InvalidOptimizerSelectionError):
        router.route(frozenset({"chat"}))

    assert registry.models[0].model_dump(mode="python") == approved_state
