import pytest

from sovereign_api.config import DeploymentEnvironment
from sovereign_api.errors import NoEligibleModelError
from sovereign_api.registry.models import ModelRegistry
from sovereign_api.routing import DeterministicModelRouter

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
