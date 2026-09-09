import pytest

from sovereign_api.config import DeploymentEnvironment
from sovereign_api.registry.models import ModelDefinition
from sovereign_api.routing import SovereignEligibilityFilter

from conftest import model_data


def build_model(**overrides: object) -> ModelDefinition:
    values = model_data("candidate")
    values.update(overrides)
    return ModelDefinition.model_validate(values)


def is_eligible(
    model: ModelDefinition,
    *,
    environment: DeploymentEnvironment = DeploymentEnvironment.DEVELOPMENT,
    capabilities: frozenset[str] = frozenset({"chat"}),
) -> bool:
    eligible = SovereignEligibilityFilter().filter(
        [model], environment=environment, required_capabilities=capabilities
    )
    return bool(eligible)


def test_disabled_model_is_rejected() -> None:
    assert not is_eligible(build_model(enabled=False))


def test_model_outside_active_environment_is_rejected() -> None:
    assert not is_eligible(
        build_model(environments=["development"]),
        environment=DeploymentEnvironment.ON_PREM,
    )


def test_model_missing_required_capability_is_rejected() -> None:
    assert not is_eligible(
        build_model(capabilities=["chat"]),
        capabilities=frozenset({"coding"}),
    )


@pytest.mark.parametrize(
    "environment",
    [DeploymentEnvironment.ON_PREM, DeploymentEnvironment.AIR_GAPPED],
)
def test_misconfigured_mock_model_is_rejected_before_ranking(
    environment: DeploymentEnvironment,
) -> None:
    model = build_model(provider="mock", environments=[environment.value])

    assert not is_eligible(model, environment=environment)
