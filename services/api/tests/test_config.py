import pytest

from sovereign_api.config import DeploymentEnvironment, load_settings, parse_environment
from sovereign_api.errors import InvalidEnvironmentError
from sovereign_api.main import create_app

from conftest import start_app


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("development", DeploymentEnvironment.DEVELOPMENT),
        ("on-prem", DeploymentEnvironment.ON_PREM),
        ("air-gapped", DeploymentEnvironment.AIR_GAPPED),
    ],
)
def test_canonical_environment_is_accepted(
    value: str,
    expected: DeploymentEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SOVEREIGN_ENV", value)

    assert load_settings().environment is expected


def test_missing_environment_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SOVEREIGN_ENV", raising=False)

    with pytest.raises(InvalidEnvironmentError, match="must be explicitly set"):
        load_settings()


def test_unknown_environment_is_rejected() -> None:
    with pytest.raises(InvalidEnvironmentError, match="Invalid SOVEREIGN_ENV"):
        parse_environment("production")


def test_application_startup_fails_for_unknown_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SOVEREIGN_ENV", "production")

    with pytest.raises(InvalidEnvironmentError, match="Invalid SOVEREIGN_ENV"):
        start_app(create_app())


def test_application_startup_fails_for_missing_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SOVEREIGN_ENV", raising=False)

    with pytest.raises(InvalidEnvironmentError, match="must be explicitly set"):
        start_app(create_app())
