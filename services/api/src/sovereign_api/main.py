"""FastAPI composition root for the first backend slice."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from sovereign_api.api_models import (
    ErrorResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    RoutingInformation,
)
from sovereign_api.body_limit import GenerateBodyLimitMiddleware
from sovereign_api.config import DeploymentEnvironment, load_settings
from sovereign_api.contracts import ModelRequest
from sovereign_api.errors import (
    InvalidOptimizerSelectionError,
    NoEligibleModelError,
    ProviderError,
    UnsupportedProviderError,
)
from sovereign_api.providers import (
    LocalOpenAICompatibleProvider,
    MockProvider,
    ModelProvider,
    VLLMProvider,
)
from sovereign_api.providers.local_openai_compatible import (
    PROVIDER_KEY as LOCAL_OPENAI_COMPATIBLE_PROVIDER_KEY,
)
from sovereign_api.providers.vllm import PROVIDER_KEY as VLLM_PROVIDER_KEY
from sovereign_api.registry import ModelRegistry, load_registry
from sovereign_api.routing import DeterministicModelRouter
from sovereign_api.task_classification import (
    DeterministicTaskClassifier,
    TaskClass,
    TaskClassifier,
    required_capabilities_for,
)


@dataclass(frozen=True, slots=True)
class Runtime:
    environment: DeploymentEnvironment
    router: DeterministicModelRouter
    providers: Mapping[str, ModelProvider]
    task_classifier: TaskClassifier


def configure_providers(
    registry: ModelRegistry, environment: DeploymentEnvironment
) -> dict[str, ModelProvider]:
    providers: dict[str, ModelProvider] = {}
    if environment is DeploymentEnvironment.DEVELOPMENT:
        providers["mock"] = MockProvider()

    if _provider_is_enabled(
        registry, environment, LOCAL_OPENAI_COMPATIBLE_PROVIDER_KEY
    ):
        providers[LOCAL_OPENAI_COMPATIBLE_PROVIDER_KEY] = (
            LocalOpenAICompatibleProvider.from_environment()
        )
    if _provider_is_enabled(registry, environment, VLLM_PROVIDER_KEY):
        providers[VLLM_PROVIDER_KEY] = VLLMProvider.from_environment()
    return providers


def _provider_is_enabled(
    registry: ModelRegistry,
    environment: DeploymentEnvironment,
    provider_key: str,
) -> bool:
    return any(
        model.enabled
        and environment in model.environments
        and model.provider == provider_key
        for model in registry.models
    )


def create_app(
    *,
    registry_path: Path | None = None,
    task_classifier: TaskClassifier | None = None,
) -> FastAPI:
    configured_task_classifier = (
        task_classifier
        if task_classifier is not None
        else DeterministicTaskClassifier()
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        settings = load_settings(registry_path=registry_path)
        registry = load_registry(settings.registry_path)
        application.state.runtime = Runtime(
            environment=settings.environment,
            router=DeterministicModelRouter(registry, settings.environment),
            providers=configure_providers(registry, settings.environment),
            task_classifier=configured_task_classifier,
        )
        yield

    application = FastAPI(title="Sovereign API", lifespan=lifespan)
    application.add_middleware(GenerateBodyLimitMiddleware)

    @application.exception_handler(RequestValidationError)
    async def malformed_request_handler(
        _request: Request, _error: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "malformed_request",
                    "message": "Request body failed validation",
                }
            },
        )

    @application.exception_handler(NoEligibleModelError)
    async def no_eligible_model_handler(
        _request: Request, error: NoEligibleModelError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"error": {"code": error.code, "message": str(error)}},
        )

    @application.exception_handler(InvalidOptimizerSelectionError)
    async def invalid_optimizer_selection_handler(
        _request: Request, error: InvalidOptimizerSelectionError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"error": {"code": error.code, "message": str(error)}},
        )

    @application.exception_handler(UnsupportedProviderError)
    async def unsupported_provider_handler(
        _request: Request, error: UnsupportedProviderError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"error": {"code": error.code, "message": str(error)}},
        )

    @application.exception_handler(ProviderError)
    async def provider_error_handler(
        _request: Request, error: ProviderError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=502,
            content={"error": {"code": error.code, "message": str(error)}},
        )

    @application.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="healthy", service="sovereign-api")

    @application.post(
        "/v1/generate",
        response_model=GenerateResponse,
        response_model_exclude_none=True,
        responses={
            413: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
            502: {"model": ErrorResponse},
        },
    )
    async def generate(request: GenerateRequest) -> GenerateResponse:
        runtime: Runtime = application.state.runtime
        task_class: TaskClass | None = None
        if request.required_capabilities is None:
            task_class = runtime.task_classifier.classify(request.prompt)
            required_capabilities = list(required_capabilities_for(task_class))
            capability_source = "inferred"
        else:
            required_capabilities = request.required_capabilities
            capability_source = "explicit"

        decision = runtime.router.route(frozenset(required_capabilities))
        provider = runtime.providers.get(decision.model.provider)
        if provider is None:
            raise UnsupportedProviderError(
                f"No provider adapter is configured for {decision.model.provider!r}"
            )

        result = await provider.generate(
            ModelRequest(model_id=decision.model.id, prompt=request.prompt)
        )
        return GenerateResponse(
            model_id=result.model_id,
            provider=decision.model.provider,
            content=result.content,
            routing=RoutingInformation(
                environment=runtime.environment,
                required_capabilities=required_capabilities,
                capability_source=(
                    capability_source
                    if runtime.environment is DeploymentEnvironment.DEVELOPMENT
                    else None
                ),
                task_class=(
                    task_class
                    if runtime.environment is DeploymentEnvironment.DEVELOPMENT
                    else None
                ),
            ),
        )

    return application


app = create_app()
