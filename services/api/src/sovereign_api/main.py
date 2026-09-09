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
from sovereign_api.errors import NoEligibleModelError, UnsupportedProviderError
from sovereign_api.providers import MockProvider, ModelProvider
from sovereign_api.registry import load_registry
from sovereign_api.routing import DeterministicModelRouter


@dataclass(frozen=True, slots=True)
class Runtime:
    environment: DeploymentEnvironment
    router: DeterministicModelRouter
    providers: Mapping[str, ModelProvider]


def create_app(*, registry_path: Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        settings = load_settings(registry_path=registry_path)
        registry = load_registry(settings.registry_path)
        application.state.runtime = Runtime(
            environment=settings.environment,
            router=DeterministicModelRouter(registry, settings.environment),
            providers=(
                {"mock": MockProvider()}
                if settings.environment is DeploymentEnvironment.DEVELOPMENT
                else {}
            ),
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

    @application.exception_handler(UnsupportedProviderError)
    async def unsupported_provider_handler(
        _request: Request, error: UnsupportedProviderError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"error": {"code": error.code, "message": str(error)}},
        )

    @application.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="healthy", service="sovereign-api")

    @application.post(
        "/v1/generate",
        response_model=GenerateResponse,
        responses={
            413: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )
    async def generate(request: GenerateRequest) -> GenerateResponse:
        runtime: Runtime = application.state.runtime
        decision = runtime.router.route(frozenset(request.required_capabilities))
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
                required_capabilities=request.required_capabilities,
            ),
        )

    return application


app = create_app()
