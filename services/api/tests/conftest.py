import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx
import yaml
from fastapi import FastAPI


def write_registry(tmp_path: Path, data: Mapping[str, Any]) -> Path:
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(yaml.safe_dump(dict(data)), encoding="utf-8")
    return registry_path


def registry_data(*models: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema_version": 1, "models": list(models)}


def model_data(
    model_id: str,
    *,
    provider: str = "mock",
    enabled: bool = True,
    environments: list[str] | None = None,
    capabilities: list[str] | None = None,
    priority: int = 10,
) -> dict[str, Any]:
    return {
        "id": model_id,
        "provider": provider,
        "enabled": enabled,
        "environments": environments or ["development"],
        "capabilities": capabilities or ["chat"],
        "context_length": 8192,
        "priority": priority,
        "metadata": {"description": "Test model"},
    }


def request_app(
    app: FastAPI,
    method: str,
    path: str,
    *,
    json: dict[str, Any] | None = None,
) -> httpx.Response:
    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                return await client.request(method, path, json=json)

    return asyncio.run(request())


def start_app(app: FastAPI) -> None:
    async def start() -> None:
        async with app.router.lifespan_context(app):
            pass

    asyncio.run(start())
