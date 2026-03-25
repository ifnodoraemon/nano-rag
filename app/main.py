from contextlib import asynccontextmanager

import os

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes_business import router as business_router
from app.api.routes_debug import router as debug_router
from app.core.config import AppContainer
from app.core.exceptions import ModelGatewayError
from app.core.logging import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    app.state.container = AppContainer.from_env()
    yield
    await app.state.container.close()


app = FastAPI(title="nano-rag", version="0.1.0", lifespan=lifespan)
app.include_router(debug_router)
app.include_router(business_router)


async def _probe_gateway(
    base_url: str, api_key: str, probe_paths: tuple[str, ...]
) -> tuple[bool, str | None]:
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            errors: list[str] = []
            for path in probe_paths:
                response = await client.get(
                    f"{base_url.rstrip('/')}{path}",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                if response.status_code < 500:
                    return True, None
                errors.append(f"{path}: {response.status_code}")
            return False, "; ".join(errors) if errors else None
    except httpx.HTTPError as exc:
        return False, str(exc)


@app.exception_handler(ModelGatewayError)
async def handle_model_gateway_error(
    _: Request, exc: ModelGatewayError
) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.get("/health")
async def health(request: Request) -> dict[str, object]:
    container = request.app.state.container
    capability_gateways: dict[str, dict[str, object]] = {}
    phoenix_ok = False
    phoenix_error: str | None = None

    required_capabilities = ["generation", "embedding"]
    if container.config.rerank_enabled:
        required_capabilities.append("rerank")

    gateway_ok = True
    for capability in required_capabilities:
        gateway = container.config.gateway_for(capability)
        reachable, error = await _probe_gateway(
            gateway["base_url"],
            gateway["api_key"],
            container.config.gateway_models_probe_paths,
        )
        capability_gateways[capability] = {
            "reachable": reachable,
            "error": error,
        }
        gateway_ok = gateway_ok and reachable

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            headers = {}
            if container.config.phoenix_ui_host_header:
                headers["Host"] = container.config.phoenix_ui_host_header
            response = await client.get(
                container.config.phoenix_ui_endpoint, headers=headers
            )
            phoenix_ok = response.status_code < 500
    except httpx.HTTPError as exc:
        phoenix_error = str(exc)

    vectorstore_status = "ok"
    vectorstore_error: str | None = None
    vectorstore_stats: dict[str, object] = {}
    try:
        vectorstore_stats = container.repository.stats()
    except Exception as exc:  # pragma: no cover
        vectorstore_status = "error"
        vectorstore_error = str(exc)

    status = (
        "ok" if gateway_ok and phoenix_ok and vectorstore_status == "ok" else "degraded"
    )
    return {
        "status": status,
        "service": "nano-rag",
        "vectorstore_backend": os.getenv("VECTORSTORE_BACKEND", "memory"),
        "gateway_mode": container.config.gateway_mode,
        "gateway": {
            "reachable": gateway_ok,
            "error": next(
                (
                    details["error"]
                    for details in capability_gateways.values()
                    if details["error"]
                ),
                None,
            ),
        },
        "phoenix": {
            "reachable": phoenix_ok,
            "error": phoenix_error,
        },
        "vectorstore": {
            "status": vectorstore_status,
            "error": vectorstore_error,
            "details": vectorstore_stats,
        },
        "trace_count": container.trace_store.list().total,
    }
