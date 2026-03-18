from contextlib import asynccontextmanager

import os

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes_chat import router as chat_router
from app.api.routes_debug import router as debug_router
from app.api.routes_ingest import router as ingest_router
from app.core.config import AppContainer
from app.core.exceptions import ModelGatewayError
from app.core.logging import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    app.state.container = AppContainer.from_env()
    yield


app = FastAPI(title="nano-rag", version="0.1.0", lifespan=lifespan)
app.include_router(chat_router)
app.include_router(ingest_router)
app.include_router(debug_router)


@app.exception_handler(ModelGatewayError)
async def handle_model_gateway_error(_: Request, exc: ModelGatewayError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.get("/health")
async def health(request: Request) -> dict[str, object]:
    container = request.app.state.container
    gateway_ok = False
    gateway_error: str | None = None
    phoenix_ok = False
    phoenix_error: str | None = None
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(
                f"{container.config.gateway_base_url.rstrip('/')}/v1/models",
                headers={"Authorization": f"Bearer {container.config.gateway_api_key}"},
            )
            gateway_ok = response.status_code < 500
    except httpx.HTTPError as exc:
        gateway_error = str(exc)

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(container.config.phoenix_ui_endpoint)
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

    status = "ok" if gateway_ok and phoenix_ok and vectorstore_status == "ok" else "degraded"
    return {
        "status": status,
        "service": "nano-rag",
        "vectorstore_backend": os.getenv("VECTORSTORE_BACKEND", "memory"),
        "gateway_mode": container.config.gateway_mode,
        "gateway": {
            "base_url": container.config.gateway_base_url,
            "reachable": gateway_ok,
            "error": gateway_error,
        },
        "phoenix": {
            "collector_endpoint": container.config.phoenix_collector_endpoint,
            "ui_endpoint": container.config.phoenix_ui_endpoint,
            "reachable": phoenix_ok,
            "error": phoenix_error,
        },
        "vectorstore": {
            "status": vectorstore_status,
            "error": vectorstore_error,
            "details": vectorstore_stats,
        },
        "parsed_dir": str(container.config.parsed_dir),
        "trace_count": len(container.trace_store.list()),
    }
