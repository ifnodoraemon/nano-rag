from contextlib import asynccontextmanager

import os
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.api.auth import is_auth_disabled, require_api_key
from app.api.routes_business import router as business_router
from app.api.routes_debug import router as debug_router
from app.core.config import AppContainer
from app.core.exceptions import ConfigurationError, ModelGatewayError, ParsingError
from app.core.logging import configure_logging


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response


FRONTEND_DIST = Path(
    os.getenv(
        "FRONTEND_DIST_DIR", Path(__file__).resolve().parents[1] / "frontend" / "dist"
    )
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    app.state.container = AppContainer.from_env()
    yield
    await app.state.container.close()


app = FastAPI(title="nano-rag", version="0.1.0", lifespan=lifespan)

allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "")
if allowed_origins_str.strip():
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins_str.split(","),
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "X-API-Key", "X-Api-Key", "Content-Type"],
    )
app.add_middleware(SecurityHeadersMiddleware)

app.include_router(debug_router)
app.include_router(business_router)

if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/favicon.svg")
    async def favicon():
        favicon_path = FRONTEND_DIST / "favicon.svg"
        if favicon_path.exists():
            return FileResponse(favicon_path)
        return JSONResponse(status_code=404, content={"detail": "not found"})

    @app.get("/icons.svg")
    async def icons():
        icons_path = FRONTEND_DIST / "icons.svg"
        if icons_path.exists():
            return FileResponse(icons_path)
        return JSONResponse(status_code=404, content={"detail": "not found"})

    @app.get("/")
    async def index():
        index_path = FRONTEND_DIST / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        return JSONResponse(status_code=404, content={"detail": "frontend not built"})


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
                if 200 <= response.status_code < 400:
                    return True, None
                detail = response.text.strip()
                if detail:
                    errors.append(f"{path}: {response.status_code} {detail}")
                else:
                    errors.append(f"{path}: {response.status_code}")
            return False, "; ".join(errors) if errors else None
    except httpx.HTTPError as exc:
        detail = str(exc).strip() or exc.__class__.__name__
        return False, detail


@app.exception_handler(ModelGatewayError)
async def handle_model_gateway_error(
    _: Request, exc: ModelGatewayError
) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.exception_handler(ParsingError)
async def handle_parsing_error(_: Request, exc: ParsingError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.get("/health")
async def health(request: Request) -> dict[str, object]:
    app_obj = getattr(request, "app", None)
    app_state = getattr(app_obj, "state", None)
    container = getattr(app_state, "container", None)
    if container is None:
        return {"status": "ok"}
    return {"status": "ok", **_auth_state(container.config)}


def _auth_state(config) -> dict[str, object]:  # noqa: ANN001
    auth_disabled = is_auth_disabled()
    auth_configured = bool(config.business_api_keys)
    if auth_disabled:
        auth_status = "disabled"
    elif auth_configured:
        auth_status = "configured"
    else:
        auth_status = "missing_keys"
    return {
        "auth_enabled": not auth_disabled,
        "auth_configured": auth_configured,
        "auth_status": auth_status,
    }


@app.get("/health/detail", dependencies=[Depends(require_api_key)])
async def health_detail(request: Request) -> dict[str, object]:
    container = request.app.state.container
    config = container.config
    auth_state = _auth_state(config)
    capability_gateways: dict[str, dict[str, object]] = {}
    phoenix_ok = False
    phoenix_error: str | None = None
    phoenix_enabled = bool(config.phoenix_ui_endpoint)

    required_capabilities = ["generation", "embedding"]
    if config.rerank_enabled:
        required_capabilities.append("rerank")

    gateway_ok = True
    if config.gateway_mode == "mock":
        for capability in required_capabilities:
            capability_gateways[capability] = {
                "base_url": config.gateway_base_url,
                "reachable": True,
                "error": None,
            }
    else:
        for capability in required_capabilities:
            try:
                gateway = config.gateway_for(capability)
            except ConfigurationError as exc:
                capability_gateways[capability] = {
                    "base_url": None,
                    "reachable": False,
                    "error": str(exc),
                }
                gateway_ok = False
                continue
            reachable, error = await _probe_gateway(
                gateway["base_url"],
                gateway["api_key"],
                config.gateway_models_probe_paths,
            )
            capability_gateways[capability] = {
                "base_url": gateway["base_url"],
                "reachable": reachable,
                "error": error,
            }
            gateway_ok = gateway_ok and reachable

    if phoenix_enabled:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                headers = {}
                if config.phoenix_ui_host_header:
                    headers["Host"] = config.phoenix_ui_host_header
                response = await client.get(
                    config.phoenix_ui_endpoint, headers=headers
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

    features = {
        "wiki": bool(getattr(config, "wiki_enabled", False)),
        "hybrid_search": bool(getattr(config, "hybrid_search_enabled", False)),
        "semantic_chunker": bool(getattr(config, "semantic_chunker_enabled", False)),
        "query_rewrite": bool(getattr(config, "query_rewrite_enabled", False)),
        "diagnosis": bool(getattr(config, "diagnosis_enabled", False)),
        "eval": bool(getattr(config, "eval_enabled", False)),
        "benchmark": bool(getattr(config, "benchmark_enabled", False)),
    }

    status = "ok" if gateway_ok and vectorstore_status == "ok" else "degraded"
    generation_gateway = capability_gateways.get("generation", {})
    return {
        "status": status,
        "service": "nano-rag",
        "gateway_mode": config.gateway_mode,
        **auth_state,
        "vectorstore_backend": vectorstore_stats.get("backend", "unknown"),
        "parsed_dir": str(config.parsed_dir),
        "gateway": {
            "base_url": generation_gateway.get("base_url"),
            "reachable": gateway_ok,
            "error": next(
                (
                    details["error"]
                    for details in capability_gateways.values()
                    if details["error"]
                ),
                None,
            ),
            "capabilities": capability_gateways,
        },
        "phoenix": {
            "enabled": phoenix_enabled,
            "collector_endpoint": config.phoenix_collector_endpoint or None,
            "ui_endpoint": config.phoenix_ui_endpoint or None,
            "reachable": phoenix_ok,
            "error": phoenix_error,
        },
        "vectorstore": {
            "status": vectorstore_status,
            "error": vectorstore_error,
            "details": vectorstore_stats,
        },
        "features": features,
        "trace_count": container.trace_store.list().total,
    }
