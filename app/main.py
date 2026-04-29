from contextlib import asynccontextmanager

import logging
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

logger = logging.getLogger(__name__)


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
    _log_startup_readiness(app.state.container.config)
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


def _log_startup_readiness(config) -> None:  # noqa: ANN001
    document_parser = config.document_parser_configured
    if document_parser["enabled"] and not document_parser["configured"]:
        logger.warning(
            "Startup readiness: document parser is not configured. provider=%s model=%s missing=%s",
            document_parser.get("provider"),
            document_parser.get("model"),
            ", ".join(document_parser.get("missing", [])),
        )

    capabilities = ["generation", "embedding"]
    if config.rerank_enabled:
        capabilities.append("rerank")
    for capability in capabilities:
        try:
            config.gateway_for(capability)
        except ConfigurationError as exc:
            logger.warning(
                "Startup readiness: %s provider is not configured. %s",
                capability,
                exc,
            )

    if config.langfuse_otel_endpoint and not config.langfuse_otel_headers:
        logger.warning(
            "Startup readiness: Langfuse OTEL endpoint is configured but credentials are missing."
        )

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


async def _probe_embedding_gateway(config, gateway: dict[str, str]) -> tuple[bool, str | None]:  # noqa: ANN001
    provider = (
        os.getenv("EMBEDDING_PROVIDER")
        or getattr(config, "models", {}).get("embedding", {}).get("provider")
        or "gemini"
    ).lower()
    if provider != "gemini":
        return await _probe_gateway(
            gateway["base_url"],
            gateway["api_key"],
            config.gateway_models_probe_paths,
        )

    alias = (
        os.getenv("EMBEDDING_MODEL_ALIAS")
        or getattr(config, "models", {}).get("embedding", {}).get("default_alias")
        or "gemini-embedding-2-preview"
    )
    base_url = gateway["base_url"].rstrip("/")
    for suffix in ("/v1beta/openai", "/v1/openai", "/openai", "/v1beta", "/v1"):
        if base_url.endswith(suffix):
            base_url = base_url[: -len(suffix)]
            break
    url = f"{base_url}/v1beta/models/{alias}"
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(url, headers={"x-goog-api-key": gateway["api_key"]})
            if 200 <= response.status_code < 400:
                return True, None
            return False, f"{response.status_code} {response.text.strip()}"
    except httpx.HTTPError as exc:
        detail = str(exc).strip() or exc.__class__.__name__
        return False, detail


def _append_path(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


async def _probe_langfuse(
    ui_endpoint: str,
    otel_endpoint: str,
    otel_headers: dict[str, str],
) -> dict[str, object]:
    ui_ok = False
    ui_error: str | None = None
    otel_ok = False
    otel_error: str | None = None

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            if ui_endpoint:
                response = await client.get(_append_path(ui_endpoint, "/api/public/health"))
                ui_ok = response.status_code < 500
                if not ui_ok:
                    ui_error = f"HTTP {response.status_code} {response.text.strip()}"
            if otel_endpoint:
                response = await client.post(
                    otel_endpoint,
                    headers=otel_headers,
                    json={},
                )
                otel_ok = 200 <= response.status_code < 400
                if not otel_ok:
                    otel_error = f"HTTP {response.status_code} {response.text.strip()}"
    except httpx.HTTPError as exc:
        detail = str(exc).strip() or exc.__class__.__name__
        if ui_endpoint and not ui_ok:
            ui_error = detail
        if otel_endpoint and not otel_ok:
            otel_error = detail

    errors = "; ".join(error for error in (ui_error, otel_error) if error)
    return {
        "ui_reachable": ui_ok,
        "otel_reachable": otel_ok,
        "reachable": (not ui_endpoint or ui_ok) and (not otel_endpoint or otel_ok),
        "error": errors or None,
    }


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
    langfuse_enabled = bool(config.langfuse_ui_endpoint or config.langfuse_otel_endpoint)
    langfuse_status: dict[str, object] = {
        "ui_reachable": False,
        "otel_reachable": False,
        "reachable": False,
        "error": None,
    }
    document_parser = config.document_parser_configured

    required_capabilities = ["generation", "embedding"]
    if config.rerank_enabled:
        required_capabilities.append("rerank")

    gateway_ok = True
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
        if capability == "embedding":
            reachable, error = await _probe_embedding_gateway(config, gateway)
        else:
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

    if document_parser["enabled"] and not document_parser["configured"]:
        gateway_ok = False

    if langfuse_enabled:
        langfuse_status = await _probe_langfuse(
            config.langfuse_ui_endpoint,
            config.langfuse_otel_endpoint,
            getattr(config, "langfuse_otel_headers", {}),
        )

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
        "providers": {
            "document_parser": document_parser,
        },
        "langfuse": {
            "enabled": langfuse_enabled,
            "otel_endpoint": config.langfuse_otel_endpoint or None,
            "ui_endpoint": config.langfuse_ui_endpoint or None,
            **langfuse_status,
        },
        "vectorstore": {
            "status": vectorstore_status,
            "error": vectorstore_error,
            "details": vectorstore_stats,
        },
        "features": features,
        "trace_count": container.trace_store.list().total,
    }
