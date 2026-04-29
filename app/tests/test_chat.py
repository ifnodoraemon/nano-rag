from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

from app.main import app as fastapi_app
from app.main import health, health_detail


@pytest.mark.asyncio
async def test_health_returns_ok() -> None:
    result = await health(SimpleNamespace())
    assert result == {"status": "ok"}


@pytest.mark.asyncio
async def test_health_reports_auth_status_over_http(monkeypatch) -> None:
    monkeypatch.delenv("RAG_AUTH_DISABLED", raising=False)
    previous_container = getattr(fastapi_app.state, "container", None)
    fastapi_app.state.container = SimpleNamespace(
        config=SimpleNamespace(business_api_keys=set())
    )
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=fastapi_app),
            base_url="http://testserver",
        ) as client:
            response = await client.get("/health")
    finally:
        fastapi_app.state.container = previous_container

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "auth_enabled": True,
        "auth_configured": False,
        "auth_status": "missing_keys",
    }


@pytest.mark.asyncio
async def test_health_detail_rejects_missing_keys_over_http(monkeypatch) -> None:
    monkeypatch.delenv("RAG_AUTH_DISABLED", raising=False)
    previous_container = getattr(fastapi_app.state, "container", None)
    fastapi_app.state.container = SimpleNamespace(
        config=SimpleNamespace(business_api_keys=set())
    )
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=fastapi_app),
            base_url="http://testserver",
        ) as client:
            response = await client.get("/health/detail")
    finally:
        fastapi_app.state.container = previous_container

    assert response.status_code == 503
    assert "RAG_API_KEYS not configured" in response.json()["detail"]


@pytest.mark.asyncio
async def test_health_route(monkeypatch) -> None:
    monkeypatch.delenv("RAG_AUTH_DISABLED", raising=False)
    fake_trace_list = SimpleNamespace(total=0, items=[])
    fake_container = SimpleNamespace(
        config=SimpleNamespace(
            gateway_models_probe_paths=("/v1/models", "/models"),
            gateway_mode="live",
            gateway_base_url="http://gateway.local",
            gateway_for=lambda capability: {  # noqa: ARG005
                "base_url": "http://gateway.local",
                "api_key": "secret",
            },
            langfuse_ui_endpoint="http://langfuse:3000",
            langfuse_otel_endpoint="http://langfuse:3000/api/public/otel/v1/traces",
            document_parser_configured={
                "enabled": True,
                "provider": "gemini",
                "model": "gemini-3.1-pro-preview",
                "base_url": "https://generativelanguage.googleapis.com",
                "configured": True,
                "missing": [],
            },
            parsed_dir="/tmp/parsed",
            rerank_enabled=False,
            business_api_keys=set(),
        ),
        repository=SimpleNamespace(
            stats=lambda: {"backend": "memory", "documents": 0, "chunks": 0}
        ),
        trace_store=SimpleNamespace(list=lambda: fake_trace_list),
    )

    class FakeResponse:
        def __init__(self, status_code: int, text: str = ""):
            self.status_code = status_code
            self.text = text

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003, ARG002
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):  # noqa: ANN001, ARG002
            return None

        async def get(self, url, headers=None):  # noqa: ANN001, ARG002
            if "gateway.local" in url or "langfuse:3000" in url:
                return FakeResponse(200)
            return FakeResponse(404)

        async def post(self, url, headers=None, json=None):  # noqa: ANN001, ARG002
            if "langfuse:3000" in url:
                return FakeResponse(200)
            return FakeResponse(404)

    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(container=fake_container))
    )

    with patch("app.main.httpx.AsyncClient", FakeAsyncClient):
        payload = await health_detail(request)
    assert payload["service"] == "nano-rag"
    assert payload["status"] == "ok"
    assert payload["auth_enabled"] is True
    assert payload["auth_configured"] is False
    assert payload["auth_status"] == "missing_keys"
    assert payload["vectorstore_backend"] == "memory"
    assert payload["parsed_dir"] == "/tmp/parsed"
    assert payload["gateway"]["reachable"] is True
    assert payload["gateway"]["base_url"] == "http://gateway.local"
    assert payload["langfuse"]["reachable"] is True
    assert payload["langfuse"]["ui_reachable"] is True
    assert payload["langfuse"]["otel_reachable"] is True
    assert payload["langfuse"]["enabled"] is True
    assert payload["providers"]["document_parser"]["configured"] is True
    assert payload["vectorstore"]["details"]["backend"] == "memory"
    assert payload["features"]["wiki"] is False
    assert payload["features"]["eval"] is False
    assert payload["features"]["diagnosis"] is False


@pytest.mark.asyncio
async def test_health_route_marks_gateway_4xx_as_degraded(monkeypatch) -> None:
    monkeypatch.delenv("RAG_AUTH_DISABLED", raising=False)
    fake_trace_list = SimpleNamespace(total=0, items=[])
    fake_container = SimpleNamespace(
        config=SimpleNamespace(
            gateway_models_probe_paths=("/v1/models", "/models"),
            gateway_mode="live",
            gateway_base_url="http://gateway.local",
            gateway_for=lambda capability: {  # noqa: ARG005
                "base_url": "http://gateway.local",
                "api_key": "secret",
            },
            langfuse_ui_endpoint="http://langfuse:3000",
            langfuse_otel_endpoint="http://langfuse:3000/api/public/otel/v1/traces",
            document_parser_configured={
                "enabled": True,
                "provider": "gemini",
                "model": "gemini-3.1-pro-preview",
                "base_url": "https://generativelanguage.googleapis.com",
                "configured": True,
                "missing": [],
            },
            parsed_dir="/tmp/parsed",
            rerank_enabled=False,
            business_api_keys=set(),
        ),
        repository=SimpleNamespace(
            stats=lambda: {"backend": "memory", "documents": 0, "chunks": 0}
        ),
        trace_store=SimpleNamespace(list=lambda: fake_trace_list),
    )

    class FakeResponse:
        def __init__(self, status_code: int, text: str = ""):
            self.status_code = status_code
            self.text = text

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003, ARG002
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):  # noqa: ANN001, ARG002
            return None

        async def get(self, url, headers=None):  # noqa: ANN001, ARG002
            if "gateway.local" in url:
                return FakeResponse(
                    400,
                    '{"error":{"message":"User location is not supported for the API use."}}',
                )
            if "langfuse:3000" in url:
                return FakeResponse(200)
            return FakeResponse(404)

        async def post(self, url, headers=None, json=None):  # noqa: ANN001, ARG002
            if "langfuse:3000" in url:
                return FakeResponse(200)
            return FakeResponse(404)

    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(container=fake_container))
    )

    with patch("app.main.httpx.AsyncClient", FakeAsyncClient):
        payload = await health_detail(request)
    assert payload["status"] == "degraded"
    assert payload["gateway"]["reachable"] is False
    assert payload["gateway"]["error"] is not None
    assert "400" in payload["gateway"]["error"]
    assert "User location is not supported" in payload["gateway"]["error"]


@pytest.mark.asyncio
async def test_health_route_does_not_degrade_when_langfuse_is_disabled(monkeypatch) -> None:
    monkeypatch.delenv("RAG_AUTH_DISABLED", raising=False)
    fake_trace_list = SimpleNamespace(total=0, items=[])
    fake_container = SimpleNamespace(
        config=SimpleNamespace(
            gateway_models_probe_paths=("/v1/models", "/models"),
            gateway_mode="live",
            gateway_base_url="http://gateway.local",
            gateway_for=lambda capability: {  # noqa: ARG005
                "base_url": "http://gateway.local",
                "api_key": "secret",
            },
            langfuse_ui_endpoint="",
            langfuse_otel_endpoint="",
            document_parser_configured={
                "enabled": True,
                "provider": "gemini",
                "model": "gemini-3.1-pro-preview",
                "base_url": "https://generativelanguage.googleapis.com",
                "configured": True,
                "missing": [],
            },
            parsed_dir="/tmp/parsed",
            rerank_enabled=False,
            business_api_keys=set(),
        ),
        repository=SimpleNamespace(
            stats=lambda: {"backend": "memory", "documents": 0, "chunks": 0}
        ),
        trace_store=SimpleNamespace(list=lambda: fake_trace_list),
    )

    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(container=fake_container))
    )

    class FakeResponse:
        status_code = 200
        text = ""

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003, ARG002
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):  # noqa: ANN001, ARG002
            return None

        async def get(self, url, headers=None):  # noqa: ANN001, ARG002
            return FakeResponse()

    with patch("app.main.httpx.AsyncClient", FakeAsyncClient):
        payload = await health_detail(request)

    assert payload["status"] == "ok"
    assert payload["langfuse"]["enabled"] is False
    assert payload["langfuse"]["reachable"] is False


@pytest.mark.asyncio
async def test_health_route_respects_auth_disabled_override(monkeypatch) -> None:
    monkeypatch.setenv("RAG_AUTH_DISABLED", "true")
    fake_trace_list = SimpleNamespace(total=0, items=[])
    fake_container = SimpleNamespace(
        config=SimpleNamespace(
            gateway_models_probe_paths=("/v1/models", "/models"),
            gateway_mode="live",
            gateway_base_url="http://gateway.local",
            gateway_for=lambda capability: {  # noqa: ARG005
                "base_url": "http://gateway.local",
                "api_key": "secret",
            },
            langfuse_ui_endpoint="",
            langfuse_otel_endpoint="",
            document_parser_configured={
                "enabled": True,
                "provider": "gemini",
                "model": "gemini-3.1-pro-preview",
                "base_url": "https://generativelanguage.googleapis.com",
                "configured": True,
                "missing": [],
            },
            parsed_dir="/tmp/parsed",
            rerank_enabled=False,
            business_api_keys={"secret"},
        ),
        repository=SimpleNamespace(
            stats=lambda: {"backend": "memory", "documents": 0, "chunks": 0}
        ),
        trace_store=SimpleNamespace(list=lambda: fake_trace_list),
    )

    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(container=fake_container))
    )

    class FakeResponse:
        status_code = 200
        text = ""

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003, ARG002
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):  # noqa: ANN001, ARG002
            return None

        async def get(self, url, headers=None):  # noqa: ANN001, ARG002
            return FakeResponse()

    with patch("app.main.httpx.AsyncClient", FakeAsyncClient):
        payload = await health_detail(request)

    assert payload["auth_enabled"] is False
    assert payload["auth_configured"] is True
    assert payload["auth_status"] == "disabled"


@pytest.mark.asyncio
async def test_health_route_reports_configured_auth(monkeypatch) -> None:
    monkeypatch.delenv("RAG_AUTH_DISABLED", raising=False)
    fake_trace_list = SimpleNamespace(total=0, items=[])
    fake_container = SimpleNamespace(
        config=SimpleNamespace(
            gateway_models_probe_paths=("/v1/models", "/models"),
            gateway_mode="live",
            gateway_base_url="http://gateway.local",
            gateway_for=lambda capability: {  # noqa: ARG005
                "base_url": "http://gateway.local",
                "api_key": "secret",
            },
            langfuse_ui_endpoint="",
            langfuse_otel_endpoint="",
            document_parser_configured={
                "enabled": True,
                "provider": "gemini",
                "model": "gemini-3.1-pro-preview",
                "base_url": "https://generativelanguage.googleapis.com",
                "configured": True,
                "missing": [],
            },
            parsed_dir="/tmp/parsed",
            rerank_enabled=False,
            business_api_keys={"secret"},
        ),
        repository=SimpleNamespace(
            stats=lambda: {"backend": "memory", "documents": 0, "chunks": 0}
        ),
        trace_store=SimpleNamespace(list=lambda: fake_trace_list),
    )
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(container=fake_container))
    )

    class FakeResponse:
        status_code = 200
        text = ""

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003, ARG002
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):  # noqa: ANN001, ARG002
            return None

        async def get(self, url, headers=None):  # noqa: ANN001, ARG002
            return FakeResponse()

    with patch("app.main.httpx.AsyncClient", FakeAsyncClient):
        payload = await health_detail(request)

    assert payload["auth_enabled"] is True
    assert payload["auth_configured"] is True
    assert payload["auth_status"] == "configured"
