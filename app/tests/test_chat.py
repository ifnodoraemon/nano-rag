from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.main import health


@pytest.mark.asyncio
async def test_health_route() -> None:
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
            phoenix_ui_endpoint="http://phoenix:6006",
            phoenix_ui_host_header="localhost",
            phoenix_collector_endpoint="http://phoenix:4317",
            parsed_dir="/tmp/parsed",
            rerank_enabled=False,
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
            if "gateway.local" in url or "phoenix:6006" in url:
                return FakeResponse(200)
            return FakeResponse(404)

    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(container=fake_container))
    )

    with patch("app.main.httpx.AsyncClient", FakeAsyncClient):
        payload = await health(request)
    assert payload["service"] == "nano-rag"
    assert payload["status"] == "ok"
    assert payload["gateway"]["base_url"] == "http://gateway.local"
    assert payload["phoenix"]["collector_endpoint"] == "http://phoenix:4317"
    assert payload["phoenix"]["ui_endpoint"] == "http://phoenix:6006"
    assert payload["parsed_dir"] == "/tmp/parsed"


@pytest.mark.asyncio
async def test_health_route_marks_gateway_4xx_as_degraded() -> None:
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
            phoenix_ui_endpoint="http://phoenix:6006",
            phoenix_ui_host_header="localhost",
            phoenix_collector_endpoint="http://phoenix:4317",
            parsed_dir="/tmp/parsed",
            rerank_enabled=False,
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
            if "phoenix:6006" in url:
                return FakeResponse(200)
            return FakeResponse(404)

    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(container=fake_container))
    )

    with patch("app.main.httpx.AsyncClient", FakeAsyncClient):
        payload = await health(request)
    assert payload["status"] == "degraded"
    assert payload["gateway"]["reachable"] is False
    assert payload["gateway"]["error"] is not None
    assert "400" in payload["gateway"]["error"]
    assert "User location is not supported" in payload["gateway"]["error"]
