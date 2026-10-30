import pytest

from app.core.config import AppConfig
from app.core.config import AppContainer
from app.core.exceptions import ConfigurationError
from app.model_client.base import GatewayClient


def _config(**models) -> AppConfig:
    return AppConfig(
        config_dir=None,  # type: ignore[arg-type]
        settings={},
        models=models or {
            "model_gateway": {"base_url": "http://localhost:4000", "api_key": "secret"},
            "generation": {"base_url": "http://localhost:4000", "api_key": "secret"},
        },
        prompts={},
    )


def test_gateway_client_headers_include_bearer_token() -> None:
    client = GatewayClient(_config(), timeout_seconds=5, capability="generation")
    assert client.headers["Authorization"] == "Bearer secret"


def test_gateway_api_key_prefers_capability_specific_env(monkeypatch) -> None:
    monkeypatch.setenv("GENERATION_API_KEY", "gemini-secret")
    client = GatewayClient(_config(), timeout_seconds=5, capability="generation")
    assert client.headers["Authorization"] == "Bearer gemini-secret"


def test_langfuse_endpoints_are_disabled_by_default() -> None:
    config = _config()
    assert config.langfuse_otel_endpoint == ""
    assert config.langfuse_ui_endpoint == ""
    assert config.langfuse_otel_headers == {}


def test_capability_gateway_requires_explicit_capability_config(monkeypatch) -> None:
    monkeypatch.delenv("DOCUMENT_PARSER_API_BASE_URL", raising=False)
    monkeypatch.delenv("DOCUMENT_PARSER_API_KEY", raising=False)
    config = _config(
        model_gateway={"base_url": "http://localhost:4000", "api_key": "secret"},
        document_parser={"default_alias": "configured-model"},
    )
    with pytest.raises(ConfigurationError, match="DOCUMENT_PARSER_API_BASE_URL"):
        config.gateway_for("document_parser")


def test_generation_gateway_prefers_capability_specific_env(monkeypatch) -> None:
    monkeypatch.setenv("GENERATION_API_BASE_URL", "http://gen.example.com/v1")
    monkeypatch.setenv("GENERATION_API_KEY", "gen-secret")
    config = _config(
        model_gateway={"base_url": "http://localhost:4000", "api_key": "secret"},
        generation={"default_alias": "configured-model"},
    )
    assert config.gateway_for("generation") == {
        "base_url": "http://gen.example.com/v1",
        "api_key": "gen-secret",
    }


@pytest.mark.asyncio
async def test_app_container_close_closes_every_client() -> None:
    """One model to run them all: generation, document parser, graph store.

    There is no embedding client and no rerank client to close - this build runs
    a single generation model.
    """
    closed: list[str] = []

    class AsyncClosable:
        def __init__(self, name: str) -> None:
            self.name = name

        async def close(self) -> None:
            closed.append(self.name)

    class SyncGraphStore:
        def close(self) -> None:
            closed.append("graph_store")

    container = AppContainer(
        config=None,  # type: ignore[arg-type]
        generation_client=AsyncClosable("generation"),  # type: ignore[arg-type]
        document_parser=AsyncClosable("document_parser"),  # type: ignore[arg-type]
        ingestion_pipeline=None,  # type: ignore[arg-type]
        chat_pipeline=None,  # type: ignore[arg-type]
        eval_runner=None,
        trace_store=None,  # type: ignore[arg-type]
        tracing_manager=None,  # type: ignore[arg-type]
        diagnosis_service=None,
        feedback_store=None,  # type: ignore[arg-type]
        knowledge_base_catalog=None,  # type: ignore[arg-type]
        graph_store=SyncGraphStore(),
    )

    await container.close()

    assert closed == ["generation", "document_parser", "graph_store"]
