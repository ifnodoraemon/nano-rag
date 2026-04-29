from app.core.config import AppConfig
from app.model_client.base import GatewayClient


def test_gateway_client_headers_include_bearer_token() -> None:
    config = AppConfig(
        config_dir=None,  # type: ignore[arg-type]
        settings={},
        models={
            "model_gateway": {"base_url": "http://localhost:4000", "api_key": "secret"},
            "generation": {},
        },
        prompts={},
    )
    client = GatewayClient(config, timeout_seconds=5, capability="generation")
    assert client.headers["Authorization"] == "Bearer secret"


def test_gateway_api_key_prefers_explicit_env(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_GATEWAY_API_KEY", "gemini-secret")
    config = AppConfig(
        config_dir=None,  # type: ignore[arg-type]
        settings={},
        models={
            "model_gateway": {"base_url": "http://localhost:4000", "api_key": "secret"},
            "generation": {},
        },
        prompts={},
    )
    client = GatewayClient(config, timeout_seconds=5, capability="generation")
    assert client.headers["Authorization"] == "Bearer gemini-secret"


def test_phoenix_endpoints_are_disabled_by_default() -> None:
    config = AppConfig(
        config_dir=None,  # type: ignore[arg-type]
        settings={},
        models={
            "model_gateway": {"base_url": "http://localhost:4000", "api_key": "secret"},
            "generation": {},
        },
        prompts={},
    )
    assert config.phoenix_collector_endpoint == ""
    assert config.phoenix_ui_endpoint == ""
    assert config.phoenix_ui_host_header == ""


def test_capability_gateway_prefers_capability_specific_env(monkeypatch) -> None:
    monkeypatch.setenv("RERANK_API_BASE_URL", "http://rerank.example.com/v1")
    monkeypatch.setenv("RERANK_API_KEY", "rerank-secret")
    config = AppConfig(
        config_dir=None,  # type: ignore[arg-type]
        settings={},
        models={
            "model_gateway": {"base_url": "http://localhost:4000", "api_key": "secret"},
            "rerank": {"default_alias": "bge-reranker-v2"},
        },
        prompts={},
    )
    assert config.gateway_for("rerank") == {
        "base_url": "http://rerank.example.com/v1",
        "api_key": "rerank-secret",
    }


def test_capability_gateway_falls_back_to_global_gateway() -> None:
    config = AppConfig(
        config_dir=None,  # type: ignore[arg-type]
        settings={},
        models={
            "model_gateway": {"base_url": "http://localhost:4000", "api_key": "secret"},
            "embedding": {"default_alias": "gemini-embedding-2"},
        },
        prompts={},
    )
    assert config.gateway_for("embedding") == {
        "base_url": "http://localhost:4000",
        "api_key": "secret",
    }
