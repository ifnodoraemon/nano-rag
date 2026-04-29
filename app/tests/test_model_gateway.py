import pytest

from app.core.config import AppConfig
from app.core.exceptions import ConfigurationError
from app.model_client.base import GatewayClient
from app.model_client.rerank import RerankClient


def test_gateway_client_headers_include_bearer_token() -> None:
    config = AppConfig(
        config_dir=None,  # type: ignore[arg-type]
        settings={},
        models={
            "model_gateway": {"base_url": "http://localhost:4000", "api_key": "secret"},
            "generation": {"base_url": "http://localhost:4000", "api_key": "secret"},
        },
        prompts={},
    )
    client = GatewayClient(config, timeout_seconds=5, capability="generation")
    assert client.headers["Authorization"] == "Bearer secret"


def test_gateway_api_key_prefers_capability_specific_env(monkeypatch) -> None:
    monkeypatch.setenv("GENERATION_API_KEY", "gemini-secret")
    config = AppConfig(
        config_dir=None,  # type: ignore[arg-type]
        settings={},
        models={
            "model_gateway": {"base_url": "http://localhost:4000", "api_key": "secret"},
            "generation": {"base_url": "http://localhost:4000", "api_key": "secret"},
        },
        prompts={},
    )
    client = GatewayClient(config, timeout_seconds=5, capability="generation")
    assert client.headers["Authorization"] == "Bearer gemini-secret"


def test_langfuse_endpoints_are_disabled_by_default() -> None:
    config = AppConfig(
        config_dir=None,  # type: ignore[arg-type]
        settings={},
        models={
            "model_gateway": {"base_url": "http://localhost:4000", "api_key": "secret"},
            "generation": {"base_url": "http://localhost:4000", "api_key": "secret"},
        },
        prompts={},
    )
    assert config.langfuse_otel_endpoint == ""
    assert config.langfuse_ui_endpoint == ""
    assert config.langfuse_otel_headers == {}


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


def test_capability_gateway_requires_explicit_capability_config() -> None:
    config = AppConfig(
        config_dir=None,  # type: ignore[arg-type]
        settings={},
        models={
            "model_gateway": {"base_url": "http://localhost:4000", "api_key": "secret"},
            "embedding": {"default_alias": "gemini-embedding-2"},
        },
        prompts={},
    )
    with pytest.raises(ConfigurationError, match="EMBEDDING_API_BASE_URL"):
        config.gateway_for("embedding")


@pytest.mark.asyncio
async def test_rerank_client_uses_configured_qwen_endpoint(monkeypatch) -> None:
    monkeypatch.delenv("DISABLE_RERANK", raising=False)
    config = AppConfig(
        config_dir=None,  # type: ignore[arg-type]
        settings={"timeout": {"rerank_seconds": 5}},
        models={
            "model_gateway": {"base_url": "", "api_key": ""},
            "rerank": {
                "default_alias": "qwen3-rerank",
                "base_url": "https://dashscope-intl.aliyuncs.com/compatible-api/v1",
                "api_key": "dashscope-secret",
                "path": "/reranks",
            },
        },
        prompts={},
    )

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.5},
                ]
            }

    class FakeAsyncClient:
        def __init__(self) -> None:
            self.calls = []

        async def post(self, url: str, **kwargs):  # noqa: ANN003
            self.calls.append({"url": url, **kwargs})
            return FakeResponse()

    client = RerankClient(config)
    fake_http = FakeAsyncClient()
    client.provider_client._client = fake_http  # noqa: SLF001

    results = await client.rerank("报销", ["规则 A", "规则 B"], top_k=2)

    assert [result.index for result in results] == [1, 0]
    assert fake_http.calls[0]["url"] == (
        "https://dashscope-intl.aliyuncs.com/compatible-api/v1/reranks"
    )
    assert fake_http.calls[0]["json"] == {
        "model": "qwen3-rerank",
        "query": "报销",
        "documents": ["规则 A", "规则 B"],
        "top_n": 2,
    }
