from app.core.config import AppConfig
from app.model_client.base import GatewayClient


def test_gateway_client_headers_include_bearer_token() -> None:
    config = AppConfig(
        config_dir=None,  # type: ignore[arg-type]
        settings={},
        models={"model_gateway": {"base_url": "http://localhost:4000", "api_key": "secret"}},
        prompts={},
    )
    client = GatewayClient(config, timeout_seconds=5)
    assert client.headers["Authorization"] == "Bearer secret"
