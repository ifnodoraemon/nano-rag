from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx

from app.core.exceptions import ModelGatewayError
from app.model_client.mock_gateway import mock_chat, mock_embeddings, mock_rerank

if TYPE_CHECKING:
    from app.core.config import AppConfig


class GatewayClient:
    def __init__(self, config: AppConfig, timeout_seconds: int, capability: str) -> None:
        self.config = config
        self.capability = capability
        gateway = config.gateway_for(capability)
        self.base_url = gateway["base_url"].rstrip("/")
        self.api_key = gateway["api_key"]
        self.timeout = timeout_seconds

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self.config.gateway_mode == "mock":
            return self._mock_post(path, payload)
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(f"{self.base_url}{path}", headers=self.headers, json=payload)
                response.raise_for_status()
                return response.json()
        except httpx.TimeoutException as exc:
            raise ModelGatewayError(
                f"{self.capability} gateway timeout on {path}. Check upstream provider settings."
            ) from exc
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text.strip()
            raise ModelGatewayError(
                f"{self.capability} gateway request failed on {path}: {exc.response.status_code} {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ModelGatewayError(f"{self.capability} gateway connection failed on {path}: {exc}") from exc

    def _mock_post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if path == "/embeddings":
            dimension = int(self.config.models.get("embedding", {}).get("dimension", 32))
            return mock_embeddings(payload.get("input", []), dimensions=dimension)
        if path == "/rerank":
            return mock_rerank(payload.get("query", ""), payload.get("documents", []), int(payload.get("top_n", 5)))
        if path == "/chat/completions":
            return mock_chat(payload.get("messages", []))
        raise ModelGatewayError(f"Mock model gateway does not support {path}")
