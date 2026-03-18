from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx

from app.core.exceptions import ModelGatewayError

if TYPE_CHECKING:
    from app.core.config import AppConfig


class GatewayClient:
    def __init__(self, config: AppConfig, timeout_seconds: int) -> None:
        self.base_url = config.gateway_base_url.rstrip("/")
        self.api_key = config.gateway_api_key
        self.timeout = timeout_seconds

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(f"{self.base_url}{path}", headers=self.headers, json=payload)
                response.raise_for_status()
                return response.json()
        except httpx.TimeoutException as exc:
            raise ModelGatewayError(f"Model gateway timeout on {path}. Check LiteLLM upstream provider settings.") from exc
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text.strip()
            raise ModelGatewayError(f"Model gateway request failed on {path}: {exc.response.status_code} {detail}") from exc
        except httpx.HTTPError as exc:
            raise ModelGatewayError(f"Model gateway connection failed on {path}: {exc}") from exc
