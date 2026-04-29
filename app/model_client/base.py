from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx

from app.core.exceptions import ModelGatewayError

if TYPE_CHECKING:
    from app.core.config import AppConfig


@dataclass(frozen=True)
class ProviderConfig:
    capability: str
    provider: str
    model: str
    base_url: str
    api_key: str

    def require_ready(self, *, require_model: bool = True) -> None:
        if not self.base_url:
            raise ModelGatewayError(
                f"{self.capability.upper()}_API_BASE_URL must be configured explicitly."
            )
        if not self.api_key:
            raise ModelGatewayError(
                f"{self.capability.upper()}_API_KEY must be configured explicitly."
            )
        if require_model and not self.model:
            raise ModelGatewayError(
                f"{self.capability.upper()} model alias must be configured explicitly."
            )


class AsyncJsonProviderClient:
    def __init__(self, provider: ProviderConfig, timeout_seconds: int) -> None:
        self.provider = provider
        self.timeout = timeout_seconds
        self._client: httpx.AsyncClient | None = None

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.provider.api_key}",
            "Content-Type": "application/json",
        }

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.provider.require_ready(require_model=False)
        try:
            response = await self._get_client().post(
                f"{self.provider.base_url}{path}", headers=self.headers, json=payload
            )
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException as exc:
            raise ModelGatewayError(
                f"{self.provider.capability} provider timeout on {path}. Check upstream provider settings."
            ) from exc
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text.strip()
            raise ModelGatewayError(
                f"{self.provider.capability} provider request failed on {path}: "
                f"{exc.response.status_code} {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ModelGatewayError(
                f"{self.provider.capability} provider connection failed on {path}: {exc}"
            ) from exc

    async def chat_completions(
        self, messages: list[dict[str, Any]], model_alias: str | None = None, **extra: Any
    ) -> dict[str, Any]:
        payload = {
            "model": model_alias or self.provider.model,
            "messages": messages,
            **extra,
        }
        return await self.post_json("/chat/completions", payload)


class GatewayClient:
    def __init__(
        self, config: AppConfig, timeout_seconds: int, capability: str
    ) -> None:
        self.config = config
        self.capability = capability
        gateway = config.gateway_for(capability, validate=False)
        model = (
            config.models.get(capability, {}).get("default_alias")
            or config.models.get("model_gateway", {}).get("default_alias")
            or ""
        )
        self.provider_config = ProviderConfig(
            capability=capability,
            provider=str(config.models.get(capability, {}).get("provider") or "openai-compatible"),
            model=str(model),
            base_url=gateway["base_url"].rstrip("/"),
            api_key=gateway["api_key"],
        )
        self.provider_client = AsyncJsonProviderClient(
            self.provider_config, timeout_seconds
        )
        self.base_url = self.provider_config.base_url
        self.api_key = self.provider_config.api_key
        self.timeout = timeout_seconds
        self._client: httpx.AsyncClient | None = None

    @property
    def headers(self) -> dict[str, str]:
        return self.provider_client.headers

    def _get_client(self) -> httpx.AsyncClient:
        return self.provider_client._get_client()

    async def close(self) -> None:
        await self.provider_client.close()
        self._client = None

    async def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return await self.provider_client.post_json(path, payload)
