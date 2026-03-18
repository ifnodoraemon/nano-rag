from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.model_client.base import GatewayClient

if TYPE_CHECKING:
    from app.core.config import AppConfig


class GenerationClient(GatewayClient):
    def __init__(self, config: AppConfig) -> None:
        super().__init__(config, config.settings["timeout"]["generation_seconds"])
        self.alias = config.models["generation"]["default_alias"]

    async def generate(self, messages: list[dict[str, str]], model_alias: str | None = None) -> dict[str, Any]:
        payload = {"model": model_alias or self.alias, "messages": messages}
        data = await self.post("/chat/completions", payload)
        choice = data["choices"][0]["message"]
        return {"content": choice["content"], "raw": data}
