from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.model_client.base import GatewayClient

if TYPE_CHECKING:
    from app.core.config import AppConfig


class GenerationClient(GatewayClient):
    def __init__(self, config: AppConfig) -> None:
        super().__init__(
            config, config.settings["timeout"]["generation_seconds"], "generation"
        )
        self.alias = config.models["generation"]["default_alias"]

    async def generate(
        self, messages: list[dict[str, str]], model_alias: str | None = None
    ) -> dict[str, Any]:
        payload = {"model": model_alias or self.alias, "messages": messages}
        data = await self.post("/chat/completions", payload)
        choices = data.get("choices", [])
        if not choices:
            return {
                "content": "",
                "finish_reason": None,
                "usage": data.get("usage") or {},
                "model": data.get("model") or model_alias or self.alias,
                "raw": data,
            }
        choice = choices[0]
        message = choice.get("message", {})
        return {
            "content": message.get("content", ""),
            "finish_reason": choice.get("finish_reason"),
            "usage": data.get("usage") or {},
            "model": data.get("model") or model_alias or self.alias,
            "raw": data,
        }
