from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from app.model_client.base import GatewayClient

if TYPE_CHECKING:
    from app.core.config import AppConfig

logger = logging.getLogger(__name__)


class GenerationClient(GatewayClient):
    def __init__(self, config: AppConfig) -> None:
        super().__init__(
            config, config.settings["timeout"]["generation_seconds"], "generation"
        )
        self.alias = config.models["generation"]["default_alias"]

    async def generate(
        self, messages: list[dict[str, Any]], model_alias: str | None = None
    ) -> dict[str, Any]:
        data = await self.provider_client.chat_completions(messages, model_alias or self.alias)
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
        finish_reason = choice.get("finish_reason")
        if finish_reason == "length":
            logger.warning(
                "generation truncated (finish_reason=length) for model %s",
                model_alias or self.alias,
            )
        message = choice.get("message", {})
        return {
            "content": message.get("content", ""),
            "finish_reason": choice.get("finish_reason"),
            "usage": data.get("usage") or {},
            "model": data.get("model") or model_alias or self.alias,
            "raw": data,
        }
