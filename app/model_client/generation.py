from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from app.core.exceptions import ConfigurationError, ModelGatewayError
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
        raw_max_tokens = os.getenv("GENERATION_MAX_TOKENS", "0")
        try:
            self.max_tokens = int(raw_max_tokens)
        except ValueError as exc:
            raise ConfigurationError(
                f"GENERATION_MAX_TOKENS must be an integer, got {raw_max_tokens!r}"
            ) from exc
        if self.max_tokens < 0:
            raise ConfigurationError("GENERATION_MAX_TOKENS must be >= 0")

    async def generate(
        self, messages: list[dict[str, Any]], model_alias: str | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        if self.max_tokens > 0:
            kwargs.setdefault("max_tokens", self.max_tokens)
        data = await self.provider_client.chat_completions(messages, model_alias or self.alias, **kwargs)
        choices = data.get("choices", [])
        if not choices:
            raise ModelGatewayError(
                "generation provider returned no choices for model "
                f"{model_alias or self.alias}"
            )
        choice = choices[0]
        finish_reason = choice.get("finish_reason")
        if finish_reason == "length":
            raise ModelGatewayError(
                "generation truncated (finish_reason=length) for model "
                f"{model_alias or self.alias}: the structured answer did not fit "
                "the token budget. Raise GENERATION_MAX_TOKENS or narrow the context."
            )
        message = choice.get("message", {})
        return {
            "content": message.get("content", ""),
            "finish_reason": choice.get("finish_reason"),
            "usage": data.get("usage") or {},
            "model": data.get("model") or model_alias or self.alias,
            "raw": data,
        }
