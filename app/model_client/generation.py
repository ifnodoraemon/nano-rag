from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from app.core.timeouts import timeout_seconds
from app.core.exceptions import (
    ConfigurationError,
    ModelGatewayError,
    ModelOutputError,
)
from app.model_client.base import GatewayClient
from app.utils.json_utils import parse_json_object

if TYPE_CHECKING:
    from app.core.config import AppConfig

logger = logging.getLogger(__name__)


class GenerationClient(GatewayClient):
    def __init__(self, config: AppConfig) -> None:
        super().__init__(
            config,
            timeout_seconds(
                config.settings,
                "generation_seconds",
                env_name="GENERATION_TIMEOUT_SECONDS",
                default=120,
            ),
            "generation",
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

        response_format = kwargs.get("response_format")
        result = await self._generate_once(messages, model_alias, **kwargs)
        if not _is_json_contract(response_format):
            return result
        if _parses_as_json(result["content"]):
            return result

        # An OpenAI-compatible gateway may accept `response_format` and silently
        # ignore it (common with smaller/self-hosted models). That is not a
        # parse failure on our side: the model simply never produced JSON.
        # Retry once with an explicit instruction appended - same strict
        # contract, stronger prompt. If it still returns prose, fail loudly and
        # say the provider ignored the schema, which is the actionable fact.
        logger.warning(
            "model %s returned prose despite response_format=%s; retrying once "
            "with an explicit JSON-only instruction",
            result.get("model"),
            _contract_name(response_format),
        )
        retried = await self._generate_once(
            _with_json_only_instruction(messages), model_alias, **kwargs
        )
        if _parses_as_json(retried["content"]):
            retried["structured_retry"] = True
            return retried
        raise ModelOutputError(
            "generation provider ignored the json_schema response format for "
            f"model {result.get('model')!r}: the response is not a JSON object "
            f"even after an explicit retry. Provider returned: "
            f"{_snippet(retried['content'])}"
        )

    async def _generate_once(
        self, messages: list[dict[str, Any]], model_alias: str | None = None, **kwargs: Any
    ) -> dict[str, Any]:
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
        message = choice.get("message", {}) or {}
        content = message.get("content") or ""
        if not str(content).strip() and not message.get("tool_calls"):
            # A reasoning model can spend the whole budget on reasoning tokens
            # and return an empty content with finish_reason=stop. Reported as
            # its own failure: downstream this looks like "model returned no
            # JSON", which sends operators chasing the wrong problem.
            reasoning = str(message.get("reasoning_content") or "")
            raise ModelGatewayError(
                "generation provider returned empty content for model "
                f"{model_alias or self.alias} (finish_reason={finish_reason!r}, "
                f"usage={data.get('usage')}). A reasoning model can consume the "
                "whole token budget before emitting content - raise "
                "GENERATION_MAX_TOKENS. "
                f"reasoning_chars={len(reasoning)}"
            )
        return {
            "content": content,
            "finish_reason": choice.get("finish_reason"),
            "usage": data.get("usage") or {},
            "model": data.get("model") or model_alias or self.alias,
            "raw": data,
        }


JSON_ONLY_INSTRUCTION = (
    "IMPORTANT: Your reply must be a single JSON object and nothing else - no "
    "prose before or after, no markdown fences. Start with '{' and end with '}'."
)


def _is_json_contract(response_format: object) -> bool:
    """True when the caller demanded a JSON schema payload."""
    if not isinstance(response_format, dict):
        return False
    return response_format.get("type") in {"json_schema", "json_object"}


def _contract_name(response_format: object) -> str:
    if isinstance(response_format, dict):
        return str(response_format.get("type"))
    return "none"


def _parses_as_json(content: object) -> bool:
    try:
        parsed = parse_json_object(str(content or ""))
    except ModelOutputError:
        return False
    return isinstance(parsed, dict)


def _with_json_only_instruction(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Append the JSON-only instruction to the last user message."""
    retried = [dict(message) for message in messages]
    for message in reversed(retried):
        if message.get("role") == "user":
            message["content"] = f"{message.get('content', '')}\n\n{JSON_ONLY_INSTRUCTION}"
            return retried
    retried.append({"role": "user", "content": JSON_ONLY_INSTRUCTION})
    return retried


def _snippet(content: object, limit: int = 300) -> str:
    text = str(content or "").strip().replace("\n", " ")
    return text[:limit] if text else "<empty>"
