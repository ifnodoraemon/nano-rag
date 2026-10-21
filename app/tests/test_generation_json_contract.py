"""The JSON contract when a provider ignores `response_format`.

OpenAI-compatible gateways commonly accept a `json_schema` response format and
silently ignore it (smaller / self-hosted models are the usual case). The
contract must stay strict - a prose answer is never accepted as a structured
answer - but a provider that ignored the parameter should be retried once with
an explicit instruction before the request fails, and the failure must say which
provider behaviour caused it.
"""

from __future__ import annotations

import pytest

from app.core.exceptions import ModelOutputError
from app.model_client.generation import GenerationClient

SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "answer",
        "schema": {"type": "object", "properties": {}, "additionalProperties": False},
        "strict": True,
    },
}


class _FakeProvider:
    """Returns scripted contents in order."""

    def __init__(self, contents: list[str]) -> None:
        self.contents = list(contents)
        self.calls: list[tuple[list[dict], dict]] = []

    async def chat_completions(self, messages, model_alias, **extra):  # noqa: ANN001, ANN003
        self.calls.append((messages, extra))
        content = self.contents.pop(0) if self.contents else ""
        return {
            "choices": [
                {"message": {"content": content}, "finish_reason": "stop"}
            ],
            "usage": {},
            "model": model_alias,
        }


def _client(contents: list[str]) -> tuple[GenerationClient, _FakeProvider]:
    client = GenerationClient.__new__(GenerationClient)
    client.alias = "test-model"
    client.max_tokens = 0
    provider = _FakeProvider(contents)
    client.provider_client = provider  # type: ignore[assignment]
    return client, provider


@pytest.mark.asyncio
async def test_json_schema_answer_is_returned_without_a_retry() -> None:
    client, provider = _client(['{"is_answerable": true, "extracted_answer": "15d"}'])

    result = await client.generate(
        [{"role": "user", "content": "q"}], response_format=SCHEMA
    )

    assert result["content"].startswith("{")
    assert len(provider.calls) == 1
    assert "structured_retry" not in result


@pytest.mark.asyncio
async def test_prose_triggers_one_retry_with_an_explicit_instruction() -> None:
    # The gateway accepted response_format and ignored it: first reply is prose.
    client, provider = _client(
        ["出差结束后 15 个自然日内提交 [C1]。", '{"is_answerable": true, "extracted_answer": "15d [C1]"}']
    )

    result = await client.generate(
        [{"role": "user", "content": "q"}], response_format=SCHEMA
    )

    assert result["structured_retry"] is True
    assert len(provider.calls) == 2
    retried_messages = provider.calls[1][0]
    assert "single JSON object" in retried_messages[-1]["content"]


@pytest.mark.asyncio
async def test_prose_twice_fails_loudly_and_names_the_cause() -> None:
    client, _provider = _client(["prose one", "prose two"])

    with pytest.raises(ModelOutputError) as exc_info:
        await client.generate([{"role": "user", "content": "q"}], response_format=SCHEMA)

    message = str(exc_info.value)
    # The operator needs to know the provider ignored the schema, not that we
    # failed to parse something.
    assert "ignored the json_schema response format" in message
    assert "prose two" in message


@pytest.mark.asyncio
async def test_non_json_calls_are_not_retried() -> None:
    # A plain-text request (query contextualizer, subquery planner) has no JSON
    # contract; prose is a valid answer and must not trigger a retry.
    client, provider = _client(["just some text"])

    result = await client.generate([{"role": "user", "content": "q"}])

    assert result["content"] == "just some text"
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_empty_content_names_the_reasoning_budget_cause() -> None:
    """A reasoning model can return finish_reason=stop with empty content.

    Downstream that surfaced as "model returned no JSON object", which points at
    the parser instead of the token budget. The cause must be named.
    """
    from app.core.exceptions import ModelGatewayError

    client = GenerationClient.__new__(GenerationClient)
    client.alias = "reasoning-model"
    client.max_tokens = 512

    class _EmptyContentProvider:
        async def chat_completions(self, messages, model_alias, **extra):  # noqa: ANN001, ANN003
            return {
                "choices": [
                    {
                        "message": {"content": "", "reasoning_content": "..." * 100},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"completion_tokens": 512},
                "model": model_alias,
            }

    client.provider_client = _EmptyContentProvider()  # type: ignore[assignment]

    with pytest.raises(ModelGatewayError) as exc_info:
        await client.generate([{"role": "user", "content": "q"}], response_format=SCHEMA)

    message = str(exc_info.value)
    assert "empty content" in message
    assert "GENERATION_MAX_TOKENS" in message
    assert "reasoning_chars=300" in message
