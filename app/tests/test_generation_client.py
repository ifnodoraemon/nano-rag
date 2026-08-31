from typing import Any

import pytest

from app.core.config import AppConfig
from app.core.exceptions import ModelGatewayError
from app.model_client.generation import GenerationClient


def _client() -> GenerationClient:
    config = AppConfig(
        config_dir=None,  # type: ignore[arg-type]
        settings={"timeout": {"generation_seconds": 5}},
        models={
            "generation": {
                "default_alias": "gemini-flash-lite-latest",
                "base_url": "http://localhost:4000/v1",
                "api_key": "secret",
            }
        },
        prompts={},
    )
    return GenerationClient(config)


def test_generate_is_a_class_method() -> None:
    assert "generate" in GenerationClient.__dict__, "generate must be defined on the class"
    assert callable(GenerationClient.__dict__["generate"])


@pytest.mark.asyncio
async def test_generate_passes_through_kwargs_and_returns_shape() -> None:
    client = _client()
    captured: dict[str, Any] = {}

    async def fake_chat_completions(messages, model_alias, **kwargs):
        captured["messages"] = messages
        captured["model_alias"] = model_alias
        captured.update(kwargs)
        return {
            "choices": [
                {"message": {"content": "answer"}, "finish_reason": "stop"}
            ],
            "usage": {"total_tokens": 3},
            "model": "gemini-flash-lite-latest",
        }

    client.provider_client.chat_completions = fake_chat_completions  # type: ignore[method-assign]

    response = await client.generate(
        [{"role": "user", "content": "hi"}],
        response_format={"type": "json_schema", "json_schema": {"name": "x", "strict": True}},
    )

    # The JSON-schema structured-output path (agentic/service.py) depends on
    # response_format reaching the provider payload untouched.
    assert captured["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "x", "strict": True},
    }
    assert captured["model_alias"] == "gemini-flash-lite-latest"
    assert response["content"] == "answer"
    assert response["finish_reason"] == "stop"
    assert response["usage"] == {"total_tokens": 3}
    assert response["model"] == "gemini-flash-lite-latest"
    assert "raw" in response


@pytest.mark.asyncio
async def test_generate_raises_when_provider_returns_no_choices() -> None:
    client = _client()

    async def fake_chat_completions(messages, model_alias, **kwargs):  # noqa: ANN001, ARG001
        return {"choices": [], "usage": {}}

    client.provider_client.chat_completions = fake_chat_completions  # type: ignore[method-assign]

    with pytest.raises(ModelGatewayError, match="no choices"):
        await client.generate([{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_generate_raises_on_truncated_output() -> None:
    # finish_reason=length means the structured JSON was cut off mid-object;
    # parsing it would produce garbage. No silent pass-through of truncated
    # content — fail visibly.
    client = _client()

    async def fake_chat_completions(messages, model_alias, **kwargs):  # noqa: ANN001, ARG001
        return {
            "choices": [
                {"message": {"content": '{"is_answerable": tru'}, "finish_reason": "length"}
            ],
            "usage": {},
        }

    client.provider_client.chat_completions = fake_chat_completions  # type: ignore[method-assign]

    with pytest.raises(ModelGatewayError, match="truncated"):
        await client.generate([{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_generate_applies_configured_max_tokens() -> None:
    client = _client()
    captured: dict[str, Any] = {}

    async def fake_chat_completions(messages, model_alias, **kwargs):
        captured.update(kwargs)
        return {
            "choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}],
            "usage": {},
        }

    client.provider_client.chat_completions = fake_chat_completions  # type: ignore[method-assign]
    client.max_tokens = 4096

    await client.generate([{"role": "user", "content": "hi"}])
    assert captured["max_tokens"] == 4096
