from typing import Any

import pytest

from app.core.config import AppConfig
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


def test_generate_and_stream_generate_are_class_methods() -> None:
    # Regression: 1c10bef inserted stream_generate at column 0, which demoted
    # `generate` into a nested dead function and pushed `stream_generate` to
    # module scope, leaving the class with no public methods.
    assert "generate" in GenerationClient.__dict__, "generate must be defined on the class"
    assert "stream_generate" in GenerationClient.__dict__, "stream_generate must be defined on the class"
    assert callable(GenerationClient.__dict__["generate"])
    assert callable(GenerationClient.__dict__["stream_generate"])


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
async def test_stream_generate_yields_chunks() -> None:
    client = _client()
    captured: dict[str, Any] = {}

    async def fake_stream_chat_completions(messages, model_alias, **kwargs):
        captured["model_alias"] = model_alias
        captured.update(kwargs)
        for content in ("Hel", "lo", ""):
            yield {
                "choices": [
                    {"delta": {"content": content}, "finish_reason": None}
                ]
            }
        yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}

    client.provider_client.stream_chat_completions = fake_stream_chat_completions  # type: ignore[method-assign]

    collected = [
        chunk
        async for chunk in client.stream_generate(
            [{"role": "user", "content": "hi"}], stream=True
        )
    ]

    assert captured["model_alias"] == "gemini-flash-lite-latest"
    assert captured["stream"] is True
    assert [c["content"] for c in collected] == ["Hel", "lo", "", ""]
    assert collected[:3] == [
        {"content": "Hel", "finish_reason": None},
        {"content": "lo", "finish_reason": None},
        {"content": "", "finish_reason": None},
    ]
    # The final chunk carries the finish reason with no new content.
    assert collected[-1] == {"content": "", "finish_reason": "stop"}
