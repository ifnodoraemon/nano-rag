import json
import pytest
from types import SimpleNamespace
from fastapi.responses import StreamingResponse

from app.api.auth import RequestContext
from app.api.routes_openai import openai_chat_completions, OpenAIChatRequest, OpenAIChatMessage
from app.schemas.chat import ChatResponse, Citation

CONTEXT = RequestContext(auth_mode="api_key")

def _request_with_container(container) -> SimpleNamespace:
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(container=container)))

@pytest.mark.asyncio
async def test_openai_chat_completions_non_streaming() -> None:
    async def fake_chat_run(payload):
        return ChatResponse(
            answer="hello response",
            citations=[],
            contexts=[],
            trace_id="trace-123",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

    container = SimpleNamespace(
        chat_pipeline=SimpleNamespace(run=fake_chat_run),
        knowledge_base_catalog=SimpleNamespace(exists=lambda kb: True),
    )

    response = await openai_chat_completions(
        OpenAIChatRequest(
            messages=[OpenAIChatMessage(role="user", content="hello")],
            stream=False,
        ),
        _request_with_container(container),
        CONTEXT,
    )

    assert response["object"] == "chat.completion"
    assert response["model"] == "nano-rag"
    assert response["choices"][0]["message"]["content"] == "hello response"
    # Real token usage from the generation result, not a hardcoded zero.
    assert response["usage"]["total_tokens"] == 15


@pytest.mark.asyncio
async def test_openai_chat_completions_streaming() -> None:
    async def fake_chat_run(payload):
        return ChatResponse(
            answer="streaming response",
            citations=[Citation(chunk_id="c1", source="doc.md")],
            contexts=[],
            trace_id="trace-456",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

    container = SimpleNamespace(
        chat_pipeline=SimpleNamespace(run=fake_chat_run),
        knowledge_base_catalog=SimpleNamespace(exists=lambda kb: True),
    )

    response = await openai_chat_completions(
        OpenAIChatRequest(
            messages=[OpenAIChatMessage(role="user", content="hello")],
            stream=True,
        ),
        _request_with_container(container),
        CONTEXT,
    )

    assert isinstance(response, StreamingResponse)

    chunks = []
    done = False
    async for chunk in response.body_iterator:
        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8")
        for line in chunk.splitlines():
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str.strip() == "[DONE]":
                done = True
                break
            chunks.append(json.loads(data_str))
        if done:
            break

    assert len(chunks) >= 3
    assert chunks[0]["object"] == "chat.completion.chunk"
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    # The full answer is delivered in content chunks.
    content = "".join(
        c["choices"][0]["delta"].get("content", "")
        for c in chunks
        if c["choices"][0]["delta"].get("content")
    )
    assert content == "streaming response"
    # The final chunk carries finish_reason and the real usage.
    final = chunks[-1]
    assert final["choices"][0]["finish_reason"] == "stop"
    assert final["usage"]["total_tokens"] == 15


@pytest.mark.asyncio
async def test_openai_chat_completions_requires_user_message() -> None:
    container = SimpleNamespace(
        chat_pipeline=SimpleNamespace(run=None),
        knowledge_base_catalog=SimpleNamespace(exists=lambda kb: True),
    )

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        await openai_chat_completions(
            OpenAIChatRequest(
                messages=[OpenAIChatMessage(role="assistant", content="only assistant")],
                stream=False,
            ),
            _request_with_container(container),
            CONTEXT,
        )
    assert exc_info.value.status_code == 422
