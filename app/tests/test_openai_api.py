import json
import pytest
from types import SimpleNamespace
from fastapi.responses import StreamingResponse

from app.api.auth import RequestContext
from app.api.routes_openai import openai_chat_completions, OpenAIChatRequest, OpenAIChatMessage
from app.schemas.chat import ChatResponse

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


@pytest.mark.asyncio
async def test_openai_chat_completions_streaming() -> None:
    async def fake_ainvoke(state):
        return {"response": ChatResponse(
            answer="streaming response",
            citations=[],
            contexts=[],
            trace_id="trace-456",
        )}

    container = SimpleNamespace(
        chat_pipeline=SimpleNamespace(
            workflow=SimpleNamespace(ainvoke=fake_ainvoke)
        ),
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
    async for chunk in response.body_iterator:
        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8")
        if chunk.startswith("data: "):
            data_str = chunk[6:]
            if data_str.strip() == "[DONE]":
                break
            chunks.append(json.loads(data_str))
            
    assert len(chunks) > 0
    assert chunks[0]["object"] == "chat.completion.chunk"

