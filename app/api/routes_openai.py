from __future__ import annotations

import asyncio
import json
import logging
import time
from uuid import uuid4
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.api.auth import RequestContext, require_api_key
from app.api.routes_business import _ensure_kb_access
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(tags=["openai"])
logger = logging.getLogger(__name__)


class OpenAIChatMessage(BaseModel):
    role: str
    content: str


class OpenAIChatRequest(BaseModel):
    model: str = "nano-rag"
    messages: list[OpenAIChatMessage]
    stream: bool = False
    temperature: float | None = 0.0
    top_p: float | None = 1.0
    
    # Extra parameters for RAG
    kb_id: str = "default"
    session_id: str | None = None
    top_k: int | None = None
    metadata_filters: dict[str, Any] | None = None


@router.post("/v1/chat/completions")
async def openai_chat_completions(
    payload: OpenAIChatRequest,
    request: Request,
    context: RequestContext = Depends(require_api_key),
):
    """
    OpenAI 兼容接口。
    根据请求中的 `stream` 参数决定是返回 JSON (非流式) 还是 SSE (流式)。
    """
    container = request.app.state.container
    _ensure_kb_access(container, payload.kb_id, context)

    query = ""
    for msg in reversed(payload.messages):
        if msg.role == "user":
            query = msg.content
            break
    if not query and payload.messages:
        query = payload.messages[-1].content

    chat_req = ChatRequest(
        query=query,
        top_k=payload.top_k,
        kb_id=payload.kb_id,
        session_id=payload.session_id,
        metadata_filters=payload.metadata_filters,
    )

    if not payload.stream:
        response = await container.chat_pipeline.run(chat_req)
        return {
            "id": f"chatcmpl-{response.trace_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": payload.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response.answer
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }
    else:
        async def event_generator():
            from time import perf_counter
            
            queue = asyncio.Queue()
            state_input = {"payload": chat_req, "started_at": perf_counter(), "stream_queue": queue}
            
            async def run_workflow():
                try:
                    state = await container.chat_pipeline.workflow.ainvoke(state_input)
                    await queue.put(state["response"])
                except Exception as e:
                    logger.error(f"Stream error: {e}")
                    await queue.put(e)
                    
            task = asyncio.create_task(run_workflow())
            
            chunk_id = f"chatcmpl-{uuid4().hex}"
            created = int(time.time())
            
            def make_chunk(delta_content: str | None, finish_reason: str | None = None) -> str:
                return "data: " + json.dumps({
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": payload.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": delta_content} if delta_content is not None else {},
                            "finish_reason": finish_reason
                        }
                    ]
                }) + "\n\n"
            
            try:
                # First chunk with role
                yield "data: " + json.dumps({
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": payload.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant"},
                            "finish_reason": None
                        }
                    ]
                }) + "\n\n"

                while True:
                    chunk = await queue.get()
                    if isinstance(chunk, BaseException):
                        # Full detail stays server-side (logged in run_workflow);
                        # only a safe marker crosses the wire.
                        yield make_chunk("\n[upstream generation error]", "error")
                        break
                    if isinstance(chunk, ChatResponse):
                        yield make_chunk(None, "stop")
                        break
                    if isinstance(chunk, str):
                        yield make_chunk(chunk, None)
                    # Fail loud on unexpected queue payloads instead of
                    # spinning the consumer loop forever.
                    logger.error("unexpected stream queue payload: %r", type(chunk))
                    yield make_chunk(None, "error")
                    break
            except asyncio.CancelledError:
                task.cancel()
                raise
            finally:
                yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")
