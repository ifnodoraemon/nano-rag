from __future__ import annotations

import json
import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.api.auth import RequestContext, require_api_key
from app.api.routes_business import _ensure_kb_access
from app.schemas.chat import ChatRequest

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

    Both modes run the identical structured pipeline (json_schema-enforced
    synthesis with citations); the stream variant only changes the transport,
    so there is no lower-quality streaming shortcut.
    """
    container = request.app.state.container
    _ensure_kb_access(container, payload.kb_id, context)

    query = ""
    for msg in reversed(payload.messages):
        if msg.role == "user":
            query = msg.content
            break
    if not query and payload.messages:
        raise HTTPException(
            status_code=422,
            detail="messages must contain at least one user message",
        )

    chat_req = ChatRequest(
        query=query,
        top_k=payload.top_k,
        kb_id=payload.kb_id,
        session_id=payload.session_id,
        metadata_filters=payload.metadata_filters,
    )

    def _usage_of(response) -> dict[str, int]:
        usage = response.usage or {}
        return {
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        }

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
            "usage": _usage_of(response),
        }

    async def event_generator():
        response = None
        try:
            response = await container.chat_pipeline.run(chat_req)
        except Exception:  # noqa: BLE001 - surfaced as a safe SSE error marker
            logger.exception("openai-compatible stream pipeline failed")

        created = int(time.time())

        def make_chunk(
            chunk_id: str,
            delta: dict[str, Any],
            finish_reason: str | None = None,
            usage: dict[str, int] | None = None,
        ) -> str:
            body: dict[str, Any] = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": payload.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": finish_reason,
                    }
                ],
            }
            if usage is not None:
                body["usage"] = usage
            return "data: " + json.dumps(body, ensure_ascii=False) + "\n\n"

        if response is None:
            chunk_id = "chatcmpl-error"
            yield make_chunk(chunk_id, {"role": "assistant"})
            yield make_chunk(
                chunk_id,
                {"content": "\n[upstream generation error]"},
                finish_reason="error",
            )
            yield "data: [DONE]\n\n"
            return

        chunk_id = f"chatcmpl-{response.trace_id}"
        # First chunk with role
        yield make_chunk(chunk_id, {"role": "assistant"})
        # The complete structured answer, delivered in transport chunks.
        answer = response.answer
        step = 512
        for start in range(0, len(answer), step):
            yield make_chunk(chunk_id, {"content": answer[start : start + step]})
        # Final chunk carries finish_reason and the real token usage.
        yield make_chunk(
            chunk_id, {}, finish_reason="stop", usage=_usage_of(response)
        )
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
