from fastapi import APIRouter, Request

from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    container = request.app.state.container
    return await container.chat_pipeline.run(payload)
