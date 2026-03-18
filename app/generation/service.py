from __future__ import annotations

from typing import TYPE_CHECKING

from app.generation.answer_formatter import AnswerFormatter
from app.generation.prompt_builder import PromptBuilder
from app.model_client.generation import GenerationClient
from app.retrieval.pipeline import RetrievalPipeline
from app.schemas.chat import ChatRequest, ChatResponse

if TYPE_CHECKING:
    from app.core.config import AppConfig
    from app.core.tracing import TraceStore


class GenerationService:
    def __init__(
        self,
        config: AppConfig,
        retrieval_pipeline: RetrievalPipeline,
        generation_client: GenerationClient,
        prompt_builder: PromptBuilder,
        answer_formatter: AnswerFormatter,
        trace_store: TraceStore,
    ) -> None:
        self.config = config
        self.retrieval_pipeline = retrieval_pipeline
        self.generation_client = generation_client
        self.prompt_builder = prompt_builder
        self.answer_formatter = answer_formatter
        self.trace_store = trace_store

    async def run(self, payload: ChatRequest) -> ChatResponse:
        contexts, trace = await self.retrieval_pipeline.run(payload.query, payload.top_k)
        messages = self.prompt_builder.build_messages(payload.query, contexts)
        result = await self.generation_client.generate(messages)
        response = self.answer_formatter.format(
            answer=result["content"],
            contexts=contexts,
            trace_id=str(trace["trace_id"]),
        )
        record = self.trace_store.get(str(trace["trace_id"]))
        if record is not None:
            record.answer = response.answer
            record.citations = [citation.model_dump() for citation in response.citations]
            record.model_alias = self.generation_client.alias
            record.prompt_version = str(self.config.settings["prompt"]["version"])
        return response
