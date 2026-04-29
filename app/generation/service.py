from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING

from app.generation.answer_formatter import AnswerFormatter
from app.generation.prompt_builder import PromptBuilder
from app.model_client.generation import GenerationClient
from app.retrieval.pipeline import RetrievalPipeline
from app.schemas.chat import ChatRequest, ChatResponse

if TYPE_CHECKING:
    from app.core.config import AppConfig
    from app.core.tracing import TraceStore, TracingManager


class GenerationService:
    def __init__(
        self,
        config: AppConfig,
        retrieval_pipeline: RetrievalPipeline,
        generation_client: GenerationClient,
        prompt_builder: PromptBuilder,
        answer_formatter: AnswerFormatter,
        trace_store: TraceStore,
        tracing_manager: TracingManager,
    ) -> None:
        self.config = config
        self.retrieval_pipeline = retrieval_pipeline
        self.generation_client = generation_client
        self.prompt_builder = prompt_builder
        self.answer_formatter = answer_formatter
        self.trace_store = trace_store
        self.tracing_manager = tracing_manager

    async def run(self, payload: ChatRequest) -> ChatResponse:
        with self.tracing_manager.span(
            "generation.run", {"generation.query": payload.query}
        ):
            contexts, trace = await self.retrieval_pipeline.run(
                payload.query,
                payload.top_k,
                kb_id=payload.kb_id or "default",
                session_id=payload.session_id,
                sample_id=payload.sample_id,
                metadata_filters=payload.metadata_filters,
            )
            messages = self.prompt_builder.build_messages(payload.query, contexts)
            generation_started = perf_counter()
            result = await self.generation_client.generate(messages)
            generation_seconds = round(perf_counter() - generation_started, 4)
            response = self.answer_formatter.format(
                answer=result["content"],
                contexts=contexts,
                trace_id=str(trace["trace_id"]),
            )
            record = self.trace_store.get(str(trace["trace_id"]))
            if record is not None:
                updated = record.model_copy(
                    update={
                        "answer": response.answer,
                        "citations": [
                            citation.model_dump()
                            for citation in response.citations
                        ],
                        "supporting_claims": [
                            claim.model_dump()
                            for claim in response.supporting_claims
                        ],
                        "model_alias": self.generation_client.alias,
                        "kb_id": payload.kb_id or record.kb_id,
                        "session_id": payload.session_id or record.session_id,
                        "sample_id": payload.sample_id or record.sample_id,
                        "prompt_version": str(
                            self.config.settings["prompt"]["version"]
                        ),
                        "prompt_messages": messages,
                        "generation_finish_reason": (
                            str(result["finish_reason"])
                            if result.get("finish_reason") is not None
                            else None
                        ),
                        "generation_usage": result.get("usage") or {},
                        "step_latencies": {
                            **record.step_latencies,
                            "generation_seconds": generation_seconds,
                            "end_to_end_seconds": float(
                                record.latency_seconds or 0.0
                            ),
                        },
                    }
                )
                self.trace_store.update(updated)
            return response
