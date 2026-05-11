from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.agentic import AgenticReasoningService
from app.core.tracing import TraceStore
from app.generation.answer_formatter import AnswerFormatter
from app.generation.prompt_builder import PromptBuilder
from app.schemas.chat import ChatRequest


class FakeTracingManager:
    @contextmanager
    def span(self, name, attributes=None):  # noqa: ANN001
        yield


class FakeRetrievalPipeline:
    def __init__(self, trace_store: TraceStore) -> None:
        self.trace_store = trace_store
        self.queries: list[str] = []

    async def run(self, query, top_k=None, **kwargs):  # noqa: ANN001, ANN003
        self.queries.append(query)
        trace_id = f"trace-{len(self.queries)}"
        contexts = []
        if len(self.queries) > 1:
            contexts = [
                {
                    "chunk_id": "node-1",
                    "node_id": "node-1",
                    "source": "spec.pdf",
                    "score": 0.92,
                    "text": "组件A 属于 系统B，并且需要满足 3.1 节的验收条件。",
                    "title": "第3章 3.1节",
                    "hierarchy_path": ["第3章", "3.1节"],
                    "page_number": 7,
                    "bounding_box": {
                        "x0": 10,
                        "y0": 20,
                        "x1": 200,
                        "y1": 80,
                    },
                }
            ]
        self.trace_store.save_raw(
            {
                "trace_id": trace_id,
                "query": query,
                "kb_id": kwargs.get("kb_id"),
                "contexts": contexts,
                "retrieval_params": {"requested_top_k": top_k},
                "step_latencies": {"retrieval_seconds": 0.01},
            }
        )
        return contexts, {"trace_id": trace_id}


class FakeGenerationClient:
    alias = "fake-generator"

    async def generate(self, messages):  # noqa: ANN001
        rendered = str(messages)
        if "Break the user question" in rendered:
            return {
                "content": (
                    '{"subqueries":["组件A属于哪个系统",'
                    '"组件A需要满足什么条件"]}'
                )
            }
        if "evidence auditor" in rendered:
            return {
                "content": (
                    '{"sufficient":true,"coverage_ratio":0.9,'
                    '"missing_terms":[],"follow_up_queries":[],"reason":"closed"}'
                )
            }
        assert "Agent evidence check" in rendered
        return {
            "content": (
                "Final Answer:\n"
                "组件A属于系统B，并需要满足3.1节的验收条件。[C1]\n\n"
                "Supporting Claims:\n"
                "- [factual] 组件A与系统B的归属关系和3.1节要求来自同一证据。[C1]"
            ),
            "finish_reason": "stop",
            "usage": {"total_tokens": 12},
        }


@pytest.mark.asyncio
async def test_agent_runs_corrective_retrieval_and_records_state() -> None:
    trace_store = TraceStore()
    retrieval = FakeRetrievalPipeline(trace_store)
    service = AgenticReasoningService(
        config=SimpleNamespace(
            parsed_dir=Path("/tmp/nonexistent-agent-test-parsed"),
            settings={
                "agent": {
                    "max_retrieval_loops": 2,
                    "max_subqueries": 4,
                    "min_coverage_ratio": 0.38,
                },
                "prompt": {"version": "test"},
            }
        ),
        retrieval_pipeline=retrieval,
        generation_client=FakeGenerationClient(),
        prompt_builder=PromptBuilder({"chat": {"system": "system"}}),
        answer_formatter=AnswerFormatter(),
        trace_store=trace_store,
        tracing_manager=FakeTracingManager(),
    )

    response = await service.run(ChatRequest(query="组件A属于哪个系统，以及需要满足什么条件"))
    record = trace_store.get("trace-1")

    assert len(retrieval.queries) == 2
    assert "组件A属于系统B" in response.answer
    assert response.citations[0].node_id == "node-1"
    assert record is not None
    assert record.contexts[0]["citation_label"] == "C1"
    assert record.retrieval_params["agent"]["retrieval_queries"] == retrieval.queries
    assert "graph_expanded_node_ids" in record.retrieval_params["agent"]
    assert record.retrieval_params["agent"]["verification"]["sufficient"] is True
