from types import SimpleNamespace

import pytest

from app.core.tracing import TraceStore
from app.eval.service import materialize_eval_records
from app.eval.ragas_runner import RagasRunner


def test_ragas_runner_returns_aggregate_metrics() -> None:
    runner = RagasRunner()
    report = runner.run(
        [
            {
                "sample_id": "sample-1",
                "trace_id": "trace-1",
                "query": "q1",
                "reference_answer": "a1",
                "answer": "a1",
                "reference_contexts": ["ctx"],
                "retrieved_contexts": ["ctx"],
                "conflicting_context_count": 1,
                "conflict_claim_count": 1,
                "insufficiency_claim_count": 0,
            }
        ]
    )

    assert report["status"] == "ok"
    assert report["records"] == 1
    assert report["aggregate"]["answer_exact_match"] == 1.0
    assert report["aggregate"]["conflicting_context_count_avg"] == 1.0
    assert report["aggregate"]["conflicting_hit_rate"] == 1.0
    assert report["aggregate"]["conflict_claim_count_avg"] == 1.0
    assert report["aggregate"]["conflict_claim_hit_rate"] == 1.0
    assert report["aggregate"]["insufficiency_claim_count_avg"] == 0.0
    assert report["aggregate"]["insufficiency_claim_hit_rate"] == 0.0
    assert report["results"][0]["sample_id"] == "sample-1"
    assert report["results"][0]["trace_id"] == "trace-1"
    assert report["results"][0]["conflicting_context_count"] == 1
    assert report["results"][0]["conflict_claim_count"] == 1


def test_ragas_runner_ignores_trailing_citation_marker_for_exact_match() -> None:
    runner = RagasRunner()
    report = runner.run(
        [
            {
                "query": "q1",
                "reference_answer": "员工应在出差结束后 15 个自然日内提交差旅报销申请。",
                "answer": "员工应在出差结束后 15 个自然日内提交差旅报销申请。 [chunk-1:0]",
                "reference_contexts": ["员工应在出差结束后 15 个自然日内提交差旅报销申请。"],
                "retrieved_contexts": ["员工应在出差结束后 15 个自然日内提交差旅报销申请。"],
            }
        ]
    )

    assert report["aggregate"]["answer_exact_match"] == 1.0


def test_ragas_runner_ignores_markdown_and_trailing_punctuation_noise() -> None:
    runner = RagasRunner()
    report = runner.run(
        [
            {
                "query": "q1",
                "reference_answer": "病假超过 3 天需要提供医院证明。",
                "answer": "病假超过 **3天** 需要提供医院证明 [chunk-1:0]。",
                "reference_contexts": ["病假超过 3 天需要提供医院证明。"],
                "retrieved_contexts": ["病假超过 3 天需要提供医院证明。"],
            }
        ]
    )

    assert report["aggregate"]["answer_exact_match"] == 1.0


def test_ragas_runner_formats_numeric_ragas_scores_only() -> None:
    class FakeFrame:
        def to_dict(self, orient: str) -> list[dict]:
            assert orient == "records"
            return [
                {
                    "user_input": "q1",
                    "response": "a1",
                    "faithfulness": 0.8751,
                    "answer_relevancy": "0.5",
                    "context_precision": None,
                }
            ]

    class FakeResult:
        def to_pandas(self) -> FakeFrame:
            return FakeFrame()

    runner = RagasRunner()
    report = runner._format_result(  # noqa: SLF001
        [
            {
                "sample_id": "sample-1",
                "query": "q1",
                "answer": "a1",
                "reference_answer": "a1",
                "reference_contexts": ["ctx"],
                "retrieved_contexts": ["ctx"],
            }
        ],
        FakeResult(),
    )

    row = report["results"][0]
    assert row["faithfulness"] == 0.8751
    assert row["answer_relevancy"] == 0.5
    assert "context_precision" not in row
    assert "user_input" not in row
    assert report["aggregate"]["faithfulness"] == 0.8751
    assert report["aggregate"]["answer_relevancy"] == 0.5


@pytest.mark.asyncio
async def test_materialize_eval_records_passes_business_context() -> None:
    trace_store = TraceStore()

    async def fake_chat_run(payload):  # noqa: ANN001
        trace_store.save_raw(
            {
                "trace_id": "trace-eval-1",
                "query": payload.query,
                "kb_id": payload.kb_id,
                "tenant_id": payload.tenant_id,
                "session_id": payload.session_id,
                "contexts": [{"chunk_id": "c1", "wiki_status": "conflicting"}],
                "supporting_claims": [
                    {
                        "claim_type": "conflict",
                        "text": "Sources disagree.",
                        "citation_labels": ["C1"],
                    }
                ],
            }
        )
        return SimpleNamespace(
            answer="a1",
            contexts=[{"text": "ctx-1", "wiki_status": "conflicting"}],
            supporting_claims=[
                {
                    "claim_type": "conflict",
                    "text": "Sources disagree.",
                    "citation_labels": ["C1"],
                }
            ],
            trace_id="trace-eval-1",
        )

    async def fake_debug(query, top_k, kb_id="default", tenant_id=None, session_id=None):  # noqa: ANN001
        return SimpleNamespace(
            contexts=[{"text": f"{query}:{top_k}:{kb_id}:{tenant_id}:{session_id}"}]
        )

    container = SimpleNamespace(
        chat_pipeline=SimpleNamespace(run=fake_chat_run),
        retrieval_pipeline=SimpleNamespace(debug=fake_debug),
        trace_store=trace_store,
    )

    records = await materialize_eval_records(
        container,
        [
            {
                "sample_id": "sample-kb-1",
                "query": "q1",
                "reference_answer": "a1",
                "reference_contexts": ["ctx-1"],
                "kb_id": "kb-a",
                "tenant_id": "tenant-a",
                "session_id": "session-a",
            }
        ],
    )

    assert records[0]["trace_id"] == "trace-eval-1"
    trace = trace_store.get("trace-eval-1")
    assert trace is not None
    assert trace.sample_id == "sample-kb-1"
    assert trace.kb_id == "kb-a"
    assert trace.tenant_id == "tenant-a"
    assert trace.session_id == "session-a"
    assert records[0]["conflicting_context_count"] == 1
    assert records[0]["conflict_claim_count"] == 1
    assert records[0]["insufficiency_claim_count"] == 0
