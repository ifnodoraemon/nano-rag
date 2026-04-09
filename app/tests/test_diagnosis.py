import pytest

from types import SimpleNamespace

from app.api.routes_debug import diagnose_auto
from app.core.tracing import TraceStore
from app.diagnostics.service import DiagnosisService
from app.schemas.diagnosis import AutoDiagnosisRequest
from app.schemas.trace import TraceRecord


def test_diagnose_trace_flags_refusal_with_context() -> None:
    service = DiagnosisService()
    trace = TraceRecord(
        trace_id="trace-1",
        query="差旅报销多久内提交？",
        retrieved_chunk_ids=["c1"],
        reranked_chunk_ids=["c1"],
        retrieved=[{"chunk_id": "c1", "text": "员工应在出差结束后 15 个自然日内提交差旅报销申请。"}],
        reranked=[{"chunk_id": "c1", "text": "员工应在出差结束后 15 个自然日内提交差旅报销申请。"}],
        contexts=[{"chunk_id": "c1", "text": "员工应在出差结束后 15 个自然日内提交差旅报销申请。"}],
        answer="无法确认。现有证据不足。",
    )

    diagnosis = service.diagnose_trace(trace)

    assert diagnosis.target_type == "trace"
    assert diagnosis.findings[0].category == "generation_refusal_with_context"


def test_diagnose_trace_flags_conflicting_wiki_contexts() -> None:
    service = DiagnosisService()
    trace = TraceRecord(
        trace_id="trace-2",
        query="PTO 可以结转吗？",
        retrieved_chunk_ids=["wiki:topic:leave-policy"],
        reranked_chunk_ids=["wiki:topic:leave-policy"],
        retrieved=[
            {
                "chunk_id": "wiki:topic:leave-policy",
                "text": "Topic page with conflicting facts.",
                "wiki_status": "conflicting",
                "source": "wiki/topics/default--leave-policy.md",
            }
        ],
        reranked=[
            {
                "chunk_id": "wiki:topic:leave-policy",
                "text": "Topic page with conflicting facts.",
                "wiki_status": "conflicting",
                "source": "wiki/topics/default--leave-policy.md",
            }
        ],
        contexts=[
            {
                "chunk_id": "wiki:topic:leave-policy",
                "text": "Topic page with conflicting facts.",
                "wiki_status": "conflicting",
                "source": "wiki/topics/default--leave-policy.md",
            }
        ],
        answer="PTO 可以结转 5 天。",
    )

    diagnosis = service.diagnose_trace(trace)

    categories = [finding.category for finding in diagnosis.findings]
    assert "wiki_conflict_detected" in categories
    assert "conflict_claim_missing" in categories
    assert "conflict_not_reflected_in_answer" in categories


def test_diagnose_trace_flags_missing_insufficiency_claim() -> None:
    service = DiagnosisService()
    trace = TraceRecord(
        trace_id="trace-3",
        query="What is the contractor carryover policy?",
        retrieved_chunk_ids=["c1"],
        reranked_chunk_ids=["c1"],
        retrieved=[{"chunk_id": "c1", "text": "No contractor policy is documented."}],
        reranked=[{"chunk_id": "c1", "text": "No contractor policy is documented."}],
        contexts=[{"chunk_id": "c1", "text": "No contractor policy is documented."}],
        answer="Insufficient evidence to determine the contractor carryover policy.",
        supporting_claims=[
            {
                "claim_type": "factual",
                "text": "No explicit contractor rule was retrieved.",
                "citation_labels": ["C1"],
            }
        ],
    )

    diagnosis = service.diagnose_trace(trace)

    categories = [finding.category for finding in diagnosis.findings]
    assert "generation_refusal_with_context" in categories
    assert "insufficiency_claim_missing" in categories


def test_diagnose_eval_flags_generation_misalignment_when_context_recall_is_full() -> None:
    service = DiagnosisService()
    report = {
        "results": [
            {
                "sample_id": "sample-1",
                "trace_id": "trace-1",
                "query": "q1",
                "answer_exact_match": 0.0,
                "reference_context_recall": 1.0,
                "answer": "根据提供的上下文，无法确认。",
                "reference_answer": "员工应在出差结束后 15 个自然日内提交差旅报销申请。",
            }
        ]
    }

    diagnosis = service.diagnose_eval_result(report, 0)

    categories = [finding.category for finding in diagnosis.findings]
    assert diagnosis.target_type == "eval_result"
    assert "generation_misalignment" in categories
    assert "refusal_bad_case" in categories


@pytest.mark.asyncio
async def test_add_ai_suggestion_degrades_gracefully_on_generation_error() -> None:
    class BrokenGenerationClient:
        async def generate(self, messages):  # noqa: ANN001, ARG002
            raise RuntimeError("upstream unavailable")

    service = DiagnosisService(generation_client=BrokenGenerationClient())
    diagnosis = service.diagnose_eval_result(
        {
            "results": [
                {
                    "sample_id": "sample-1",
                    "trace_id": "trace-1",
                    "query": "q1",
                    "answer_exact_match": 1.0,
                    "reference_context_recall": 1.0,
                    "answer": "a1",
                    "reference_answer": "a1",
                }
            ]
        },
        0,
    )

    diagnosis = await service.add_ai_suggestion(diagnosis, {"foo": "bar"})

    assert diagnosis.ai_suggestion == "AI diagnosis unavailable: upstream unavailable"


@pytest.mark.asyncio
async def test_diagnose_auto_prefers_latest_eval_bad_case() -> None:
    container = SimpleNamespace(
        diagnosis_service=DiagnosisService(),
        trace_store=TraceStore(),
    )
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(container=container)))

    from unittest.mock import patch

    with patch("app.api.routes_debug.list_eval_reports", return_value=[{"path": "data/reports/latest.json"}]), patch(
        "app.api.routes_debug.load_json",
        return_value={
            "results": [
                {
                    "sample_id": "sample-1",
                    "trace_id": "trace-1",
                    "answer_exact_match": 0.0,
                    "reference_context_recall": 1.0,
                    "answer": "无法确认。",
                    "reference_answer": "a1",
                }
            ]
        },
    ), patch("app.api.routes_debug.resolve_eval_report_path", side_effect=lambda path: path):
        diagnosis = await diagnose_auto(AutoDiagnosisRequest(include_ai=False), request)

    assert diagnosis.target_type == "eval_result"
