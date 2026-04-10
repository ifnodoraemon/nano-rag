from types import SimpleNamespace

from app.benchmark.service import build_benchmark_report
from app.core.tracing import TraceStore
from app.diagnostics.service import DiagnosisService


def test_build_benchmark_report_aggregates_latency_and_diagnosis_counts() -> None:
    trace_store = TraceStore()
    trace_store.save_raw(
        {
            "trace_id": "trace-1",
            "query": "q1",
            "latency_seconds": 1.2,
            "step_latencies": {"retrieval_seconds": 0.2, "generation_seconds": 1.0},
            "contexts": [{"chunk_id": "wiki:topic:1", "wiki_status": "conflicting"}],
        }
    )
    trace_store.save_raw(
        {
            "trace_id": "trace-2",
            "query": "q2",
            "latency_seconds": 0.8,
            "step_latencies": {"retrieval_seconds": 0.1, "generation_seconds": 0.7},
        }
    )

    report = build_benchmark_report(
        dataset_path="data/eval/sample.jsonl",
        eval_report={
            "records": 2,
            "aggregate": {
                "answer_exact_match": 0.5,
                "reference_context_recall": 1.0,
                "retrieved_context_count_avg": 1.0,
            },
            "results": [
                {
                    "sample_id": "sample-1",
                    "trace_id": "trace-1",
                    "query": "q1",
                    "answer_exact_match": 0.0,
                    "reference_context_recall": 1.0,
                    "conflicting_context_count": 1,
                    "answer": "无法确认。",
                    "reference_answer": "a1",
                },
                {
                    "sample_id": "sample-2",
                    "trace_id": "trace-2",
                    "query": "q2",
                    "answer_exact_match": 1.0,
                    "reference_context_recall": 1.0,
                    "answer": "a2",
                    "reference_answer": "a2",
                },
            ],
        },
        trace_store=trace_store,
        diagnosis_service=DiagnosisService(),
    )

    assert report["aggregate"]["bad_case_count"] == 1
    assert report["aggregate"]["conflicting_bad_case_count"] == 1
    assert report["aggregate"]["conflicting_context_count_avg"] == 0.5
    assert report["aggregate"]["conflicting_hit_rate"] == 0.5
    assert report["aggregate"]["latency_seconds_avg"] == 1.0
    assert report["aggregate"]["latency_seconds_p95"] in {0.8, 1.2}
    assert report["diagnosis_counts"]["generation_misalignment"] == 1
    assert report["results"][0]["diagnosis"]["target_type"] == "eval_result"
