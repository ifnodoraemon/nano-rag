from __future__ import annotations

from statistics import mean
from typing import TYPE_CHECKING

from app.utils.constants import P95_PERCENTILE
from app.utils.text import safe_float

if TYPE_CHECKING:
    from app.core.tracing import TraceStore
    from app.diagnostics.service import DiagnosisService


def build_benchmark_report(
    dataset_path: str,
    eval_report: dict,
    trace_store: TraceStore,
    diagnosis_service: DiagnosisService,
) -> dict:
    results = eval_report.get("results", []) if isinstance(eval_report, dict) else []
    bad_case_count = 0
    conflicting_bad_case_count = 0
    latency_values: list[float] = []
    diagnosis_counts: dict[str, int] = {}
    enriched_results: list[dict] = []
    conflicting_counts: list[int] = []

    for index, result in enumerate(results):
        result_payload = dict(result)
        conflicting_context_count = int(
            result_payload.get("conflicting_context_count", 0) or 0
        )
        trace_id = result_payload.get("trace_id")
        trace = trace_store.get(str(trace_id)) if trace_id else None
        if trace and trace.latency_seconds is not None:
            latency_values.append(float(trace.latency_seconds))
            result_payload["latency_seconds"] = trace.latency_seconds
            result_payload["step_latencies"] = trace.step_latencies
            result_payload["model_alias"] = trace.model_alias
        if trace:
            conflicting_context_count = max(
                conflicting_context_count,
                sum(
                    1
                    for context in trace.contexts
                    if isinstance(context, dict)
                    and context.get("wiki_status") == "conflicting"
                ),
            )
        result_payload["conflicting_context_count"] = conflicting_context_count
        conflicting_counts.append(conflicting_context_count)

        if (
            safe_float(result_payload.get("answer_exact_match")) < 1.0
            or safe_float(result_payload.get("reference_context_recall")) < 1.0
        ):
            bad_case_count += 1
            if conflicting_context_count > 0:
                conflicting_bad_case_count += 1
            diagnosis = diagnosis_service.diagnose_eval_result(eval_report, index)
            result_payload["diagnosis"] = diagnosis.model_dump()
            for finding in diagnosis.findings:
                diagnosis_counts[finding.category] = (
                    diagnosis_counts.get(finding.category, 0) + 1
                )

        enriched_results.append(result_payload)

    latency_avg = round(mean(latency_values), 4) if latency_values else 0.0
    latency_p95 = 0.0
    conflicting_context_count_avg = (
        round(mean(conflicting_counts), 4) if conflicting_counts else 0.0
    )
    conflicting_hit_rate = (
        round(
            sum(1 for value in conflicting_counts if value > 0) / len(conflicting_counts),
            4,
        )
        if conflicting_counts
        else 0.0
    )
    if latency_values:
        ordered = sorted(latency_values)
        p95_index = int(len(ordered) * P95_PERCENTILE)
        rank = min(p95_index, len(ordered) - 1)
        latency_p95 = round(float(ordered[rank]), 4)

    return {
        "status": "ok",
        "dataset_path": dataset_path,
        "records": int(eval_report.get("records", 0)),
        "aggregate": {
            **(
                eval_report.get("aggregate", {})
                if isinstance(eval_report.get("aggregate", {}), dict)
                else {}
            ),
            "bad_case_count": bad_case_count,
            "conflicting_bad_case_count": conflicting_bad_case_count,
            "conflicting_context_count_avg": conflicting_context_count_avg,
            "conflicting_hit_rate": conflicting_hit_rate,
            "latency_seconds_avg": latency_avg,
            "latency_seconds_p95": latency_p95,
        },
        "diagnosis_counts": diagnosis_counts,
        "results": enriched_results,
    }
