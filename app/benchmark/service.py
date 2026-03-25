from __future__ import annotations

from statistics import mean


def build_benchmark_report(
    dataset_path: str,
    eval_report: dict,
    trace_store,
    diagnosis_service,
) -> dict:
    results = eval_report.get("results", []) if isinstance(eval_report, dict) else []
    bad_case_count = 0
    latency_values: list[float] = []
    diagnosis_counts: dict[str, int] = {}
    enriched_results: list[dict] = []

    for index, result in enumerate(results):
        result_payload = dict(result)
        trace_id = result_payload.get("trace_id")
        trace = trace_store.get(str(trace_id)) if trace_id else None
        if trace and trace.latency_seconds is not None:
            latency_values.append(float(trace.latency_seconds))
            result_payload["latency_seconds"] = trace.latency_seconds
            result_payload["step_latencies"] = trace.step_latencies
            result_payload["model_alias"] = trace.model_alias

        if float(result_payload.get("answer_exact_match", 0.0) or 0.0) < 1.0 or float(
            result_payload.get("reference_context_recall", 0.0) or 0.0
        ) < 1.0:
            bad_case_count += 1
            diagnosis = diagnosis_service.diagnose_eval_result(eval_report, index)
            result_payload["diagnosis"] = diagnosis.model_dump()
            for finding in diagnosis.findings:
                diagnosis_counts[finding.category] = diagnosis_counts.get(finding.category, 0) + 1

        enriched_results.append(result_payload)

    latency_avg = round(mean(latency_values), 4) if latency_values else 0.0
    latency_p95 = 0.0
    if latency_values:
        ordered = sorted(latency_values)
        rank = max(0, min(len(ordered) - 1, int(len(ordered) * 0.95) - 1))
        latency_p95 = round(float(ordered[rank]), 4)

    return {
        "status": "ok",
        "dataset_path": dataset_path,
        "records": int(eval_report.get("records", 0)),
        "aggregate": {
            **(eval_report.get("aggregate", {}) if isinstance(eval_report.get("aggregate", {}), dict) else {}),
            "bad_case_count": bad_case_count,
            "latency_seconds_avg": latency_avg,
            "latency_seconds_p95": latency_p95,
        },
        "diagnosis_counts": diagnosis_counts,
        "results": enriched_results,
    }
