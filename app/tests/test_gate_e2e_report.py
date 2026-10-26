"""The end-to-end gate uses only deterministic metrics - no judge can change its verdict.

Motivation, measured: three versions of an LLM judge gave 67%, 67% and 54% SUPPORTED on
the SAME report, and sampling the disputed cases showed the judge was mostly wrong (a
supporting clause at character 4473 of a 6000-character context never gets read). A gate
built on that number would move for reasons unrelated to the system.
"""
from __future__ import annotations

import json

from scripts.gate_e2e_report import compute, evaluate


def _row(**over):
    row = {
        "sample_id": "s1",
        "query": "q",
        "answer": "答 [C1]",
        "citation_integrity": 1.0,
        "citation_precision": 1.0,
        "latency_seconds": 5.0,
    }
    row.update(over)
    return row


def test_metrics_are_computed_from_the_report_alone():
    report = {"results": [_row(), _row(sample_id="s2"), _row(sample_id="s3", answer="文档中未包含")]}
    metrics = compute(report)
    assert metrics["total"] == 3
    assert metrics["answered_rate"] == 2 / 3
    assert metrics["errored_rate"] == 0.0
    assert metrics["citation_integrity"] == 1.0


def test_a_refusal_is_not_counted_as_answered():
    metrics = compute({"results": [_row(answer="文档中未包含关于X的信息。")]})
    assert metrics["answered_rate"] == 0.0


def test_an_errored_question_is_counted_separately():
    report = {"results": [_row(), _row(error="ModelGatewayError: 502")]}
    metrics = compute(report)
    assert metrics["errored_rate"] == 0.5
    assert metrics["answered_rate"] == 0.5


def test_a_fabricated_citation_fails_the_gate():
    report = {"results": [_row(citation_integrity=0.5)]}
    results = evaluate(
        compute(report),
        min_citation_integrity=0.95,
        min_citation_precision=0.90,
        answered_rate_band=None,
        max_errored_rate=0.02,
        max_latency_p95=None,
    )
    integrity = [item for item in results if item.name == "citation_integrity"][0]
    assert not integrity.passed


def test_the_answered_rate_band_fails_at_both_ends():
    """A jump in either direction is a signal: too few answers, or the auditor stopped refusing."""
    too_few = {"results": [_row(answer="文档中未包含") for _ in range(9)] + [_row()]}
    results = evaluate(
        compute(too_few),
        min_citation_integrity=0.95,
        min_citation_precision=0.90,
        answered_rate_band=(0.6, 0.95),
        max_errored_rate=0.02,
        max_latency_p95=None,
    )
    band = [item for item in results if item.name == "answered_rate"][0]
    assert not band.passed, "0.10 answered rate must fail a 0.60-0.95 band"

    too_many = {"results": [_row() for _ in range(10)]}
    results = evaluate(
        compute(too_many),
        min_citation_integrity=0.95,
        min_citation_precision=0.90,
        answered_rate_band=(0.6, 0.95),
        max_errored_rate=0.02,
        max_latency_p95=None,
    )
    band = [item for item in results if item.name == "answered_rate"][0]
    assert not band.passed, "1.00 answered rate must fail a 0.60-0.95 band"


def test_a_missing_metric_fails_rather_than_passes_silently():
    report = {"results": [{"sample_id": "s", "query": "q", "answer": "答"}]}
    results = evaluate(
        compute(report),
        min_citation_integrity=0.95,
        min_citation_precision=0.90,
        answered_rate_band=None,
        max_errored_rate=0.02,
        max_latency_p95=None,
    )
    integrity = [item for item in results if item.name == "citation_integrity"][0]
    assert not integrity.passed
    assert integrity.value is None


def test_latency_p95_is_gated_when_requested():
    report = {"results": [_row(latency_seconds=float(i)) for i in range(1, 21)]}
    results = evaluate(
        compute(report),
        min_citation_integrity=0.95,
        min_citation_precision=0.90,
        answered_rate_band=None,
        max_errored_rate=0.02,
        max_latency_p95=15.0,
    )
    latency = [item for item in results if item.name == "latency_p95"][0]
    assert not latency.passed


def test_the_committed_report_passes_its_own_gate():
    """Regression guard on the archived 270-question run."""
    from pathlib import Path

    path = Path("data/eval/cnlaw/e2e270_final.json")
    if not path.exists():  # pragma: no cover - archived fixture may be absent
        import pytest

        pytest.skip("archived report not present")
    metrics = compute(json.loads(path.read_text(encoding="utf-8")))
    results = evaluate(
        metrics,
        min_citation_integrity=0.95,
        min_citation_precision=0.90,
        answered_rate_band=(0.60, 0.95),
        max_errored_rate=0.02,
        max_latency_p95=None,
    )
    failed = [item.name for item in results if not item.passed]
    assert not failed, f"archived report regressed: {failed}"
