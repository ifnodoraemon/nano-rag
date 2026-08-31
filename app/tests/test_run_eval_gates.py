"""Gate semantics for scripts/run_eval.py: a gate on a missing metric must
fail (the old code defaulted to 0.0, so --max-conflicting-hit-rate could
never fail and --min-context-recall always failed in deterministic mode)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from run_eval import _failed_thresholds  # noqa: E402


def _args(**overrides) -> argparse.Namespace:
    ns = argparse.Namespace(
        min_context_recall=None,
        min_answer_relevancy=None,
        max_conflicting_hit_rate=None,
    )
    ns.__dict__.update(overrides)
    return ns


def test_missing_metric_fails_the_gate() -> None:
    failures = _failed_thresholds(
        {"answer_exact_match": 1.0},
        _args(max_conflicting_hit_rate=0.2),
    )
    assert any("conflicting_hit_rate is missing" in f for f in failures)


def test_context_recall_gate_reads_deterministic_metric() -> None:
    failures = _failed_thresholds(
        {"reference_context_recall": 0.5}, _args(min_context_recall=0.8)
    )
    assert failures == ["reference_context_recall=0.5000 is below threshold 0.8000"]


def test_context_recall_gate_falls_back_to_llm_naming() -> None:
    failures = _failed_thresholds(
        {"context_recall": 0.9}, _args(min_context_recall=0.8)
    )
    assert failures == []


def test_conflicting_hit_rate_gate_can_fail() -> None:
    failures = _failed_thresholds(
        {"conflicting_hit_rate": 0.5}, _args(max_conflicting_hit_rate=0.2)
    )
    assert failures == [
        "conflicting_hit_rate=0.5000 is above threshold 0.2000"
    ]
