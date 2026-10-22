"""Paired bootstrap for metric comparisons.

The BM25 sweep picked different optima on two independent samples, and nothing in
the reports could say whether either was real. These tests pin the properties that
make the answer trustworthy: a genuine improvement is detected, noise is not, and
a mean that hides offsetting per-question wins/losses is reported as such.
"""

from __future__ import annotations

import pytest

from app.eval.significance import (
    compare_reports,
    extract_metric,
    paired_bootstrap,
)


def test_a_real_consistent_improvement_is_significant() -> None:
    baseline = [0.0] * 200
    candidate = [1.0] * 200

    result = paired_bootstrap("recall@4", baseline, candidate)

    assert result.mean_delta == pytest.approx(1.0)
    assert result.significant
    assert result.wins == 200
    assert result.losses == 0
    assert result.p_value < 0.01


def test_identical_runs_are_never_significant() -> None:
    values = [0.0, 1.0] * 100
    result = paired_bootstrap("recall@4", values, list(values))

    assert result.mean_delta == 0.0
    assert not result.significant
    assert result.ties == len(values)


def test_pure_noise_is_not_reported_as_an_improvement() -> None:
    """Alternating wins and losses must not pass, however tempting the mean."""
    import random

    rng = random.Random(7)
    baseline = [float(rng.random()) for _ in range(200)]
    candidate = [float(rng.random()) for _ in range(200)]

    result = paired_bootstrap("mrr", baseline, candidate)

    assert not result.significant
    # Symmetric noise: both sides win a comparable number of rows.
    assert abs(result.wins - result.losses) < 60


def test_offsetting_wins_and_losses_are_visible() -> None:
    """A positive mean can still be a coin flip - wins/losses must expose that."""
    baseline = [0.0] * 50 + [1.0] * 50
    candidate = [1.0] * 50 + [0.0] * 50

    result = paired_bootstrap("recall@1", baseline, candidate)

    assert result.mean_delta == pytest.approx(0.0)
    assert result.wins == 50
    assert result.losses == 50
    assert not result.significant


def test_a_small_consistent_gain_on_many_rows_is_detected() -> None:
    """A 3-point gain that every single question shares is a real effect."""
    baseline = [0.0] * 300
    candidate = [0.3] * 300

    result = paired_bootstrap("ndcg@4", baseline, candidate)

    assert result.significant
    assert result.ci_low > 0.0


def test_a_gain_concentrated_in_a_few_rows_is_not_significant() -> None:
    """Three huge wins out of 300 rows is an outlier, not an improvement."""
    baseline = [0.0] * 300
    candidate = [0.0] * 297 + [1.0, 1.0, 1.0]

    result = paired_bootstrap("recall@4", baseline, candidate)

    assert result.mean_delta == pytest.approx(0.01)
    assert not result.significant


def test_mismatched_row_counts_are_rejected() -> None:
    with pytest.raises(ValueError, match="needs identical rows"):
        paired_bootstrap("mrr", [0.0, 1.0], [0.0])


def test_empty_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="no rows"):
        paired_bootstrap("mrr", [], [])


def test_extract_metric_reads_retrieval_rows() -> None:
    report = {
        "results": [
            {"sample_id": "a", "recall_at_k": {"4": 0.5}, "ndcg_at_k": {"4": 0.7}, "mrr": 0.25},
            {"sample_id": "b", "recall_at_k": {"4": 1.0}, "ndcg_at_k": {"4": 1.0}},
        ]
    }
    assert extract_metric(report, "recall@4") == {"a": 0.5, "b": 1.0}
    assert extract_metric(report, "ndcg@4") == {"a": 0.7, "b": 1.0}
    # mrr is derived from first_relevant_rank; a row with neither means nothing
    # relevant was retrieved, which is genuinely MRR 0 - not a missing value.
    assert extract_metric(report, "mrr") == {"a": 0.25, "b": 0.0}


def test_extract_metric_reads_answer_level_rows() -> None:
    report = {"results": [{"sample_id": "a", "answer_exact_match": 1.0, "faithfulness": None}]}
    assert extract_metric(report, "answer_exact_match") == {"a": 1.0}
    # A None metric is not a zero: the row is excluded, not counted as a failure.
    assert extract_metric(report, "faithfulness") == {}


def test_mrr_is_derived_from_the_first_relevant_rank_when_absent() -> None:
    report = {"results": [{"sample_id": "a", "first_relevant_rank": 4}]}
    assert extract_metric(report, "mrr") == {"a": 0.25}


def test_compare_reports_only_uses_shared_rows() -> None:
    baseline = {
        "results": [
            {"sample_id": "a", "recall_at_k": {"4": 0.0}},
            {"sample_id": "b", "recall_at_k": {"4": 0.0}},
        ]
    }
    candidate = {
        "results": [
            {"sample_id": "a", "recall_at_k": {"4": 1.0}},
            {"sample_id": "c", "recall_at_k": {"4": 1.0}},
        ]
    }

    results, skipped = compare_reports(baseline, candidate, ["recall@4"])

    assert len(results) == 1
    assert results[0].rows == 1, "only sample a is shared"
    assert skipped == []


def test_compare_reports_reports_what_it_could_not_compare() -> None:
    baseline = {"results": [{"sample_id": "a", "recall_at_k": {"4": 0.0}}]}
    candidate = {"results": [{"sample_id": "a", "recall_at_k": {"4": 1.0}}]}

    results, skipped = compare_reports(baseline, candidate, ["faithfulness"])

    assert results == []
    assert skipped and "faithfulness" in skipped[0]
