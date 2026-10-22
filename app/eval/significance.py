"""Is a metric difference real, or sampling noise?

Every evaluation here runs on a few hundred rows sampled from a much larger
corpus, so two configurations always differ by *something*. The BM25 parameter
sweep made this concrete: two independent samples picked different optima, and
nothing in the reports could say whether the 0.3-point gap was a real effect.

Two questions have to be separated, and neither is answered by comparing means:

- **Paired comparison** - do the same questions get better? A configuration can
  win on average while losing on more questions than it wins, which is a sign the
  gain is concentrated in a few outliers.
- **Uncertainty** - how large could the effect plausibly be? A paired bootstrap
  over the per-row deltas gives a confidence interval without assuming the
  deltas are normally distributed (metric values are bounded and lumpy, so they
  are not).

The bootstrap is paired on purpose: both runs see identical rows, so resampling
*rows* (never the two runs independently) preserves that pairing and removes the
question-to-question variance that otherwise swamps the difference.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

# 10k resamples puts the 2.5%/97.5% quantiles within ~1% of their true value for
# the sample sizes used here, and costs milliseconds in pure Python.
DEFAULT_RESAMPLES = 10_000


@dataclass(frozen=True)
class PairedResult:
    """Outcome of comparing one metric between a baseline and a candidate run."""

    metric: str
    rows: int
    baseline_mean: float
    candidate_mean: float
    mean_delta: float
    ci_low: float
    ci_high: float
    p_value: float
    wins: int
    losses: int
    ties: int
    significant: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "rows": self.rows,
            "baseline_mean": round(self.baseline_mean, 4),
            "candidate_mean": round(self.candidate_mean, 4),
            "mean_delta": round(self.mean_delta, 4),
            "ci95_low": round(self.ci_low, 4),
            "ci95_high": round(self.ci_high, 4),
            "p_value": round(self.p_value, 4),
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "significant": self.significant,
        }

    def describe(self) -> str:
        direction = "+" if self.mean_delta >= 0 else ""
        verdict = "SIGNIFICANT" if self.significant else "not significant"
        return (
            f"{self.metric}: {self.baseline_mean:.4f} -> {self.candidate_mean:.4f} "
            f"({direction}{self.mean_delta:.4f}, 95% CI [{self.ci_low:+.4f}, "
            f"{self.ci_high:+.4f}], p={self.p_value:.4f}) "
            f"wins={self.wins} losses={self.losses} ties={self.ties} -> {verdict}"
        )


def paired_bootstrap(
    metric: str,
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = 0.95,
    seed: int = 20240517,
) -> PairedResult:
    """Compare one metric across two runs over the same rows.

    ``baseline`` and ``candidate`` must be row-aligned: index i is the same
    question in both.
    """
    if len(baseline) != len(candidate):
        raise ValueError(
            f"{metric}: baseline has {len(baseline)} rows but candidate has "
            f"{len(candidate)}; a paired comparison needs identical rows"
        )
    if not baseline:
        raise ValueError(f"{metric}: no rows to compare")

    deltas = [float(b) - float(a) for a, b in zip(baseline, candidate)]
    rows = len(deltas)
    mean_delta = sum(deltas) / rows

    rng = random.Random(seed)
    resampled_means: list[float] = []
    for _ in range(resamples):
        total = sum(deltas[rng.randrange(rows)] for _ in range(rows))
        resampled_means.append(total / rows)
    resampled_means.sort()

    alpha = (1.0 - confidence) / 2.0
    low_index = max(0, int(alpha * resamples) - 1)
    high_index = min(resamples - 1, int((1.0 - alpha) * resamples))
    ci_low = resampled_means[low_index]
    ci_high = resampled_means[high_index]

    # Two-sided bootstrap p-value: how often the resampled effect lands on the
    # other side of zero from the observed one. Reported alongside the interval
    # because a CI that barely excludes zero and a p of 0.049 should not be read
    # as a decisive result.
    if mean_delta == 0:
        p_value = 1.0
    else:
        opposite = sum(1 for value in resampled_means if (value <= 0 if mean_delta > 0 else value >= 0))
        p_value = min(1.0, 2.0 * opposite / resamples)

    wins = sum(1 for delta in deltas if delta > 0)
    losses = sum(1 for delta in deltas if delta < 0)
    return PairedResult(
        metric=metric,
        rows=rows,
        baseline_mean=sum(float(v) for v in baseline) / rows,
        candidate_mean=sum(float(v) for v in candidate) / rows,
        mean_delta=mean_delta,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        wins=wins,
        losses=losses,
        ties=rows - wins - losses,
        # "Significant" = the interval excludes zero. Deliberately conservative:
        # it is the only claim this many rows can support.
        significant=ci_low > 0.0 or ci_high < 0.0,
    )


def extract_metric(report: dict, metric: str) -> dict[str, float]:
    """Pull per-row values of one metric out of an eval report, keyed by sample.

    Handles both report shapes: retrieval rows carry nested ``recall_at_k`` /
    ``ndcg_at_k`` dicts, answer-level rows carry the metric as a flat key.
    """
    rows = report.get("results") or []
    values: dict[str, float] = {}
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            continue
        value = _row_metric(row, metric)
        if value is not None:
            values[sample_id] = value
    return values


def _row_metric(row: dict, metric: str) -> float | None:
    if metric in {"recall@", "hit@", "ndcg@"}:
        raise ValueError("metric needs a cutoff, e.g. recall@4")
    for prefix, key in (("recall", "recall_at_k"), ("hit", "hit_at_k"), ("ndcg", "ndcg_at_k")):
        if metric.startswith(prefix + "@"):
            cutoff = metric.split("@", 1)[1]
            nested = row.get(key) or {}
            value = nested.get(cutoff)
            return None if value is None else float(value)
    if metric == "mrr":
        if row.get("mrr") is not None:
            return float(row["mrr"])
        rank = row.get("first_relevant_rank")
        return 1.0 / float(rank) if rank else 0.0
    value = row.get(metric)
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compare_reports(
    baseline: dict,
    candidate: dict,
    metrics: Sequence[str],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 20240517,
) -> tuple[list[PairedResult], list[str]]:
    """Compare a candidate report against a baseline over shared rows.

    Returns the results plus the metrics that could not be compared (missing on
    either side, or no shared rows) rather than silently reporting fewer numbers.
    """
    results: list[PairedResult] = []
    skipped: list[str] = []
    for metric in metrics:
        base_values = extract_metric(baseline, metric)
        cand_values = extract_metric(candidate, metric)
        shared = sorted(set(base_values) & set(cand_values))
        if not shared:
            skipped.append(f"{metric}: no shared rows with this metric")
            continue
        results.append(
            paired_bootstrap(
                metric,
                [base_values[key] for key in shared],
                [cand_values[key] for key in shared],
                resamples=resamples,
                seed=seed,
            )
        )
    return results, skipped
