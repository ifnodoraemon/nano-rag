#!/usr/bin/env python3
"""Compare two evaluation reports with a paired bootstrap.

Use this instead of eyeballing two aggregate means. Two runs over the same rows
always differ; this says whether the difference survives resampling, and how many
questions actually got better or worse.

    # answer-level reports
    python scripts/compare_eval_reports.py --baseline a.json --candidate b.json

    # retrieval reports (recall@k / ndcg@k / mrr are picked up automatically)
    python scripts/compare_eval_reports.py --baseline a.json --candidate b.json \
        --metrics recall@4,ndcg@4,mrr

Exit code is 0 when every compared metric is indistinguishable from noise, and 1
when at least one metric shows a significant change - so a CI gate can use it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.eval.significance import compare_reports

DEFAULT_METRICS = (
    "recall@1",
    "recall@4",
    "recall@12",
    "ndcg@4",
    "ndcg@12",
    "mrr",
    "answer_exact_match",
    "reference_context_recall",
    "citation_integrity",
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument(
        "--metrics",
        default=None,
        help="comma-separated (default: every metric either report contains)",
    )
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20240517)
    parser.add_argument(
        "--require-significant",
        action="store_true",
        help="exit 1 unless at least one metric changed significantly",
    )
    args = parser.parse_args()

    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    metrics = (
        [item.strip() for item in args.metrics.split(",") if item.strip()]
        if args.metrics
        else list(DEFAULT_METRICS)
    )

    results, skipped = compare_reports(
        baseline, candidate, metrics, resamples=args.resamples, seed=args.seed
    )

    print(f"baseline : {args.baseline}")
    print(f"candidate: {args.candidate}")
    print()
    if not results:
        print("no metric present in both reports", file=sys.stderr)
        for line in skipped:
            print(f"  - {line}", file=sys.stderr)
        return 1

    width = max(len(result.metric) for result in results)
    print(f"{'metric'.ljust(width)}  {'baseline':>9} {'candidate':>10} {'delta':>8}  "
          f"{'95% CI':>20} {'p':>7}  {'win/loss/tie':>14}  verdict")
    print("-" * (width + 92))
    for result in results:
        ci = f"[{result.ci_low:+.4f}, {result.ci_high:+.4f}]"
        wlt = f"{result.wins}/{result.losses}/{result.ties}"
        verdict = "significant" if result.significant else "noise"
        print(
            f"{result.metric.ljust(width)}  {result.baseline_mean:>9.4f} "
            f"{result.candidate_mean:>10.4f} {result.mean_delta:>+8.4f}  "
            f"{ci:>20} {result.p_value:>7.4f}  {wlt:>14}  {verdict}"
        )

    for line in skipped:
        print(f"\nskipped: {line}", file=sys.stderr)

    significant = [result for result in results if result.significant]
    print()
    if significant:
        print(f"{len(significant)} metric(s) changed significantly:")
        for result in significant:
            direction = "better" if result.mean_delta > 0 else "worse"
            print(f"  - {result.metric}: {direction} by {result.mean_delta:+.4f}")
    else:
        print(
            "no metric changed significantly: the difference is within sampling "
            "noise, so these runs do not justify a change."
        )

    if args.require_significant:
        return 0 if significant else 1
    return 0 if not significant else 1


if __name__ == "__main__":
    raise SystemExit(main())
