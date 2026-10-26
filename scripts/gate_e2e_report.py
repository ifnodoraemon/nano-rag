#!/usr/bin/env python3
"""Gate an end-to-end report on the metrics that do not need a judge.

Why this exists. Answer quality was being assessed by asking a model "is this answer
supported?". Three versions of that judge produced three different verdicts on the same
report, and sampling the disputed cases showed the judge - not the system - was wrong
most of the time: a supporting clause sits at character 4473 of a 6000-character context
and never gets read. Numbers from such a judge cannot gate a release, because a change
that really helps can lower them and a change that hurts can raise them.

The metrics here are computed from the report alone, by rules, and a model cannot
disagree with them:

    citation_integrity   every [Cn] label resolves to a context that was provided
    citation_precision   the cited text shares vocabulary with the sentence citing it
    answered_rate        the share of questions that produced an answer rather than a
                         refusal - a refusal is a legitimate outcome, but a jump means
                         the auditor started rejecting evidence it can see
    errored_rate         questions that failed outright
    latency_p95          p95 end-to-end seconds

`answered_rate` is deliberately NOT "higher is better": an answer that is wrong and
confident is worse than a refusal. It is gated as a BAND, so a large move in either
direction fails and someone looks.

Usage:
    python scripts/gate_e2e_report.py --report data/eval/cnlaw/e2e270_final.json
    python scripts/gate_e2e_report.py --report r.json --min-citation-integrity 0.95 \
        --answered-rate-band 0.70,0.92
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GateResult:
    name: str
    value: float | None
    requirement: str
    passed: bool


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return ordered[index]


def compute(report: dict) -> dict[str, float | None]:
    rows = report.get("results") or []
    if not rows:
        raise SystemExit("report contains no results")
    total = len(rows)
    errored = [row for row in rows if row.get("error")]

    answered = [
        row
        for row in rows
        if not row.get("error")
        and row.get("answer")
        and "未包含" not in str(row.get("answer"))
    ]
    integrity = [
        row["citation_integrity"]
        for row in rows
        if isinstance(row.get("citation_integrity"), (int, float))
    ]
    precision = [
        row["citation_precision"]
        for row in rows
        if isinstance(row.get("citation_precision"), (int, float))
    ]
    latencies = [
        float(row["latency_seconds"])
        for row in rows
        if isinstance(row.get("latency_seconds"), (int, float))
    ]

    return {
        "total": float(total),
        "errored_rate": len(errored) / total,
        "answered_rate": len(answered) / total,
        "citation_integrity": (
            round(sum(integrity) / len(integrity), 4) if integrity else None
        ),
        "citation_precision": (
            round(sum(precision) / len(precision), 4) if precision else None
        ),
        "latency_p95": _p95(latencies),
    }


def evaluate(
    metrics: dict[str, float | None],
    *,
    min_citation_integrity: float,
    min_citation_precision: float,
    answered_rate_band: tuple[float, float] | None,
    max_errored_rate: float,
    max_latency_p95: float | None,
) -> list[GateResult]:
    results: list[GateResult] = []

    def at_least(name: str, threshold: float, label: str) -> None:
        value = metrics.get(name)
        if value is None:
            results.append(
                GateResult(name, None, f">= {threshold}", False)
            )
            return
        results.append(
            GateResult(name, float(value), f">= {threshold}", float(value) >= threshold)
        )

    def at_most(name: str, threshold: float) -> None:
        value = metrics.get(name)
        if value is None:
            results.append(GateResult(name, None, f"<= {threshold}", False))
            return
        results.append(
            GateResult(name, float(value), f"<= {threshold}", float(value) <= threshold)
        )

    at_least("citation_integrity", min_citation_integrity, "integrity")
    at_least("citation_precision", min_citation_precision, "precision")
    at_most("errored_rate", max_errored_rate)

    if answered_rate_band is not None:
        low, high = answered_rate_band
        value = metrics.get("answered_rate")
        requirement = f"between {low} and {high}"
        passed = value is not None and low <= float(value) <= high
        results.append(
            GateResult(
                "answered_rate",
                None if value is None else float(value),
                requirement,
                passed,
            )
        )

    if max_latency_p95 is not None:
        at_most("latency_p95", max_latency_p95)

    return results


def _parse_band(raw: str) -> tuple[float, float]:
    parts = raw.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"expected 'LOW,HIGH', got {raw!r}")
    low, high = float(parts[0]), float(parts[1])
    if low >= high:
        raise argparse.ArgumentTypeError(f"band low must be below high, got {raw!r}")
    return low, high


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, help="end-to-end report JSON")
    parser.add_argument("--min-citation-integrity", type=float, default=0.95)
    parser.add_argument("--min-citation-precision", type=float, default=0.90)
    parser.add_argument(
        "--answered-rate-band",
        type=_parse_band,
        default=(0.60, 0.95),
        help="accepted range for answered_rate; both ends matter",
    )
    parser.add_argument("--max-errored-rate", type=float, default=0.02)
    parser.add_argument("--max-latency-p95", type=float, default=None)
    parser.add_argument(
        "--json", action="store_true", help="emit the result as JSON"
    )
    args = parser.parse_args(argv)

    report_path = Path(args.report)
    if not report_path.exists():
        print(f"ERROR: report not found: {report_path}", file=sys.stderr)
        return 2
    report = json.loads(report_path.read_text(encoding="utf-8"))

    metrics = compute(report)
    results = evaluate(
        metrics,
        min_citation_integrity=args.min_citation_integrity,
        min_citation_precision=args.min_citation_precision,
        answered_rate_band=args.answered_rate_band,
        max_errored_rate=args.max_errored_rate,
        max_latency_p95=args.max_latency_p95,
    )

    if args.json:
        print(
            json.dumps(
                {
                    "report": str(report_path),
                    "metrics": metrics,
                    "gates": [
                        {
                            "name": item.name,
                            "value": item.value,
                            "requirement": item.requirement,
                            "passed": item.passed,
                        }
                        for item in results
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if all(item.passed for item in results) else 1

    print(f"report: {report_path}")
    print(f"questions: {int(metrics['total'] or 0)}")
    print()
    print(f"{'metric':<22}{'value':>10}  {'requirement':<22}verdict")
    print("-" * 66)
    for item in results:
        value = "n/a" if item.value is None else f"{item.value:.4f}"
        verdict = "PASS" if item.passed else "FAIL"
        print(f"{item.name:<22}{value:>10}  {item.requirement:<22}{verdict}")

    failed = [item for item in results if not item.passed]
    print()
    if failed:
        print(f"FAILED: {len(failed)} gate(s)")
        for item in failed:
            print(f"  - {item.name} ({item.value}) required {item.requirement}")
        return 1
    print("all gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
