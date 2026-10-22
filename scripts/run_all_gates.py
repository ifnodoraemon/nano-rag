#!/usr/bin/env python3
"""Run every deterministic evaluation gate in one command.

Metrics that live in separate scripts get run inconsistently, and a gate nobody
runs is documentation. This is the single entry point: it discovers the datasets
that exist, runs each deterministic stage, applies the thresholds, and prints one
summary.

Deterministic stages only (no model calls, no deepeval): these are cheap enough to
run on every change, and they cover the failures that are structural -

    retrieval   recall@k / MRR / nDCG  - was the right document found at all
    read plan   chunk_recall / section_coverage - was the passage that answers it
                                                 actually read
    answer      exact match / context recall / citation integrity & precision /
                abstention / latency / cost

Missing datasets are reported as SKIPPED, loudly. They are not a pass: a gate that
silently disappears when its data is absent is worse than no gate, because the
pipeline still reports green. Pass ``--strict`` to turn a skip into a failure, which
is what a release check should use.

    python scripts/run_all_gates.py
    python scripts/run_all_gates.py --strict
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ROOT_DIR = ROOT


@dataclass
class GateResult:
    stage: str
    status: str  # "pass" | "fail" | "skip"
    detail: str = ""
    metrics: dict[str, object] = field(default_factory=dict)


def _dataset_dir() -> Path:
    return Path(os.getenv("EVAL_DATASET_DIR", ROOT_DIR / "data" / "eval"))


def _report_dir() -> Path:
    return Path(os.getenv("EVAL_REPORT_DIR", ROOT_DIR / "data" / "reports" / "eval"))


def _run_module_script(script: str, argv: list[str]) -> int:
    """Run one of the eval scripts in-process, returning its exit code.

    In-process keeps the container/config wiring in one place and makes the output
    ordering deterministic; each script is written to be import-safe for this.
    """
    path = ROOT_DIR / "scripts" / script
    spec = importlib.util.spec_from_file_location(f"_gate_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    previous = sys.argv
    sys.argv = [str(path), *argv]
    try:
        return int(module.main())
    finally:
        sys.argv = previous


def _read_report(pattern: str) -> dict:
    reports = sorted(_report_dir().glob(pattern))
    if not reports:
        return {}
    try:
        payload = json.loads(reports[-1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _gate_retrieval(args: argparse.Namespace) -> GateResult:
    dataset = _dataset_dir() / args.retrieval_dataset
    if not dataset.exists():
        return GateResult("retrieval", "skip", f"dataset not found: {dataset}")
    argv = ["--dataset", str(dataset), "--output", str(_report_dir() / "retrieval_gate.json")]
    for cutoff, threshold in args.min_recall_at:
        argv += ["--min-recall-at", f"{cutoff}={threshold}"]
    if args.min_mrr is not None:
        argv += ["--min-mrr", str(args.min_mrr)]
    if args.min_ndcg_at:
        for cutoff, threshold in args.min_ndcg_at:
            argv += ["--min-ndcg-at", f"{cutoff}={threshold}"]
    code = _run_module_script("run_retrieval_eval.py", argv)
    report = _read_report("retrieval_gate.json")
    aggregate = report.get("aggregate", {}) if isinstance(report, dict) else {}
    return GateResult(
        "retrieval",
        "pass" if code == 0 else "fail",
        "" if code == 0 else "threshold not met",
        {key: aggregate.get(key) for key in ("recall@1", "recall@4", "mrr", "ndcg@4")},
    )


def _gate_read_plan(args: argparse.Namespace) -> GateResult:
    dataset = _dataset_dir() / args.read_plan_dataset
    if not dataset.exists():
        return GateResult("read_plan", "skip", f"dataset not found: {dataset}")
    argv = ["--dataset", str(dataset), "--output", str(_report_dir() / "read_plan_gate.json")]
    if args.min_chunk_recall is not None:
        argv += ["--min-chunk-recall", str(args.min_chunk_recall)]
    if args.min_section_coverage is not None:
        argv += ["--min-section-coverage", str(args.min_section_coverage)]
    code = _run_module_script("run_read_plan_eval.py", argv)
    report = _read_report("read_plan_gate.json")
    aggregate = report.get("aggregate", {}) if isinstance(report, dict) else {}
    return GateResult(
        "read_plan",
        "pass" if code == 0 else "fail",
        "" if code == 0 else "threshold not met",
        {
            key: aggregate.get(key)
            for key in ("chunk_recall", "section_coverage_avg", "rows_with_error")
        },
    )


def _gate_answer(args: argparse.Namespace) -> GateResult:
    dataset = _dataset_dir() / args.answer_dataset
    if not dataset.exists():
        return GateResult("answer", "skip", f"dataset not found: {dataset}")
    argv = [
        "--dataset",
        str(dataset),
        "--output",
        str(_report_dir() / "answer_gate.json"),
        "--mode",
        "deterministic",
    ]
    if args.min_context_recall is not None:
        argv += ["--min-context-recall", str(args.min_context_recall)]
    if args.min_exact_match is not None:
        argv += ["--min-exact-match", str(args.min_exact_match)]
    if args.min_citation_integrity is not None:
        argv += ["--min-citation-integrity", str(args.min_citation_integrity)]
    if args.min_citation_precision is not None:
        argv += ["--min-citation-precision", str(args.min_citation_precision)]
    code = _run_module_script("run_eval.py", argv)
    report = _read_report("answer_gate.json")
    aggregate = report.get("aggregate", {}) if isinstance(report, dict) else {}
    return GateResult(
        "answer",
        "pass" if code == 0 else "fail",
        "" if code == 0 else "threshold not met",
        {
            key: aggregate.get(key)
            for key in (
                "answer_exact_match",
                "reference_context_recall",
                "citation_integrity",
                "citation_precision",
                "false_refusal_rate",
                "latency_p95_seconds",
            )
        },
    )


def _print_summary(results: list[GateResult]) -> None:
    print()
    print("=" * 78)
    print("deterministic eval gates")
    print("=" * 78)
    for result in results:
        label = {"pass": "PASS", "fail": "FAIL", "skip": "SKIP"}[result.status]
        detail = f"  ({result.detail})" if result.detail else ""
        print(f"{label:5s} {result.stage}{detail}")
        for key, value in result.metrics.items():
            if value is not None:
                print(f"        {key} = {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-dataset", default="retrieval_employee_handbook.jsonl")
    parser.add_argument("--read-plan-dataset", default="read_plan.jsonl")
    parser.add_argument("--answer-dataset", default="employee_handbook_eval.jsonl")
    parser.add_argument(
        "--min-recall-at",
        nargs="*",
        default=["4=0.8"],
        metavar="CUTOFF=THRESHOLD",
        help="retrieval recall gates (default: 4=0.8)",
    )
    parser.add_argument("--min-mrr", type=float, default=None)
    parser.add_argument(
        "--min-ndcg-at",
        nargs="*",
        default=[],
        metavar="CUTOFF=THRESHOLD",
        help="nDCG gates, e.g. 4=0.7",
    )
    parser.add_argument("--min-chunk-recall", type=float, default=0.9)
    parser.add_argument("--min-section-coverage", type=float, default=None)
    parser.add_argument("--min-context-recall", type=float, default=None)
    parser.add_argument("--min-exact-match", type=float, default=None)
    parser.add_argument("--min-citation-integrity", type=float, default=0.9)
    parser.add_argument("--min-citation-precision", type=float, default=None)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="treat a skipped gate (missing dataset) as a failure",
    )
    parser.add_argument("--only", default=None, help="comma-separated stages to run")
    args = parser.parse_args()

    def _pairs(values: list[str]) -> list[tuple[int, float]]:
        parsed: list[tuple[int, float]] = []
        for raw in values or []:
            try:
                cutoff, threshold = str(raw).split("=", 1)
                parsed.append((int(cutoff), float(threshold)))
            except ValueError as exc:
                raise SystemExit(
                    f"expected CUTOFF=THRESHOLD, got {raw!r}"
                ) from exc
        return parsed

    args.min_recall_at = _pairs(args.min_recall_at)
    args.min_ndcg_at = _pairs(args.min_ndcg_at)

    selected = {item.strip() for item in (args.only or "").split(",") if item.strip()}
    stages = [
        ("retrieval", _gate_retrieval),
        ("read_plan", _gate_read_plan),
        ("answer", _gate_answer),
    ]
    results: list[GateResult] = []
    for name, gate in stages:
        if selected and name not in selected:
            continue
        try:
            results.append(gate(args))
        except SystemExit as exc:  # a gate script's own argument failure
            results.append(GateResult(name, "fail", f"gate crashed: {exc}"))

    _print_summary(results)

    failed = [r for r in results if r.status == "fail"]
    skipped = [r for r in results if r.status == "skip"]
    if failed:
        print(f"\n{len(failed)} gate(s) FAILED")
    if skipped:
        print(f"\n{len(skipped)} gate(s) SKIPPED (missing dataset):")
        for result in skipped:
            print(f"  - {result.stage}: {result.detail}")
    if args.strict and skipped:
        print("\n--strict: a skipped gate is a failure")
        return 2
    if failed:
        return 2
    if not any(r.status == "pass" for r in results):
        print("\nno gate ran: nothing was evaluated")
        return 2
    print("\nall executed gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
