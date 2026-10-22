#!/usr/bin/env python3
"""Sweep BM25 ranking parameters (k1, b) on a retrieval benchmark.

``k1`` (term-frequency saturation) and ``b`` (length normalisation) have shipped
at the textbook defaults 1.5/0.75 and were never measured against real judged
queries. They are pure ranking parameters - no model, no re-ingest - so a sweep
is cheap and reproducible, and the winner can be adopted without changing the
architecture.

This runs the retrieval evaluation once per grid point and prints a table. It
touches no config file: the parameters travel through the environment that
``WikiSearcher`` already reads, so nothing is mutated on disk and the run is safe
to repeat.

    python scripts/sweep_bm25_params.py --benchmark /tmp/t2r-bench

The defaults are only changed if a grid point wins on *two independent samples* -
one sample can be noise, and shipping a new default on noise is worse than
shipping the textbook value.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("recall@1", "recall@4", "recall@12", "mrr")


def _provider_env(benchmark: Path, report_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    env_file = ROOT / "docker" / ".env"
    values: dict[str, str] = {}
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                key, _, value = line.partition("=")
                values[key.strip()] = value.strip()
    base_url = values.get("GENERATION_API_BASE_URL", "").replace(
        "host.docker.internal", "localhost"
    )
    env.update(
        {
            "APP_CONFIG_DIR": str(ROOT / "configs"),
            "RAG_GRAPH_BACKEND": "none",
            "RAG_WIKI_ENABLED": "true",
            "RAG_ENV": "prod",
            "RAG_API_KEYS": "sweep-business",
            "RAG_ADMIN_API_KEYS": "sweep-admin",
            "GENERATION_API_KEY": values.get("GENERATION_API_KEY", ""),
            "GENERATION_MODEL_ALIAS": values.get("GENERATION_MODEL_ALIAS", ""),
            "GENERATION_API_BASE_URL": base_url,
            "DOCUMENT_PARSER_API_KEY": values.get("DOCUMENT_PARSER_API_KEY", ""),
            "DOCUMENT_PARSER_API_BASE_URL": base_url,
            "PARSED_OUTPUT_DIR": str(benchmark / "parsed"),
            "WIKI_OUTPUT_DIR": str(benchmark / "wiki"),
            "TRACE_STORE_DIR": str(benchmark / "sweep-traces"),
            "EVAL_REPORT_DIR": str(report_dir),
        }
    )
    return env


def _run_point(
    benchmark: Path,
    dataset: Path,
    k1: float,
    b: float,
    env: dict[str, str],
    report_dir: Path,
    title_weight: float = 0.0,
) -> dict[str, float] | None:
    point_env = {
        **env,
        "RAG_BM25_K1": str(k1),
        "RAG_BM25_B": str(b),
        "RAG_BM25_TITLE_WEIGHT": str(title_weight),
    }
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_retrieval_eval.py"),
            "--dataset",
            str(dataset),
        ],
        cwd=str(ROOT),
        env=point_env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(result.stderr[-800:], file=sys.stderr)
        return None
    reports = sorted(report_dir.glob("*.json"))
    if not reports:
        return None
    aggregate = json.loads(reports[-1].read_text(encoding="utf-8")).get("aggregate", {})
    return {metric: aggregate.get(metric) for metric in METRICS}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--dataset", default=None, help="defaults to <benchmark>/retrieval.jsonl")
    parser.add_argument("--k1", default="0.9,1.2,1.5,1.8,2.2")
    parser.add_argument("--b", default="0.4,0.6,0.75,0.9")
    parser.add_argument(
        "--title-weight",
        default="0",
        help="title field weight: 0 disables it (the historical behaviour)",
    )
    parser.add_argument(
        "--baseline",
        default="1.5,0.75",
        help="k1,b of the shipped default, reported for comparison",
    )
    args = parser.parse_args()

    benchmark = Path(args.benchmark).resolve()
    dataset = Path(args.dataset) if args.dataset else benchmark / "retrieval.jsonl"
    if not dataset.exists():
        raise SystemExit(f"dataset not found: {dataset}")

    report_dir = benchmark / "sweep" / "bm25"
    report_dir.mkdir(parents=True, exist_ok=True)
    env = _provider_env(benchmark, report_dir)

    baseline_k1, baseline_b = (float(part) for part in args.baseline.split(","))
    k1_values = [float(value) for value in args.k1.split(",") if value]
    b_values = [float(value) for value in args.b.split(",") if value]

    title_weights = [float(value) for value in args.title_weight.split(",") if value]
    results: dict[tuple[float, float, float], dict[str, float]] = {}
    print(
        f"sweeping {len(k1_values) * len(b_values) * len(title_weights)} points "
        f"on {dataset.name}",
        flush=True,
    )
    for k1, b, title_weight in itertools.product(k1_values, b_values, title_weights):
        metrics = _run_point(benchmark, dataset, k1, b, env, report_dir, title_weight)
        if metrics is None:
            print(f"  k1={k1} b={b} tw={title_weight}: FAILED", flush=True)
            continue
        results[(k1, b, title_weight)] = metrics
        marker = (
            " (shipped)"
            if (k1, b) == (baseline_k1, baseline_b) and title_weight == 0.0
            else ""
        )
        print(
            f"  k1={k1:<4} b={b:<5} tw={title_weight:<4} "
            + " ".join(f"{metric}={metrics[metric]:.4f}" for metric in METRICS if metrics[metric] is not None)
            + marker,
            flush=True,
        )

    if not results:
        print("no successful points", file=sys.stderr)
        return 1

    baseline = results.get((baseline_k1, baseline_b, 0.0))
    print("\n" + "=" * 90)
    print("ranked by MRR (then recall@4)")
    print("=" * 90)
    ranked = sorted(
        results.items(),
        key=lambda item: (
            item[1].get("mrr") or 0.0,
            item[1].get("recall@4") or 0.0,
        ),
        reverse=True,
    )
    for (k1, b, title_weight), metrics in ranked[:8]:
        delta = ""
        if baseline and baseline.get("mrr") is not None and metrics.get("mrr") is not None:
            delta = f"  mrr_delta={metrics['mrr'] - baseline['mrr']:+.4f}"
        print(
            f"  k1={k1:<4} b={b:<5} tw={title_weight:<4} "
            + " ".join(f"{metric}={metrics[metric]:.4f}" for metric in METRICS if metrics[metric] is not None)
            + delta
        )

    best_key = ranked[0][0]
    print(
        f"\nbest on this sample: k1={best_key[0]} b={best_key[1]} tw={best_key[2]}"
        f"{' (already the shipped default)' if best_key == (baseline_k1, baseline_b, 0.0) else ''}"
    )
    print("Adopt only if it also wins on a second, independently sampled benchmark.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
