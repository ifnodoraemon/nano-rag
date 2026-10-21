#!/usr/bin/env python3
"""Run the retrieval A/B experiments (tokenizer, rerank) reproducibly.

Every experiment goes through the same path: build (or reuse) a benchmark
corpus, then run scripts/run_retrieval_eval.py once per variant and print a
comparison table. The env plumbing lives here so a comparison never depends on
remembering which variables a variant needs.

    # tokenizer A/B on a public Chinese corpus
    python scripts/ab_retrieval.py tokenizer --benchmark /tmp/t2r-bench

    # rerank on/off (needs a reachable rerank endpoint)
    python scripts/ab_retrieval.py rerank --benchmark /tmp/t2r-bench \
        --rerank-base-url http://127.0.0.1:8099 --rerank-path /reranks

Reports land in <benchmark>/ab/<variant>/ so each run is inspectable.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    "recall@1",
    "recall@4",
    "recall@12",
    "mrr",
    "separation_ratio_avg",
    "candidate_count_avg",
    "unretrieved_records",
)


def _provider_env() -> dict[str, str]:
    """Read provider settings from docker/.env so the A/B uses the real model."""
    env = dict(os.environ)
    env_file = ROOT / "docker" / ".env"
    if not env_file.exists():
        raise SystemExit("docker/.env not found - run scripts/init_env.py first")
    values: dict[str, str] = {}
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()

    env.setdefault("APP_CONFIG_DIR", str(ROOT / "configs"))
    env.setdefault("RAG_GRAPH_BACKEND", "none")
    env.setdefault("RAG_WIKI_ENABLED", "true")
    env.setdefault("RAG_ENV", "prod")
    env.setdefault("RAG_API_KEYS", "ab-business")
    env.setdefault("RAG_ADMIN_API_KEYS", "ab-admin")
    # A gateway reached via host.docker.internal is not resolvable on the host,
    # where this script runs.
    base_url = values.get("GENERATION_API_BASE_URL", "").replace(
        "host.docker.internal", "localhost"
    )
    env.setdefault("GENERATION_API_KEY", values.get("GENERATION_API_KEY", ""))
    env.setdefault("GENERATION_MODEL_ALIAS", values.get("GENERATION_MODEL_ALIAS", ""))
    env.setdefault("GENERATION_API_BASE_URL", base_url)
    env.setdefault("DOCUMENT_PARSER_API_BASE_URL", base_url)
    env.setdefault("DOCUMENT_PARSER_API_KEY", values.get("GENERATION_API_KEY", ""))
    return env


def _run_variant(
    benchmark: Path, dataset: Path, variant: str, overrides: dict[str, str]
) -> dict[str, object] | None:
    report_dir = benchmark / "ab" / variant
    report_dir.mkdir(parents=True, exist_ok=True)
    env = _provider_env()
    env.update(
        {
            "PARSED_OUTPUT_DIR": str(benchmark / "parsed"),
            "WIKI_OUTPUT_DIR": str(benchmark / "wiki"),
            "TRACE_STORE_DIR": str(benchmark / "ab" / f"traces-{variant}"),
            "EVAL_REPORT_DIR": str(report_dir),
            **overrides,
        }
    )
    print(f"\n--- {variant} ---", flush=True)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_retrieval_eval.py"),
            "--dataset",
            str(dataset),
        ],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,  # a failing variant is reported, not raised
    )
    if result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr[-2000:], file=sys.stderr)
        print(f"variant {variant} FAILED (exit {result.returncode})")
        return None
    reports = sorted(report_dir.glob("*.json"))
    if not reports:
        print(f"variant {variant} produced no report")
        return None
    return json.loads(reports[-1].read_text(encoding="utf-8"))


def _table(results: dict[str, dict[str, object]]) -> None:
    header = f"{'variant':14s}" + "".join(f"{m:>14s}" for m in METRICS)
    print("\n" + header)
    print("-" * len(header))
    for variant, report in results.items():
        aggregate = report.get("aggregate", {})
        row = f"{variant:14s}"
        for metric in METRICS:
            value = aggregate.get(metric)
            row += f"{value:>14.4f}" if isinstance(value, (int, float)) else f"{'-':>14s}"
        print(row)


def tokenizer_experiment(benchmark: Path, dataset: Path, modes: list[str]) -> int:
    results: dict[str, dict[str, object]] = {}
    for mode in modes:
        report = _run_variant(
            benchmark,
            dataset,
            f"tokenizer-{mode}",
            {"RAG_BM25_TOKENIZER": mode},
        )
        if report:
            results[mode] = report
    _table(results)
    return 0 if results else 1


def rerank_experiment(
    benchmark: Path,
    dataset: Path,
    *,
    base_url: str,
    path: str,
    model: str,
    api_key: str,
    tokenizer: str,
) -> int:
    results: dict[str, dict[str, object]] = {}
    shared = {"RAG_BM25_TOKENIZER": tokenizer}
    for variant, extra in (
        ("rerank-off", {"DISABLE_RERANK": "true"}),
        (
            "rerank-on",
            {
                "DISABLE_RERANK": "false",
                "RERANK_MODEL_ALIAS": model,
                "RERANK_API_BASE_URL": base_url,
                "RERANK_API_PATH": path,
                "RERANK_API_KEY": api_key,
            },
        ),
    ):
        report = _run_variant(benchmark, dataset, variant, {**shared, **extra})
        if report:
            results[variant] = report
    _table(results)
    return 0 if results else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("tokenizer", "rerank"))
    parser.add_argument("--benchmark", required=True, help="built benchmark directory")
    parser.add_argument("--dataset", default=None, help="defaults to <benchmark>/retrieval.jsonl")
    parser.add_argument(
        "--modes",
        default="cjk_ngram,cjk_bigram,jieba",
        help="tokenizer variants (tokenizer experiment)",
    )
    parser.add_argument("--tokenizer", default="cjk_ngram", help="tokenizer for the rerank experiment")
    parser.add_argument("--rerank-base-url", default="http://127.0.0.1:8099")
    parser.add_argument("--rerank-path", default="/reranks")
    parser.add_argument("--rerank-model", default="llm-rerank-shim")
    parser.add_argument("--rerank-api-key", default="shim")
    args = parser.parse_args()

    benchmark = Path(args.benchmark).resolve()
    dataset = Path(args.dataset) if args.dataset else benchmark / "retrieval.jsonl"
    if not dataset.exists():
        raise SystemExit(f"dataset not found: {dataset}")

    if args.experiment == "tokenizer":
        return tokenizer_experiment(
            benchmark, dataset, [m for m in args.modes.split(",") if m]
        )
    return rerank_experiment(
        benchmark,
        dataset,
        base_url=args.rerank_base_url,
        path=args.rerank_path,
        model=args.rerank_model,
        api_key=args.rerank_api_key,
        tokenizer=args.tokenizer,
    )


if __name__ == "__main__":
    raise SystemExit(main())
