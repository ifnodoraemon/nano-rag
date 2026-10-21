#!/usr/bin/env python3
"""Measure reranking offline, decoupled from the serving path.

Why not just enable rerank in the app? Because the app is *correctly* fail-loud:
a single rerank error aborts the whole request. With an LLM-as-judge substitute
over a flaky gateway, one refusal killed a 50-query benchmark repeatedly.

This script instead re-scores the baseline candidate lists that the retrieval
evaluation already recorded, so the judge runs in a retry-tolerant batch:

  1. read <benchmark>/ab/baseline/*.json (the rerank-off report, which stores the
     per-query BM25 ranking);
  2. for each query, ask the model to pick the single most useful candidate among
     the top N (a one-integer answer - the least refusable form);
  3. move the pick to the front, keep the BM25 order otherwise, and recompute
     recall@k / MRR against the judged relevance in the dataset;
  4. report how many queries the judge failed on (they keep BM25 order).

Consequence: this measures "does an LLM reranker improve top-k on this corpus",
with the judge's own failures visible in the report. It is NOT a cross-encoder
benchmark and says nothing about reranker latency in production.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_shim():
    spec = importlib.util.spec_from_file_location(
        "llm_rerank_shim", ROOT / "scripts" / "llm_rerank_shim.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def _metrics(
    rankings: dict[str, list[str]],
    relevance: dict[str, set[str]],
    cutoffs: tuple[int, ...],
) -> dict[str, float]:
    rows = [qid for qid in relevance if qid in rankings]
    if not rows:
        return {}
    out: dict[str, float] = {}
    mrr = 0.0
    for qid in rows:
        ranking = rankings[qid]
        expected = relevance[qid]
        first = next(
            (pos for pos, doc in enumerate(ranking, start=1) if doc in expected), None
        )
        if first:
            mrr += 1.0 / first
        for cutoff in cutoffs:
            hit = sum(1 for doc in ranking[:cutoff] if doc in expected)
            out[f"recall@{cutoff}"] = out.get(f"recall@{cutoff}", 0.0) + (
                hit / len(expected)
            )
    out["mrr"] = mrr
    return {key: round(value / len(rows), 4) for key, value in out.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--baseline-report", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0, help="0 = all queries")
    parser.add_argument("--gateway", default="http://localhost:8080/v1")
    parser.add_argument("--model", default="deepseek-v4.1-flash")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    benchmark = Path(args.benchmark).resolve()
    dataset = (
        Path(args.dataset) if args.dataset else benchmark / "retrieval50.jsonl"
    )
    rows = [
        json.loads(line)
        for line in dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = json.loads(Path(args.baseline_report).read_text(encoding="utf-8"))
    by_sample = {row["sample_id"]: row for row in report.get("results", [])}

    # doc_id -> passage text, read straight from the artifacts
    texts: dict[str, str] = {}
    for artifact in (benchmark / "parsed").glob("*.json"):
        document = json.loads(artifact.read_text(encoding="utf-8"))["document"]
        texts[document["doc_id"]] = document.get("content", "")

    shim = _load_shim()
    baseline_rankings: dict[str, list[str]] = {}
    relevance: dict[str, set[str]] = {}
    reranked: dict[str, list[str]] = {}
    failures = 0
    processed = 0

    for row in rows:
        sample_id = row["sample_id"]
        baseline = by_sample.get(sample_id)
        if not baseline:
            continue
        ranking = [doc for doc in baseline.get("ranking", []) if doc in texts]
        if not ranking:
            continue
        baseline_rankings[sample_id] = ranking
        relevance[sample_id] = set(row.get("expected_doc_ids") or [])
        reranked[sample_id] = ranking

        if args.limit and processed >= args.limit:
            continue
        processed += 1
        candidates = ranking[: args.top_n]
        documents = [texts[doc][:600] for doc in candidates]
        try:
            scores = shim._score(
                args.gateway, args.api_key, args.model, row["query"], documents
            )
            best = max(range(len(scores)), key=lambda i: scores[i])
            reordered = [candidates[best], *[d for d in ranking if d != candidates[best]]]
            reranked[sample_id] = reordered
        except Exception as exc:  # noqa: BLE001 - recorded, order left unchanged
            failures += 1
            print(f"  judge failed for {sample_id}: {exc}", flush=True)
        if processed % 10 == 0:
            print(f"  ... {processed} queries scored ({failures} failures)", flush=True)
        time.sleep(0.2)

    cutoffs = (1, 4, 12)
    before = _metrics(baseline_rankings, relevance, cutoffs)
    after = _metrics(reranked, relevance, cutoffs)

    print("\n" + "=" * 62)
    print(f"queries scored        : {processed}  (judge failures: {failures})")
    print(f"{'metric':<14s}{'bm25':>12s}{'bm25+llm-rerank':>20s}")
    for key in ("recall@1", "recall@4", "recall@12", "mrr"):
        print(f"{key:<14s}{before.get(key, 0):>12.4f}{after.get(key, 0):>20.4f}")

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "mode": "offline-llm-rerank",
                    "queries": processed,
                    "judge_failures": failures,
                    "baseline": before,
                    "reranked": after,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
