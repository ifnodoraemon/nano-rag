#!/usr/bin/env python3
"""Build an answerability dataset: questions the corpus can and cannot answer.

The abstention metrics need both kinds of row, and inventing "unanswerable"
questions by hand is exactly the sort of guesswork that makes a metric
untrustworthy. This derives them from judged relevance instead:

- **answerable**   : the query's judged passages ARE in the index.
- **unanswerable** : the query's judged passages were deliberately left OUT of the
  index. The evidence provably is not there, so a correct system must decline.

That makes the label objective: it comes from a relevance judgement, not from
someone's opinion about what the corpus covers.

    python scripts/build_answerability_dataset.py \\
        --benchmark /tmp/t2r-bench --out data/eval/answerability.jsonl

Output rows carry ``expected_answerable`` and the query; run them end-to-end so
the model's structured verdict lands on each result row, then the harness computes
false_refusal_rate / correct_abstention_rate / unsupported_answer_rate.

The rows are drawn from the SAME query pool as the retrieval benchmark, so
answerable and unanswerable rows are stylistically indistinguishable - otherwise a
model could separate them from the phrasing alone and the metric would measure
nothing.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _indexed_doc_ids(parsed_dir: Path) -> set[str]:
    return {path.stem for path in parsed_dir.glob("*.json")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, help="built benchmark directory")
    parser.add_argument("--queries", default="/tmp/t2r/queries.parquet")
    parser.add_argument("--qrels", default="/tmp/t2r/qrels.parquet")
    parser.add_argument("--out", required=True)
    parser.add_argument("--answerable", type=int, default=30)
    parser.add_argument("--unanswerable", type=int, default=30)
    parser.add_argument("--kb-id", default="default")
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("reading the source qrels requires pyarrow") from exc

    benchmark = Path(args.benchmark).resolve()
    parsed_dir = benchmark / "parsed"
    indexed = _indexed_doc_ids(parsed_dir)
    if not indexed:
        raise SystemExit(f"no parsed artifacts under {parsed_dir}; build the benchmark first")

    # The same content-hash scheme the benchmark builder uses, so a doc_id can be
    # matched back to its judged passage.
    import hashlib

    def doc_id_for(passage_id: str) -> str:
        digest = hashlib.sha256(f"t2retrieval|{passage_id}".encode()).hexdigest()
        return f"doc-{digest[:24]}"

    queries = {
        str(row["id"]): str(row["text"]) for row in pq.read_table(args.queries).to_pylist()
    }
    judged: dict[str, list[str]] = {}
    for row in pq.read_table(args.qrels).to_pylist():
        judged.setdefault(str(row["qid"]), []).append(str(row["pid"]))

    answerable: list[dict[str, object]] = []
    unanswerable: list[dict[str, object]] = []
    for qid, passage_ids in judged.items():
        if qid not in queries:
            continue
        mapped = [doc_id_for(pid) for pid in passage_ids]
        present = [doc_id for doc_id in mapped if doc_id in indexed]
        absent = [doc_id for doc_id in mapped if doc_id not in indexed]
        if len(present) == len(mapped) and present:
            answerable.append(
                {
                    "query": queries[qid],
                    "expected_answerable": True,
                    "present_doc_ids": present,
                }
            )
        elif len(absent) == len(mapped) and absent:
            unanswerable.append(
                {
                    "query": queries[qid],
                    "expected_answerable": False,
                    "absent_doc_ids": absent,
                }
            )

    if len(answerable) < args.answerable:
        raise SystemExit(
            f"only {len(answerable)} answerable queries available, need {args.answerable}"
        )
    if len(unanswerable) < args.unanswerable:
        raise SystemExit(
            f"only {len(unanswerable)} unanswerable queries available, "
            f"need {args.unanswerable}"
        )

    rng = random.Random(args.seed)
    rng.shuffle(answerable)
    rng.shuffle(unanswerable)
    chosen = answerable[: args.answerable] + unanswerable[: args.unanswerable]
    rng.shuffle(chosen)  # interleave, so ordering carries no signal

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(chosen):
            payload = {
                "sample_id": f"ans-{index + 1}",
                "kb_id": args.kb_id,
                **row,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    print(f"wrote {len(chosen)} rows -> {out_path}")
    print(f"  answerable   : {args.answerable} (judged passages are indexed)")
    print(f"  unanswerable : {args.unanswerable} (judged passages deliberately excluded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
