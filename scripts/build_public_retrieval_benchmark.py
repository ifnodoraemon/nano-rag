#!/usr/bin/env python3
"""Build a Chinese retrieval benchmark from a public dataset (C-MTEB/T2Retrieval).

Why: the tokenizer A/B experiment needs a corpus with *real* user
queries and judged relevance. The repo only ships a few KB of Chinese prose, and
verbatim-fragment queries cannot discriminate ranking strategies (recall
saturates at 1.0). T2Retrieval is a public Chinese text-retrieval benchmark:
118k passages, 22k queries, 118k relevance judgements.

This builds a *sample* of it in the layout the retrieval evaluation expects:

    <out>/parsed/*.json      one artifact per sampled passage
    <out>/wiki/              compiled wiki pages (bootstrap, no model calls)
    <out>/retrieval.jsonl    queries with expected_doc_ids from the qrels

Sampling: N queries that have judged passages, every judged passage for those
queries, plus distractor passages up to a cap. Larger corpora make the ranking
task more realistic, so the cap is configurable.

    python scripts/build_public_retrieval_benchmark.py --out /tmp/t2r-bench

The passages are web text (they contain markup such as <br><img>), which is
realistic for enterprise ingestion and is left as-is on purpose.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.logging import configure_logging
from app.schemas.chunk import Chunk
from app.schemas.document import Document
from app.wiki.compiler import WikiCompiler

MAX_PASSAGE_CHARS = 1500


def _require_pyarrow():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - dependency is present in dev
        raise SystemExit(
            "reading T2Retrieval parquet files requires pyarrow "
            "(pip install pyarrow)"
        ) from exc
    return pq


def _read_parquet(path: Path) -> list[dict]:
    pq = _require_pyarrow()
    return pq.read_table(path).to_pylist()


def _doc_id(passage_id: str) -> str:
    digest = hashlib.sha256(f"t2retrieval|{passage_id}".encode()).hexdigest()
    return f"doc-{digest[:24]}"


def build(
    corpus_path: Path,
    queries_path: Path,
    qrels_path: Path,
    out_dir: Path,
    *,
    query_limit: int,
    distractors: int,
    seed: int,
) -> tuple[Path, Path, int]:
    corpus = {str(row["id"]): str(row["text"]) for row in _read_parquet(corpus_path)}
    queries = {str(row["id"]): str(row["text"]) for row in _read_parquet(queries_path)}
    qrels: dict[str, list[str]] = {}
    # The qrels carry a graded relevance score; recall and MRR would discard it,
    # so it is kept for nDCG ("are the most valuable passages near the top").
    grades: dict[str, dict[str, int]] = {}
    for row in _read_parquet(qrels_path):
        qid, pid = str(row["qid"]), str(row["pid"])
        if qid in queries and pid in corpus:
            qrels.setdefault(qid, []).append(pid)
            try:
                grade = int(row.get("score") or 1)
            except (TypeError, ValueError):
                grade = 1
            if grade > 0:
                grades.setdefault(qid, {})[pid] = grade

    # Prefer queries with a small judged set: a single relevant passage among
    # thousands of distractors is a sharper test than ten of them.
    usable = sorted(
        (qid for qid, pids in qrels.items() if 1 <= len(pids) <= 3),
        key=lambda qid: (len(qrels[qid]), qid),
    )
    rng = random.Random(seed)
    rng.shuffle(usable)
    selected_queries = usable[:query_limit]
    if not selected_queries:
        raise SystemExit("no usable queries found (need 1-3 judged passages each)")

    relevant_ids: set[str] = set()
    for qid in selected_queries:
        relevant_ids.update(qrels[qid])
    candidates = [pid for pid in corpus if pid not in relevant_ids]
    rng.shuffle(candidates)
    distractor_ids = candidates[:distractors]
    kept_ids = sorted(relevant_ids | set(distractor_ids))

    parsed_dir = out_dir / "parsed"
    wiki_dir = out_dir / "wiki"
    parsed_dir.mkdir(parents=True, exist_ok=True)

    documents: list[tuple[Document, list[Chunk]]] = []
    id_map: dict[str, str] = {}
    for passage_id in kept_ids:
        text = corpus[passage_id][:MAX_PASSAGE_CHARS]
        if not text.strip():
            continue
        doc_id = _doc_id(passage_id)
        id_map[passage_id] = doc_id
        title = text.strip().splitlines()[0][:60] or f"passage-{passage_id}"
        document = Document(
            doc_id=doc_id,
            source_path=f"t2retrieval/{passage_id}.txt",
            title=title,
            content=text,
            metadata={
                "kb_id": "default",
                "doc_type": "passage",
                "source_key": f"t2retrieval {passage_id}",
                "source_content_hash": hashlib.sha256(text.encode()).hexdigest(),
            },
        )
        chunk = Chunk(
            chunk_id=f"{doc_id}:0",
            doc_id=doc_id,
            chunk_index=0,
            text=text,
            source_path=document.source_path,
            title=title,
            metadata={"kb_id": "default", "doc_type": "passage"},
        )
        documents.append((document, [chunk]))

    for document, chunks in documents:
        (parsed_dir / f"{document.doc_id}.json").write_text(
            json.dumps(
                {
                    "document": document.model_dump(),
                    "chunks": [chunk.model_dump() for chunk in chunks],
                    "structured_document": None,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    compiler = WikiCompiler(wiki_dir)
    compiler.bootstrap_from_parsed_dir(parsed_dir)

    rows: list[dict] = []
    for qid in selected_queries:
        expected = [id_map[pid] for pid in qrels[qid] if pid in id_map]
        if not expected:
            continue
        row: dict[str, object] = {
            "sample_id": f"t2r-{qid}",
            "kb_id": "default",
            "query": queries[qid],
            "expected_doc_ids": expected,
        }
        graded = {
            id_map[pid]: grade
            for pid, grade in grades.get(qid, {}).items()
            if pid in id_map
        }
        if graded:
            row["relevance_grades"] = graded
        rows.append(row)

    dataset_path = out_dir / "retrieval.jsonl"
    dataset_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return dataset_path, parsed_dir, len(documents)


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="/tmp/t2r/corpus.parquet")
    parser.add_argument("--queries", default="/tmp/t2r/queries.parquet")
    parser.add_argument("--qrels", default="/tmp/t2r/qrels.parquet")
    parser.add_argument("--out", required=True)
    parser.add_argument("--queries-limit", type=int, default=200)
    parser.add_argument(
        "--distractors",
        type=int,
        default=4000,
        help="random non-relevant passages to make ranking non-trivial",
    )
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    dataset_path, parsed_dir, doc_count = build(
        Path(args.corpus),
        Path(args.queries),
        Path(args.qrels),
        Path(args.out).resolve(),
        query_limit=args.queries_limit,
        distractors=args.distractors,
        seed=args.seed,
    )
    rows = len(dataset_path.read_text(encoding="utf-8").splitlines())
    print(f"corpus   : {doc_count} documents -> {parsed_dir}")
    print(f"dataset  : {rows} queries -> {dataset_path}")
    print()
    print("Run:")
    print(
        f"  PARSED_OUTPUT_DIR={parsed_dir} WIKI_OUTPUT_DIR={Path(args.out).resolve() / 'wiki'} "
        f"python scripts/run_retrieval_eval.py --dataset {dataset_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
