#!/usr/bin/env python3
"""Build an enterprise-shaped benchmark: whole documents, not lone web snippets.

Why: the public T2Retrieval sample is 4,408 *unrelated* short passages, so every
query is a needle-in-a-haystack over many different topics. Real enterprise corpora
look different - a document is a whole page or report about one subject, and the
documents in a library share a domain. If recall@1 is low mainly because the corpus
is a pile of disconnected snippets, tuning ranking on it optimises the wrong thing.

FINDING - this tool does NOT answer that question, and its output should not be used
as evidence. Two results that together say why:

  threshold 0.18 -> 4,408 passages collapse into 4,330 documents. The passages are
  genuinely unrelated, so there is no latent "whole document" structure to recover.
  threshold 0.02 -> 605 documents averaging 4,582 chars, i.e. documents glued
  together from unrelated topics. Retrieval degraded (recall@1 0.47 -> 0.23) and 87
  of 200 queries had their judged passage split across documents, so the labels
  themselves stopped being meaningful.

The honest conclusion: corpus granularity cannot be tested by re-grouping this
dataset. Answering "is recall@1 low because of corpus shape?" needs a real
enterprise corpus (whole documents, several judged queries each) of a few hundred
documents - the bundled corpus has 11 documents and 6 queries, where recall@1 is
1.0 but the sample saturates and proves nothing.

Kept as a starting point for that experiment, not as a benchmark generator.

This groups passages into documents by content overlap (documents about the same
subject end up together), then maps each query's judged passages onto the documents
that contain them. The queries and their relevance judgements are unchanged; only
the document granularity moves toward what an enterprise corpus looks like.

    python scripts/build_enterprise_shaped_benchmark.py \\
        --benchmark /tmp/t2r-bench --out /tmp/t2r-ent
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Deliveries share this much vocabulary to be considered the same subject.
SIMILARITY_THRESHOLD = 0.18
MAX_DOC_CHARS = 6000


def _tokens(text: str) -> set[str]:
    """Character bigrams: cheap, language-agnostic, no tokenizer dependency."""
    stripped = "".join(text.split())
    return {stripped[i : i + 2] for i in range(len(stripped) - 1)}


def _merge(units: list[tuple[str, str]], threshold: float) -> tuple[dict[str, str], dict[str, str]]:
    """Greedy single-pass grouping; returns (passage->doc, doc->text)."""
    passage_to_doc: dict[str, str] = {}
    doc_text: dict[str, str] = {}
    signatures: dict[str, set[str]] = {}
    for passage_id, text in units:
        signature = _tokens(text)
        best_doc = None
        best = threshold
        for doc_id, doc_signature in signatures.items():
            if len(doc_text[doc_id]) + len(text) > MAX_DOC_CHARS:
                continue
            shared = len(signature & doc_signature)
            union = len(signature | doc_signature) or 1
            score = shared / union
            if score >= best:
                best = score
                best_doc = doc_id
        if best_doc is None:
            doc_id = f"ent-{hashlib.sha256(passage_id.encode()).hexdigest()[:20]}"
            doc_text[doc_id] = text
            signatures[doc_id] = signature
        else:
            doc_id = best_doc
            doc_text[doc_id] = doc_text[doc_id] + "\n\n" + text
            signatures[doc_id] |= signature
        passage_to_doc[passage_id] = doc_id
    return passage_to_doc, doc_text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, help="benchmark built by build_public_retrieval_benchmark.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--threshold", type=float, default=SIMILARITY_THRESHOLD)
    args = parser.parse_args()

    source = Path(args.benchmark).resolve()
    parsed_dir = source / "parsed"
    out = Path(args.out).resolve()

    # passage doc_id -> text, from the existing artifacts
    passage_text: dict[str, str] = {}
    for artifact in parsed_dir.glob("*.json"):
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        document = payload.get("document") or {}
        passage_text[str(document.get("doc_id"))] = str(document.get("content") or "")
    if not passage_text:
        raise SystemExit(f"no artifacts in {parsed_dir}")

    passage_to_doc, doc_text = _merge(list(passage_text.items()), args.threshold)

    # dataset: same queries, expectations remapped onto the merged documents
    rows = [
        json.loads(line)
        for line in (source / "retrieval.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    out_parsed = out / "parsed"
    out_parsed.mkdir(parents=True, exist_ok=True)
    for doc_id, text in doc_text.items():
        artifact = {
            "document": {
                "doc_id": doc_id,
                "source_path": f"enterprise/{doc_id}.md",
                "title": text.strip().splitlines()[0][:60],
                "content": text,
                "metadata": {"kb_id": "default", "doc_type": "document"},
            },
            "chunks": [
                {
                    "chunk_id": f"{doc_id}:0",
                    "doc_id": doc_id,
                    "chunk_index": 0,
                    "text": text,
                    "source_path": f"enterprise/{doc_id}.md",
                    "title": text.strip().splitlines()[0][:60],
                    "metadata": {"kb_id": "default", "doc_type": "document"},
                }
            ],
            "structured_document": None,
        }
        (out_parsed / f"{doc_id}.json").write_text(
            json.dumps(artifact, ensure_ascii=False), encoding="utf-8"
        )

    # wiki pages via the committed compiler (no model calls)
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from app.wiki.compiler import WikiCompiler

    WikiCompiler(out / "wiki").bootstrap_from_parsed_dir(out_parsed)

    kept = 0
    with (out / "retrieval.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            expected = {
                passage_to_doc[pid]
                for pid in row.get("expected_doc_ids") or []
                if pid in passage_to_doc
            }
            if not expected:
                continue
            kept += 1
            handle.write(
                json.dumps(
                    {
                        "sample_id": row["sample_id"],
                        "kb_id": row.get("kb_id", "default"),
                        "query": row["query"],
                        "expected_doc_ids": sorted(expected),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    avg_chars = sum(len(text) for text in doc_text.values()) / len(doc_text)
    print(f"passages : {len(passage_text)}")
    print(f"documents: {len(doc_text)}  (avg {avg_chars:.0f} chars each)")
    print(f"queries  : {kept}")
    print(f"out      : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
