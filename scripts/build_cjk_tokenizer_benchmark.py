#!/usr/bin/env python3
"""Build a Chinese retrieval benchmark corpus from real text in this repo.

The tokenizer A/B needs documents that share vocabulary, so a query has to be
discriminated by *precise terms* rather than by one common character. This
script slices real Chinese source text into paragraphs, writes them as parsed
artifacts + wiki pages (no model calls: it takes the local extraction path), and
emits a retrieval dataset whose queries are quotes from specific paragraphs.

Queries are verbatim fragments of the target paragraph on purpose: the question
is "does the tokenizer let BM25 find the paragraph a phrase came from", not
"can a model paraphrase". With verbatim queries a *good* tokenizer wins by
precision (it does not reward unrelated paragraphs that merely share common
characters), which is exactly the axis that differs between modes.

    python scripts/build_cjk_tokenizer_benchmark.py --out /tmp/cjkbench
    RAG_BM25_TOKENIZER=cjk_bigram python scripts/run_retrieval_eval.py \
        --dataset /tmp/cjkbench/retrieval.jsonl

The generated corpus is small (this repo only ships a few KB of Chinese prose),
so treat the numbers as indicative of the mechanism, not as a production
benchmark. Re-run the same commands against a real corpus for a decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.logging import configure_logging
from app.schemas.chunk import Chunk
from app.schemas.document import Document
from app.wiki.compiler import WikiCompiler

# Minimum length for a paragraph to be usable as a retrieval unit.
MIN_PASSAGE_CHARS = 60
# How much of the target paragraph becomes the query.
QUERY_CHARS = 18
MAX_PASSAGES = 60

_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]")
_WHITESPACE_RE = re.compile(r"\s+")


def _extract_text(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() in {".html", ".htm"}:
        raw = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", raw)
        raw = re.sub(r"(?s)<[^>]+>", "\n", raw)
        raw = re.sub(r"&[a-zA-Z]+;", " ", raw)
    return raw


def _cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(_CJK_RUN_RE.findall(text)) / len(text)


def _passages(text: str) -> list[str]:
    """Split into usable Chinese passages, de-duplicated."""
    candidates: list[str] = []
    for block in re.split(r"\n{2,}|\r\n{2,}", text):
        normalized = _WHITESPACE_RE.sub(" ", block).strip()
        if len(normalized) < MIN_PASSAGE_CHARS:
            continue
        # Chinese prose only: mixed navigation/boilerplate lines are noise.
        if _cjk_ratio(normalized) < 0.35:
            continue
        candidates.append(normalized)

    seen: set[str] = set()
    passages: list[str] = []
    for candidate in candidates:
        key = candidate[:80]
        if key in seen:
            continue
        seen.add(key)
        passages.append(candidate)
        if len(passages) >= MAX_PASSAGES:
            break
    return passages


def _query_for(passage: str, query_chars: int) -> str:
    """A verbatim fragment from the middle of the passage."""
    cjk_only = "".join(_CJK_RUN_RE.findall(passage))
    if len(cjk_only) < query_chars:
        return cjk_only
    start = max(0, (len(passage) - query_chars) // 2)
    return passage[start : start + query_chars].strip()


def _doc_id(source: str, index: int) -> str:
    digest = hashlib.sha256(f"{source}|{index}".encode("utf-8")).hexdigest()
    return f"doc-{digest[:24]}"


def build(out_dir: Path, query_chars: int = QUERY_CHARS) -> tuple[Path, Path]:
    sources = [
        ROOT / "data/raw/employee_handbook.md",
        ROOT / "data/raw/spec_pressure/engineering_acceptance_matrix.md",
        ROOT / "data/raw/spec_pressure/control_requirements.txt",
        *sorted(ROOT.glob("data/raw/*ACL*.html")),
    ]

    parsed_dir = out_dir / "parsed"
    wiki_dir = out_dir / "wiki"
    parsed_dir.mkdir(parents=True, exist_ok=True)

    documents: list[tuple[Document, list[Chunk]]] = []
    dataset: list[dict] = []
    for source in sources:
        if not source.exists():
            continue
        passages = _passages(_extract_text(source))
        for index, passage in enumerate(passages):
            doc_id = _doc_id(str(source), index)
            title = f"{source.stem} #{index + 1}"
            document = Document(
                doc_id=doc_id,
                source_path=f"{source.parent.name}/{source.name}#{index + 1}",
                title=title,
                content=passage,
                metadata={
                    "kb_id": "default",
                    "doc_type": "article",
                    "source_key": f"{source.stem} {index + 1}".lower(),
                    "source_content_hash": hashlib.sha256(
                        passage.encode("utf-8")
                    ).hexdigest(),
                },
            )
            chunk = Chunk(
                chunk_id=f"{doc_id}:0",
                doc_id=doc_id,
                chunk_index=0,
                text=passage,
                source_path=document.source_path,
                title=title,
                metadata={"kb_id": "default", "doc_type": "article"},
            )
            documents.append((document, [chunk]))
            dataset.append(
                {
                    "sample_id": f"cjk-{len(dataset) + 1}",
                    "kb_id": "default",
                    "query": _query_for(passage, query_chars),
                    "expected_doc_ids": [doc_id],
                }
            )

    if not documents:
        raise SystemExit("no Chinese passages found; nothing to benchmark")

    for document, chunks in documents:
        (parsed_dir / f"{document.doc_id}.json").write_text(
            json.dumps(
                {
                    "document": document.model_dump(),
                    "chunks": [chunk.model_dump() for chunk in chunks],
                    "structured_document": None,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    compiler = WikiCompiler(wiki_dir)
    compiler.bootstrap_from_parsed_dir(parsed_dir)

    dataset_path = out_dir / "retrieval.jsonl"
    dataset_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in dataset) + "\n",
        encoding="utf-8",
    )
    return dataset_path, parsed_dir


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument(
        "--query-chars",
        type=int,
        default=QUERY_CHARS,
        help="length of the quoted query fragment (default %(default)s)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out).resolve()
    dataset_path, parsed_dir = build(out_dir, query_chars=args.query_chars)
    rows = dataset_path.read_text(encoding="utf-8").count("\n")
    print(f"corpus:   {parsed_dir}")
    print(f"dataset:  {dataset_path}  ({rows} queries)")
    print()
    print("Run the A/B:")
    for mode in ("cjk_ngram", "cjk_bigram", "jieba"):
        print(
            f"  RAG_BM25_TOKENIZER={mode} "
            f"RAG_WIKI_ENABLED=true PARSED_OUTPUT_DIR={parsed_dir} "
            f"WIKI_OUTPUT_DIR={out_dir / 'wiki'} "
            f"python scripts/run_retrieval_eval.py --dataset {dataset_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
