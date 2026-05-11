from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


RAGBENCH_DATASETS = {
    "ragbench-delucionqa": ("rungalileo/ragbench", "delucionqa", "test"),
    "ragbench-emanual": ("rungalileo/ragbench", "emanual", "test"),
    "ragbench-expertqa": ("rungalileo/ragbench", "expertqa", "test"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize public RAG eval corpora into nano-rag raw/eval files."
    )
    parser.add_argument(
        "--dataset",
        choices=[*RAGBENCH_DATASETS.keys(), "hotpotqa"],
        default="ragbench-delucionqa",
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--kb-id", default=None)
    parser.add_argument("--raw-dir", default="data/raw/public_eval")
    parser.add_argument("--eval-dir", default="data/eval")
    parser.add_argument("--top-k", type=int, default=8)
    return parser.parse_args()


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug[:80] or "sample"


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )


def load_dataset(name: str, subset: str | None, split: str):
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required. Install project requirements first."
        ) from exc
    return hf_load_dataset(name, subset, split=split) if subset else hf_load_dataset(name, split=split)


def normalize_contexts(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, dict):
        text = value.get("text") or value.get("content") or value.get("passage")
        return normalize_contexts(text)
    if isinstance(value, list):
        contexts: list[str] = []
        for item in value:
            contexts.extend(normalize_contexts(item))
        return [item for item in contexts if item.strip()]
    return [str(value).strip()] if str(value).strip() else []


def prepare_ragbench(args: argparse.Namespace, raw_base: Path) -> list[dict[str, Any]]:
    dataset_name, subset, split = RAGBENCH_DATASETS[args.dataset]
    dataset = load_dataset(dataset_name, subset, split)
    kb_id = args.kb_id or args.dataset
    records: list[dict[str, Any]] = []
    for index, row in enumerate(dataset.select(range(min(args.limit, len(dataset))))):
        query = str(row.get("question") or row.get("query") or "").strip()
        reference = str(row.get("response") or row.get("answer") or "").strip()
        contexts = normalize_contexts(
            row.get("documents") or row.get("contexts") or row.get("retrieved_contexts")
        )
        if not query or not reference or not contexts:
            continue
        sample_id = f"{args.dataset}-{index + 1:04d}"
        source = raw_base / kb_id / f"{sample_id}.md"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(
            f"# {query}\n\n" + "\n\n---\n\n".join(contexts),
            encoding="utf-8",
        )
        records.append(
            {
                "sample_id": sample_id,
                "kb_id": kb_id,
                "query": query,
                "reference_answer": reference,
                "reference_contexts": contexts[:3],
                "top_k": args.top_k,
                "source_path": str(source.relative_to(ROOT)),
                "public_dataset": args.dataset,
            }
        )
    return records


def prepare_hotpotqa(args: argparse.Namespace, raw_base: Path) -> list[dict[str, Any]]:
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", "validation")
    kb_id = args.kb_id or "hotpotqa"
    records: list[dict[str, Any]] = []
    for index, row in enumerate(dataset.select(range(min(args.limit, len(dataset))))):
        query = str(row.get("question") or "").strip()
        answer = str(row.get("answer") or "").strip()
        context = row.get("context") or {}
        titles = list(context.get("title") or [])
        sentence_groups = list(context.get("sentences") or [])
        docs: list[str] = []
        for title, sentences in zip(titles, sentence_groups, strict=False):
            text = " ".join(str(sentence) for sentence in sentences)
            if text.strip():
                docs.append(f"## {title}\n\n{text}")
        if not query or not answer or not docs:
            continue
        sample_id = f"hotpotqa-{index + 1:04d}-{slugify(query)}"
        source = raw_base / kb_id / f"{sample_id}.md"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"# {query}\n\n" + "\n\n".join(docs), encoding="utf-8")
        records.append(
            {
                "sample_id": sample_id,
                "kb_id": kb_id,
                "query": query,
                "reference_answer": answer,
                "reference_contexts": docs[:3],
                "top_k": args.top_k,
                "source_path": str(source.relative_to(ROOT)),
                "public_dataset": "hotpotqa",
            }
        )
    return records


def main() -> int:
    args = parse_args()
    raw_base = (ROOT / args.raw_dir).resolve()
    eval_dir = (ROOT / args.eval_dir).resolve()
    records = (
        prepare_hotpotqa(args, raw_base)
        if args.dataset == "hotpotqa"
        else prepare_ragbench(args, raw_base)
    )
    output = eval_dir / f"{args.dataset}.jsonl"
    write_jsonl(output, records)
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "records": len(records),
                "raw_dir": str(raw_base),
                "eval_path": str(output),
                "kb_id": args.kb_id or args.dataset,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
