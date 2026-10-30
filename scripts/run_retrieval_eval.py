#!/usr/bin/env python3
"""Score the retrieval (discovery) stage on a dataset.

This is the retrieval analogue of scripts/run_eval.py: it runs the wiki BM25
discovery + deterministic version filter with no
generation calls, so it is cheap, reproducible, and safe to gate CI on.

Dataset (JSONL), one row per question:

    {"sample_id": "r-1", "kb_id": "default", "query": "差旅报销多久内提交？",
     "expected_source_keys": ["policies travel"]}

Either ``expected_source_keys`` (the version-stable identity used by the wiki
ledger) or ``expected_doc_ids`` may be used.

Usage:
    python scripts/run_retrieval_eval.py --dataset data/eval/retrieval.jsonl
    python scripts/run_retrieval_eval.py --dataset ... --min-recall-at 12=0.9 --min-mrr 0.7
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import AppContainer, load_config
from app.agentic.discovery import AgenticDiscovery
from app.wiki.compiler import WikiCompiler
from app.wiki.search import WikiSearcher
from app.core.exceptions import ConfigurationError, RetrievalError, StoreError
from app.core.logging import configure_logging
from app.eval.dataset import (
    get_eval_report_dir,
    load_jsonl_dataset,
    resolve_data_path,
    save_json,
)
from app.eval.retrieval import (
    RetrievedDocument,
    evaluate_retrieval,
    failed_retrieval_gates,
    parse_retrieval_record,
)
from app.retrieval.filters import merge_metadata_filters, sanitize_metadata_filters
from app.retrieval.hits import SearchHit

logger = logging.getLogger("run_retrieval_eval")


def _parse_cutoff_gate(raw: str) -> tuple[int, float]:
    try:
        cutoff, threshold = raw.split("=", 1)
        return int(cutoff), float(threshold)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected CUTOFF=THRESHOLD (e.g. 12=0.9), got {raw!r}"
        ) from exc


def _document_from_hit(hit: SearchHit) -> RetrievedDocument:
    metadata = hit.chunk.metadata or {}
    source_key = metadata.get("source_key")
    return RetrievedDocument(
        doc_id=hit.chunk.doc_id,
        title=hit.chunk.title or "",
        source_key=str(source_key) if source_key else None,
        score=float(hit.score),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="path to a retrieval JSONL dataset")
    parser.add_argument("--llm", action="store_true", help="explicitly enable live model-assisted discovery")
    parser.add_argument("--output", default=None, help="report path (default: EVAL_REPORT_DIR)")
    parser.add_argument("--cutoffs", default="1,4,12", help="comma-separated cutoffs")
    parser.add_argument(
        "--min-recall-at",
        action="append",
        type=_parse_cutoff_gate,
        default=None,
        metavar="CUTOFF=THRESHOLD",
        help="gate on recall@k (repeatable), e.g. --min-recall-at 12=0.9",
    )
    parser.add_argument(
        "--min-ndcg-at",
        action="append",
        type=_parse_cutoff_gate,
        default=None,
        metavar="CUTOFF=THRESHOLD",
        help="fail when nDCG@cutoff is below threshold (repeatable)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(os.getenv("RAG_EVAL_CONCURRENCY", "8")),
        help="records evaluated in parallel (default 8; model calls are I/O-bound)",
    )
    parser.add_argument("--min-mrr", type=float, default=None, help="gate on MRR")
    return parser


async def _run(args: argparse.Namespace) -> int:
    dataset_path = resolve_data_path(args.dataset)
    raw_rows = load_jsonl_dataset(str(dataset_path))
    if not raw_rows:
        logger.error("dataset %s contains no records", dataset_path)
        return 2
    records = [parse_retrieval_record(row, index) for index, row in enumerate(raw_rows)]
    cutoffs = tuple(
        int(item) for item in str(args.cutoffs).split(",") if str(item).strip()
    )

    container = AppContainer.from_env() if args.llm else None
    if container is None:
        config = load_config()
        if not config.wiki_enabled:
            raise ConfigurationError("retrieval evaluation requires the wiki index")
        compiler = WikiCompiler(config.wiki_dir)
        compiler.bootstrap_from_parsed_dir(config.parsed_dir)
        searcher = WikiSearcher(config.wiki_dir)
        discovery = AgenticDiscovery(config=config, wiki_searcher=searcher,
            generation_client=None, trace_store=None, tracing_manager=None)
        discovery.query_expansion_enabled = False
        discovery.scope_preference_enabled = False
        discovery.hyde_enabled = False
        discovery.llm_rerank_enabled = False
        discovery.candidate_filter_enabled = False
    else:
        searcher = container.wiki_searcher
        discovery = container.chat_pipeline.agentic_discovery
    try:
        if searcher is None:
            raise ConfigurationError(
                "retrieval evaluation needs the wiki discovery layer "
                "(RAG_WIKI_ENABLED=true)"
            )
        available = searcher.stats()
        logger.info("discovery index: %s", available)

        async def discover(record):  # noqa: ANN001, ANN202
            filters = merge_metadata_filters(record.metadata_filters, None)
            staged = await discovery.discovery_candidates(
                record.query,
                record.kb_id,
                filters,
            )
            logger.debug(
                "%s -> %d candidates (filters=%s)",
                record.sample_id,
                len(staged.candidates),
                sanitize_metadata_filters(filters),
            )
            return [_document_from_hit(hit) for hit in staged.candidates]

        report = await evaluate_retrieval(
        discover, records, cutoffs, concurrency=args.concurrency
    )
    finally:
        if container is not None:
            await container.close()
        elif hasattr(searcher.index, "close"):
            searcher.index.close()

    output_path = (
        resolve_data_path(args.output)
        if args.output
        else get_eval_report_dir() / f"{dataset_path.stem}_retrieval.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(str(output_path), report)

    aggregate = report["aggregate"]
    logger.info("recall/hit/MRR: %s", aggregate)
    logger.info("report: %s", output_path)

    gates = failed_retrieval_gates(
        aggregate,  # type: ignore[arg-type]
        min_recall_at=dict(args.min_recall_at or []),
        min_mrr=args.min_mrr,
        min_ndcg_at=dict(args.min_ndcg_at or []),
    )
    for failure in gates:
        logger.error("%s", failure)
    return 2 if gates else 0


def main() -> int:
    configure_logging()
    args = _build_parser().parse_args()
    try:
        return asyncio.run(_run(args))
    except (ConfigurationError, RetrievalError, StoreError) as exc:
        logger.error("%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
