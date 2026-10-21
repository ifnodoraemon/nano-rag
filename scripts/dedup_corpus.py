#!/usr/bin/env python3
"""Find (and optionally remove) documents that duplicate another document's content.

A duplicate is a second artifact whose document text is byte-identical to
another artifact in the same knowledge base. They appear when the same file is
ingested under a different path — an upload next to a raw copy, a renamed file,
a reorganized corpus — or when a historical build used a different doc_id
scheme for the same source.

Duplicates are not merely untidy: each one takes a discovery slot and splits
BM25 scores, so the copies crowd the real answer out of the top-k. Run
``scripts/run_retrieval_eval.py`` before and after to see the effect.

Dry run by default (prints the plan). Use --apply to delete.

    python scripts/dedup_corpus.py
    python scripts/dedup_corpus.py --apply
    python scripts/dedup_corpus.py --kb-id default --apply
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import build_graph_store, load_config
from app.core.exceptions import ConfigurationError, StoreError
from app.core.logging import configure_logging
from app.ingestion.dedup import (
    find_duplicate_documents,
    iter_fingerprints,
    remove_document_everywhere,
)
from app.wiki.compiler import WikiCompiler
import logging

logger = logging.getLogger("dedup_corpus")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="delete the redundant documents (default: dry run)",
    )
    parser.add_argument("--kb-id", default=None, help="limit to one knowledge base")
    parser.add_argument(
        "--keep",
        default=None,
        metavar="DOC_ID",
        help="force which document id to keep within a group",
    )
    return parser


def main() -> int:
    configure_logging()
    args = _build_parser().parse_args()

    config = load_config()
    parsed_dir = config.parsed_dir
    if not parsed_dir.exists():
        logger.error("parsed dir does not exist: %s", parsed_dir)
        return 1

    fingerprints = iter_fingerprints(parsed_dir)
    if args.kb_id:
        fingerprints = [
            fingerprint for fingerprint in fingerprints if fingerprint.kb_id == args.kb_id
        ]
    groups = find_duplicate_documents(fingerprints)

    if not groups:
        logger.info("no duplicate documents found in %s", parsed_dir)
        return 0

    total_redundant = sum(len(group.redundant) for group in groups)
    logger.info(
        "%d duplicate group(s), %d redundant document(s) in %s",
        len(groups),
        total_redundant,
        parsed_dir,
    )

    wiki_compiler = WikiCompiler(config.wiki_dir) if args.apply else None
    # The graph store is a third copy of the same document; leaving its rows
    # behind would keep the removed document expandable from search.
    graph_store = build_graph_store(config) if args.apply else None
    if args.apply and graph_store is not None:
        logger.info("graph store: %s", type(graph_store).__name__)
    removed = 0
    for group in groups:
        members = (group.keep, *group.redundant)
        if args.keep:
            forced = [member for member in members if member.doc_id == args.keep]
            if not forced:
                logger.warning(
                    "--keep %s is not part of the duplicate group %s/%s; "
                    "using the automatic choice",
                    args.keep,
                    group.kb_id,
                    group.identity[:12],
                )
            keep = forced[0] if forced else group.keep
        else:
            keep = group.keep
        print(
            f"\nkb={group.kb_id} reason={group.reason} "
            f"identity={group.identity[:60]}\n"
            f"  keep    {keep.doc_id}  ({keep.source_path or 'no source_path'})"
        )
        for member in members:
            if member.doc_id == keep.doc_id:
                continue
            print(
                f"  remove  {member.doc_id}  "
                f"({member.source_path or 'no source_path'})  [{group.reason}]"
            )
            if not args.apply:
                continue
            touched = remove_document_everywhere(
                doc_id=member.doc_id,
                kb_id=member.kb_id,
                parsed_dir=parsed_dir,
                graph_store=graph_store,
                wiki_compiler=wiki_compiler,
            )
            removed += 1
            logger.info("removed %s (stores: %s)", member.doc_id, ",".join(touched))

    if not args.apply:
        print("\nDry run. Re-run with --apply to delete the redundant documents.")
        return 0

    if graph_store is not None:
        graph_store.close()
    logger.info("removed %d duplicate document(s)", removed)
    if removed:
        logger.info(
            "re-run scripts/run_retrieval_eval.py to measure the change in "
            "discovery recall"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ConfigurationError, StoreError) as exc:
        logging.getLogger("dedup_corpus").error("%s", exc)
        raise SystemExit(1) from exc
