#!/usr/bin/env python3
"""Score the deterministic half of the read plan: chunk selection.

This is the stage that decides which chunks of a chosen document get read. It has
no model calls, so it is cheap and safe to gate on - and it is where a real defect
lived: a focus section spanning several chunks contributed only its first chunk,
so evidence in a sibling chunk never reached the model and the answer said "not
found". Document-level recall cannot see that (the document was found), which is
why this stage needs its own metric.

Dataset (JSONL), one row per document question:

    {"sample_id": "rp-1", "doc_id": "doc-...", "expected_text": "15 个自然日",
     "focus_sections": ["差旅报销"]}

``expected_text`` is a verbatim fragment of the passage needed to answer; the
metric asks whether the chunk holding it was selected for reading.

    python scripts/run_read_plan_eval.py --dataset data/eval/read_plan.jsonl
    python scripts/run_read_plan_eval.py --dataset ... --min-chunk-recall 0.9
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.logging import configure_logging
from app.eval.dataset import get_eval_report_dir, resolve_data_path, save_json
from app.eval.read_plan import evaluate_read_plan, load_read_plan_cases

logger = logging.getLogger("run_read_plan_eval")


def _parsed_dir() -> Path:
    return Path(os.getenv("PARSED_OUTPUT_DIR", "data/parsed"))


def _max_chunks_per_doc() -> int:
    raw = os.getenv("RAG_READ_PLAN_MAX_CHUNKS_PER_DOC")
    if raw is None or not raw.strip():
        # Mirrors settings.yaml agent.max_read_chunks_per_doc.
        return 24
    try:
        return max(1, int(raw))
    except ValueError as exc:
        raise SystemExit(
            f"RAG_READ_PLAN_MAX_CHUNKS_PER_DOC must be an integer, got {raw!r}"
        ) from exc


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--min-chunk-recall",
        type=float,
        default=None,
        help="fail when chunk_recall is below this",
    )
    parser.add_argument(
        "--min-section-coverage",
        type=float,
        default=None,
        help="fail when section_coverage_avg is below this",
    )
    args = parser.parse_args()

    parsed_dir = _parsed_dir()
    if not parsed_dir.exists():
        logger.error("parsed artifact directory does not exist: %s", parsed_dir)
        return 2

    cases = load_read_plan_cases(resolve_data_path(args.dataset))
    report = evaluate_read_plan(
        cases,
        parsed_dir=parsed_dir,
        max_chunks_per_doc=_max_chunks_per_doc(),
    )

    output_path = (
        resolve_data_path(args.output)
        if args.output
        else get_eval_report_dir() / "read_plan_eval.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(str(output_path), report)

    aggregate = report["aggregate"]
    logger.info("read plan: %s", aggregate)
    logger.info("report: %s", output_path)

    failures = 0
    if args.min_chunk_recall is not None:
        value = aggregate.get("chunk_recall")
        if value is None or float(value) < args.min_chunk_recall:
            logger.error(
                "gate failed: chunk_recall=%s < %s", value, args.min_chunk_recall
            )
            failures += 1
    if args.min_section_coverage is not None:
        value = aggregate.get("section_coverage_avg")
        if value is None or float(value) < args.min_section_coverage:
            logger.error(
                "gate failed: section_coverage_avg=%s < %s",
                value,
                args.min_section_coverage,
            )
            failures += 1
    if aggregate.get("rows_missing_expected_chunk"):
        logger.error(
            "%s row(s) declare an expected_text that appears in no chunk: the "
            "dataset is stale or the text is wrong",
            aggregate["rows_missing_expected_chunk"],
        )
        failures += 1
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
