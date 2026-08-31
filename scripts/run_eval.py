import argparse
import asyncio
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import AppContainer
from app.eval.dataset import get_eval_report_dir, load_jsonl_dataset, save_json
from app.eval.service import materialize_eval_records

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def resolve_project_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run offline evaluation for nano-rag.")
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset.")
    parser.add_argument("--output", required=False, help="Path to output JSON report.")
    parser.add_argument(
        "--no-ragas-lib",
        action="store_true",
        help="Disable the RAGAS library metrics and only use built-in deterministic metrics.",
    )
    parser.add_argument(
        "--min-context-recall",
        type=float,
        default=None,
        help=(
            "Fail with exit code 2 if the aggregate context recall is below "
            "this value. Checks reference_context_recall (deterministic, "
            "always computed) and falls back to the LLM context_recall."
        ),
    )
    parser.add_argument(
        "--min-answer-relevancy",
        type=float,
        default=None,
        help=(
            "Fail with exit code 2 if the aggregate answer_relevancy is below "
            "this value. LLM-metric only: requires the ragas/deepeval mode "
            "(do not combine with --no-ragas-lib)."
        ),
    )
    parser.add_argument(
        "--max-conflicting-hit-rate",
        type=float,
        default=None,
        help="Fail with exit code 2 if aggregate conflicting_hit_rate is above this value.",
    )
    args = parser.parse_args()
    if args.no_ragas_lib and args.min_answer_relevancy is not None:
        parser.error(
            "--min-answer-relevancy requires LLM metrics and cannot be combined "
            "with --no-ragas-lib"
        )

    dataset_path = resolve_project_path(args.dataset)
    if args.output:
        output_path = resolve_project_path(args.output)
    else:
        report_dir = get_eval_report_dir()
        report_dir.mkdir(parents=True, exist_ok=True)
        output_path = report_dir / f"{dataset_path.stem}_manual.json"
    dataset = load_jsonl_dataset(str(dataset_path))
    container = AppContainer.from_env()
    
    runner = container.eval_runner
    if not runner:
        from app.eval.deepeval_runner import DeepevalRunner
        runner = DeepevalRunner(generation_client=container.generation_client)
        
    evaluated_records = asyncio.run(materialize_eval_records(container, dataset))
    report = (
        runner.run(evaluated_records)
        if args.no_ragas_lib
        else asyncio.run(runner.run_async(evaluated_records))
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(str(output_path), report)
    logger.info("%s", output_path)
    failed_thresholds = _failed_thresholds(report.get("aggregate", {}), args)
    if failed_thresholds:
        for failure in failed_thresholds:
            logger.error("%s", failure)
        return 2
    return 0


def _failed_thresholds(aggregate: dict, args: argparse.Namespace) -> list[str]:
    checks = [
        ("reference_context_recall", args.min_context_recall, "min"),
        ("answer_relevancy", args.min_answer_relevancy, "min"),
        ("conflicting_hit_rate", args.max_conflicting_hit_rate, "max"),
    ]
    failures: list[str] = []
    for metric, threshold, mode in checks:
        if threshold is None:
            continue
        value = aggregate.get(metric)
        if value is None and metric == "reference_context_recall":
            # LLM-mode reports may carry the deepeval naming instead.
            value = aggregate.get("context_recall")
        if value is None:
            # A gate on a metric the report does not contain must fail, not
            # silently pass with a default 0.0.
            failures.append(
                f"{metric} is missing from the report; gate cannot be evaluated"
            )
            continue
        value = float(value)
        if mode == "min" and value < threshold:
            failures.append(f"{metric}={value:.4f} is below threshold {threshold:.4f}")
        if mode == "max" and value > threshold:
            failures.append(f"{metric}={value:.4f} is above threshold {threshold:.4f}")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
