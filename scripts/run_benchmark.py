import argparse
import asyncio
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.benchmark.service import build_benchmark_report
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
    parser = argparse.ArgumentParser(description="Run benchmark for nano-rag.")
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset.")
    parser.add_argument(
        "--output", required=False, help="Path to output benchmark JSON."
    )
    args = parser.parse_args()

    dataset_path = resolve_project_path(args.dataset)
    if args.output:
        output_path = resolve_project_path(args.output)
    else:
        output_dir = get_eval_report_dir() / "benchmarks"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{dataset_path.stem}_benchmark.json"

    container = AppContainer.from_env()
    dataset = load_jsonl_dataset(str(dataset_path))
    evaluated_records = asyncio.run(materialize_eval_records(container, dataset))
    eval_report = container.ragas_runner.run(evaluated_records)
    benchmark_report = build_benchmark_report(
        dataset_path=str(dataset_path.relative_to(ROOT)),
        eval_report=eval_report,
        trace_store=container.trace_store,
        diagnosis_service=container.diagnosis_service,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(str(output_path), benchmark_report)
    logger.info("%s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
