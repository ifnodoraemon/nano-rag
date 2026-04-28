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
from app.eval.ragas_runner import RagasRunner

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
        "--ragas-lib",
        action="store_true",
        help="Use the RAGAS library metrics in addition to built-in deterministic metrics.",
    )
    args = parser.parse_args()

    dataset_path = resolve_project_path(args.dataset)
    if args.output:
        output_path = resolve_project_path(args.output)
    else:
        report_dir = get_eval_report_dir()
        report_dir.mkdir(parents=True, exist_ok=True)
        output_path = report_dir / f"{dataset_path.stem}_manual.json"
    dataset = load_jsonl_dataset(str(dataset_path))
    container = AppContainer.from_env()
    runner = container.ragas_runner or RagasRunner(
        generation_client=container.generation_client
    )
    evaluated_records = asyncio.run(materialize_eval_records(container, dataset))
    report = (
        asyncio.run(runner.run_async(evaluated_records))
        if args.ragas_lib
        else runner.run(evaluated_records)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(str(output_path), report)
    logger.info("%s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
