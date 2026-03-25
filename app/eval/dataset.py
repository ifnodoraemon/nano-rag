import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def get_eval_dataset_dir() -> Path:
    return Path(os.getenv("EVAL_DATASET_DIR", ROOT / "data" / "eval"))


def get_eval_report_dir() -> Path:
    return Path(os.getenv("EVAL_REPORT_DIR", ROOT / "data" / "reports" / "eval"))


def get_benchmark_report_dir() -> Path:
    return get_eval_report_dir() / "benchmarks"


def load_jsonl_dataset(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def save_json(path: str, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def resolve_data_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return ROOT / raw_path


def _resolve_within(base_dir: Path, raw_path: str, label: str) -> Path:
    base = base_dir.resolve()
    target = resolve_data_path(raw_path).resolve()
    try:
        target.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"{label} path must be inside {base}") from exc
    return target


def resolve_eval_dataset_path(raw_path: str) -> Path:
    return _resolve_within(get_eval_dataset_dir(), raw_path, "eval dataset")


def resolve_eval_report_path(raw_path: str) -> Path:
    return _resolve_within(get_eval_report_dir(), raw_path, "eval report")


def resolve_benchmark_report_path(raw_path: str) -> Path:
    return _resolve_within(get_benchmark_report_dir(), raw_path, "benchmark report")


def summarize_jsonl_dataset(path: Path) -> dict[str, Any]:
    records = load_jsonl_dataset(str(path))
    sample_queries = [str(record.get("query", "")).strip() for record in records[:3] if str(record.get("query", "")).strip()]
    return {
        "name": path.name,
        "path": str(path.relative_to(ROOT)),
        "records": len(records),
        "sample_queries": sample_queries,
        "updated_at": int(path.stat().st_mtime),
    }


def list_eval_datasets() -> list[dict[str, Any]]:
    dataset_dir = get_eval_dataset_dir()
    if not dataset_dir.exists():
        return []
    return [summarize_jsonl_dataset(path) for path in sorted(dataset_dir.glob("*.jsonl"))]


def summarize_eval_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    aggregate = payload.get("aggregate", {}) if isinstance(payload, dict) else {}
    return {
        "name": path.name,
        "path": str(path.relative_to(ROOT)),
        "records": int(payload.get("records", 0)) if isinstance(payload, dict) else 0,
        "status": str(payload.get("status", "unknown")) if isinstance(payload, dict) else "unknown",
        "aggregate": aggregate if isinstance(aggregate, dict) else {},
        "updated_at": int(path.stat().st_mtime),
    }


def list_eval_reports() -> list[dict[str, Any]]:
    report_dir = get_eval_report_dir()
    if not report_dir.exists():
        return []
    reports = [summarize_eval_report(path) for path in sorted(report_dir.glob("*.json"))]
    reports.sort(key=lambda item: int(item["updated_at"]), reverse=True)
    return reports


def list_benchmark_reports() -> list[dict[str, Any]]:
    report_dir = get_benchmark_report_dir()
    if not report_dir.exists():
        return []
    reports = [summarize_eval_report(path) for path in sorted(report_dir.glob("*.json"))]
    reports.sort(key=lambda item: int(item["updated_at"]), reverse=True)
    return reports
