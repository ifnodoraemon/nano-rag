import os
from pathlib import Path


SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt", ".html"}
_DEFAULT_TEST_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


def _get_allowed_ingest_dirs() -> list[Path]:
    raw = os.getenv("RAG_INGEST_ALLOWED_DIRS", "")
    if not raw.strip():
        if os.getenv("PYTEST_CURRENT_TEST") or os.getenv("TESTING"):
            return [_DEFAULT_TEST_DIR]
        raise RuntimeError(
            "RAG_INGEST_ALLOWED_DIRS environment variable is not set. "
            "Please configure the allowed directories for document ingestion, "
            "e.g., RAG_INGEST_ALLOWED_DIRS=/data/docs,/data/pdfs"
        )
    return [Path(item.strip()).resolve() for item in raw.split(",") if item.strip()]


_ALLOWED_INGEST_DIRS: list[Path] | None = None


def _get_allowed_dirs() -> list[Path]:
    global _ALLOWED_INGEST_DIRS
    if _ALLOWED_INGEST_DIRS is None:
        _ALLOWED_INGEST_DIRS = _get_allowed_ingest_dirs()
    return _ALLOWED_INGEST_DIRS


def _is_path_allowed(path: Path) -> bool:
    resolved = path.resolve()
    for allowed_dir in _get_allowed_dirs():
        try:
            resolved.relative_to(allowed_dir)
            return True
        except ValueError:
            continue
    return False


class IngestPathError(RuntimeError):
    pass


def discover_files(root: str) -> list[Path]:
    base = Path(root).resolve()
    if not _is_path_allowed(base):
        allowed_str = ", ".join(str(d) for d in _get_allowed_dirs())
        raise IngestPathError(
            f"Path '{root}' is not within allowed ingest directories. "
            f"Allowed directories: {allowed_str}. "
            "Set RAG_INGEST_ALLOWED_DIRS env var to configure additional directories."
        )
    if base.is_file():
        return [base]
    return sorted(
        path
        for path in base.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
