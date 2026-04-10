import os
from pathlib import Path


SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt", ".html", ".png", ".jpg", ".jpeg", ".webp"}
_DEFAULT_TEST_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
_DEFAULT_UPLOAD_DIR = Path(__file__).resolve().parents[2] / "data" / "uploads"


def _get_allowed_ingest_dirs() -> list[Path]:
    raw = os.getenv("RAG_INGEST_ALLOWED_DIRS", "")
    upload_dir = Path(os.getenv("UPLOAD_OUTPUT_DIR", _DEFAULT_UPLOAD_DIR)).resolve()
    if not raw.strip():
        if os.getenv("PYTEST_CURRENT_TEST") or os.getenv("TESTING"):
            return [_DEFAULT_TEST_DIR.resolve(), upload_dir]
        raise RuntimeError(
            "RAG_INGEST_ALLOWED_DIRS environment variable is not set. "
            "Please configure the allowed directories for document ingestion, "
            "e.g., RAG_INGEST_ALLOWED_DIRS=/data/docs,/data/pdfs"
        )
    allowed_dirs = [Path(item.strip()).resolve() for item in raw.split(",") if item.strip()]
    if upload_dir not in allowed_dirs:
        allowed_dirs.append(upload_dir)
    return allowed_dirs


def _get_allowed_dirs() -> list[Path]:
    return _get_allowed_ingest_dirs()


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
        raise IngestPathError(
            f"Path is not within allowed ingest directories. "
            "Set RAG_INGEST_ALLOWED_DIRS env var to configure additional directories."
        )
    if base.is_file():
        return [base]
    return sorted(
        path
        for path in base.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
