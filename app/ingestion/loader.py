import os
from pathlib import Path


SUPPORTED_EXTENSIONS = {
    # text-bearing
    ".pdf", ".md", ".txt", ".html",
    # image
    ".png", ".jpg", ".jpeg", ".webp",
    # audio
    ".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac",
    # video
    ".mp4", ".mov", ".webm", ".mkv", ".avi", ".mpeg", ".mpg",
}
_DEFAULT_TEST_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
_DEFAULT_UPLOAD_DIR = Path(__file__).resolve().parents[2] / "data" / "uploads"

_cached_allowed_dirs: list[Path] | None = None


def _get_allowed_ingest_dirs() -> list[Path]:
    global _cached_allowed_dirs
    if _cached_allowed_dirs is not None:
        return _cached_allowed_dirs
    raw = os.getenv("RAG_INGEST_ALLOWED_DIRS", "")
    upload_dir = Path(os.getenv("UPLOAD_OUTPUT_DIR", _DEFAULT_UPLOAD_DIR)).resolve()
    if not raw.strip():
        if os.getenv("PYTEST_CURRENT_TEST") or os.getenv("TESTING"):
            _cached_allowed_dirs = [_DEFAULT_TEST_DIR.resolve(), upload_dir]
            return _cached_allowed_dirs
        raise RuntimeError(
            "RAG_INGEST_ALLOWED_DIRS environment variable is not set. "
            "Please configure the allowed directories for document ingestion, "
            "e.g., RAG_INGEST_ALLOWED_DIRS=/data/docs,/data/pdfs"
        )
    allowed_dirs = [Path(item.strip()).resolve() for item in raw.split(",") if item.strip()]
    if upload_dir not in allowed_dirs:
        allowed_dirs.append(upload_dir)
    for d in allowed_dirs:
        if str(d) == "/":
            raise RuntimeError(
                "RAG_INGEST_ALLOWED_DIRS must not include the filesystem root '/'."
            )
    _cached_allowed_dirs = allowed_dirs
    return _cached_allowed_dirs


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


def list_allowed_ingest_sources() -> list[dict[str, object]]:
    sources: list[dict[str, object]] = []
    seen: set[str] = set()
    for allowed_dir in _get_allowed_dirs():
        if not allowed_dir.exists():
            continue
        candidates = [allowed_dir] if allowed_dir.is_file() else sorted(allowed_dir.rglob("*"))
        for path in candidates:
            if (
                not path.is_file()
                or path.suffix.lower() not in SUPPORTED_EXTENSIONS
                or not _is_path_allowed(path)
            ):
                continue
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            stat = path.stat()
            sources.append(
                {
                    "path": resolved,
                    "name": path.name,
                    "extension": path.suffix.lower(),
                    "size_bytes": stat.st_size,
                    "updated_at": stat.st_mtime,
                }
            )
    return sources


def discover_files(root: str) -> list[Path]:
    base = Path(root).resolve()
    if not _is_path_allowed(base):
        raise IngestPathError(
            "Path is not within allowed ingest directories."
        )
    if base.is_file():
        return [base]
    return sorted(
        path
        for path in base.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and _is_path_allowed(path.resolve())
    )
    if base.is_file():
        return [base]
    return sorted(
        path
        for path in base.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
