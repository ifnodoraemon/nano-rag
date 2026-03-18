from pathlib import Path


SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt", ".html"}


def discover_files(root: str) -> list[Path]:
    base = Path(root)
    if base.is_file():
        return [base]
    return sorted(path for path in base.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)
