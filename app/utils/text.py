from __future__ import annotations

import re
import unicodedata


def normalize_text(raw: object) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    text = re.sub(r"(\*\*|__|`)", "", text)
    text = re.sub(r"\s*\[[^\[\]]+:[^\[\]]+\](?=[。！？.!?]?\s*$)", "", text)
    text = re.sub(r"[。！？.!?]+\s*$", "", text)
    text = re.sub(r"\s+", "", text)
    return text.strip()


def normalize_for_comparison(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = re.sub(r"\s+", "", normalized)
    normalized = re.sub(r"[^\w\u4e00-\u9fff]", "", normalized)
    return normalized.lower()


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value) if value is not None else default
        return result if result == result else default
    except (ValueError, TypeError):
        return default


def parse_bool_env(value: str | None) -> bool:
    if not value:
        return False
    return value.strip().lower() in ("1", "true", "yes", "on")
