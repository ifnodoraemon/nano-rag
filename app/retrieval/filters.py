from __future__ import annotations

import re
from datetime import date, datetime

DOC_TYPE_MARKERS: dict[str, tuple[str, ...]] = {
    "faq": ("faq", "q&a", "常见问题", "问答"),
    "policy": ("policy", "政策", "制度", "办法", "规范"),
    "handbook": ("handbook", "手册", "员工手册"),
    "guide": ("guide", "guideline", "指南", "指引"),
    "procedure": ("procedure", "流程", "操作步骤", "操作规程"),
    "contract": ("contract", "agreement", "合同", "协议"),
    "form": ("form", "template", "表单", "模板"),
}
DATE_PATTERNS = (
    re.compile(r"((?:19|20)\d{2}-\d{1,2}-\d{1,2})"),
    re.compile(r"((?:19|20)\d{2}/\d{1,2}/\d{1,2})"),
    re.compile(r"((?:19|20)\d{2}年\d{1,2}月\d{1,2}日)"),
)
YEAR_PATTERN = re.compile(r"\b((?:19|20)\d{2})\b")
VERSION_PATTERNS = (
    re.compile(r"\b(v\d+(?:\.\d+)*)\b", re.IGNORECASE),
    re.compile(r"\b(version\s+\d+(?:\.\d+)*)\b", re.IGNORECASE),
    re.compile(r"(版本[:：]?\s*[A-Za-z0-9._-]+)"),
)


def infer_metadata_filters(query: str) -> dict[str, object]:
    lowered = query.lower()
    inferred: dict[str, object] = {}

    doc_types = [
        doc_type
        for doc_type, markers in DOC_TYPE_MARKERS.items()
        if any(marker in lowered for marker in markers)
    ]
    if doc_types:
        inferred["doc_types"] = doc_types
        inferred["doc_type_match_mode"] = "soft"

    explicit_date = _extract_first_match(query, DATE_PATTERNS)
    if explicit_date:
        normalized = normalize_date_string(explicit_date)
        if normalized:
            inferred["effective_date_to"] = normalized
    else:
        year_match = YEAR_PATTERN.search(query)
        if year_match:
            year = year_match.group(1)
            inferred["effective_date_from"] = f"{year}-01-01"
            inferred["effective_date_to"] = f"{year}-12-31"

    version = _extract_first_match(query, VERSION_PATTERNS)
    if version:
        inferred["version"] = normalize_version(version)

    return inferred


def merge_metadata_filters(
    explicit_filters: dict[str, object] | None,
    inferred_filters: dict[str, object] | None,
) -> dict[str, object] | None:
    merged: dict[str, object] = {}
    if inferred_filters:
        merged.update(
            {
                key: value
                for key, value in inferred_filters.items()
                if value not in (None, "", [], {})
            }
        )
    if explicit_filters:
        merged.update(
            {
                key: value
                for key, value in explicit_filters.items()
                if value not in (None, "", [], {})
            }
        )
        if explicit_filters.get("doc_types"):
            merged.pop("doc_type_match_mode", None)
    return merged or None


def match_metadata_filters(
    metadata: dict[str, object] | None,
    filters: dict[str, object] | None,
) -> bool:
    if not filters:
        return True
    metadata = metadata or {}

    doc_types = filters.get("doc_types")
    if isinstance(doc_types, list) and doc_types:
        wanted_doc_types = {str(item).lower() for item in doc_types}
        actual_doc_type = str(metadata.get("doc_type", "")).lower()
        actual_doc_types = metadata.get("doc_types")
        if isinstance(actual_doc_types, list):
            candidate_doc_types = {
                str(item).lower() for item in actual_doc_types if str(item).strip()
            }
        else:
            candidate_doc_types = {actual_doc_type} if actual_doc_type else set()
        if (
            filters.get("doc_type_match_mode") == "soft"
            and not candidate_doc_types
        ):
            return True
        if not candidate_doc_types.intersection(wanted_doc_types):
            return False

    version = filters.get("version")
    if version:
        actual_version = normalize_version(metadata.get("version"))
        if actual_version != normalize_version(version):
            return False

    effective_date = parse_date(metadata.get("effective_date"))
    effective_date_from = parse_date(filters.get("effective_date_from"))
    effective_date_to = parse_date(filters.get("effective_date_to"))
    if effective_date_from or effective_date_to:
        if effective_date is None:
            return False
        if effective_date_from and effective_date < effective_date_from:
            return False
        if effective_date_to and effective_date > effective_date_to:
            return False

    return True


def parse_date(value: object) -> date | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = normalize_date_string(value)
    if not normalized:
        return None
    try:
        return datetime.strptime(normalized, "%Y-%m-%d").date()
    except ValueError:
        return None


def normalize_date_string(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    if "年" in raw and "月" in raw and "日" in raw:
        match = re.match(r"((?:19|20)\d{2})年(\d{1,2})月(\d{1,2})日", raw)
        if match:
            year, month, day = match.groups()
            return f"{year}-{int(month):02d}-{int(day):02d}"
    for separator in ("-", "/"):
        parts = raw.split(separator)
        if len(parts) == 3 and all(part.isdigit() for part in parts):
            year, month, day = parts
            return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
    return None


def normalize_version(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    normalized = normalized.replace("版本", "").replace("version", "").replace("：", ":")
    normalized = normalized.replace(":", "").strip()
    return normalized or None


def sanitize_metadata_filters(
    filters: dict[str, object] | None,
) -> dict[str, object] | None:
    if not filters:
        return None
    cleaned = {
        key: value
        for key, value in filters.items()
        if not str(key).endswith("_mode")
    }
    return cleaned or None


def _extract_first_match(text: str, patterns: tuple[re.Pattern[str], ...]) -> str | None:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    return None
