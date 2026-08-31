from __future__ import annotations

import re
from datetime import date, datetime

DATE_PATTERNS = (
    re.compile(r"((?:19|20)\d{2}-\d{1,2}-\d{1,2})"),
    re.compile(r"((?:19|20)\d{2}/\d{1,2}/\d{1,2})"),
    re.compile(r"((?:19|20)\d{2}年\d{1,2}月\d{1,2}日)"),
)
VERSION_PATTERNS = (
    re.compile(r"\b(v\d+(?:\.\d+)*)\b", re.IGNORECASE),
    re.compile(r"\b(version\s+\d+(?:\.\d+)*)\b", re.IGNORECASE),
    re.compile(r"(版本[:：]?\s*[A-Za-z0-9._-]+)"),
)


def infer_metadata_filters(query: str) -> dict[str, object]:
    """Syntax-level filter inference from the query.

    Only EXPLICIT full dates produce a (soft) date window. A bare year is
    deliberately NOT expanded into a date range: "2013年制定的规定现在还有
    效吗" would otherwise hard-exclude the newer current version that the
    version ledger promotes — temporal semantics belong to the LLM reading
    plan, not to a regex guess.
    """
    inferred: dict[str, object] = {}

    explicit_date = _extract_first_match(query, DATE_PATTERNS)
    if explicit_date:
        normalized = normalize_date_string(explicit_date)
        if normalized:
            inferred["effective_date_to"] = normalized
            inferred["effective_date_match_mode"] = "soft"

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
        if explicit_filters.get("effective_date_from") or explicit_filters.get("effective_date_to"):
            merged.pop("effective_date_match_mode", None)
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
        if not candidate_doc_types:
            # Soft mode: a document without any doc_type does not disqualify
            # itself here — but unlike the old early-return, the version and
            # date checks below still apply to it.
            if filters.get("doc_type_match_mode") != "soft":
                return False
        elif not candidate_doc_types.intersection(wanted_doc_types):
            return False

    version = filters.get("version")
    if version:
        if not _version_matches(metadata.get("version"), version):
            return False

    effective_date = parse_date(metadata.get("effective_date"))
    effective_date_from = parse_date(filters.get("effective_date_from"))
    effective_date_to = parse_date(filters.get("effective_date_to"))
    if effective_date_from or effective_date_to:
        if effective_date is None:
            if filters.get("effective_date_match_mode") == "soft":
                return True
            return False
        if effective_date_from and effective_date < effective_date_from:
            return False
        if effective_date_to and effective_date > effective_date_to:
            return False

    return True


def _version_matches(actual: object, wanted: object) -> bool:
    """Version comparison with one shared semantic for filter and ranking.

    "v2" and "2.0" are the same version (numeric tuple comparison with
    trailing zeros canonicalized, so "2.0" == "2"); only when either side has
    no parseable numbers does it fall back to exact string equality.
    """
    actual_normalized = normalize_version(actual)
    wanted_normalized = normalize_version(wanted)
    if actual_normalized is None or wanted_normalized is None:
        return actual_normalized == wanted_normalized
    actual_key = _canonical_version_key(actual_normalized)
    wanted_key = _canonical_version_key(wanted_normalized)
    if actual_key and wanted_key:
        return actual_key == wanted_key
    return actual_normalized == wanted_normalized


def _canonical_version_key(value: str) -> tuple[int, ...]:
    key = version_key(value)
    while key and key[-1] == 0:
        key = key[:-1]
    return key


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
    # "v2" and "2.0" must normalize to comparable forms: the leading "v" is
    # notation, not identity.
    if re.fullmatch(r"v\d+(?:\.\d+)*", normalized):
        normalized = normalized[1:]
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


def version_key(value: object) -> tuple[int, ...]:
    """Extract a comparable version tuple from a free-form version string.

    Shared by the retrieval freshness ranking, the metadata filter, and the
    wiki version ledger so all three apply the same "higher numeric version
    wins / equal numeric version matches" rule.
    """
    if not isinstance(value, str):
        return ()
    parts = re.findall(r"\d+", value)
    if not parts:
        return ()
    return tuple(int(part) for part in parts)


def _extract_first_match(text: str, patterns: tuple[re.Pattern[str], ...]) -> str | None:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    return None
