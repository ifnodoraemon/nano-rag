"""Small, total coercions for values that come out of untyped dicts.

Metadata in this project flows through ``dict[str, object]`` (parsed artifact
frontmatter, trace records, model payloads). Reaching into those dicts with
``for item in value`` or ``float(value)`` is unsound: a malformed value crashes
a request instead of degrading a single field. These helpers make every such
read total and explicit.
"""

from __future__ import annotations


def as_list(value: object) -> list[object]:
    """Return the value if it is a list, else an empty list."""
    return value if isinstance(value, list) else []


def as_str_list(value: object) -> list[str]:
    """Return the non-empty string forms of a list-ish value."""
    return [str(item).strip() for item in as_list(value) if str(item).strip()]


def as_str(value: object, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def as_int(value: object, default: int = 0) -> int:
    """Integer coercion for a scalar that may be missing or malformed."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str)):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    return default


def as_float(value: object, default: float = 0.0) -> float:
    """Float coercion for a scalar that may be missing or malformed."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, str)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    return default


def as_dict(value: object) -> dict[str, object]:
    """Return the value if it is a mapping, else an empty dict."""
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    return {}
