from __future__ import annotations

import json

from app.core.exceptions import ModelOutputError


def parse_json_object(content: str) -> dict[str, object]:
    """Parse a JSON object out of LLM output. Fail loud, never degrade.

    Strips markdown code fences and any prose around the outermost object,
    then requires a valid JSON object. Any violation raises ModelOutputError
    so callers surface the failure instead of silently falling back.
    """
    text = (content or "").strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ModelOutputError("model output contains no JSON object")
    text = text[start : end + 1]
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ModelOutputError(f"model output is not valid JSON: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ModelOutputError("model JSON output is not an object")
    return loaded
