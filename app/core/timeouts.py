"""HTTP timeout resolution shared by the model clients.

Kept in a dependency-free leaf module on purpose: ``app.core.config`` imports the
clients, so the clients cannot import helpers back from it (circular import).
"""

from __future__ import annotations

import os
from typing import Any, Mapping

from app.core.exceptions import ConfigurationError


def timeout_seconds(
    settings: Mapping[str, Any],
    key: str,
    *,
    env_name: str,
    default: float,
) -> float:
    """Resolve an HTTP timeout: env var wins, then settings.yaml, then default.

    Timeouts used to be editable only in configs/settings.yaml, so tuning one for
    a slower provider (or for an experiment) meant editing a mounted file and
    restarting - and a slow reranker just timed out with no override.
    """
    raw: object = os.getenv(env_name)
    if raw is None:
        timeout_section = settings.get("timeout") or {}
        raw = timeout_section.get(key) if isinstance(timeout_section, Mapping) else None
    if raw is None:
        return float(default)
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        raise ConfigurationError(
            f"{env_name} (or settings.yaml timeout.{key}) must be a number, got {raw!r}"
        )
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            f"{env_name} (or settings.yaml timeout.{key}) must be a number, "
            f"got {raw!r}"
        ) from exc
    if value <= 0:
        raise ConfigurationError(f"{env_name} must be > 0, got {value}")
    return value
