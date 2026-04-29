from __future__ import annotations

from dataclasses import dataclass
import hmac
import logging
import os

from fastapi import Header, HTTPException, Request

logger = logging.getLogger(__name__)
AUTH_TRUE_VALUES = {"true", "1"}


@dataclass(frozen=True)
class RequestContext:
    auth_mode: str
    principal_id: str | None = None
    external_org_id: str | None = None
    allowed_kb_ids: set[str] | None = None


def _constant_time_check(token: str, keys: set[str]) -> bool:
    return any(hmac.compare_digest(token, key) for key in keys)


def is_auth_disabled() -> bool:
    return os.getenv("RAG_AUTH_DISABLED", "").lower() in AUTH_TRUE_VALUES


def require_api_key(
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> RequestContext:
    container = request.app.state.container
    if is_auth_disabled():
        logger.warning(
            "API authentication explicitly disabled via RAG_AUTH_DISABLED=true."
        )
        return RequestContext(auth_mode="disabled")
    keys = container.config.business_api_keys
    if not keys:
        raise HTTPException(
            status_code=503,
            detail=(
                "RAG_API_KEYS not configured. Set RAG_API_KEYS for API access "
                "or set RAG_AUTH_DISABLED=true only for local development."
            ),
        )
    token = x_api_key
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    if not token or not _constant_time_check(token, keys):
        raise HTTPException(status_code=401, detail="invalid or missing api key")
    return RequestContext(auth_mode="api_key")
