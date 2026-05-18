from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import logging
import os
import threading
from time import monotonic

from fastapi import Header, HTTPException, Request

logger = logging.getLogger(__name__)
AUTH_TRUE_VALUES = {"true", "1"}
INSECURE_DEFAULT_KEYS = frozenset({"", "change-me", "nano-rag-local", "sk-xxx", "your-api-key"})
_RATE_LOCK = threading.Lock()
_RATE_WINDOW: dict[str, tuple[float, int]] = {}


@dataclass(frozen=True)
class RequestContext:
    auth_mode: str
    principal_id: str | None = None
    external_org_id: str | None = None
    allowed_kb_ids: set[str] | None = None


def _constant_time_check(token: str, keys: set[str]) -> bool:
    return any(hmac.compare_digest(token, key) for key in keys)


def _is_production() -> bool:
    return os.getenv("RAG_ENV", os.getenv("ENVIRONMENT", "")).lower() in {
        "prod",
        "production",
    }


def _reject_insecure_defaults(keys: set[str], *, capability: str) -> None:
    if not _is_production():
        return
    insecure = sorted(keys.intersection(INSECURE_DEFAULT_KEYS))
    if insecure:
        raise HTTPException(
            status_code=503,
            detail=(
                f"{capability} contains insecure default credentials in production. "
                "Set a strong secret and rotate any exposed defaults."
            ),
        )


def _admin_api_keys() -> set[str]:
    raw = os.getenv("RAG_ADMIN_API_KEYS", "")
    return {item.strip() for item in raw.split(",") if item.strip()}


def _trusted_proxy_secret() -> str:
    return os.getenv("RAG_TRUSTED_PROXY_SECRET", "").strip()


def _split_kb_ids(value: str | None) -> set[str] | None:
    if value is None:
        return None
    items = {item.strip() for item in value.split(",") if item.strip()}
    return items or set()


def _rate_limit_enabled() -> int:
    raw = os.getenv("RAG_RATE_LIMIT_REQUESTS_PER_MINUTE", "0")
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _enforce_rate_limit(key: str) -> None:
    limit = _rate_limit_enabled()
    if limit <= 0:
        return
    now = monotonic()
    bucket = hashlib.sha256(key.encode("utf-8")).hexdigest()
    with _RATE_LOCK:
        if len(_RATE_WINDOW) > 10000:
            stale = [
                bucket
                for bucket, (window_start, _) in _RATE_WINDOW.items()
                if now - window_start >= 120
            ]
            for bucket in stale:
                _RATE_WINDOW.pop(bucket, None)
        window_start, count = _RATE_WINDOW.get(bucket, (now, 0))
        if now - window_start >= 60:
            window_start, count = now, 0
        if count >= limit:
            raise HTTPException(status_code=429, detail="rate limit exceeded")
        _RATE_WINDOW[bucket] = (window_start, count + 1)


def is_auth_disabled() -> bool:
    return os.getenv("RAG_AUTH_DISABLED", "").lower() in AUTH_TRUE_VALUES


def _proxy_context(
    *,
    x_rag_proxy_secret: str | None,
    x_rag_principal_id: str | None,
    x_rag_org_id: str | None,
    x_rag_allowed_kb_ids: str | None,
) -> RequestContext | None:
    secret = _trusted_proxy_secret()
    if not secret:
        return None
    if not x_rag_proxy_secret or not hmac.compare_digest(x_rag_proxy_secret, secret):
        return None
    principal = (x_rag_principal_id or "").strip() or None
    org_id = (x_rag_org_id or "").strip() or None
    return RequestContext(
        auth_mode="trusted_proxy",
        principal_id=principal,
        external_org_id=org_id,
        allowed_kb_ids=_split_kb_ids(x_rag_allowed_kb_ids),
    )


def require_api_key(
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
    x_rag_proxy_secret: str | None = Header(default=None),
    x_rag_principal_id: str | None = Header(default=None),
    x_rag_org_id: str | None = Header(default=None),
    x_rag_allowed_kb_ids: str | None = Header(default=None),
) -> RequestContext:
    container = request.app.state.container
    if is_auth_disabled():
        logger.warning(
            "API authentication explicitly disabled via RAG_AUTH_DISABLED=true."
        )
        return RequestContext(auth_mode="disabled")
    proxy_context = _proxy_context(
        x_rag_proxy_secret=x_rag_proxy_secret,
        x_rag_principal_id=x_rag_principal_id,
        x_rag_org_id=x_rag_org_id,
        x_rag_allowed_kb_ids=x_rag_allowed_kb_ids,
    )
    if proxy_context is not None:
        _enforce_rate_limit(proxy_context.principal_id or proxy_context.external_org_id or "trusted_proxy")
        return proxy_context
    keys = container.config.business_api_keys
    _reject_insecure_defaults(keys, capability="RAG_API_KEYS")
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
    _enforce_rate_limit(token)
    return RequestContext(auth_mode="api_key")


def require_admin_key(
    request: Request,
    authorization: str | None = Header(default=None),
    x_admin_api_key: str | None = Header(default=None),
    x_rag_admin_key: str | None = Header(default=None),
    x_rag_proxy_secret: str | None = Header(default=None),
    x_rag_principal_id: str | None = Header(default=None),
    x_rag_org_id: str | None = Header(default=None),
    x_rag_allowed_kb_ids: str | None = Header(default=None),
) -> RequestContext:
    if is_auth_disabled():
        logger.warning(
            "Admin API authentication explicitly disabled via RAG_AUTH_DISABLED=true."
        )
        return RequestContext(auth_mode="disabled")
    proxy_context = _proxy_context(
        x_rag_proxy_secret=x_rag_proxy_secret,
        x_rag_principal_id=x_rag_principal_id,
        x_rag_org_id=x_rag_org_id,
        x_rag_allowed_kb_ids=x_rag_allowed_kb_ids,
    )
    if proxy_context is not None:
        return RequestContext(
            auth_mode="trusted_proxy_admin",
            principal_id=proxy_context.principal_id,
            external_org_id=proxy_context.external_org_id,
            allowed_kb_ids=proxy_context.allowed_kb_ids,
        )
    keys = _admin_api_keys()
    _reject_insecure_defaults(keys, capability="RAG_ADMIN_API_KEYS")
    if not keys:
        raise HTTPException(
            status_code=503,
            detail=(
                "RAG_ADMIN_API_KEYS not configured. Admin, debug, eval and replay "
                "endpoints require a separate admin key."
            ),
        )
    token = x_rag_admin_key or x_admin_api_key
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    if not token or not _constant_time_check(token, keys):
        raise HTTPException(status_code=401, detail="invalid or missing admin api key")
    _enforce_rate_limit(f"admin:{token}")
    return RequestContext(auth_mode="admin_api_key")
