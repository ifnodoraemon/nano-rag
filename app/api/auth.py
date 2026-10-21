from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import logging
import os
import threading
from time import monotonic

from fastapi import Header, HTTPException, Request

from app.utils.constants import INSECURE_DEFAULT_KEYS

logger = logging.getLogger(__name__)
AUTH_TRUE_VALUES = {"true", "1"}
# Environments where a known default key is acceptable for development.
# Everything else (unset, "prod", "production", or any other value) is treated
# as production and rejects insecure defaults.
_LOCAL_ENVS = {"local", "dev", "development", "test", "testing", "unittest"}
_RATE_LOCK = threading.Lock()
_RATE_WINDOW: dict[str, tuple[float, int]] = {}
_RATE_SWEEP_INTERVAL_SECONDS = 60.0
_LAST_RATE_SWEEP = 0.0


@dataclass(frozen=True)
class RequestContext:
    auth_mode: str
    principal_id: str | None = None
    external_org_id: str | None = None
    allowed_kb_ids: set[str] | None = None


def _constant_time_check(token: str, keys: set[str]) -> bool:
    # http headers arrive latin-1 decoded: non-ASCII tokens would make
    # hmac.compare_digest raise TypeError (unauthenticated 500). They can
    # never match a configured ASCII key, so treat them as a plain miss.
    if not token.isascii():
        return False
    return any(hmac.compare_digest(token, key) for key in keys)


def _allows_insecure_defaults() -> bool:
    """Fail-closed: insecure default keys are only tolerated in explicit
    local/dev/test environments. An unset or unrecognized RAG_ENV (including
    "prod"/"production") rejects them, so a deployment that forgets to set a
    strong secret cannot silently serve with a known key."""
    env = os.getenv("RAG_ENV", os.getenv("ENVIRONMENT", "")).strip().lower()
    return env in _LOCAL_ENVS


def _reject_insecure_defaults(keys: set[str], *, capability: str) -> None:
    if _allows_insecure_defaults():
        return
    insecure = sorted(keys.intersection(INSECURE_DEFAULT_KEYS))
    if insecure:
        raise HTTPException(
            status_code=503,
            detail=(
                f"{capability} contains insecure default credentials outside a "
                "local/dev/test environment. Set a strong secret and rotate any "
                "exposed defaults."
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
    # On by default: leaving the limiter off meant unlimited admin-key guessing.
    raw = os.getenv("RAG_RATE_LIMIT_REQUESTS_PER_MINUTE", "300")
    try:
        return max(0, int(raw))
    except ValueError as exc:
        # A malformed limit must never silently disable the limiter.
        raise HTTPException(
            status_code=500,
            detail=(
                "RAG_RATE_LIMIT_REQUESTS_PER_MINUTE is misconfigured and must "
                "be an integer; refusing to serve with the limiter disabled"
            ),
        ) from exc


def _client_address(request: Request | None) -> str:
    """Best-effort client identity for rate limiting.

    Behind the bundled nginx the socket address is the proxy, so the
    nginx-supplied ``X-Real-IP`` is preferred — but only as a bucketing hint,
    never for authorization.
    """
    if request is None:
        return "unknown"
    headers = getattr(request, "headers", None)
    forwarded = None
    if headers is not None and hasattr(headers, "get"):
        forwarded = headers.get("x-real-ip") or headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip() or "unknown"
    client = getattr(request, "client", None)
    host = getattr(client, "host", None)
    return str(host) if host else "unknown"


def _rate_limit_subject(request: Request | None, credential: str | None) -> str:
    """Bucket by client address AND credential so one shared proxy-injected
    key does not put every browser in a single global bucket."""
    address = _client_address(request)
    return f"{address}|{credential or 'anonymous'}"


def _enforce_rate_limit(key: str) -> None:
    global _LAST_RATE_SWEEP
    limit = _rate_limit_enabled()
    if limit <= 0:
        return
    now = monotonic()
    bucket = hashlib.sha256(key.encode("utf-8")).hexdigest()
    with _RATE_LOCK:
        # Time-based sweep (not only when huge): a proxy client rotating
        # principal ids must not be able to keep the window unbounded.
        if len(_RATE_WINDOW) > 10000 or now - _LAST_RATE_SWEEP >= _RATE_SWEEP_INTERVAL_SECONDS:
            stale = [
                old_bucket
                for old_bucket, (window_start, _) in _RATE_WINDOW.items()
                if now - window_start >= 120
            ]
            for old_bucket in stale:
                _RATE_WINDOW.pop(old_bucket, None)
            _LAST_RATE_SWEEP = now
        window_start, count = _RATE_WINDOW.get(bucket, (now, 0))
        if now - window_start >= 60:
            window_start, count = now, 0
        if count >= limit:
            raise HTTPException(status_code=429, detail="rate limit exceeded")
        _RATE_WINDOW[bucket] = (window_start, count + 1)


def is_auth_disabled() -> bool:
    """``RAG_AUTH_DISABLED=true`` is honoured only in an explicit local/dev/
    test environment.

    Insecure *keys* are already rejected outside those environments; letting
    the kill switch through everywhere would be the same fail-open hole with
    a bigger blast radius. A deployment that sets it in production keeps
    authentication enforced (and the startup readiness log says so).
    """
    raw = os.getenv("RAG_AUTH_DISABLED", "").strip().lower()
    if raw not in AUTH_TRUE_VALUES:
        return False
    if _allows_insecure_defaults():
        return True
    logger.error(
        "RAG_AUTH_DISABLED=true is ignored outside a local/dev/test environment "
        "(RAG_ENV=%r); authentication remains enforced.",
        os.getenv("RAG_ENV", os.getenv("ENVIRONMENT", "")),
    )
    return False


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
    if not x_rag_proxy_secret or not _constant_time_check(x_rag_proxy_secret, {secret}):
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
        _enforce_rate_limit(
            _rate_limit_subject(
                request,
                proxy_context.principal_id
                or proxy_context.external_org_id
                or "trusted_proxy",
            )
        )
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
    _enforce_rate_limit(_rate_limit_subject(request, token))
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
    """Operator authentication.

    Deliberately NOT satisfiable by the trusted-proxy secret: the proxy is a
    pass-through for business calls only. Operator routes must be
    authenticated by the caller with an admin key, so a proxy (or anything
    that can set headers on the public port) can never escalate to
    admin/debug/eval.

    The `x_rag_*` proxy headers stay in the signature so FastAPI keeps
    accepting the requests, but they are ignored here — the proxy identity
    is never trusted for admin access.
    """
    del x_rag_proxy_secret, x_rag_principal_id, x_rag_org_id, x_rag_allowed_kb_ids
    if is_auth_disabled():
        logger.warning(
            "Admin API authentication explicitly disabled via RAG_AUTH_DISABLED=true."
        )
        return RequestContext(auth_mode="disabled")
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
    _enforce_rate_limit(_rate_limit_subject(request, f"admin:{token}"))
    return RequestContext(auth_mode="admin_api_key")


def require_api_or_admin_key(
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
    x_admin_api_key: str | None = Header(default=None),
    x_rag_admin_key: str | None = Header(default=None),
    x_rag_proxy_secret: str | None = Header(default=None),
    x_rag_principal_id: str | None = Header(default=None),
    x_rag_org_id: str | None = Header(default=None),
    x_rag_allowed_kb_ids: str | None = Header(default=None),
) -> RequestContext:
    """Authenticate with either credential.

    Used by read-only endpoints an operator drives directly (ingest job status):
    the browser presents the proxy-injected business key, while a CLI operator
    presents the admin key. The KB scope of whichever credential authenticated
    is still enforced by the handler.
    """
    try:
        return require_api_key(
            request,
            authorization=authorization,
            x_api_key=x_api_key,
            x_rag_proxy_secret=x_rag_proxy_secret,
            x_rag_principal_id=x_rag_principal_id,
            x_rag_org_id=x_rag_org_id,
            x_rag_allowed_kb_ids=x_rag_allowed_kb_ids,
        )
    except HTTPException as business_error:
        if business_error.status_code not in {401, 503}:
            raise
        return require_admin_key(
            request,
            authorization=authorization,
            x_admin_api_key=x_admin_api_key,
            x_rag_admin_key=x_rag_admin_key,
        )
