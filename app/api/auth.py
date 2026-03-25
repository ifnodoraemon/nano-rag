from __future__ import annotations

import logging

from fastapi import Header, HTTPException, Request

logger = logging.getLogger(__name__)


def require_api_key(
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> None:
    container = request.app.state.container
    keys = container.config.business_api_keys
    if not keys:
        logger.warning(
            "RAG_API_KEYS not configured. API authentication is disabled. "
            "This is not recommended for production environments."
        )
        return
    token = x_api_key
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    if not token or token not in keys:
        raise HTTPException(status_code=401, detail="invalid or missing api key")
