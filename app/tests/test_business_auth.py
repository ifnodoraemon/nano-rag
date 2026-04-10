from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.api.auth import require_api_key


def _request_with_keys(keys: set[str]):
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(container=SimpleNamespace(config=SimpleNamespace(business_api_keys=keys)))))


def test_business_auth_allows_missing_key_when_not_configured(monkeypatch) -> None:
    monkeypatch.setenv("RAG_AUTH_DISABLED", "true")
    require_api_key(_request_with_keys(set()), authorization=None, x_api_key=None)


def test_business_auth_accepts_bearer_token() -> None:
    require_api_key(_request_with_keys({"secret-token"}), authorization="Bearer secret-token", x_api_key=None)


def test_business_auth_rejects_invalid_token() -> None:
    with pytest.raises(HTTPException) as exc_info:
        require_api_key(_request_with_keys({"secret-token"}), authorization="Bearer nope", x_api_key=None)

    assert exc_info.value.status_code == 401
