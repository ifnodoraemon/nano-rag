from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.api.auth import require_admin_key, require_api_key


def _request_with_keys(keys: set[str]):
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(container=SimpleNamespace(config=SimpleNamespace(business_api_keys=keys)))))


def test_business_auth_rejects_missing_key_when_not_configured(monkeypatch) -> None:
    monkeypatch.delenv("RAG_AUTH_DISABLED", raising=False)
    with pytest.raises(HTTPException) as exc_info:
        require_api_key(_request_with_keys(set()), authorization=None, x_api_key=None)

    assert exc_info.value.status_code == 503


def test_business_auth_can_be_disabled_explicitly(monkeypatch) -> None:
    monkeypatch.setenv("RAG_AUTH_DISABLED", "true")
    require_api_key(_request_with_keys(set()), authorization=None, x_api_key=None)


def test_business_auth_accepts_bearer_token(monkeypatch) -> None:
    monkeypatch.delenv("RAG_AUTH_DISABLED", raising=False)
    require_api_key(_request_with_keys({"secret-token"}), authorization="Bearer secret-token", x_api_key=None)


def test_business_auth_rejects_invalid_token(monkeypatch) -> None:
    monkeypatch.delenv("RAG_AUTH_DISABLED", raising=False)
    with pytest.raises(HTTPException) as exc_info:
        require_api_key(_request_with_keys({"secret-token"}), authorization="Bearer nope", x_api_key=None)

    assert exc_info.value.status_code == 401


def test_business_auth_rejects_local_default_in_production(monkeypatch) -> None:
    monkeypatch.setenv("RAG_ENV", "production")
    monkeypatch.delenv("RAG_AUTH_DISABLED", raising=False)
    with pytest.raises(HTTPException) as exc_info:
        require_api_key(
            _request_with_keys({"nano-rag-local"}),
            authorization="Bearer nano-rag-local",
            x_api_key=None,
        )

    assert exc_info.value.status_code == 503


def test_trusted_proxy_context_scopes_kbs(monkeypatch) -> None:
    monkeypatch.delenv("RAG_AUTH_DISABLED", raising=False)
    monkeypatch.setenv("RAG_TRUSTED_PROXY_SECRET", "proxy-secret")

    context = require_api_key(
        _request_with_keys(set()),
        authorization=None,
        x_api_key=None,
        x_rag_proxy_secret="proxy-secret",
        x_rag_principal_id="user-1",
        x_rag_org_id="org-1",
        x_rag_allowed_kb_ids="default,finance",
    )

    assert context.auth_mode == "trusted_proxy"
    assert context.principal_id == "user-1"
    assert context.allowed_kb_ids == {"default", "finance"}


def test_admin_auth_requires_separate_admin_key(monkeypatch) -> None:
    monkeypatch.delenv("RAG_AUTH_DISABLED", raising=False)
    monkeypatch.delenv("RAG_ADMIN_API_KEYS", raising=False)

    with pytest.raises(HTTPException) as exc_info:
        require_admin_key(
            _request_with_keys({"business-secret"}),
            authorization=None,
            x_admin_api_key=None,
            x_rag_admin_key=None,
        )

    assert exc_info.value.status_code == 503


def test_admin_auth_accepts_admin_key(monkeypatch) -> None:
    monkeypatch.delenv("RAG_AUTH_DISABLED", raising=False)
    monkeypatch.setenv("RAG_ADMIN_API_KEYS", "admin-secret")

    context = require_admin_key(
        _request_with_keys({"business-secret"}),
        authorization=None,
        x_admin_api_key=None,
        x_rag_admin_key="admin-secret",
    )

    assert context.auth_mode == "admin_api_key"
