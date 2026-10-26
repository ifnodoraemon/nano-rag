"""Test configuration.

Several pipeline stages now make their own model calls (query-scope classification,
scope extraction, article splitting, doc2query, HyDE). Unit tests use stub clients that
answer what their own test needs, so those stages would fail or, worse, silently change
what the test is measuring.

They are switched off HERE rather than by adding a fallback inside the code: the runtime
must fail loudly when a configured stage cannot run, and a test that forgets to disable a
stage should see that failure rather than a quietly different code path.

Tests that exercise one of these stages set the flag back to "true" themselves.
"""

from __future__ import annotations

import pytest

_STAGES_OFF_BY_DEFAULT = (
    "RAG_SCOPE_PREFERENCE_ENABLED",
    "RAG_SCOPE_EXTRACTION_ENABLED",
    "RAG_HYDE_ENABLED",
    "RAG_CANDIDATE_FILTER_ENABLED",
    "RAG_ARTICLE_INDEX_ENABLED",
    "RAG_DOC2QUERY_ENABLED",
    "RAG_MODEL_CHUNKING_ENABLED",
    "RAG_QUERY_EXPANSION_ENABLED",
)


@pytest.fixture(autouse=True)
def _disable_model_calling_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _STAGES_OFF_BY_DEFAULT:
        monkeypatch.setenv(name, "false")
    # Keep tests off the network even if a stage is enabled by accident.
    monkeypatch.setenv("MODEL_GATEWAY_MODE", "live")


@pytest.fixture
def enable_stage(monkeypatch: pytest.MonkeyPatch):
    """Turn one model-calling stage back on for a single test."""

    def _enable(name: str) -> None:
        monkeypatch.setenv(name, "true")

    return _enable


@pytest.fixture
def tmp_env(monkeypatch: pytest.MonkeyPatch):
    """Set arbitrary environment variables for a test."""

    def _set(**values: str) -> None:
        for key, value in values.items():
            monkeypatch.setenv(key, value)

    return _set
