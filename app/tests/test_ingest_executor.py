"""Ingest job bookkeeping must never mask the failure being reported."""
from __future__ import annotations

import pytest

from app.ingestion.executor import run_ingest_paths


class _BrokenStore:
    """The record disappears mid-job: writing the terminal state raises."""

    def mark_running(self, job_id: str, *, stage: str) -> None:  # noqa: ARG002
        return None

    def heartbeat(self, job_id: str, *, stage: str) -> None:  # noqa: ARG002
        return None

    def mark_completed(self, job_id: str, **kwargs) -> None:  # noqa: ANN003, ARG002
        raise KeyError("ingest job not found")

    def mark_failed(self, job_id: str, **kwargs) -> None:  # noqa: ANN003, ARG002
        raise KeyError("ingest job not found")


class _Store:
    def __init__(self) -> None:
        self.failed: list[dict] = []
        self.completed: list[dict] = []

    def mark_running(self, job_id: str, *, stage: str) -> None:  # noqa: ARG002
        return None

    def heartbeat(self, job_id: str, *, stage: str) -> None:  # noqa: ARG002
        return None

    def mark_completed(self, job_id: str, **kwargs) -> None:  # noqa: ANN003, ARG002
        self.completed.append(kwargs)

    def mark_failed(self, job_id: str, **kwargs) -> None:  # noqa: ANN003, ARG002
        self.failed.append(kwargs)


class _Pipeline:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    async def run(self, path, **kwargs):  # noqa: ANN001, ANN003
        if self.error is not None:
            raise self.error
        raise AssertionError("unexpected success path")


class _Container:
    def __init__(self, store, pipeline) -> None:  # noqa: ANN001
        self.ingest_job_store = store
        self.ingestion_pipeline = pipeline


@pytest.mark.asyncio
async def test_a_broken_job_store_does_not_replace_the_real_error() -> None:
    """Regression: mark_failed raised KeyError and hid the provider failure."""
    container = _Container(_BrokenStore(), _Pipeline(RuntimeError("PROVIDER FAILURE")))

    with pytest.raises(RuntimeError, match="PROVIDER FAILURE"):
        await run_ingest_paths(container, "job-1", ["/x"], "default", None, force=False)


@pytest.mark.asyncio
async def test_the_failure_is_recorded_when_the_store_works() -> None:
    store = _Store()
    container = _Container(store, _Pipeline(RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="boom"):
        await run_ingest_paths(container, "job-1", ["/x"], "default", None, force=False)

    assert store.failed and store.failed[0]["error"] == "boom"


@pytest.mark.asyncio
async def test_completion_counts_are_recorded() -> None:
    store = _Store()

    class _OkPipeline:
        async def run(self, path, **kwargs):  # noqa: ANN001, ANN003
            class _Response:
                documents = 2
                chunks = 5
                unchanged_doc_ids = ["d3"]
                usage = {"prompt_tokens": 10, "completion_tokens": 4}

            return _Response()

    container = _Container(store, _OkPipeline())
    await run_ingest_paths(container, "job-1", ["/x"], "default", None, force=False)

    assert store.completed[0]["documents"] == 2
    assert store.completed[0]["chunks"] == 5
    assert store.completed[0]["unchanged_documents"] == 1
    assert store.completed[0]["prompt_tokens"] == 10
    assert store.completed[0]["completion_tokens"] == 4
