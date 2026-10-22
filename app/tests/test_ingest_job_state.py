"""Job state machine: terminal states are final, live records are never purged."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from app.core.exceptions import StoreError
from app.ingestion.jobs import IngestJobStatus, IngestJobStore


def _store(tmp_path: Path) -> IngestJobStore:
    return IngestJobStore(tmp_path)


def test_a_completed_job_cannot_be_flipped_to_failed(tmp_path) -> None:
    """A record reaped while its worker was slow must not be rewritten later."""
    store = _store(tmp_path)
    job = store.create(kb_id="default", source="path", path="/x")
    store.mark_running(job.job_id)
    store.mark_completed(job.job_id, documents=1, chunks=2)

    with pytest.raises(StoreError, match="already completed"):
        store.mark_failed(job.job_id, stage="failed", error="late")


def test_a_failed_job_cannot_be_flipped_to_completed(tmp_path) -> None:
    store = _store(tmp_path)
    job = store.create(kb_id="default", source="path", path="/x")
    store.mark_running(job.job_id)
    store.mark_failed(job.job_id, stage="failed", error="boom")

    with pytest.raises(StoreError, match="already failed"):
        store.mark_completed(job.job_id, documents=1, chunks=1)


def test_the_first_terminal_state_wins(tmp_path) -> None:
    store = _store(tmp_path)
    job = store.create(kb_id="default", source="path", path="/x")
    store.mark_running(job.job_id)
    store.mark_completed(job.job_id, documents=1, chunks=1)

    record = store.get(job.job_id)
    assert record is not None
    assert record.status is IngestJobStatus.COMPLETED


def test_purge_never_deletes_a_running_job(tmp_path, monkeypatch) -> None:
    """Deleting a live record turns the job into a 404 for its own worker."""
    monkeypatch.setenv("RAG_INGEST_JOB_TTL_DAYS", "1")
    store = _store(tmp_path)
    job = store.create(kb_id="default", source="path", path="/x")
    store.mark_running(job.job_id)

    record_path = tmp_path / f"{job.job_id}.json"
    old = time.time() - 5 * 86400
    import os

    os.utime(record_path, (old, old))

    # Trigger a purge via the public path (purge runs on submit).
    store.create(kb_id="default", source="path", path="/y")

    assert record_path.exists(), "a RUNNING job must never be purged"


def test_purge_removes_a_finished_job(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RAG_INGEST_JOB_TTL_DAYS", "1")
    store = _store(tmp_path)
    job = store.create(kb_id="default", source="path", path="/x")
    store.mark_running(job.job_id)
    store.mark_completed(job.job_id, documents=1, chunks=1)

    record_path = tmp_path / f"{job.job_id}.json"
    old = time.time() - 5 * 86400
    import os

    os.utime(record_path, (old, old))

    store.create(kb_id="default", source="path", path="/y")

    assert not record_path.exists()


def test_a_queued_job_with_no_heartbeat_is_reaped(tmp_path) -> None:
    """A job whose enqueue failed sits QUEUED forever unless it is reaped."""
    store = _store(tmp_path)
    job = store.create(kb_id="default", source="path", path="/x")

    reaped = store.reap_stale_running(max_idle_seconds=0.0001)

    assert job.job_id in reaped
    record = store.get(job.job_id)
    assert record is not None
    assert record.status is IngestJobStatus.FAILED
    assert "abandoned" in (record.error or "")


def test_a_heartbeat_keeps_a_running_job_alive(tmp_path) -> None:
    store = _store(tmp_path)
    job = store.create(kb_id="default", source="path", path="/x")
    store.mark_running(job.job_id)
    store.heartbeat(job.job_id, stage="ingesting 1/1")

    assert store.reap_stale_running(max_idle_seconds=3600) == []
