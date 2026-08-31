from __future__ import annotations

import fcntl
import json
import logging
import os
from enum import Enum
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from time import time
from collections.abc import Generator
from uuid import uuid4

from pydantic import BaseModel, Field

from app.core.exceptions import StoreError

logger = logging.getLogger(__name__)

_DEFAULT_JOB_TTL_DAYS = 7


class IngestJobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class IngestJobRecord(BaseModel):
    job_id: str
    kb_id: str
    source: str
    path: str
    status: IngestJobStatus = IngestJobStatus.QUEUED
    stage: str = "queued"
    documents: int = 0
    chunks: int = 0
    uploaded_files: list[str] = Field(default_factory=list)
    error: str | None = None
    submitted_at: float = Field(default_factory=time)
    started_at: float | None = None
    completed_at: float | None = None


class IngestJobStore:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    @contextmanager
    def _file_lock(self) -> Generator[None, None, None]:
        lock_path = self.root_dir / ".lock"
        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)

    def create(
        self,
        *,
        kb_id: str,
        source: str,
        path: str,
        uploaded_files: list[str] | None = None,
    ) -> IngestJobRecord:
        self._purge_expired_jobs()
        record = IngestJobRecord(
            job_id=f"job-{uuid4().hex[:16]}",
            kb_id=kb_id,
            source=source,
            path=path,
            uploaded_files=uploaded_files or [],
        )
        self.save(record)
        return record

    def get(self, job_id: str) -> IngestJobRecord | None:
        path = self._path(job_id)
        if not path.exists():
            return None
        with self._file_lock():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                return IngestJobRecord.model_validate(raw)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                # A corrupt job record is a store-integrity failure, not
                # "job not found": returning None here would tell the API the
                # job never existed and hide the corruption.
                raise StoreError(
                    f"ingest job record {job_id} is corrupt: {exc}"
                ) from exc

    def mark_running(self, job_id: str, stage: str = "running") -> IngestJobRecord:
        record = self._require(job_id)
        record.status = IngestJobStatus.RUNNING
        record.stage = stage
        record.started_at = record.started_at or time()
        record.error = None
        self.save(record)
        return record

    def mark_completed(
        self, job_id: str, *, documents: int, chunks: int
    ) -> IngestJobRecord:
        record = self._require(job_id)
        record.status = IngestJobStatus.COMPLETED
        record.stage = "completed"
        record.documents = documents
        record.chunks = chunks
        record.completed_at = time()
        record.error = None
        self.save(record)
        return record

    def mark_failed(self, job_id: str, *, stage: str, error: str) -> IngestJobRecord:
        record = self._require(job_id)
        record.status = IngestJobStatus.FAILED
        record.stage = stage
        record.error = error
        record.completed_at = time()
        self.save(record)
        return record

    def save(self, record: IngestJobRecord) -> None:
        with self._lock:
            with self._file_lock():
                path = self._path(record.job_id)
                tmp_path = path.with_suffix(".json.tmp")
                tmp_path.write_text(
                    json.dumps(record.model_dump(mode="json"), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                os.replace(str(tmp_path), str(path))

    def _require(self, job_id: str) -> IngestJobRecord:
        record = self.get(job_id)
        if record is None:
            raise KeyError(f"ingest job not found: {job_id}")
        return record

    def _path(self, job_id: str) -> Path:
        return self.root_dir / f"{job_id}.json"

    def _purge_expired_jobs(self) -> None:
        """Once per submit: drop finished job records past the TTL so the
        store does not grow unbounded (0 = keep forever)."""
        ttl_days = _job_ttl_days()
        if ttl_days <= 0:
            return
        cutoff = time() - ttl_days * 86400
        try:
            entries = list(self.root_dir.glob("job-*.json"))
        except OSError:
            return
        for record_path in entries:
            try:
                if record_path.stat().st_mtime < cutoff:
                    record_path.unlink(missing_ok=True)
            except OSError:
                continue


def _job_ttl_days() -> int:
    raw = os.getenv("RAG_INGEST_JOB_TTL_DAYS", str(_DEFAULT_JOB_TTL_DAYS))
    try:
        return max(0, int(raw))
    except ValueError as exc:
        raise StoreError(
            f"RAG_INGEST_JOB_TTL_DAYS must be an integer, got {raw!r}"
        ) from exc
