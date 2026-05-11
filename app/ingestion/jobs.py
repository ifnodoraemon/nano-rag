from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from threading import Lock
from time import time
from uuid import uuid4

from pydantic import BaseModel, Field


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

    def create(
        self,
        *,
        kb_id: str,
        source: str,
        path: str,
        uploaded_files: list[str] | None = None,
    ) -> IngestJobRecord:
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
        try:
            return IngestJobRecord.model_validate(
                json.loads(path.read_text(encoding="utf-8"))
            )
        except (OSError, json.JSONDecodeError, ValueError):
            return None

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
            self._path(record.job_id).write_text(
                json.dumps(record.model_dump(mode="json"), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def _require(self, job_id: str) -> IngestJobRecord:
        record = self.get(job_id)
        if record is None:
            raise KeyError(f"ingest job not found: {job_id}")
        return record

    def _path(self, job_id: str) -> Path:
        return self.root_dir / f"{job_id}.json"
