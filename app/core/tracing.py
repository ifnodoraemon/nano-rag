from __future__ import annotations

import fcntl
import json
import logging
import os
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from time import time
from typing import Any, Generic, TypeVar
from uuid import uuid4

from pydantic import BaseModel

from app.schemas.business import FeedbackRecord
from app.schemas.common import PaginatedResponse
from app.schemas.trace import TraceRecord, TraceSummary

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

TRACE_MAX_RECORDS = int(os.getenv("RAG_TRACE_MAX_RECORDS", "200"))
FEEDBACK_MAX_RECORDS = int(os.getenv("RAG_FEEDBACK_MAX_RECORDS", "500"))
OTLP_TLS_ENABLED = os.getenv("OTLP_TLS_ENABLED", "false").lower() in (
    "true",
    "1",
    "yes",
)


def _current_otel_trace_id() -> str | None:
    try:
        from opentelemetry import trace as otel_trace
    except ImportError:
        return None

    current_context = otel_trace.get_current_span().get_span_context()
    if current_context and current_context.trace_id:
        return format(current_context.trace_id, "032x")
    return None


class TraceSession:
    def __init__(self) -> None:
        self.trace_id = _current_otel_trace_id() or str(uuid4())
        self.started_at = time()
        self.events: dict[str, object] = {}

    def record(self, key: str, value: object) -> None:
        self.events[key] = value

    def finish(self) -> dict[str, object]:
        self.events["latency_seconds"] = round(time() - self.started_at, 3)
        self.events["trace_id"] = self.trace_id
        return self.events


class _PersistedStore(ABC, Generic[T]):
    def __init__(
        self, max_records: int, persist_dir: Path | None, record_id_field: str
    ) -> None:
        self.max_records = max(max_records, 1)
        self.persist_dir = persist_dir
        self._record_id_field = record_id_field
        self._records: OrderedDict[str, T] = OrderedDict()
        self._lock = threading.Lock()
        if self.persist_dir is not None:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._load_persisted()

    @contextmanager
    def _file_lock(self) -> Generator[None, None, None]:
        if self.persist_dir is None:
            yield
            return
        lock_path = self.persist_dir / ".lock"
        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)

    def _load_persisted(self) -> None:
        assert self.persist_dir is not None
        files = sorted(
            self.persist_dir.glob("*.json"), key=lambda path: path.stat().st_mtime
        )
        for path in files[-self.max_records :]:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            record = self._validate_record(payload)
            record_id = getattr(record, self._record_id_field)
            self._records[record_id] = record
        self._prune_persisted()

    def _prune_persisted(self) -> None:
        assert self.persist_dir is not None
        files = sorted(
            self.persist_dir.glob("*.json"), key=lambda path: path.stat().st_mtime
        )
        if len(files) > self.max_records:
            for path in files[: len(files) - self.max_records]:
                path.unlink(missing_ok=True)

    @abstractmethod
    def _validate_record(self, payload: dict[str, object]) -> T: ...

    def _get_record_id(self, record: T) -> str:
        return getattr(record, self._record_id_field)

    def _write_record(self, record: T) -> None:
        assert self.persist_dir is not None
        record_id = self._get_record_id(record)
        path = self.persist_dir / f"{record_id}.json"
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        os.replace(str(tmp_path), str(path))

    def _save_internal(self, record: T) -> str | None:
        record_id = self._get_record_id(record)
        removed_id: str | None = None
        if record_id in self._records:
            self._records.pop(record_id)
        self._records[record_id] = record
        while len(self._records) > self.max_records:
            removed_id, _ = self._records.popitem(last=False)
        return removed_id

    def save(self, record: T) -> T:
        with self._lock:
            if self.persist_dir is not None:
                with self._file_lock():
                    removed_id = self._save_internal(record)
                    self._write_record(record)
                    if removed_id:
                        stale_file = self.persist_dir / f"{removed_id}.json"
                        if stale_file.exists():
                            stale_file.unlink()
                    self._prune_persisted()
            else:
                self._save_internal(record)
        return record


class TraceStore(_PersistedStore[TraceRecord]):
    def __init__(
        self, max_records: int = TRACE_MAX_RECORDS, persist_dir: Path | None = None
    ) -> None:
        super().__init__(max_records, persist_dir, "trace_id")

    def _validate_record(self, payload: dict[str, object]) -> TraceRecord:
        return TraceRecord.model_validate(payload)

    def save_raw(self, record: dict[str, object]) -> TraceRecord:
        trace = TraceRecord.model_validate(record)
        return super().save(trace)

    def update(self, record: TraceRecord) -> TraceRecord:
        return super().save(record)

    def get(self, trace_id: str) -> TraceRecord | None:
        with self._lock:
            return self._records.get(trace_id)

    def list(
        self,
        page: int = 1,
        page_size: int = 20,
        kb_id: str | None = None,
        tenant_id: str | None = None,
    ) -> PaginatedResponse[TraceSummary]:
        with self._lock:
            all_records = list(reversed(self._records.values()))
        if kb_id is not None:
            all_records = [
                record for record in all_records if (record.kb_id or "default") == kb_id
            ]
        if tenant_id is not None:
            all_records = [
                record for record in all_records if record.tenant_id == tenant_id
            ]
        total = len(all_records)
        start = (page - 1) * page_size
        end = start + page_size
        summaries: list[TraceSummary] = []
        for record in all_records[start:end]:
            conflicting_context_count = sum(
                1
                for context in record.contexts
                if isinstance(context, dict)
                and context.get("wiki_status") == "conflicting"
            )
            conflict_claim_count = sum(
                1
                for claim in record.supporting_claims
                if isinstance(claim, dict) and claim.get("claim_type") == "conflict"
            )
            insufficiency_claim_count = sum(
                1
                for claim in record.supporting_claims
                if isinstance(claim, dict)
                and claim.get("claim_type") == "insufficiency"
            )
            conditional_claim_count = sum(
                1
                for claim in record.supporting_claims
                if isinstance(claim, dict)
                and claim.get("claim_type") == "conditional"
            )
            summaries.append(
                TraceSummary(
                    trace_id=record.trace_id,
                    latency_seconds=record.latency_seconds,
                    query=record.query,
                    kb_id=record.kb_id,
                    tenant_id=record.tenant_id,
                    session_id=record.session_id,
                    model_alias=record.model_alias,
                    prompt_version=record.prompt_version,
                    context_count=len(record.contexts),
                    conflicting_context_count=conflicting_context_count,
                    conflict_claim_count=conflict_claim_count,
                    insufficiency_claim_count=insufficiency_claim_count,
                    conditional_claim_count=conditional_claim_count,
                )
            )
        return PaginatedResponse.create(summaries, total, page, page_size)


class FeedbackStore(_PersistedStore[FeedbackRecord]):
    def __init__(
        self, max_records: int = FEEDBACK_MAX_RECORDS, persist_dir: Path | None = None
    ) -> None:
        super().__init__(max_records, persist_dir, "feedback_id")

    def _validate_record(self, payload: dict[str, object]) -> FeedbackRecord:
        return FeedbackRecord.model_validate(payload)

    def save_raw(self, record: dict[str, object]) -> FeedbackRecord:
        feedback = FeedbackRecord.model_validate(record)
        return super().save(feedback)

    def list(self) -> list[FeedbackRecord]:
        with self._lock:
            return list(reversed(self._records.values()))


class TracingManager:
    def __init__(self, service_name: str, collector_endpoint: str) -> None:
        self.service_name = service_name
        self.collector_endpoint = collector_endpoint
        self.tracer: Any | None = None

        if not collector_endpoint:
            return

        try:
            from opentelemetry import trace as otel_trace
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
        except ImportError as exc:
            logger.warning(
                "Tracing dependencies are not installed; disabling OTLP export: %s",
                exc,
            )
            return

        provider = TracerProvider(
            resource=Resource.create({"service.name": service_name})
        )
        provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(
                    endpoint=collector_endpoint, insecure=not OTLP_TLS_ENABLED
                )
            )
        )
        otel_trace.set_tracer_provider(provider)
        self.tracer = otel_trace.get_tracer(service_name)

    @contextmanager
    def span(
        self, name: str, attributes: dict[str, object] | None = None
    ) -> Generator[Any | None, None, None]:
        if self.tracer is None:
            yield None
            return

        with self.tracer.start_as_current_span(name) as span:
            for key, value in (attributes or {}).items():
                if value is None:
                    continue
                if isinstance(value, (str, bool, int, float)):
                    span.set_attribute(key, value)
                else:
                    span.set_attribute(key, str(value))
            yield span
