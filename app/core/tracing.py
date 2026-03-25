from __future__ import annotations

import json
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from time import time
from uuid import uuid4

from opentelemetry import trace as otel_trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from app.schemas.business import FeedbackRecord
from app.schemas.trace import TraceRecord, TraceSummary


class TraceSession:
    def __init__(self) -> None:
        current_context = otel_trace.get_current_span().get_span_context()
        if current_context and current_context.trace_id:
            self.trace_id = format(current_context.trace_id, "032x")
        else:
            self.trace_id = str(uuid4())
        self.started_at = time()
        self.events: dict[str, object] = {}

    def record(self, key: str, value: object) -> None:
        self.events[key] = value

    def finish(self) -> dict[str, object]:
        self.events["latency_seconds"] = round(time() - self.started_at, 3)
        self.events["trace_id"] = self.trace_id
        return self.events


class TraceStore:
    def __init__(self, max_records: int = 200, persist_dir: Path | None = None) -> None:
        self.max_records = max_records
        self.persist_dir = persist_dir
        self._records: OrderedDict[str, TraceRecord] = OrderedDict()
        if self.persist_dir is not None:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._load_persisted()

    def save(self, record: dict[str, object]) -> TraceRecord:
        trace = TraceRecord.model_validate(record)
        removed_trace_id: str | None = None
        if trace.trace_id in self._records:
            self._records.pop(trace.trace_id)
        self._records[trace.trace_id] = trace
        while len(self._records) > self.max_records:
            removed_trace_id, _ = self._records.popitem(last=False)
        if self.persist_dir is not None:
            (self.persist_dir / f"{trace.trace_id}.json").write_text(
                trace.model_dump_json(indent=2),
                encoding="utf-8",
            )
            if removed_trace_id:
                stale_file = self.persist_dir / f"{removed_trace_id}.json"
                if stale_file.exists():
                    stale_file.unlink()
            self._prune_persisted()
        return trace

    def get(self, trace_id: str) -> TraceRecord | None:
        return self._records.get(trace_id)

    def list(self) -> list[TraceSummary]:
        summaries: list[TraceSummary] = []
        for record in reversed(self._records.values()):
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
                )
            )
        return summaries

    def _load_persisted(self) -> None:
        assert self.persist_dir is not None
        files = sorted(self.persist_dir.glob("*.json"), key=lambda path: path.stat().st_mtime)
        for path in files[-self.max_records :]:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            trace = TraceRecord.model_validate(payload)
            self._records[trace.trace_id] = trace
        self._prune_persisted()

    def _prune_persisted(self) -> None:
        assert self.persist_dir is not None
        files = sorted(self.persist_dir.glob("*.json"), key=lambda path: path.stat().st_mtime)
        for path in files[:-self.max_records]:
            path.unlink(missing_ok=True)


class FeedbackStore:
    def __init__(self, max_records: int = 500, persist_dir: Path | None = None) -> None:
        self.max_records = max_records
        self.persist_dir = persist_dir
        self._records: OrderedDict[str, FeedbackRecord] = OrderedDict()
        if self.persist_dir is not None:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._load_persisted()

    def save(self, record: dict[str, object]) -> FeedbackRecord:
        feedback = FeedbackRecord.model_validate(record)
        removed_feedback_id: str | None = None
        if feedback.feedback_id in self._records:
            self._records.pop(feedback.feedback_id)
        self._records[feedback.feedback_id] = feedback
        while len(self._records) > self.max_records:
            removed_feedback_id, _ = self._records.popitem(last=False)
        if self.persist_dir is not None:
            (self.persist_dir / f"{feedback.feedback_id}.json").write_text(
                feedback.model_dump_json(indent=2),
                encoding="utf-8",
            )
            if removed_feedback_id:
                stale_file = self.persist_dir / f"{removed_feedback_id}.json"
                if stale_file.exists():
                    stale_file.unlink()
            self._prune_persisted()
        return feedback

    def list(self) -> list[FeedbackRecord]:
        return list(reversed(self._records.values()))

    def _load_persisted(self) -> None:
        assert self.persist_dir is not None
        files = sorted(self.persist_dir.glob("*.json"), key=lambda path: path.stat().st_mtime)
        for path in files[-self.max_records :]:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            feedback = FeedbackRecord.model_validate(payload)
            self._records[feedback.feedback_id] = feedback
        self._prune_persisted()

    def _prune_persisted(self) -> None:
        assert self.persist_dir is not None
        files = sorted(self.persist_dir.glob("*.json"), key=lambda path: path.stat().st_mtime)
        for path in files[:-self.max_records]:
            path.unlink(missing_ok=True)


class TracingManager:
    def __init__(self, service_name: str, collector_endpoint: str) -> None:
        self.service_name = service_name
        self.collector_endpoint = collector_endpoint
        if collector_endpoint:
            provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=collector_endpoint, insecure=True)))
            otel_trace.set_tracer_provider(provider)
        self.tracer = otel_trace.get_tracer(service_name)

    @contextmanager
    def span(self, name: str, attributes: dict[str, object] | None = None):
        with self.tracer.start_as_current_span(name) as span:
            for key, value in (attributes or {}).items():
                if value is None:
                    continue
                if isinstance(value, (str, bool, int, float)):
                    span.set_attribute(key, value)
                else:
                    span.set_attribute(key, str(value))
            yield span
