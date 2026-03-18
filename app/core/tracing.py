from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from time import time
from uuid import uuid4

from opentelemetry import trace as otel_trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

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
    def __init__(self, max_records: int = 200) -> None:
        self.max_records = max_records
        self._records: OrderedDict[str, TraceRecord] = OrderedDict()

    def save(self, record: dict[str, object]) -> TraceRecord:
        trace = TraceRecord.model_validate(record)
        if trace.trace_id in self._records:
            self._records.pop(trace.trace_id)
        self._records[trace.trace_id] = trace
        while len(self._records) > self.max_records:
            self._records.popitem(last=False)
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
                    model_alias=record.model_alias,
                    prompt_version=record.prompt_version,
                    context_count=len(record.contexts),
                )
            )
        return summaries


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
