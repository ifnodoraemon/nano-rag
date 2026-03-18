from __future__ import annotations

from collections import OrderedDict
from time import time
from uuid import uuid4

from app.schemas.trace import TraceRecord, TraceSummary


class TraceSession:
    def __init__(self) -> None:
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
                )
            )
        return summaries
