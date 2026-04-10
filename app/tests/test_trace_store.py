import json

from app.core.tracing import TraceStore


def test_trace_store_saves_and_lists_latest_first() -> None:
    store = TraceStore()
    store.save_raw({"trace_id": "t1", "query": "q1", "latency_seconds": 0.1})
    store.save_raw({"trace_id": "t2", "query": "q2", "latency_seconds": 0.2})

    result = store.list()
    summaries = result.items

    assert summaries[0].trace_id == "t2"
    assert store.get("t1") is not None


def test_trace_store_loads_persisted_records(tmp_path) -> None:
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    (trace_dir / "trace-1.json").write_text(
        json.dumps({"trace_id": "trace-1", "query": "persisted"}),
        encoding="utf-8",
    )

    store = TraceStore(persist_dir=trace_dir)

    assert store.get("trace-1") is not None
    result = store.list()
    assert result.items[0].trace_id == "trace-1"


def test_trace_store_prunes_old_persisted_files(tmp_path) -> None:
    trace_dir = tmp_path / "traces"
    store = TraceStore(max_records=2, persist_dir=trace_dir)

    store.save_raw({"trace_id": "t1"})
    store.save_raw({"trace_id": "t2"})
    store.save_raw({"trace_id": "t3"})

    persisted = sorted(path.name for path in trace_dir.glob("*.json"))
    assert persisted == ["t2.json", "t3.json"]
