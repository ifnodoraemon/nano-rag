from app.core.tracing import TraceStore


def test_trace_store_saves_and_lists_latest_first() -> None:
    store = TraceStore()
    store.save({"trace_id": "t1", "query": "q1", "latency_seconds": 0.1})
    store.save({"trace_id": "t2", "query": "q2", "latency_seconds": 0.2})

    summaries = store.list()

    assert summaries[0].trace_id == "t2"
    assert store.get("t1") is not None
