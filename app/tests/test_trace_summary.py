from app.core.tracing import TraceStore


def test_trace_summary_contains_context_count() -> None:
    store = TraceStore()
    store.save({"trace_id": "t1", "query": "q1", "contexts": [{"chunk_id": "c1"}]})

    summaries = store.list()

    assert summaries[0].context_count == 1
