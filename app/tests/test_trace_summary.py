from app.core.tracing import TraceStore


def test_trace_summary_contains_context_count() -> None:
    store = TraceStore()
    store.save_raw(
        {
            "trace_id": "t1",
            "query": "q1",
            "contexts": [
                {"chunk_id": "c1"},
                {"chunk_id": "c2", "wiki_status": "conflicting"},
            ],
            "supporting_claims": [
                {"claim_type": "conflict", "text": "a", "citation_labels": ["C1"]},
                {"claim_type": "insufficiency", "text": "b", "citation_labels": ["C2"]},
                {"claim_type": "conditional", "text": "c", "citation_labels": ["C3"]},
            ],
        }
    )

    result = store.list()
    summaries = result.items

    assert summaries[0].context_count == 2
    assert summaries[0].conflicting_context_count == 1
    assert summaries[0].conflict_claim_count == 1
    assert summaries[0].insufficiency_claim_count == 1
    assert summaries[0].conditional_claim_count == 1
