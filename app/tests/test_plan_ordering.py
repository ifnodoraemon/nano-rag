"""The read plan's judgement becomes the ranking, not just the read list.

Measured over 435 questions: the plan's first choice is the correct document 70.3% of
the time against BM25's 45.1% on rank 1. A fresh 60-question slice reproduced it
(66.7% vs 51.7%). The plan was already making that judgement; the change is to let it
reorder what everything downstream sees.
"""
from __future__ import annotations

from app.agentic.discovery import AgenticDiscovery
from app.schemas.chunk import Chunk
from app.wiki.search import SearchHit


def _hit(doc_id: str, score: float = 1.0) -> SearchHit:
    return SearchHit(
        chunk=Chunk(
            chunk_id=f"{doc_id}:0",
            doc_id=doc_id,
            chunk_index=0,
            text="text",
            source_path=f"{doc_id}.md",
            title=doc_id,
            metadata={"wiki_kind": "source"},
        ),
        score=score,
    )


def test_the_plans_first_choice_comes_first():
    candidates = [_hit("a"), _hit("b"), _hit("c")]
    plan = {"selected_docs": [{"doc_id": "c"}, {"doc_id": "a"}]}
    ordered = AgenticDiscovery._order_by_plan(plan, candidates)
    assert [hit.chunk.doc_id for hit in ordered] == ["c", "a", "b"]


def test_documents_the_plan_did_not_name_are_kept_in_lexical_order():
    """Reordering must never drop a candidate: it is not a filter."""
    candidates = [_hit("a", 3.0), _hit("b", 2.0), _hit("c", 1.0)]
    plan = {"selected_docs": [{"doc_id": "c"}]}
    ordered = AgenticDiscovery._order_by_plan(plan, candidates)
    ids = [hit.chunk.doc_id for hit in ordered]
    assert ids[0] == "c"
    assert sorted(ids[1:]) == ["a", "b"]
    assert len(ordered) == len(candidates)


def test_an_empty_plan_leaves_the_ranking_untouched():
    candidates = [_hit("a"), _hit("b")]
    assert AgenticDiscovery._order_by_plan({"selected_docs": []}, candidates) == candidates


def test_an_unknown_doc_id_in_the_plan_is_ignored():
    """A plan that names a document outside the candidate list must not invent one."""
    candidates = [_hit("a"), _hit("b")]
    plan = {"selected_docs": [{"doc_id": "zzz"}, {"doc_id": "b"}]}
    ordered = AgenticDiscovery._order_by_plan(plan, candidates)
    assert [hit.chunk.doc_id for hit in ordered] == ["b", "a"]


def test_a_repeated_doc_id_is_ordered_once():
    candidates = [_hit("a"), _hit("b")]
    plan = {"selected_docs": [{"doc_id": "b"}, {"doc_id": "b"}]}
    ordered = AgenticDiscovery._order_by_plan(plan, candidates)
    assert [hit.chunk.doc_id for hit in ordered] == ["b", "a"]


def test_an_empty_candidate_list_stays_empty():
    assert AgenticDiscovery._order_by_plan({"selected_docs": [{"doc_id": "a"}]}, []) == []
