"""Lock the KB-scoping invariant that two review false-positives rested on.

Both reviewers flagged (1) a cross-KB graph_node collision via the global
node_id primary key, and (2) the global orphan-entity GC deleting entities
owned by other KBs. Both were assessed false — but the assessment only held
because every graph id embeds its KB scope by construction:

    doc_id    = sha256(kb_id | source_path)      (IngestionPipeline)
    node_id   = f"{doc_id}:node:{sha256(doc_id:source_ref)}"   (StructuredDocumentParser)
    entity_id = f"entity:{sha1(kb_id:name)}"       (GraphExtractor)

That invariant was asserted in comments only. These tests pin it to the real
production derivations so a future refactor that drops KB scope from any layer
fails the suite instead of silently re-introducing the collision (node PK
assigned to the wrong document) or the cross-KB entity GC deletion.
"""

from app.ingestion.graph_extractor import GraphExtractor
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.structured_parser import StructuredDocumentParser


def _doc_id(source_path: str, kb_id: str) -> str:
    # Pure method: reads only its arguments, so call unbound (self unused).
    return IngestionPipeline._stable_doc_id(None, source_path, kb_id)  # noqa: SLF001


def _node_id(doc_id: str, source_ref: str) -> str:
    return StructuredDocumentParser._stable_node_id(None, doc_id, source_ref)  # noqa: SLF001


def _entity_id(kb_id: str, name: str) -> str:
    return GraphExtractor._entity_id(None, kb_id, name)  # noqa: SLF001


def test_doc_id_is_scoped_by_kb_and_deterministic() -> None:
    # Same logical document ingested under two KBs must NOT map to the same
    # row — that is what the cross-KB node_id PK "collision" worried about.
    assert _doc_id("manual.pdf", "kb1") != _doc_id("manual.pdf", "kb2")
    # Deterministic: re-ingest of the same (kb, path) is an upsert, not a new row.
    assert _doc_id("manual.pdf", "kb1") == _doc_id("manual.pdf", "kb1")


def test_node_id_embeds_doc_id() -> None:
    # A node's identity is derived from its owning doc_id, which already
    # carries kb scope — so node rows can never be reassignable across KBs.
    for kb in ("kb1", "kb2"):
        doc = _doc_id("manual.pdf", kb)
        node = _node_id(doc, "section-1")
        assert node.startswith(f"{doc}:node:")
    # Distinct docs (even with the same section ref) yield distinct nodes.
    assert _node_id(_doc_id("manual.pdf", "kb1"), "s") != _node_id(
        _doc_id("manual.pdf", "kb2"), "s"
    )


def test_entity_id_is_scoped_by_kb_and_casefolded() -> None:
    # Same entity name in two KBs mints two ids (no cross-KB sharing/GC target).
    assert _entity_id("kb1", "Reimbursement") != _entity_id("kb2", "Reimbursement")
    # Deterministic and case-insensitive within a KB.
    assert _entity_id("kb1", "Reimbursement") == _entity_id("kb1", "reimbursement")


def test_cross_kb_ids_do_not_collide() -> None:
    # The composite property that defeats both review claims: build a full
    # document's worth of ids under each of two KBs and assert the two id
    # sets are disjoint — nothing a KB-1 document writes can collide with or
    # orphan a KB-2 row.
    kb1 = {_doc_id("manual.pdf", "kb1")}
    kb1 |= {_node_id(_doc_id("manual.pdf", "kb1"), ref) for ref in ("s1", "s2")}
    kb1 |= {_entity_id("kb1", name) for name in ("Reimbursement", "Travel")}

    kb2 = {_doc_id("manual.pdf", "kb2")}
    kb2 |= {_node_id(_doc_id("manual.pdf", "kb2"), ref) for ref in ("s1", "s2")}
    kb2 |= {_entity_id("kb2", name) for name in ("Reimbursement", "Travel")}

    assert kb1 and kb2
    assert kb1.isdisjoint(kb2)
