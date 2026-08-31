import json

from app.retrieval.graph_expander import GraphExpander
from app.retrieval.graph_index import GraphIndex
from app.schemas.structured import (
    DocumentNode,
    GraphEntity,
    GraphRelation,
    KnowledgeGraph,
    NodeProvenance,
    NodeType,
    StructuredDocument,
)


def _document(doc_id: str, source_path: str, text: str) -> StructuredDocument:
    root = DocumentNode(
        node_id=f"{doc_id}:root",
        doc_id=doc_id,
        kb_id="default",
        node_type=NodeType.ROOT,
        title=doc_id,
        provenance=NodeProvenance(source_document_id=doc_id),
    )
    section = DocumentNode(
        node_id=f"{doc_id}:node:1",
        doc_id=doc_id,
        kb_id="default",
        node_type=NodeType.PARAGRAPH,
        text=text,
        parent_id=root.node_id,
        provenance=NodeProvenance(
            source_document_id=doc_id,
            page_number=1,
            hierarchy_path=[doc_id],
        ),
    )
    root.children.append(section)
    entity = GraphEntity(
        entity_id="entity:shared",
        name="Shared concept",
        entity_type="concept",
        source_node_ids=[section.node_id],
    )
    relation = GraphRelation(
        relation_id=f"rel:{doc_id}",
        source_id=section.node_id,
        target_id=entity.entity_id,
        relation_type="ABOUT",
        source_node_id=section.node_id,
    )
    return StructuredDocument(
        doc_id=doc_id,
        kb_id="default",
        source_path=source_path,
        title=doc_id,
        root=root,
        graph=KnowledgeGraph(entities=[entity], relations=[relation]),
    )


def _write_artifact(tmp_path, document: StructuredDocument) -> None:
    (tmp_path / f"{document.doc_id}.json").write_text(
        json.dumps({"structured_document": document.model_dump(mode="json")}),
        encoding="utf-8",
    )


def test_graph_expander_crosses_documents_through_shared_entities(tmp_path) -> None:
    _write_artifact(tmp_path, _document("doc-a", "a.md", "Evidence from A."))
    _write_artifact(tmp_path, _document("doc-b", "b.md", "Evidence from B."))

    expanded = GraphExpander(tmp_path).expand(
        [{"node_id": "doc-a:node:1", "chunk_id": "doc-a:node:1"}],
        kb_id="default",
    )

    assert [context["node_id"] for context in expanded] == ["doc-b:node:1"]
    assert expanded[0]["source"] == "b.md"
    assert expanded[0]["graph_relation"] == "SHARES_ENTITY"


def test_graph_neighborhood_includes_cross_document_shared_entity(tmp_path) -> None:
    _write_artifact(tmp_path, _document("doc-a", "a.md", "Evidence from A."))
    _write_artifact(tmp_path, _document("doc-b", "b.md", "Evidence from B."))

    neighborhood = GraphIndex(tmp_path).neighborhood(
        "doc-a:node:1",
        kb_id="default",
    )

    assert neighborhood is not None
    _, neighbors = neighborhood
    targets = [item["target"] for item in neighbors]
    assert any(target.get("entity_id") == "entity:shared" for target in targets)
    assert any(target.get("node_id") == "doc-b:node:1" for target in targets)


def test_graph_expander_prefers_configured_graph_store(tmp_path) -> None:
    class FakeGraphStore:
        def expand_node_ids(self, node_ids, *, kb_id, max_neighbors=8):  # noqa: ANN001
            assert node_ids == {"doc-a:node:1"}
            assert kb_id == "default"
            return [("doc-b:node:1", "STORE_EDGE")]

    _write_artifact(tmp_path, _document("doc-a", "a.md", "Evidence from A."))
    _write_artifact(tmp_path, _document("doc-b", "b.md", "Evidence from B."))

    expanded = GraphExpander(tmp_path, FakeGraphStore()).expand(
        [{"node_id": "doc-a:node:1", "chunk_id": "doc-a:node:1"}],
        kb_id="default",
    )

    assert expanded[0]["node_id"] == "doc-b:node:1"
    assert expanded[0]["graph_relation"] == "STORE_EDGE"


def test_graph_index_load_is_cached_and_reuses_unchanged_documents(tmp_path) -> None:
    """The graph view must be served from cache when artifacts are unchanged:
    the previous implementation re-read and re-validated the whole corpus on
    every load() call — once per query in the expansion path."""
    import json as _json

    from app.tests.test_graph_index import _document

    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()
    doc = _document("doc-a", "a.md", "Evidence from A.")
    artifact = {
        "document": {"doc_id": "doc-a"},
        "chunks": [],
        "structured_document": doc.model_dump(),
    }
    (parsed_dir / "doc-a.json").write_text(
        _json.dumps(artifact, ensure_ascii=False), encoding="utf-8"
    )

    index = GraphIndex(parsed_dir)
    view1 = index.load("default")
    assert "doc-a:node:1" in view1.nodes

    load_calls: list[object] = []
    original = index._load_document

    def counting_load(path):
        load_calls.append(path.name)
        return original(path)

    index._load_document = counting_load  # type: ignore[method-assign]
    view2 = index.load("default")
    assert view2 is view1
    assert load_calls == []  # nothing re-read from disk

    # A new artifact invalidates the view; unchanged artifacts are not re-parsed.
    doc_b = _document("doc-b", "b.md", "Evidence from B.")
    (parsed_dir / "doc-b.json").write_text(
        _json.dumps(
            {
                "document": {"doc_id": "doc-b"},
                "chunks": [],
                "structured_document": doc_b.model_dump(),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    view3 = index.load("default")
    assert "doc-b:node:1" in view3.nodes
    # Only doc-b was parsed from disk; doc-a was served from the document cache.
    assert load_calls == ["doc-b.json"]
