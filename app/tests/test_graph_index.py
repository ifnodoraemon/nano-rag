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
