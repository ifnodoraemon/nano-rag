import json

from app.retrieval.graph_store import PostgresGraphStore
from app.schemas.structured import (
    DocumentNode,
    GraphEntity,
    GraphRelation,
    KnowledgeGraph,
    NodeProvenance,
    NodeType,
    StructuredDocument,
    TablePayload,
)


def _table_document() -> StructuredDocument:
    """Document with a table node so the narrative-preview branch is covered."""
    root = DocumentNode(
        node_id="doc-t:root",
        doc_id="doc-t",
        kb_id="default",
        node_type=NodeType.ROOT,
        title="doc-t",
        provenance=NodeProvenance(source_document_id="doc-t"),
    )
    table = DocumentNode(
        node_id="doc-t:node:tbl",
        doc_id="doc-t",
        kb_id="default",
        node_type=NodeType.TABLE,
        title="Matrix",
        parent_id=root.node_id,
        provenance=NodeProvenance(
            source_document_id="doc-t",
            page_number=2,
            hierarchy_path=["doc-t", "Matrix"],
        ),
        table=TablePayload(rows=2, cols=2, narrative="The narrative preview."),
    )
    root.children.append(table)
    return StructuredDocument(
        doc_id="doc-t",
        kb_id="default",
        source_path="t.md",
        title="doc-t",
        root=root,
        graph=KnowledgeGraph(),
    )


def test_upsert_statements_write_nodes_entities_and_relations() -> None:
    from app.tests.test_graph_index import _document

    statements = PostgresGraphStore._upsert_statements(  # noqa: SLF001
        _document("doc-a", "a.md", "Evidence from A.")
    )
    params = [call[1] for call in statements]
    # First statement rewrites the document's node rows from scratch.
    assert statements[0][0].startswith("DELETE FROM graph_node")
    # Stale relations from the previous (non-deterministic) extraction are
    # dropped by owning (doc_id, kb_id) before the fresh ones are written.
    assert statements[1][0].startswith("DELETE FROM graph_relation")
    assert statements[1][1] == ("doc-a", "default")
    assert params[2][1] == "doc-a"  # graph_document upsert doc_id
    assert any(item[0] == "doc-a:node:1" for item in params)  # node insert
    assert any(item[0] == "entity:shared" for item in params)  # entity insert
    # The last statement reclaims orphan entities (links from this doc were
    # just rewritten, so any orphan is stale from a prior extraction).
    assert "DELETE FROM graph_entity" in statements[-1][0]
    relation_row = next(
        item for item in params if item[0] == "rel:doc-a"
    )
    # (relation_id, source_id, target_id, kb_id, doc_id, relation_type, confidence)
    assert relation_row[4] == "doc-a"


def test_upsert_statements_use_table_narrative_and_truncate_preview() -> None:
    statements = PostgresGraphStore._upsert_statements(_table_document())
    node_row = next(
        params
        for sql, params in statements
        if sql.lstrip().startswith("INSERT INTO graph_node") and params[0] == "doc-t:node:tbl"
    )
    # (node_id, kb_id, doc_id, node_type, title, text_preview, page, hierarchy)
    assert node_row[5] == "The narrative preview."
    assert json.loads(node_row[7]) == ["doc-t", "Matrix"]
    # The root node has no narrative; falls back to empty text.
    root_row = next(
        params
        for sql, params in statements
        if sql.lstrip().startswith("INSERT INTO graph_node") and params[0] == "doc-t:root"
    )
    assert root_row[5] == ""


def test_delete_statements_drop_nodes_relations_document_and_orphan_entities() -> None:
    statements = PostgresGraphStore._delete_statements("doc-a", "default")  # noqa: SLF001
    assert statements[0][0].startswith("DELETE FROM graph_node")
    assert statements[0][1] == ("doc-a", "default")
    # Relation rows carry no FK to graph_node; they must be deleted by owning
    # (doc_id, kb_id) or they leak as dangling edges (HIGH-1 regression).
    assert statements[1][0].startswith("DELETE FROM graph_relation")
    assert statements[1][1] == ("doc-a", "default")
    assert statements[2][0].startswith("DELETE FROM graph_document")
    assert statements[3][0].lstrip().startswith("DELETE FROM graph_entity")
    # Entity GC is deliberately global (entity ids are KB-namespaced), so it
    # must NOT be kb-scoped here.
    assert "kb_id = %s" not in statements[3][0]
    assert "graph_node_entity" in statements[3][0]


def test_upsert_statements_coerce_null_title_to_empty() -> None:
    # Regression: paragraph nodes carry title=None; the graph_node.title column
    # is NOT NULL, so an explicit NULL must be coerced to "" or the upsert fails.
    from app.tests.test_graph_index import _document

    statements = PostgresGraphStore._upsert_statements(  # noqa: SLF001
        _document("doc-a", "a.md", "Evidence from A.")
    )
    node_row = next(
        params
        for sql, params in statements
        if sql.lstrip().startswith("INSERT INTO graph_node") and params[0] == "doc-a:node:1"
    )
    # (node_id, kb_id, doc_id, node_type, title, text_preview, page, hierarchy)
    assert node_row[4] == ""  # title was None in the fixture
    assert node_row[5] == "Evidence from A."


def test_partition_neighbors_splits_entity_ids_from_nodes() -> None:
    rows = [
        {"target_id": "entity:abc", "relation_type": "ABOUT"},
        {"target_id": "doc-b:node:1", "relation_type": "SHARES_ENTITY"},
        {"target_id": None, "relation_type": "ABOUT"},  # dropped
        {"target_id": "doc-c:node:1", "relation_type": ""},  # dropped
    ]
    entity_neighbors, neighbors = PostgresGraphStore._partition_neighbors(rows)  # noqa: SLF001
    assert entity_neighbors == [("entity:abc", "ABOUT")]
    assert neighbors == [("doc-b:node:1", "SHARES_ENTITY")]


def test_dedupe_neighbors_keeps_first_relation_and_caps_at_limit() -> None:
    neighbors = PostgresGraphStore._dedupe_neighbors(  # noqa: SLF001
        [
            ("doc-b:node:1", "SHARES_ENTITY"),
            ("doc-b:node:1", "ABOUT"),
            ("doc-c:node:1", "ABOUT"),
            ("doc-d:node:1", "ABOUT"),
        ],
        limit=2,
    )
    assert neighbors == [
        ("doc-b:node:1", "SHARES_ENTITY"),
        ("doc-c:node:1", "ABOUT"),
    ]
