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


def test_upsert_batches_write_documents_nodes_entities_and_relations() -> None:
    from app.tests.test_graph_index import _document

    batches = PostgresGraphStore._upsert_batches(  # noqa: SLF001
        _document("doc-a", "a.md", "Evidence from A.")
    )
    sql_shapes = [sql.lstrip() for sql, _rows in batches]
    # One batched executemany per SQL shape — a single round trip per shape
    # instead of one round trip per row.
    assert any(sql.startswith("INSERT INTO graph_document") for sql in sql_shapes)
    assert any(sql.startswith("INSERT INTO graph_node") for sql in sql_shapes)
    assert any(sql.startswith("INSERT INTO graph_entity") for sql in sql_shapes)
    assert any(sql.startswith("INSERT INTO graph_node_entity") for sql in sql_shapes)
    assert any(sql.startswith("INSERT INTO graph_relation") for sql in sql_shapes)

    node_rows = next(
        rows for sql, rows in batches if sql.lstrip().startswith("INSERT INTO graph_node")
    )
    assert any(row[0] == "doc-a:node:1" for row in node_rows)
    entity_rows = next(
        rows for sql, rows in batches if sql.lstrip().startswith("INSERT INTO graph_entity")
    )
    assert any(row[0] == "entity:shared" for row in entity_rows)
    relation_rows = next(
        rows for sql, rows in batches if sql.lstrip().startswith("INSERT INTO graph_relation")
    )
    relation_row = next(row for row in relation_rows if row[0] == "rel:doc-a")
    # (relation_id, source_id, target_id, kb_id, doc_id, relation_type, confidence)
    assert relation_row[4] == "doc-a"


def test_upsert_batches_use_table_narrative_and_truncate_preview() -> None:
    batches = PostgresGraphStore._upsert_batches(_table_document())  # noqa: SLF001
    node_rows = next(
        rows for sql, rows in batches if sql.lstrip().startswith("INSERT INTO graph_node")
    )
    node_row = next(row for row in node_rows if row[0] == "doc-t:node:tbl")
    # (node_id, kb_id, doc_id, node_type, title, text_preview, page, hierarchy)
    assert node_row[5] == "The narrative preview."
    assert json.loads(node_row[7]) == ["doc-t", "Matrix"]
    # The root node has no narrative; falls back to empty text.
    root_row = next(row for row in node_rows if row[0] == "doc-t:root")
    assert root_row[5] == ""


def test_upsert_batches_coerce_null_title_to_empty() -> None:
    # Regression: paragraph nodes carry title=None; the graph_node.title column
    # is NOT NULL, so an explicit NULL must be coerced to "" or the upsert fails.
    from app.tests.test_graph_index import _document

    batches = PostgresGraphStore._upsert_batches(  # noqa: SLF001
        _document("doc-a", "a.md", "Evidence from A.")
    )
    node_rows = next(
        rows for sql, rows in batches if sql.lstrip().startswith("INSERT INTO graph_node")
    )
    node_row = next(row for row in node_rows if row[0] == "doc-a:node:1")
    # (node_id, kb_id, doc_id, node_type, title, text_preview, page, hierarchy)
    assert node_row[4] == ""  # title was None in the fixture
    assert node_row[5] == "Evidence from A."


def test_orphan_entity_gc_is_not_part_of_per_document_upsert() -> None:
    # The global GC full-scans graph_node_entity and serializes concurrent
    # ingest transactions; it must be a separate, once-per-job maintenance
    # call (collect_orphan_entities), never embedded in the per-document
    # upsert/delete path.
    from app.retrieval import graph_store as graph_store_module

    batches = PostgresGraphStore._upsert_batches(_table_document())  # noqa: SLF001
    for sql, rows in batches:
        assert "DELETE FROM graph_entity" not in sql
        assert "graph_node_entity" not in sql or sql.lstrip().startswith("INSERT")
    assert "DELETE FROM graph_entity" in graph_store_module._GC_ORPHAN_ENTITIES_QUERY


def test_expand_queries_are_bounded() -> None:
    from app.retrieval import graph_store as graph_store_module

    assert "LIMIT %(expand_limit)s" in graph_store_module._EXPAND_QUERY
    assert "LIMIT %(expand_limit)s" in graph_store_module._BACKFILL_QUERY


def test_stats_does_not_leak_connection_details() -> None:
    # The store must be constructible-by-contract and its stats must not leak
    # host/credentials; pool bounds are reported instead of the URI.
    import inspect

    signature = inspect.signature(PostgresGraphStore.__init__)
    assert "pool_min" in signature.parameters
    assert "pool_max" in signature.parameters
    source = inspect.getsource(PostgresGraphStore.stats)
    assert "self.uri" not in source


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
