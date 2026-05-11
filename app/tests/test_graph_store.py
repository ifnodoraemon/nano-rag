from app.retrieval.graph_store import Neo4jGraphStore
from app.tests.test_graph_index import _document


class FakeTx:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def run(self, statement: str, **params):  # noqa: ANN001, ANN003
        self.calls.append((statement, params))


def test_neo4j_graph_store_writes_document_nodes_entities_and_relations() -> None:
    tx = FakeTx()
    document = _document("doc-a", "a.md", "Evidence from A.")

    Neo4jGraphStore._upsert_document_tx(tx, document)  # noqa: SLF001

    params = [call[1] for call in tx.calls]
    assert any(item.get("doc_id") == "doc-a" for item in params)
    assert any(item.get("node_id") == "doc-a:node:1" for item in params)
    assert any(item.get("entity_id") == "entity:shared" for item in params)
    assert any(item.get("relation_id") == "rel:doc-a" for item in params)


def test_neo4j_graph_store_delete_query_handles_documents_without_nodes() -> None:
    tx = FakeTx()

    Neo4jGraphStore._delete_document_tx(tx, "doc-a", "default")  # noqa: SLF001

    statement, params = tx.calls[0]
    assert "FOREACH" in statement
    assert params == {"doc_id": "doc-a", "kb_id": "default"}


def test_neo4j_graph_store_neighbor_dedupe_keeps_doc_node_ids_only() -> None:
    store = object.__new__(Neo4jGraphStore)

    neighbors = store._dedupe_neighbors(  # noqa: SLF001
        [
            ("doc-b:node:1", "SHARES_ENTITY"),
            ("doc-b:node:1", "ABOUT"),
            ("doc-c:node:1", "ABOUT"),
        ],
        limit=5,
    )

    assert neighbors == [
        ("doc-b:node:1", "SHARES_ENTITY"),
        ("doc-c:node:1", "ABOUT"),
    ]
