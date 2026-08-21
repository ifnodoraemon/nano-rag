from __future__ import annotations

import json
import os
from typing import Protocol

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.schemas.structured import ENTITY_ID_PREFIX, StructuredDocument

_DEFAULT_PG_URI = "postgresql://nanorag:nano-rag@postgres:5432/nanorag"


class GraphStore(Protocol):
    def upsert_document(self, document: StructuredDocument) -> None: ...

    def delete_document(self, *, doc_id: str, kb_id: str) -> None: ...

    def expand_node_ids(
        self,
        node_ids: set[str],
        *,
        kb_id: str,
        max_neighbors: int = 8,
    ) -> list[tuple[str, str]]: ...

    def stats(self) -> dict[str, object]: ...

    def close(self) -> None: ...


_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS graph_document (
        kb_id text NOT NULL,
        doc_id text NOT NULL,
        title text NOT NULL DEFAULT '',
        source_path text NOT NULL DEFAULT '',
        PRIMARY KEY (kb_id, doc_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS graph_node (
        node_id text PRIMARY KEY,
        kb_id text NOT NULL,
        doc_id text NOT NULL,
        node_type text NOT NULL DEFAULT '',
        title text NOT NULL DEFAULT '',
        text_preview text NOT NULL DEFAULT '',
        page_number int,
        hierarchy_path text NOT NULL DEFAULT '[]'
    )
    """,
    "CREATE INDEX IF NOT EXISTS graph_node_doc_idx ON graph_node (doc_id)",
    """
    CREATE TABLE IF NOT EXISTS graph_entity (
        entity_id text PRIMARY KEY,
        kb_id text NOT NULL,
        name text NOT NULL DEFAULT '',
        entity_type text NOT NULL DEFAULT ''
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS graph_node_entity (
        node_id text NOT NULL REFERENCES graph_node (node_id) ON DELETE CASCADE,
        entity_id text NOT NULL REFERENCES graph_entity (entity_id) ON DELETE CASCADE,
        PRIMARY KEY (node_id, entity_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS graph_node_entity_entity_idx ON graph_node_entity (entity_id)",
    """
    CREATE TABLE IF NOT EXISTS graph_relation (
        relation_id text PRIMARY KEY,
        source_id text NOT NULL,
        target_id text NOT NULL,
        kb_id text NOT NULL,
        doc_id text NOT NULL DEFAULT '',
        relation_type text NOT NULL DEFAULT '',
        confidence double precision
    )
    """,
    # Idempotent upgrade for volumes created before graph_relation carried a
    # doc_id: scopes relation deletion to the owning document (fresh installs
    # get the column from the CREATE above, so this is a no-op there).
    "ALTER TABLE graph_relation ADD COLUMN IF NOT EXISTS doc_id text NOT NULL DEFAULT ''",
    "CREATE INDEX IF NOT EXISTS graph_relation_source_idx ON graph_relation (source_id)",
    "CREATE INDEX IF NOT EXISTS graph_relation_target_idx ON graph_relation (target_id)",
    "CREATE INDEX IF NOT EXISTS graph_relation_doc_idx ON graph_relation (doc_id, kb_id)",
)

# Seed expansion: which other nodes share an entity with the seeds, plus the
# relation edges directly touching a seed (in or out). Entity-prefixed targets
# are resolved to concrete nodes by a follow-up backfill in Python.
_EXPAND_QUERY = """
WITH seeds AS (SELECT UNNEST(%(node_ids)s) AS node_id)
SELECT n.node_id AS target_id, 'SHARES_ENTITY' AS relation_type
FROM graph_node n
JOIN graph_node_entity ne ON ne.node_id = n.node_id
JOIN graph_node_entity se ON se.entity_id = ne.entity_id
JOIN seeds s ON se.node_id = s.node_id
WHERE n.kb_id = %(kb_id)s AND n.node_id <> s.node_id
UNION
SELECT r.target_id, r.relation_type
FROM graph_relation r
JOIN seeds s ON r.source_id = s.node_id
WHERE r.kb_id = %(kb_id)s
UNION
SELECT r.source_id, r.relation_type
FROM graph_relation r
JOIN seeds s ON r.target_id = s.node_id
WHERE r.kb_id = %(kb_id)s
"""

_BACKFILL_QUERY = """
SELECT ne.entity_id, ne.node_id
FROM graph_node_entity ne
JOIN graph_node n ON n.node_id = ne.node_id
WHERE ne.entity_id = ANY(%(entity_ids)s)
  AND n.kb_id = %(kb_id)s
  AND ne.node_id <> ALL(%(seed_node_ids)s)
"""

# Reclaim entity rows no node links to any more. Shared by upsert (run after
# the fresh node_entity links are inserted) and delete. Deliberately global,
# not kb-scoped: entity ids are sha1(kb_id:name) (see GraphExtractor), so an
# entity can only be referenced by node links in its own KB — "unreferenced
# anywhere" is exactly "unreferenced in this KB plus nothing else", and a
# kb-scoped subquery would wrongly delete entities still owned by other KBs.
# The KB-scoping this relies on is pinned by app/tests/test_graph_id_scoping.py
# (test_entity_id_is_scoped_by_kb_and_casefolded) — that test fails if a
# refactor ever makes entity ids non-KB-scoped, which is exactly when this
# global GC would become unsafe.
_GC_ORPHAN_ENTITIES_QUERY = """
DELETE FROM graph_entity
WHERE entity_id NOT IN (
    SELECT DISTINCT entity_id FROM graph_node_entity
)
"""


class PostgresGraphStore:
    """Native-SQL graph store backing document-structure expansion.

    Node/entity/relation rows are materialized from the LLM-extracted
    structured graph at ingest. Expansion answers "which other nodes are
    related to these seeds" via shares-entity joins and RELATED edges.

    The per-statement SQL and the Python-side expansion logic are extracted
    into static helpers so they can be exercised without a live database.
    """

    def __init__(self, uri: str) -> None:
        self.uri = uri
        self._pool = ConnectionPool(
            conninfo=uri,
            min_size=1,
            max_size=4,
            kwargs={"row_factory": dict_row},
            open=True,
        )
        with self._pool.connection() as conn:
            conn.execute("SELECT 1")
            for statement in _SCHEMA_STATEMENTS:
                conn.execute(statement)

    @classmethod
    def from_env(cls) -> "PostgresGraphStore":
        return cls(os.getenv("PG_URI", _DEFAULT_PG_URI))

    def upsert_document(self, document: StructuredDocument) -> None:
        statements = self._upsert_statements(document)
        with self._pool.connection() as conn:
            with conn.transaction():
                for sql, params in statements:
                    conn.execute(sql, params)

    def delete_document(self, *, doc_id: str, kb_id: str) -> None:
        statements = self._delete_statements(doc_id, kb_id)
        with self._pool.connection() as conn:
            with conn.transaction():
                for sql, params in statements:
                    conn.execute(sql, params)

    def expand_node_ids(
        self,
        node_ids: set[str],
        *,
        kb_id: str,
        max_neighbors: int = 8,
    ) -> list[tuple[str, str]]:
        if not node_ids:
            return []
        with self._pool.connection() as conn:
            rows = conn.execute(
                _EXPAND_QUERY,
                {"node_ids": sorted(node_ids), "kb_id": kb_id},
            ).fetchall()
        entity_neighbors, neighbors = self._partition_neighbors(rows)
        if not entity_neighbors:
            return self._dedupe_neighbors(neighbors, limit=max_neighbors)
        backfill = self._backfill_entity_neighbors(
            entity_neighbors, kb_id=kb_id, seed_node_ids=node_ids
        )
        return self._dedupe_neighbors(neighbors + backfill, limit=max_neighbors)

    def stats(self) -> dict[str, object]:
        with self._pool.connection() as conn:
            row = conn.execute(
                """
                SELECT
                    (SELECT count(*) FROM graph_document) AS documents,
                    (SELECT count(*) FROM graph_node) AS nodes,
                    (SELECT count(*) FROM graph_entity) AS entities,
                    (SELECT count(*) FROM graph_relation) AS relations
                """
            ).fetchone()
        row = row or {}
        return {
            "backend": "postgresql",
            "uri": self.uri.split("@")[-1],
            "documents": int(row.get("documents", 0)),
            "nodes": int(row.get("nodes", 0)),
            "entities": int(row.get("entities", 0)),
            "relations": int(row.get("relations", 0)),
        }

    def close(self) -> None:
        self._pool.close()

    @staticmethod
    def _node_text(node: "object") -> str:
        table = getattr(node, "table", None)
        narrative = getattr(table, "narrative", None) if table else None
        return narrative if narrative else str(getattr(node, "text", "") or "")

    @classmethod
    def _upsert_statements(
        cls, document: StructuredDocument
    ) -> list[tuple[str, tuple[object, ...]]]:
        statements: list[tuple[str, tuple[object, ...]]] = [
            (
                "DELETE FROM graph_node WHERE doc_id = %s AND kb_id = %s",
                (document.doc_id, document.kb_id),
            ),
            # Re-ingest re-runs LLM extraction, so relation ids from the
            # previous pass no longer match the fresh ones; drop this doc's
            # stale relations before re-materializing (delete_document does
            # the same). Without this, each re-ingest leaves orphaned edges
            # that expand_node_ids can still return.
            (
                "DELETE FROM graph_relation WHERE doc_id = %s AND kb_id = %s",
                (document.doc_id, document.kb_id),
            ),
            (
                """
                INSERT INTO graph_document (kb_id, doc_id, title, source_path)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (kb_id, doc_id)
                DO UPDATE SET title = EXCLUDED.title,
                              source_path = EXCLUDED.source_path
                """,
                (
                    document.kb_id,
                    document.doc_id,
                    document.title,
                    document.source_path,
                ),
            ),
        ]
        # Node ids are `{doc_id}:node:{sha256}` and doc ids are
        # sha256(source_path, kb_id), so a node_id is globally unique per
        # document+kb. The ON CONFLICT (node_id) DO UPDATE below therefore
        # can only ever reassign a row back to its own document — it is not a
        # cross-KB collision hazard despite node_id being a global PK. The
        # KB-scoping of doc_id and node_id this relies on is pinned by
        # app/tests/test_graph_id_scoping.py (test_node_id_embeds_doc_id,
        # test_cross_kb_ids_do_not_collide) — a refactor that dropped KB scope
        # from either id would fail that test before this PK could be abused.
        for node in document.iter_nodes():
            statements.append(
                (
                    """
                    INSERT INTO graph_node
                        (node_id, kb_id, doc_id, node_type, title,
                         text_preview, page_number, hierarchy_path)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (node_id)
                    DO UPDATE SET kb_id = EXCLUDED.kb_id,
                                  doc_id = EXCLUDED.doc_id,
                                  node_type = EXCLUDED.node_type,
                                  title = EXCLUDED.title,
                                  text_preview = EXCLUDED.text_preview,
                                  page_number = EXCLUDED.page_number,
                                  hierarchy_path = EXCLUDED.hierarchy_path
                    """,
                    (
                        node.node_id,
                        document.kb_id,
                        document.doc_id,
                        node.node_type.value,
                        node.title or "",
                        cls._node_text(node)[:500],
                        node.provenance.page_number,
                        json.dumps(node.provenance.hierarchy_path, ensure_ascii=False),
                    ),
                )
            )
        for entity in document.graph.entities:
            statements.append(
                (
                    """
                    INSERT INTO graph_entity (entity_id, kb_id, name, entity_type)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (entity_id)
                    DO UPDATE SET kb_id = EXCLUDED.kb_id,
                                  name = EXCLUDED.name,
                                  entity_type = EXCLUDED.entity_type
                    """,
                    (
                        entity.entity_id,
                        document.kb_id,
                        entity.name,
                        entity.entity_type,
                    ),
                )
            )
            for node_id in entity.source_node_ids:
                statements.append(
                    (
                        "INSERT INTO graph_node_entity (node_id, entity_id) "
                        "VALUES (%s, %s) ON CONFLICT DO NOTHING",
                        (node_id, entity.entity_id),
                    )
                )
        for relation in document.graph.relations:
            statements.append(
                (
                    """
                    INSERT INTO graph_relation
                        (relation_id, source_id, target_id, kb_id, doc_id,
                         relation_type, confidence)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (relation_id)
                    DO UPDATE SET kb_id = EXCLUDED.kb_id,
                                  doc_id = EXCLUDED.doc_id,
                                  relation_type = EXCLUDED.relation_type,
                                  confidence = EXCLUDED.confidence
                    """,
                    (
                        relation.relation_id,
                        relation.source_id,
                        relation.target_id,
                        document.kb_id,
                        document.doc_id,
                        relation.relation_type,
                        relation.confidence,
                    ),
                )
            )
        # Last: reclaim entities no node links to any more. The doc's node
        # rows were deleted up front (cascading their node_entity links) and
        # its fresh links were just inserted, so by here any orphan is either
        # an entity the previous extraction minted and this one dropped, or
        # one whose only referencing node belongs to a fully-deleted doc.
        statements.append((_GC_ORPHAN_ENTITIES_QUERY, ()))
        return statements

    @staticmethod
    def _delete_statements(doc_id: str, kb_id: str) -> list[tuple[str, tuple[object, ...]]]:
        # Relation rows carry no FK to graph_node, so deleting the doc's nodes
        # would leave dangling relation rows behind (queryable by
        # expand_node_ids and unreclaimable). Relations are minted per
        # document at extraction, so (doc_id, kb_id) is the owning scope.
        # Entity GC (last) is the shared global query — see its definition.
        return [
            (
                "DELETE FROM graph_node WHERE doc_id = %s AND kb_id = %s",
                (doc_id, kb_id),
            ),
            (
                "DELETE FROM graph_relation WHERE doc_id = %s AND kb_id = %s",
                (doc_id, kb_id),
            ),
            (
                "DELETE FROM graph_document WHERE doc_id = %s AND kb_id = %s",
                (doc_id, kb_id),
            ),
            (_GC_ORPHAN_ENTITIES_QUERY, ()),
        ]

    @staticmethod
    def _partition_neighbors(
        rows: list[dict[str, object]],
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        entity_neighbors: list[tuple[str, str]] = []
        neighbors: list[tuple[str, str]] = []
        for row in rows:
            target_id = row.get("target_id")
            relation = row.get("relation_type")
            if not target_id or not relation:
                continue
            target = str(target_id)
            if target.startswith(ENTITY_ID_PREFIX):
                entity_neighbors.append((target, str(relation)))
            else:
                neighbors.append((target, str(relation)))
        return entity_neighbors, neighbors

    def _backfill_entity_neighbors(
        self,
        entity_neighbors: list[tuple[str, str]],
        *,
        kb_id: str,
        seed_node_ids: set[str],
    ) -> list[tuple[str, str]]:
        entity_ids = [entity_id for entity_id, _ in entity_neighbors]
        relation_by_entity = {
            entity_id: relation for entity_id, relation in entity_neighbors
        }
        with self._pool.connection() as conn:
            backfill = conn.execute(
                _BACKFILL_QUERY,
                {
                    "entity_ids": entity_ids,
                    "kb_id": kb_id,
                    "seed_node_ids": sorted(seed_node_ids),
                },
            ).fetchall()
        return [
            (
                str(row.get("node_id")),
                relation_by_entity.get(str(row.get("entity_id")), "RELATED"),
            )
            for row in backfill
            if row.get("node_id")
        ]

    @staticmethod
    def _dedupe_neighbors(
        neighbors: list[tuple[str, str]],
        *,
        limit: int,
    ) -> list[tuple[str, str]]:
        deduped: list[tuple[str, str]] = []
        seen: set[str] = set()
        for node_id, relation in neighbors:
            if node_id in seen:
                continue
            seen.add(node_id)
            deduped.append((node_id, relation))
            if len(deduped) >= limit:
                break
        return deduped
