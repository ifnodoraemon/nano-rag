from __future__ import annotations

import json
import os
from typing import Protocol

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.core.exceptions import ConfigurationError
from app.schemas.structured import ENTITY_ID_PREFIX, StructuredDocument

_DEFAULT_PG_URI = "postgresql://nanorag:nano-rag@postgres:5432/nanorag"
_DEFAULT_POOL_MIN = 1
_DEFAULT_POOL_MAX = 4


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
    "CREATE INDEX IF NOT EXISTS graph_node_kb_idx ON graph_node (kb_id)",
    """
    CREATE TABLE IF NOT EXISTS graph_entity (
        entity_id text PRIMARY KEY,
        kb_id text NOT NULL,
        name text NOT NULL DEFAULT '',
        entity_type text NOT NULL DEFAULT ''
    )
    """,
    "CREATE INDEX IF NOT EXISTS graph_entity_kb_idx ON graph_entity (kb_id)",
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
    "CREATE INDEX IF NOT EXISTS graph_relation_kb_idx ON graph_relation (kb_id)",
)

# Seed expansion: which other nodes share an entity with the seeds, plus the
# relation edges directly touching a seed (in or out). Entity-prefixed targets
# are resolved to concrete nodes by a follow-up backfill in Python. The LIMIT
# keeps hub entities (shared by thousands of nodes) from being pulled into
# Python wholesale — the backfill and dedupe then trim to max_neighbors.
_EXPAND_QUERY = """
WITH seeds AS (SELECT UNNEST(%(node_ids)s) AS node_id)
SELECT target_id, relation_type FROM (
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
) expanded
LIMIT %(expand_limit)s
"""

_BACKFILL_QUERY = """
SELECT ne.entity_id, ne.node_id
FROM graph_node_entity ne
JOIN graph_node n ON n.node_id = ne.node_id
WHERE ne.entity_id = ANY(%(entity_ids)s)
  AND n.kb_id = %(kb_id)s
  AND ne.node_id <> ALL(%(seed_node_ids)s)
LIMIT %(expand_limit)s
"""

# Reclaim entity rows no node links to any more. NOT part of the per-document
# upsert/delete hot path (it full-scans graph_node_entity and serializes
# concurrent ingest transactions); the pipeline runs it once per ingest job,
# and it is exposed for maintenance. Deliberately global, not kb-scoped:
# entity ids are sha1(kb_id:name) (see GraphExtractor), so an entity can only
# be referenced by node links in its own KB — "unreferenced anywhere" is
# exactly "unreferenced in this KB plus nothing else", and a kb-scoped
# subquery would wrongly delete entities still owned by other KBs. The
# KB-scoping this relies on is pinned by app/tests/test_graph_id_scoping.py
# (test_entity_id_is_scoped_by_kb_and_casefolded) — that test fails if a
# refactor ever makes entity ids non-KB-scoped, which is exactly when this
# global GC would become unsafe.
_GC_ORPHAN_ENTITIES_QUERY = """
DELETE FROM graph_entity
WHERE entity_id NOT IN (
    SELECT DISTINCT entity_id FROM graph_node_entity
)
"""

_NODE_UPSERT_SQL = """
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
"""

_ENTITY_UPSERT_SQL = """
INSERT INTO graph_entity (entity_id, kb_id, name, entity_type)
VALUES (%s, %s, %s, %s)
ON CONFLICT (entity_id)
DO UPDATE SET kb_id = EXCLUDED.kb_id,
              name = EXCLUDED.name,
              entity_type = EXCLUDED.entity_type
"""

_NODE_ENTITY_INSERT_SQL = (
    "INSERT INTO graph_node_entity (node_id, entity_id) "
    "VALUES (%s, %s) ON CONFLICT DO NOTHING"
)

_RELATION_UPSERT_SQL = """
INSERT INTO graph_relation
    (relation_id, source_id, target_id, kb_id, doc_id,
     relation_type, confidence)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (relation_id)
DO UPDATE SET kb_id = EXCLUDED.kb_id,
              doc_id = EXCLUDED.doc_id,
              relation_type = EXCLUDED.relation_type,
              confidence = EXCLUDED.confidence
"""


class PostgresGraphStore:
    """Native-SQL graph store backing document-structure expansion.

    Node/entity/relation rows are materialized from the LLM-extracted
    structured graph at ingest. Expansion answers "which other nodes are
    related to these seeds" via shares-entity joins and RELATED edges.

    High-concurrency notes: the pool size is configurable (32 Celery workers
    with a hardcoded max=4 pool each would exceed PostgreSQL's default
    max_connections), per-document upserts batch their rows with executemany
    instead of one round trip per row, and the orphan-entity GC runs once per
    ingest job instead of full-scanning on every document.
    """

    def __init__(self, uri: str, *, pool_min: int | None = None, pool_max: int | None = None) -> None:
        self.uri = uri
        self.pool_min = pool_min if pool_min is not None else _pool_size_from_env(
            "RAG_PG_POOL_MIN", _DEFAULT_POOL_MIN
        )
        self.pool_max = pool_max if pool_max is not None else _pool_size_from_env(
            "RAG_PG_POOL_MAX", _DEFAULT_POOL_MAX
        )
        if self.pool_max < self.pool_min:
            raise ConfigurationError(
                f"RAG_PG_POOL_MAX ({self.pool_max}) must be >= RAG_PG_POOL_MIN "
                f"({self.pool_min})"
            )
        self._pool = ConnectionPool(
            conninfo=uri,
            min_size=self.pool_min,
            max_size=self.pool_max,
            kwargs={"row_factory": dict_row},
        )
        with self._pool.connection() as conn:
            conn.execute("SELECT 1")
            for statement in _SCHEMA_STATEMENTS:
                conn.execute(statement)

    @classmethod
    def from_env(cls) -> "PostgresGraphStore":
        return cls(os.getenv("PG_URI", _DEFAULT_PG_URI))

    def upsert_document(self, document: StructuredDocument) -> None:
        with self._pool.connection() as conn:
            with conn.transaction():
                self._upsert_document_on(conn, document)

    def delete_document(self, *, doc_id: str, kb_id: str) -> None:
        with self._pool.connection() as conn:
            with conn.transaction():
                # Relation rows carry no FK to graph_node, so deleting the
                # doc's nodes would leave dangling relation rows behind
                # (queryable by expand_node_ids and unreclaimable). Relations
                # are minted per document at extraction, so (doc_id, kb_id)
                # is the owning scope.
                conn.execute(
                    "DELETE FROM graph_node WHERE doc_id = %s AND kb_id = %s",
                    (doc_id, kb_id),
                )
                conn.execute(
                    "DELETE FROM graph_relation WHERE doc_id = %s AND kb_id = %s",
                    (doc_id, kb_id),
                )
                conn.execute(
                    "DELETE FROM graph_document WHERE doc_id = %s AND kb_id = %s",
                    (doc_id, kb_id),
                )

    def collect_orphan_entities(self) -> int:
        """Reclaim entity rows no node links to any more (once per ingest
        job / on maintenance — see _GC_ORPHAN_ENTITIES_QUERY)."""
        with self._pool.connection() as conn:
            with conn.transaction():
                cursor = conn.execute(_GC_ORPHAN_ENTITIES_QUERY)
                return cursor.rowcount or 0

    def expand_node_ids(
        self,
        node_ids: set[str],
        *,
        kb_id: str,
        max_neighbors: int = 8,
    ) -> list[tuple[str, str]]:
        if not node_ids:
            return []
        expand_limit = max(max_neighbors * 4, 32)
        # One connection for both the seed expansion and the entity backfill
        # so they observe a consistent snapshot.
        with self._pool.connection() as conn:
            rows = conn.execute(
                _EXPAND_QUERY,
                {
                    "node_ids": sorted(node_ids),
                    "kb_id": kb_id,
                    "expand_limit": expand_limit,
                },
            ).fetchall()
            entity_neighbors, neighbors = self._partition_neighbors(rows)
            if not entity_neighbors:
                return self._dedupe_neighbors(neighbors, limit=max_neighbors)
            backfill_rows = conn.execute(
                _BACKFILL_QUERY,
                {
                    "entity_ids": [entity_id for entity_id, _ in entity_neighbors],
                    "kb_id": kb_id,
                    "seed_node_ids": sorted(node_ids),
                    "expand_limit": expand_limit,
                },
            ).fetchall()
        relation_by_entity = {
            entity_id: relation for entity_id, relation in entity_neighbors
        }
        backfill = [
            (
                str(row.get("node_id")),
                relation_by_entity.get(str(row.get("entity_id")), "RELATED"),
            )
            for row in backfill_rows
            if row.get("node_id")
        ]
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
            "pool_min_size": self.pool_min,
            "pool_max_size": self.pool_max,
            "documents": int(row.get("documents", 0)),
            "nodes": int(row.get("nodes", 0)),
            "entities": int(row.get("entities", 0)),
            "relations": int(row.get("relations", 0)),
        }

    def close(self) -> None:
        self._pool.close()

    # ------------------------------------------------------------------ #
    # Upsert SQL (kept in pure helpers so they can be exercised without a  #
    # live database; see tests/test_graph_store.py)                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _node_text(node: "object") -> str:
        table = getattr(node, "table", None)
        narrative = getattr(table, "narrative", None) if table else None
        return narrative if narrative else str(getattr(node, "text", "") or "")

    @classmethod
    def _upsert_batches(
        cls, document: StructuredDocument
    ) -> list[tuple[str, list[tuple[object, ...]]]]:
        """Group the document's rows into executemany batches, one per SQL
        shape, instead of one round trip per row."""
        document_rows: list[tuple[object, ...]] = [
            (
                document.kb_id,
                document.doc_id,
                document.title,
                document.source_path,
            )
        ]
        node_rows: list[tuple[object, ...]] = []
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
            node_rows.append(
                (
                    node.node_id,
                    document.kb_id,
                    document.doc_id,
                    node.node_type.value,
                    node.title or "",
                    cls._node_text(node)[:500],
                    node.provenance.page_number,
                    json.dumps(node.provenance.hierarchy_path, ensure_ascii=False),
                )
            )
        entity_rows: list[tuple[object, ...]] = []
        node_entity_rows: list[tuple[object, ...]] = []
        for entity in document.graph.entities:
            entity_rows.append(
                (
                    entity.entity_id,
                    document.kb_id,
                    entity.name,
                    entity.entity_type,
                )
            )
            for node_id in entity.source_node_ids:
                node_entity_rows.append((node_id, entity.entity_id))
        relation_rows: list[tuple[object, ...]] = []
        for relation in document.graph.relations:
            relation_rows.append(
                (
                    relation.relation_id,
                    relation.source_id,
                    relation.target_id,
                    document.kb_id,
                    document.doc_id,
                    relation.relation_type,
                    relation.confidence,
                )
            )
        return [
            (
                """
                INSERT INTO graph_document (kb_id, doc_id, title, source_path)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (kb_id, doc_id)
                DO UPDATE SET title = EXCLUDED.title,
                              source_path = EXCLUDED.source_path
                """,
                document_rows,
            ),
            (_NODE_UPSERT_SQL, node_rows),
            (_ENTITY_UPSERT_SQL, entity_rows),
            (_NODE_ENTITY_INSERT_SQL, node_entity_rows),
            (_RELATION_UPSERT_SQL, relation_rows),
        ]

    def _upsert_document_on(self, conn, document: StructuredDocument) -> None:
        # Re-ingest re-runs LLM extraction, so relation ids from the previous
        # pass no longer match the fresh ones; drop this doc's stale rows
        # before re-materializing. Without this, each re-ingest leaves
        # orphaned edges that expand_node_ids can still return.
        conn.execute(
            "DELETE FROM graph_node WHERE doc_id = %s AND kb_id = %s",
            (document.doc_id, document.kb_id),
        )
        conn.execute(
            "DELETE FROM graph_relation WHERE doc_id = %s AND kb_id = %s",
            (document.doc_id, document.kb_id),
        )
        for sql, rows in self._upsert_batches(document):
            if rows:
                conn.cursor().executemany(sql, rows)

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


def _pool_size_from_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return max(1, int(raw))
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer, got {raw!r}") from exc
