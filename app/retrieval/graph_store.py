from __future__ import annotations

import json
import os
from typing import Protocol

from neo4j import GraphDatabase

from app.schemas.structured import StructuredDocument


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


class Neo4jGraphStore:
    def __init__(self, uri: str, username: str, password: str) -> None:
        self.uri = uri
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.driver.verify_connectivity()
        self._ensure_schema()

    @classmethod
    def from_env(cls) -> "Neo4jGraphStore":
        return cls(
            uri=os.getenv("NEO4J_URI", "bolt://neo4j:7687"),
            username=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "nano-rag-graph"),
        )

    def upsert_document(self, document: StructuredDocument) -> None:
        with self.driver.session() as session:
            session.execute_write(self._delete_document_tx, document.doc_id, document.kb_id)
            session.execute_write(self._upsert_document_tx, document)

    def delete_document(self, *, doc_id: str, kb_id: str) -> None:
        with self.driver.session() as session:
            session.execute_write(self._delete_document_tx, doc_id, kb_id)

    def expand_node_ids(
        self,
        node_ids: set[str],
        *,
        kb_id: str,
        max_neighbors: int = 8,
    ) -> list[tuple[str, str]]:
        if not node_ids:
            return []
        neighbors: list[tuple[str, str]] = []
        entity_neighbors: list[tuple[str, str]] = []
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (seed:DocNode)
                WHERE seed.node_id IN $node_ids AND seed.kb_id = $kb_id
                OPTIONAL MATCH (seed)-[:MENTIONS]->(:Entity)<-[:MENTIONS]-(shared:DocNode)
                WHERE shared.kb_id = $kb_id AND shared.node_id <> seed.node_id
                WITH seed, collect(DISTINCT shared.node_id) AS shared_ids
                OPTIONAL MATCH (seed)-[out:RELATED]->(out_target)
                WITH seed, shared_ids,
                     collect(DISTINCT {
                       id: coalesce(out_target.node_id, out_target.entity_id),
                       relation: out.relation_type
                     }) AS out_targets
                OPTIONAL MATCH (in_source)-[incoming:RELATED]->(seed)
                WITH shared_ids, out_targets,
                     collect(DISTINCT {
                       id: coalesce(in_source.node_id, in_source.entity_id),
                       relation: incoming.relation_type
                     }) AS in_sources
                RETURN shared_ids, out_targets, in_sources
                """,
                node_ids=list(node_ids),
                kb_id=kb_id,
            )
            for row in result:
                neighbors.extend(
                    (node_id, "SHARES_ENTITY")
                    for node_id in row.get("shared_ids", [])
                    if node_id
                )
                for item in [*row.get("out_targets", []), *row.get("in_sources", [])]:
                    if not isinstance(item, dict):
                        continue
                    target_id = item.get("id")
                    relation = item.get("relation")
                    if target_id and relation:
                        target = str(target_id)
                        if target.startswith("entity:"):
                            entity_neighbors.append((target, str(relation)))
                        else:
                            neighbors.append((target, str(relation)))
            if entity_neighbors:
                entity_ids = [entity_id for entity_id, _ in entity_neighbors]
                relation_by_entity = {
                    entity_id: relation for entity_id, relation in entity_neighbors
                }
                entity_result = session.run(
                    """
                    MATCH (e:Entity)<-[:MENTIONS]-(n:DocNode)
                    WHERE e.entity_id IN $entity_ids
                      AND n.kb_id = $kb_id
                      AND NOT n.node_id IN $seed_node_ids
                    RETURN e.entity_id AS entity_id, collect(DISTINCT n.node_id) AS node_ids
                    """,
                    entity_ids=entity_ids,
                    kb_id=kb_id,
                    seed_node_ids=list(node_ids),
                )
                for row in entity_result:
                    relation = relation_by_entity.get(str(row.get("entity_id")), "RELATED")
                    neighbors.extend(
                        (node_id, relation)
                        for node_id in row.get("node_ids", [])
                        if node_id
                    )
        return self._dedupe_neighbors(neighbors, limit=max_neighbors)

    def stats(self) -> dict[str, object]:
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (d:Document) WITH count(d) AS documents
                MATCH (n:DocNode) WITH documents, count(n) AS nodes
                MATCH (e:Entity) WITH documents, nodes, count(e) AS entities
                MATCH ()-[r:RELATED]->()
                RETURN documents, nodes, entities, count(r) AS relations
                """
            )
            row = result.single() or {}
        return {
            "backend": "neo4j",
            "uri": self.uri,
            "documents": int(row.get("documents", 0)),
            "nodes": int(row.get("nodes", 0)),
            "entities": int(row.get("entities", 0)),
            "relations": int(row.get("relations", 0)),
        }

    def close(self) -> None:
        self.driver.close()

    def _ensure_schema(self) -> None:
        statements = [
            "CREATE CONSTRAINT rag_document_id IF NOT EXISTS FOR (d:Document) REQUIRE (d.kb_id, d.doc_id) IS UNIQUE",
            "CREATE CONSTRAINT rag_doc_node_id IF NOT EXISTS FOR (n:DocNode) REQUIRE n.node_id IS UNIQUE",
            "CREATE CONSTRAINT rag_entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
        ]
        with self.driver.session() as session:
            for statement in statements:
                session.run(statement)

    @staticmethod
    def _delete_document_tx(tx, doc_id: str, kb_id: str) -> None:  # noqa: ANN001
        tx.run(
            """
            MATCH (d:Document {doc_id: $doc_id, kb_id: $kb_id})
            OPTIONAL MATCH (d)-[:HAS_NODE]->(n:DocNode)
            WITH d, collect(n) AS nodes
            FOREACH (node IN nodes | DETACH DELETE node)
            DETACH DELETE d
            WITH 1 AS _
            MATCH (e:Entity {kb_id: $kb_id})
            WHERE NOT (:DocNode {kb_id: $kb_id})-[:MENTIONS]->(e)
            DETACH DELETE e
            """,
            doc_id=doc_id,
            kb_id=kb_id,
        )

    @staticmethod
    def _upsert_document_tx(tx, document: StructuredDocument) -> None:  # noqa: ANN001
        tx.run(
            """
            MERGE (d:Document {doc_id: $doc_id, kb_id: $kb_id})
            SET d.title = $title,
                d.source_path = $source_path
            """,
            doc_id=document.doc_id,
            kb_id=document.kb_id,
            title=document.title,
            source_path=document.source_path,
        )
        for node in document.iter_nodes():
            text = node.table.narrative if node.table and node.table.narrative else node.text
            tx.run(
                """
                MATCH (d:Document {doc_id: $doc_id, kb_id: $kb_id})
                MERGE (n:DocNode {node_id: $node_id})
                SET n.doc_id = $doc_id,
                    n.kb_id = $kb_id,
                    n.node_type = $node_type,
                    n.title = $title,
                    n.text_preview = $text_preview,
                    n.page_number = $page_number,
                    n.hierarchy_path_json = $hierarchy_path_json
                MERGE (d)-[:HAS_NODE]->(n)
                """,
                doc_id=document.doc_id,
                kb_id=document.kb_id,
                node_id=node.node_id,
                node_type=node.node_type.value,
                title=node.title,
                text_preview=(text or "")[:500],
                page_number=node.provenance.page_number,
                hierarchy_path_json=json.dumps(
                    node.provenance.hierarchy_path,
                    ensure_ascii=False,
                ),
            )
        for entity in document.graph.entities:
            tx.run(
                """
                MERGE (e:Entity {entity_id: $entity_id})
                SET e.kb_id = $kb_id,
                    e.name = $name,
                    e.entity_type = $entity_type
                """,
                entity_id=entity.entity_id,
                kb_id=document.kb_id,
                name=entity.name,
                entity_type=entity.entity_type,
            )
            for node_id in entity.source_node_ids:
                tx.run(
                    """
                    MATCH (n:DocNode {node_id: $node_id})
                    MATCH (e:Entity {entity_id: $entity_id})
                    MERGE (n)-[:MENTIONS]->(e)
                    """,
                    node_id=node_id,
                    entity_id=entity.entity_id,
                )
        for relation in document.graph.relations:
            tx.run(
                """
                MATCH (source)
                WHERE source.node_id = $source_id OR source.entity_id = $source_id
                MATCH (target)
                WHERE target.node_id = $target_id OR target.entity_id = $target_id
                MERGE (source)-[r:RELATED {relation_id: $relation_id}]->(target)
                SET r.relation_type = $relation_type,
                    r.source_node_id = $source_node_id,
                    r.confidence = $confidence
                """,
                source_id=relation.source_id,
                target_id=relation.target_id,
                relation_id=relation.relation_id,
                relation_type=relation.relation_type,
                source_node_id=relation.source_node_id,
                confidence=relation.confidence,
            )

    def _dedupe_neighbors(
        self,
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
