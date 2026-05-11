from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from app.schemas.structured import (
    DocumentNode,
    GraphEntity,
    GraphRelation,
    StructuredDocument,
)

AddNeighbor = Callable[[str, str, dict[str, object]], None]


@dataclass
class GraphView:
    documents: dict[str, StructuredDocument] = field(default_factory=dict)
    nodes: dict[str, DocumentNode] = field(default_factory=dict)
    entities: dict[str, GraphEntity] = field(default_factory=dict)
    relations: list[GraphRelation] = field(default_factory=list)
    source_to_relations: dict[str, list[GraphRelation]] = field(default_factory=dict)
    target_to_relations: dict[str, list[GraphRelation]] = field(default_factory=dict)
    node_to_entities: dict[str, list[str]] = field(default_factory=dict)


class GraphIndex:
    def __init__(self, parsed_dir: Path) -> None:
        self.parsed_dir = parsed_dir

    def load(self, kb_id: str) -> GraphView:
        view = GraphView()
        for document in self._load_documents(kb_id):
            view.documents[document.doc_id] = document
            for node in document.iter_nodes():
                view.nodes[node.node_id] = node
            self._merge_entities(view, document)
            self._append_relations(view, document.graph.relations)
        return view

    def expand_node_ids(
        self,
        node_ids: set[str],
        *,
        kb_id: str,
        max_neighbors: int = 8,
    ) -> list[tuple[str, str]]:
        view = self.load(kb_id)
        expanded: list[tuple[str, str]] = []
        seen = set(node_ids)
        for node_id in node_ids:
            for neighbor_id, relation in self.neighbor_node_ids(view, node_id):
                if neighbor_id in seen:
                    continue
                if neighbor_id not in view.nodes:
                    continue
                seen.add(neighbor_id)
                expanded.append((neighbor_id, relation))
                if len(expanded) >= max_neighbors:
                    return expanded
        return expanded

    def neighbor_node_ids(
        self, view: GraphView, node_id: str
    ) -> list[tuple[str, str]]:
        neighbors: list[tuple[str, str]] = []
        for entity_id in view.node_to_entities.get(node_id, []):
            entity = view.entities.get(entity_id)
            if entity is None:
                continue
            for source_node_id in entity.source_node_ids:
                neighbors.append((source_node_id, "SHARES_ENTITY"))
        for relation in view.source_to_relations.get(node_id, []):
            neighbors.extend(
                self._relation_neighbors(
                    view,
                    relation.target_id,
                    relation.relation_type,
                )
            )
        for relation in view.target_to_relations.get(node_id, []):
            neighbors.extend(
                self._relation_neighbors(
                    view,
                    relation.source_id,
                    relation.relation_type,
                )
            )
        return self._dedupe_neighbors(neighbors)

    def node_context(
        self,
        view: GraphView,
        node_id: str,
        relation: str,
    ) -> dict[str, object] | None:
        node = view.nodes.get(node_id)
        if node is None:
            return None
        document = view.documents.get(node.doc_id)
        if document is None:
            return None
        text = node.table.narrative if node.table and node.table.narrative else node.text
        if not text.strip():
            return None
        bbox = (
            node.provenance.bounding_box.model_dump()
            if node.provenance.bounding_box
            else None
        )
        return {
            "chunk_id": node.node_id,
            "node_id": node.node_id,
            "text": text,
            "source": document.source_path,
            "title": " / ".join(node.provenance.hierarchy_path) or document.title,
            "score": 0.0,
            "page_number": node.provenance.page_number,
            "hierarchy_path": node.provenance.hierarchy_path,
            "bounding_box": bbox,
            "evidence_role": "supporting",
            "wiki_kind": "graph_expanded",
            "wiki_status": "n/a",
            "graph_relation": relation,
            "modality": "text",
        }

    def node_summary(self, node: DocumentNode) -> dict[str, object]:
        return {
            "node_id": node.node_id,
            "doc_id": node.doc_id,
            "node_type": node.node_type.value,
            "title": node.title,
            "text": node.text[:240],
            "page_number": node.provenance.page_number,
            "hierarchy_path": node.provenance.hierarchy_path,
        }

    def entity_summary(self, entity: GraphEntity) -> dict[str, object]:
        return {
            "entity_id": entity.entity_id,
            "name": entity.name,
            "entity_type": entity.entity_type,
            "source_node_ids": entity.source_node_ids,
            "metadata": entity.metadata,
        }

    def neighborhood(
        self,
        node_id: str,
        *,
        kb_id: str,
    ) -> tuple[dict[str, object], list[dict[str, object]]] | None:
        view = self.load(kb_id)
        node = view.nodes.get(node_id)
        if node is None:
            return None
        neighbors: list[dict[str, object]] = []
        seen: set[tuple[str, str, str]] = set()

        def add(relation: str, direction: str, target: dict[str, object]) -> None:
            target_id = str(target.get("node_id") or target.get("entity_id") or "")
            key = (relation, direction, target_id)
            if not target_id or key in seen:
                return
            seen.add(key)
            neighbors.append(
                {"relation": relation, "direction": direction, "target": target}
            )

        for entity_id in view.node_to_entities.get(node_id, []):
            entity = view.entities.get(entity_id)
            if entity:
                add("MENTIONS", "out", self.entity_summary(entity))
                for source_node_id in entity.source_node_ids:
                    if source_node_id != node_id and source_node_id in view.nodes:
                        add(
                            "SHARES_ENTITY",
                            "both",
                            self.node_summary(view.nodes[source_node_id]),
                        )
        for relation in view.source_to_relations.get(node_id, []):
            self._add_relation_target(view, relation, "out", add)
        for relation in view.target_to_relations.get(node_id, []):
            self._add_relation_target(view, relation, "in", add, use_source=True)
        return self.node_summary(node), neighbors

    def _load_documents(self, kb_id: str) -> list[StructuredDocument]:
        documents: list[StructuredDocument] = []
        if not self.parsed_dir.exists():
            return documents
        for artifact in sorted(self.parsed_dir.glob("*.json")):
            document = self._load_document(artifact)
            if document is not None and document.kb_id == kb_id:
                documents.append(document)
        return documents

    def _load_document(self, path: Path) -> StructuredDocument | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        raw = payload.get("structured_document") if isinstance(payload, dict) else None
        if not isinstance(raw, dict):
            return None
        return StructuredDocument.model_validate(raw)

    def _merge_entities(self, view: GraphView, document: StructuredDocument) -> None:
        valid_node_ids = {node.node_id for node in document.iter_nodes()}
        for entity in document.graph.entities:
            source_node_ids = [
                node_id for node_id in entity.source_node_ids if node_id in valid_node_ids
            ]
            existing = view.entities.get(entity.entity_id)
            if existing is None:
                existing = entity.model_copy(update={"source_node_ids": []})
                view.entities[entity.entity_id] = existing
            existing.source_node_ids = self._dedupe(
                [*existing.source_node_ids, *source_node_ids]
            )
            for node_id in source_node_ids:
                view.node_to_entities.setdefault(node_id, [])
                if entity.entity_id not in view.node_to_entities[node_id]:
                    view.node_to_entities[node_id].append(entity.entity_id)

    def _append_relations(
        self, view: GraphView, relations: list[GraphRelation]
    ) -> None:
        for relation in relations:
            view.relations.append(relation)
            view.source_to_relations.setdefault(relation.source_id, []).append(relation)
            view.target_to_relations.setdefault(relation.target_id, []).append(relation)

    def _relation_neighbors(
        self, view: GraphView, target_id: str, relation_type: str
    ) -> list[tuple[str, str]]:
        if target_id in view.nodes:
            return [(target_id, relation_type)]
        entity = view.entities.get(target_id)
        if entity is None:
            return []
        return [
            (source_node_id, relation_type)
            for source_node_id in entity.source_node_ids
        ]

    def _add_relation_target(
        self,
        view: GraphView,
        relation: GraphRelation,
        direction: str,
        add: AddNeighbor,
        *,
        use_source: bool = False,
    ) -> None:
        target_id = relation.source_id if use_source else relation.target_id
        node = view.nodes.get(target_id)
        entity = view.entities.get(target_id)
        if node:
            add(relation.relation_type, direction, self.node_summary(node))
        elif entity:
            add(relation.relation_type, direction, self.entity_summary(entity))

    def _dedupe_neighbors(
        self, neighbors: list[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        deduped: list[tuple[str, str]] = []
        seen: set[str] = set()
        for node_id, relation in neighbors:
            if node_id in seen:
                continue
            seen.add(node_id)
            deduped.append((node_id, relation))
        return deduped

    def _dedupe(self, items: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped
