from __future__ import annotations

import json
from pathlib import Path

from app.schemas.structured import DocumentNode, StructuredDocument


class GraphExpander:
    def __init__(self, parsed_dir: Path) -> None:
        self.parsed_dir = parsed_dir

    def expand(
        self,
        contexts: list[dict[str, object]],
        *,
        max_neighbors: int = 8,
    ) -> list[dict[str, object]]:
        expanded: list[dict[str, object]] = []
        seen_nodes = {
            str(context.get("node_id") or context.get("chunk_id"))
            for context in contexts
            if context.get("node_id") or context.get("chunk_id")
        }
        documents = self._load_documents(seen_nodes)
        for node_id in seen_nodes:
            document = documents.get(self._doc_id_from_node(node_id))
            if document is None:
                continue
            by_id = {node.node_id: node for node in document.iter_nodes()}
            for neighbor_id, relation in self._neighbor_node_ids(document, node_id):
                if neighbor_id in seen_nodes:
                    continue
                node = by_id.get(neighbor_id)
                if node is None:
                    continue
                context = self._node_context(document, node, relation)
                if context:
                    expanded.append(context)
                    seen_nodes.add(neighbor_id)
                if len(expanded) >= max_neighbors:
                    return expanded
        return expanded

    def _load_documents(
        self, node_ids: set[str]
    ) -> dict[str, StructuredDocument]:
        documents: dict[str, StructuredDocument] = {}
        for node_id in node_ids:
            doc_id = self._doc_id_from_node(node_id)
            if not doc_id or doc_id in documents:
                continue
            document = self._load_document(doc_id)
            if document:
                documents[doc_id] = document
        return documents

    def _load_document(self, doc_id: str) -> StructuredDocument | None:
        path = self.parsed_dir / f"{doc_id}.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        raw = payload.get("structured_document") if isinstance(payload, dict) else None
        if not isinstance(raw, dict):
            return None
        return StructuredDocument.model_validate(raw)

    def _neighbor_node_ids(
        self, document: StructuredDocument, node_id: str
    ) -> list[tuple[str, str]]:
        neighbors: list[tuple[str, str]] = []
        entities = {
            entity.entity_id: entity
            for entity in document.graph.entities
            if entity.source_node_ids
        }
        for relation in document.graph.relations:
            if relation.source_id == node_id:
                if relation.target_id in entities:
                    for source_node_id in entities[relation.target_id].source_node_ids:
                        neighbors.append((source_node_id, relation.relation_type))
                else:
                    neighbors.append((relation.target_id, relation.relation_type))
            if relation.target_id == node_id:
                if relation.source_id in entities:
                    for source_node_id in entities[relation.source_id].source_node_ids:
                        neighbors.append((source_node_id, relation.relation_type))
                else:
                    neighbors.append((relation.source_id, relation.relation_type))
        return self._dedupe_neighbors(neighbors)

    def _node_context(
        self,
        document: StructuredDocument,
        node: DocumentNode,
        relation: str,
    ) -> dict[str, object] | None:
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

    def _doc_id_from_node(self, node_id: str) -> str:
        if ":node:" in node_id:
            return node_id.split(":node:", 1)[0]
        return node_id.split(":", 1)[0]

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
