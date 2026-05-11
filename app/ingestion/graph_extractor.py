from __future__ import annotations

import hashlib
import json
from typing import Any

from app.model_client.generation import GenerationClient
from app.schemas.structured import (
    GraphEntity,
    GraphRelation,
    KnowledgeGraph,
    StructuredDocument,
)


MAX_GRAPH_NODES_FOR_PROMPT = 80
MAX_NODE_TEXT_CHARS = 900


class GraphExtractor:
    def __init__(self, generation_client: GenerationClient) -> None:
        self.generation_client = generation_client

    async def extract(self, document: StructuredDocument) -> KnowledgeGraph:
        payload = await self._extract_with_llm(document)
        entities = self._entities(document, payload.get("entities", []))
        relations = self._relations(document, payload.get("relations", []), entities)
        return KnowledgeGraph(
            entities=list(entities.values()),
            relations=relations,
        )

    async def _extract_with_llm(self, document: StructuredDocument) -> dict[str, Any]:
        result = await self.generation_client.generate(
            [
                {
                    "role": "system",
                    "content": (
                        "You extract document knowledge graphs for an industrial RAG system. "
                        "Return only compact JSON. Do not explain."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Extract the professional entities and relations needed for multi-step retrieval. "
                        "Use source_node_ids from the provided nodes. Include hierarchy/citation relations when useful. "
                        "Do not invent facts outside the nodes.\n"
                        "Return JSON: "
                        '{"entities":[{"name":"...","entity_type":"...","source_node_ids":["..."]}],'
                        '"relations":[{"source":"entity name or node_id","target":"entity name or node_id",'
                        '"relation_type":"...","source_node_id":"...","confidence":0.0}]}\n\n'
                        f"Document: {document.title}\n"
                        f"Nodes: {json.dumps(self._nodes_for_prompt(document), ensure_ascii=False)}"
                    ),
                },
            ]
        )
        return self._json_object(str(result.get("content") or ""))

    def _nodes_for_prompt(self, document: StructuredDocument) -> list[dict[str, object]]:
        nodes: list[dict[str, object]] = []
        for node in document.iter_nodes():
            if len(nodes) >= MAX_GRAPH_NODES_FOR_PROMPT:
                break
            text = node.table.narrative if node.table and node.table.narrative else node.text
            nodes.append(
                {
                    "node_id": node.node_id,
                    "node_type": node.node_type.value,
                    "title": node.title,
                    "hierarchy_path": node.provenance.hierarchy_path,
                    "page_number": node.provenance.page_number,
                    "text": (text or "")[:MAX_NODE_TEXT_CHARS],
                }
            )
        return nodes

    def _entities(
        self, document: StructuredDocument, raw_entities: object
    ) -> dict[str, GraphEntity]:
        entities = self._node_entities(document)
        if not isinstance(raw_entities, list):
            return entities
        for raw in raw_entities:
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name") or "").strip()
            if not name:
                continue
            entity_id = self._entity_id(document.kb_id, name)
            source_node_ids = [
                str(item)
                for item in raw.get("source_node_ids", [])
                if str(item).strip()
            ]
            entities[entity_id] = GraphEntity(
                entity_id=entity_id,
                name=name,
                entity_type=str(raw.get("entity_type") or "concept"),
                source_node_ids=self._dedupe(source_node_ids),
                metadata={"doc_id": document.doc_id, "kb_id": document.kb_id},
            )
        return entities

    def _node_entities(self, document: StructuredDocument) -> dict[str, GraphEntity]:
        entities: dict[str, GraphEntity] = {}
        for node in document.iter_nodes():
            entities[node.node_id] = GraphEntity(
                entity_id=node.node_id,
                name=node.title or node.provenance.source_ref or node.node_id,
                entity_type=f"document_{node.node_type.value}",
                source_node_ids=[node.node_id],
                metadata={
                    "doc_id": node.doc_id,
                    "kb_id": node.kb_id,
                    "page_number": node.provenance.page_number,
                    "hierarchy_path": node.provenance.hierarchy_path,
                },
            )
        return entities

    def _relations(
        self,
        document: StructuredDocument,
        raw_relations: object,
        entities: dict[str, GraphEntity],
    ) -> list[GraphRelation]:
        relations: dict[str, GraphRelation] = {}
        for node in document.iter_nodes():
            if node.parent_id:
                self._add_relation(
                    relations,
                    source_id=node.node_id,
                    target_id=node.parent_id,
                    relation_type="PART_OF",
                    source_node_id=node.node_id,
                    confidence=1.0,
                )
        if not isinstance(raw_relations, list):
            return list(relations.values())
        name_to_id = {entity.name.casefold(): entity_id for entity_id, entity in entities.items()}
        for raw in raw_relations:
            if not isinstance(raw, dict):
                continue
            source_id = self._resolve_id(str(raw.get("source") or ""), name_to_id, entities)
            target_id = self._resolve_id(str(raw.get("target") or ""), name_to_id, entities)
            relation_type = str(raw.get("relation_type") or "").strip().upper()
            if not source_id or not target_id or not relation_type:
                continue
            self._add_relation(
                relations,
                source_id=source_id,
                target_id=target_id,
                relation_type=relation_type,
                source_node_id=str(raw.get("source_node_id") or "") or None,
                confidence=self._confidence(raw.get("confidence")),
            )
        return list(relations.values())

    def _resolve_id(
        self,
        value: str,
        name_to_id: dict[str, str],
        entities: dict[str, GraphEntity],
    ) -> str | None:
        value = value.strip()
        if not value:
            return None
        if value in entities:
            return value
        return name_to_id.get(value.casefold())

    def _add_relation(
        self,
        relations: dict[str, GraphRelation],
        *,
        source_id: str,
        target_id: str,
        relation_type: str,
        source_node_id: str | None,
        confidence: float,
    ) -> None:
        raw = f"{source_id}|{relation_type}|{target_id}"
        relation_id = f"rel:{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:24]}"
        relations[relation_id] = GraphRelation(
            relation_id=relation_id,
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            source_node_id=source_node_id,
            confidence=confidence,
        )

    def _json_object(self, content: str) -> dict[str, Any]:
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end >= start:
            text = text[start : end + 1]
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return loaded if isinstance(loaded, dict) else {}

    def _confidence(self, value: object) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return 0.7
        return max(0.0, min(1.0, confidence))

    def _entity_id(self, kb_id: str, name: str) -> str:
        digest = hashlib.sha1(f"{kb_id}:{name.casefold()}".encode("utf-8")).hexdigest()
        return f"entity:{digest[:20]}"

    def _dedupe(self, items: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for item in items:
            key = item.casefold()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped
