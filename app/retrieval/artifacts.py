"""Scoped document reads, lexical node selection and structural navigation."""
from __future__ import annotations

import json
from collections import OrderedDict
from threading import RLock
from pathlib import Path

from app.core.exceptions import RetrievalError
from app.retrieval.bm25 import BM25Index
from app.retrieval.context_builder import build_contexts
from app.retrieval.hits import SearchHit
from app.schemas.chunk import Chunk


class ArtifactReader:
    _cache: OrderedDict = OrderedDict()
    _lock = RLock()
    def __init__(self, root: Path):
        self.root = Path(root)

    def load(self, doc_id: str, kb_id: str | None = None) -> dict:
        path = (self.root / f"{doc_id}.json").resolve()
        if path.parent != self.root.resolve():
            raise RetrievalError("document id must identify an artifact in the parsed directory")
        try:
            stat = path.stat()
            key = (str(path), stat.st_mtime_ns, stat.st_size)
            with self._lock:
                encoded = self._cache.get(key)
                if encoded is None:
                    encoded = path.read_text(encoding="utf-8")
                    self._cache[key] = encoded
                    while len(self._cache) > 128:
                        self._cache.popitem(last=False)
                self._cache.move_to_end(key)
            # Decode a fresh value, so a caller cannot mutate the shared cache.
            data = json.loads(encoded)
        except (OSError, ValueError) as exc:
            raise RetrievalError(f"cannot read parsed artifact {doc_id}: {exc}") from exc
        if not isinstance(data, dict) or not isinstance(data.get("chunks"), list):
            raise RetrievalError(f"malformed parsed artifact {doc_id}")
        document = data.get("document") or {}
        actual = document.get("metadata", {}).get("kb_id", "default")
        if kb_id is not None and actual != kb_id:
            raise RetrievalError("document is outside the requested knowledge base")
        for raw in data["chunks"]:
            if not isinstance(raw, dict) or raw.get("doc_id") != doc_id:
                raise RetrievalError(f"invalid chunk ownership in artifact {doc_id}")
        return data

    @staticmethod
    def rank(chunks: list[dict], query: str, seed_ids=(), seed_text: str = "") -> list[dict]:
        index = BM25Index()
        for c in chunks:
            index.add_document(c["chunk_id"], str(c.get("text", "")), str(c.get("title") or ""))
        scores = dict(index.search(query, len(chunks))) if query.strip() else {}
        def priority(c):
            text = str(c.get("text") or "")
            seeded = c["chunk_id"] in seed_ids or (seed_text and text and (
                seed_text.strip() in text or text.strip() in seed_text
            ))
            return bool(seeded), scores.get(c["chunk_id"], 0)
        return sorted(chunks, key=priority, reverse=True)

    def manifest(self, doc_id: str, query: str, kb_id: str | None = None,
                 excluded: set[str] | None = None, path: list[str] | None = None,
                 offset: int = 0, limit: int = 128) -> dict:
        data = self.load(doc_id, kb_id)
        chunks = data["chunks"]
        excluded = excluded or set()
        available = [c for c in chunks if c["chunk_id"] not in excluded]
        path = path or []
        if path:
            available = [c for c in available if
                         (c.get("metadata", {}).get("hierarchy_path") or [c.get("title") or "Document"])[:len(path)] == path]
        ranked = self.rank(available, query)
        nodes = [{"node_id": c["chunk_id"], "title": c.get("title"),
                  "path": c.get("metadata", {}).get("hierarchy_path", []),
                  "chars": len(c.get("text", "")), "preview": c.get("text", "")[:180],
                  "summary": c.get("metadata", {}).get("retrieval_summary", ""),
                  "table_id": c.get("metadata", {}).get("table_node_id")}
                 for c in ranked[:32]]
        paths = {}
        children = {}
        for c in available:
            node_path = c.get("metadata", {}).get("hierarchy_path") or [c.get("title") or "Document"]
            paths.setdefault(" / ".join(node_path), []).append(c["chunk_id"])
            branch = tuple(node_path[:len(path)+1])
            children[branch] = children.get(branch, 0) + 1
        branches = list(children.items())
        return {"doc_id": doc_id, "nodes": nodes,
                "outline": [{"path": p, "node_ids": ids} for p, ids in list(paths.items())[offset:offset+limit]],
                "children": [{"path": list(p), "node_count": count} for p, count in branches[offset:offset+limit]],
                "path": path, "next_offset": offset + min(limit, max(0, len(branches)-offset)),
                "branches_total": len(branches),
                "total_nodes": len(chunks), "remaining_nodes": len(available),
                "manifest_complete": len(available) <= 32,
                "table_ids": sorted({str(c.get("metadata", {}).get("table_node_id")
                                     or c.get("metadata", {}).get("source_node_id") or c["chunk_id"])
                                     for c in available if c.get("metadata", {}).get("table")})}

    def read_nodes(self, doc_id: str, node_ids: list[str], kb_id: str,
                   *, include_neighbors: bool = False, max_nodes: int = 24) -> list[dict]:
        data = self.load(doc_id, kb_id)
        chunks = data["chunks"]
        by_id = {c["chunk_id"]: c for c in chunks}
        unknown = set(node_ids) - set(by_id)
        if unknown:
            raise RetrievalError(f"unknown nodes in {doc_id}: {sorted(unknown)}")
        selected = [by_id[k] for k in dict.fromkeys(node_ids)]
        if include_neighbors:
            positions = {c["chunk_id"]: i for i, c in enumerate(chunks)}
            for key in node_ids:
                i = positions[key]
                selected.extend(chunks[max(0, i-1):i+2])
                parent = by_id[key].get("metadata", {}).get("parent_chunk_id")
                if parent in by_id:
                    selected.append(by_id[parent])
                source_node = by_id[key].get("metadata", {}).get("source_node_id")
                if source_node:
                    selected.extend(c for c in chunks if c.get("metadata", {}).get("source_node_id") == source_node)
        selected = list({c["chunk_id"]: c for c in selected}.values())[:max_nodes]
        contexts = build_contexts([SearchHit(chunk=Chunk.model_validate(c), score=0) for c in selected],
                                  len(selected), truncate=False)
        for c in contexts:
            c["document_read_complete"] = len(selected) == len(chunks)
            c["document_total_nodes"] = len(chunks)
        return contexts

    def references(self, doc_id: str, node_ids: list[str], kb_id: str) -> list[str]:
        chunks = self.load(doc_id, kb_id)["chunks"]
        by_id = {c["chunk_id"]: c for c in chunks}
        if set(node_ids) - set(by_id):
            raise RetrievalError("reference seed contains unknown nodes")
        text = "\n".join(by_id[k]["text"] for k in node_ids)
        targets = []
        for c in chunks:
            meta = c.get("metadata", {})
            labels = [meta.get("clause_id"), meta.get("definition_term")]
            labels += (meta.get("hierarchy_path") or [])[-1:]
            if c["chunk_id"] not in node_ids and any(
                isinstance(label, str) and len(label) >= 2 and label in text for label in labels
            ):
                targets.append(c["chunk_id"])
        for key in node_ids:
            targets.extend(k for k in by_id[key].get("metadata", {}).get("reference_node_ids", [])
                           if k in by_id)
        return list(dict.fromkeys(targets))
