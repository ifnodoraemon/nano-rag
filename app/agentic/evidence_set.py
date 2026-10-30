"""An immutable snapshot of exactly the source passages both model stages see."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace

from app.agentic.evidence_view import EvidenceDriller


@dataclass(frozen=True)
class EvidenceSet:
    # JSON strings make nested context metadata immutable as well as the container.
    items: tuple[str, ...]
    complete: bool
    source_context_count: int
    media_items: tuple[str, ...] = ()

    @classmethod
    def build(cls, contexts: list[dict], driller: EvidenceDriller,
              watch_terms: list[str]) -> "EvidenceSet":
        excerpts = driller.excerpt(contexts, watch_terms=watch_terms)
        by_key = {}
        for context in contexts:
            for key in (context.get("node_id"), context.get("chunk_id"),
                        context.get("citation_label")):
                if key is not None:
                    by_key[str(key)] = context
        items = []
        for excerpt in excerpts:
            source = by_key.get(str(excerpt.node_id)) or by_key.get(str(excerpt.label))
            if source is None:
                raise ValueError("evidence excerpt has no source context")
            passages = [p.as_dict() for p in excerpt.passages]
            identity = json.dumps([source.get("doc_id"), excerpt.node_id,
                                   source.get("source_content_hash"), passages], sort_keys=True)
            item = {**source, "text": excerpt.render(), "_evidence_frozen": True,
                    "evidence_id": "ev:" + hashlib.sha256(identity.encode()).hexdigest()[:24],
                    "evidence_excerpt": {"document_chars": excerpt.total_chars,
                                         "visible_chars": excerpt.visible_chars,
                                         "exhausted": excerpt.exhausted,
                                         "passages": passages}}
            items.append(json.dumps(item, ensure_ascii=False))
        complete = len(excerpts) == len(contexts) and bool(excerpts) and all(
            e.exhausted for e in excerpts
        ) and all(c.get("document_read_complete", True) for c in contexts)
        return cls(tuple(items), complete, len(contexts))

    def contexts(self) -> list[dict]:
        return [json.loads(item) for item in self.items]

    def with_media_parts(self, parts: list[dict]) -> "EvidenceSet":
        return replace(self, media_items=tuple(json.dumps(p) for p in parts))

    def media_parts(self) -> list[dict]:
        return [json.loads(item) for item in self.media_items]

    def prompt_items(self) -> list[dict]:
        return [{"label": c.get("citation_label"), "node_id": c.get("node_id"),
                 "doc_id": c.get("doc_id"), "source": c.get("source"),
                 "evidence_id": c["evidence_id"], "text": c["text"],
                 "excerpt_exhausted": c["evidence_excerpt"]["exhausted"],
                 "document_read_complete": c.get("document_read_complete", True),
                 "map_summaries": [f.get("summary") for f in c.get("map_facts", [])],
                 "passages": c["evidence_excerpt"]["passages"]}
                for c in self.contexts()]

    def summary(self) -> dict:
        contexts = self.contexts()
        total = sum(c["evidence_excerpt"]["document_chars"] for c in contexts)
        visible = sum(c["evidence_excerpt"]["visible_chars"] for c in contexts)
        return {"contexts": self.source_context_count, "contexts_shown": len(contexts),
                "document_chars": total, "visible_chars": visible,
                "visible_ratio": round(visible / total, 4) if total else 0,
                "all_exhausted": self.complete,
                "media_parts": len(self.media_items),
                "media_sha256": [hashlib.sha256(p.encode()).hexdigest() for p in self.media_items],
                "evidence_ids": [c["evidence_id"] for c in contexts]}
