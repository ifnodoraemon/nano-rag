from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.retrieval.bm25 import BM25Index
from app.retrieval.filters import match_metadata_filters
from app.schemas.chunk import Chunk
from app.vectorstore.repository import SearchHit, _tenant_matches
from app.wiki.compiler import WikiCompiler

MAX_WIKI_CONTEXT_CHARS = 2200


@dataclass
class WikiDocument:
    doc_id: str
    title: str
    source_path: str
    kb_id: str
    tenant_id: str | None
    page_path: str
    body: str
    kind: str
    metadata: dict[str, object]


class WikiSearcher:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.sources_dir = self.root_dir / "sources"
        self.topics_dir = self.root_dir / "topics"
        self.indexes_dir = self.root_dir / "indexes"
        self.index = BM25Index()
        self.documents: dict[str, WikiDocument] = {}
        self.refresh()

    @property
    def enabled(self) -> bool:
        return self.root_dir.exists()

    def refresh(self) -> None:
        self.index.clear()
        self.documents.clear()
        if not self.root_dir.exists():
            return
        self._index_topic_pages()
        self._index_scoped_indexes()
        self._index_source_pages()

    def search(
        self,
        query: str,
        top_k: int,
        kb_id: str = "default",
        tenant_id: str | None = None,
        metadata_filters: dict[str, object] | None = None,
    ) -> list[SearchHit]:
        scope_id = WikiCompiler.scope_id(kb_id=kb_id, tenant_id=tenant_id)
        allowed_doc_ids = {
            doc_id
            for doc_id, document in self.documents.items()
            if document.kb_id == kb_id
            and _tenant_matches(document.tenant_id, tenant_id)
            and match_metadata_filters(document.metadata, metadata_filters)
            and (
                document.kind in {"topic", "source"}
                or doc_id == f"wiki:index:{scope_id}"
            )
        }
        if not allowed_doc_ids:
            return []
        results = self.index.search(
            query, top_k=max(top_k * 4, 8), allowed_doc_ids=allowed_doc_ids
        )
        reranked = sorted(
            results,
            key=lambda item: (
                item[1] + self._kind_bonus(self.documents.get(item[0])),
                item[1],
            ),
            reverse=True,
        )[:top_k]
        hits: list[SearchHit] = []
        for doc_id, score in reranked:
            document = self.documents.get(doc_id)
            if document is None:
                continue
            hits.append(
                SearchHit(
                    chunk=Chunk(
                        chunk_id=doc_id,
                        doc_id=document.doc_id,
                        chunk_index=0,
                        text=self._limit_body(document.body),
                        source_path=document.page_path,
                        title=document.title,
                        metadata={
                            "kb_id": document.kb_id,
                            "tenant_id": document.tenant_id,
                            "wiki_kind": document.kind,
                            "wiki_status": document.metadata.get("status"),
                            "original_source_path": document.source_path,
                            "doc_type": document.metadata.get("doc_type"),
                            "source_key": document.metadata.get("source_key"),
                            "effective_date": document.metadata.get("effective_date"),
                            "version": document.metadata.get("version"),
                        },
                    ),
                    score=score,
                )
            )
        return hits

    def _index_topic_pages(self) -> None:
        if not self.topics_dir.exists():
            return
        for page_path in sorted(self.topics_dir.glob("*.md")):
            metadata, body = WikiCompiler.read_frontmatter(page_path)
            kb_id = str(metadata.get("kb_id", "default"))
            tenant_id = metadata.get("tenant_id")
            topic_name = str(metadata.get("topic", page_path.stem))
            doc_id = f"wiki:topic:{page_path.stem}"
            self.documents[doc_id] = WikiDocument(
                doc_id=page_path.stem,
                title=topic_name,
                source_path=str(page_path.relative_to(self.root_dir.parent)),
                kb_id=kb_id,
                tenant_id=str(tenant_id) if tenant_id not in (None, "") else None,
                page_path=str(page_path.relative_to(self.root_dir.parent)),
                body=body,
                kind="topic",
                metadata=metadata,
            )
            self.index.add_document(doc_id, body)

    def _index_scoped_indexes(self) -> None:
        if not self.indexes_dir.exists():
            return
        for index_path in sorted(self.indexes_dir.glob("*.md")):
            scope_id = index_path.stem
            kb_id, tenant_id = self._parse_scope_id(scope_id)
            body = index_path.read_text(encoding="utf-8")
            doc_id = f"wiki:index:{scope_id}"
            self.documents[doc_id] = WikiDocument(
                doc_id=doc_id,
                title=f"Wiki Index {scope_id}",
                source_path=str(index_path.relative_to(self.root_dir.parent)),
                kb_id=kb_id,
                tenant_id=tenant_id,
                page_path=str(index_path.relative_to(self.root_dir.parent)),
                body=body,
                kind="index",
                metadata={},
            )
            self.index.add_document(doc_id, body)

    def _index_source_pages(self) -> None:
        if not self.sources_dir.exists():
            return
        for page_path in sorted(self.sources_dir.glob("*.md")):
            metadata, body = WikiCompiler.read_frontmatter(page_path)
            kb_id = str(metadata.get("kb_id", "default"))
            tenant_id = metadata.get("tenant_id")
            doc_id = f"wiki:source:{metadata.get('doc_id', page_path.stem)}"
            self.documents[doc_id] = WikiDocument(
                doc_id=str(metadata.get("doc_id", page_path.stem)),
                title=str(metadata.get("title", page_path.stem)),
                source_path=str(metadata.get("source_path", "")),
                kb_id=kb_id,
                tenant_id=str(tenant_id) if tenant_id not in (None, "") else None,
                page_path=str(page_path.relative_to(self.root_dir.parent)),
                body=body,
                kind="source",
                metadata=metadata,
            )
            self.index.add_document(doc_id, body)

    def _parse_scope_id(self, scope_id: str) -> tuple[str, str | None]:
        if "__" not in scope_id:
            return scope_id, None
        kb_id, tenant_id = scope_id.split("__", 1)
        return kb_id, tenant_id or None

    def _limit_body(self, body: str) -> str:
        normalized = body.strip()
        if len(normalized) <= MAX_WIKI_CONTEXT_CHARS:
            return normalized
        return normalized[: MAX_WIKI_CONTEXT_CHARS - 3].rstrip() + "..."

    def _kind_bonus(self, document: WikiDocument | None) -> float:
        if document is None:
            return 0.0
        if document.kind == "topic":
            return 0.2
        if document.kind == "source":
            return 0.1
        if document.kind == "index":
            return 0.05
        return 0.0
