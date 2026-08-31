from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path

from app.core.exceptions import RetrievalError
from app.retrieval.bm25 import BM25Config, BM25Index
from app.retrieval.filters import match_metadata_filters
from app.schemas.chunk import Chunk
from app.retrieval.hits import SearchHit
from app.wiki.compiler import WikiCompiler

logger = logging.getLogger(__name__)

MAX_WIKI_CONTEXT_CHARS = 2200
DEFAULT_MAX_DOCUMENTS = 50000
DEFAULT_MAX_PAGES_PER_KB = 20000


@dataclass
class WikiDocument:
    doc_id: str
    title: str
    source_path: str
    kb_id: str
    page_path: str
    body: str
    kind: str
    metadata: dict[str, object]


class WikiSearcher:
    """In-memory BM25 index over the compiled wiki pages.

    High-concurrency design:

    - incremental sync: the on-disk page set is diffed by (path, mtime_ns)
      and only added/changed/removed pages are re-indexed. The previous
      any-change-means-full-rebuild behavior turned every ingest into a
      query-side re-index storm over the whole corpus;
    - the root-level ``log.md``/``SCHEMA.md`` files are no longer part of the
      staleness snapshot (they are not indexed, but they changed on every
      ingest and forced spurious rebuilds);
    - a reentrant lock keeps sync/search consistent across the event loop
      and to_thread callers;
    - capacity limits (global and per-KB) raise instead of silently dropping
      pages — an invisible ceiling is a silent degradation path.
    """

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.sources_dir = self.root_dir / "sources"
        self.topics_dir = self.root_dir / "topics"
        self.indexes_dir = self.root_dir / "indexes"
        self.index = BM25Index(BM25Config(max_documents=self._max_documents()))
        self.documents: dict[str, WikiDocument] = {}
        self._page_mtimes: dict[str, int] = {}
        self._path_doc_ids: dict[str, str] = {}
        self._kb_page_counts: dict[str, int] = {}
        self._max_pages_per_kb = self._max_pages_per_kb_config()
        self._lock = threading.RLock()
        self.refresh()

    @staticmethod
    def _max_documents() -> int:
        try:
            return max(1, int(os.getenv("RAG_BM25_MAX_DOCUMENTS", str(DEFAULT_MAX_DOCUMENTS))))
        except ValueError as exc:
            raise RetrievalError(
                f"RAG_BM25_MAX_DOCUMENTS must be an integer: {exc}"
            ) from exc

    @staticmethod
    def _max_pages_per_kb_config() -> int:
        try:
            return max(
                1, int(os.getenv("RAG_WIKI_MAX_PAGES_PER_KB", str(DEFAULT_MAX_PAGES_PER_KB)))
            )
        except ValueError as exc:
            raise RetrievalError(
                f"RAG_WIKI_MAX_PAGES_PER_KB must be an integer: {exc}"
            ) from exc

    @property
    def enabled(self) -> bool:
        return self.root_dir.exists()

    def _collect_page_mtimes(self) -> dict[str, int]:
        """The on-disk page set as {path: mtime_ns} for the directories that
        are actually indexed. Cheap (a few directories) and covers additions,
        edits, and removals — mtime alone would miss deletions."""
        snapshot: dict[str, int] = {}
        for directory in (self.sources_dir, self.topics_dir, self.indexes_dir):
            if not directory.exists():
                continue
            for path in directory.iterdir():
                if path.is_file():
                    try:
                        snapshot[str(path)] = path.stat().st_mtime_ns
                    except OSError:
                        continue
        return snapshot

    def _sync_if_stale(self) -> None:
        # The in-memory BM25 index is process-local, but the wiki directory is
        # shared across processes: in the standard runtime the Celery ingest
        # worker writes pages while the app serves search. Re-index only the
        # delta since the last sync — otherwise the app stays blind to every
        # document ingested after it started (or burns the whole corpus
        # re-indexing on every search during ingest).
        current = self._collect_page_mtimes()
        if current == self._page_mtimes:
            return
        removed = set(self._page_mtimes) - set(current)
        for path_str in removed:
            self._remove_page(path_str)
        for path_str, mtime in current.items():
            if self._page_mtimes.get(path_str) != mtime:
                self._index_page(Path(path_str))
        self._page_mtimes = current

    def refresh(self) -> None:
        """Full rebuild (startup, worker-side post-ingest verification)."""
        with self._lock:
            self.index.clear()
            self.documents.clear()
            self._page_mtimes.clear()
            self._path_doc_ids.clear()
            self._kb_page_counts.clear()
            if not self.root_dir.exists():
                return
            for path_str in sorted(self._collect_page_mtimes()):
                self._index_page(Path(path_str))
            self._page_mtimes = self._collect_page_mtimes()

    def stats(self) -> dict[str, object]:
        """Storage summary for the debug endpoints."""
        with self._lock:
            self._sync_if_stale()
            kinds: dict[str, int] = {}
            for document in self.documents.values():
                kinds[document.kind] = kinds.get(document.kind, 0) + 1
            return {
                "backend": "wiki-bm25",
                "document_count": len(self.documents),
                "source_pages": kinds.get("source", 0),
                "topic_pages": kinds.get("topic", 0),
                "index_pages": kinds.get("index", 0),
                "max_documents": self.index.config.max_documents,
                "max_pages_per_kb": self._max_pages_per_kb,
            }

    def search(
        self,
        query: str,
        top_k: int,
        kb_id: str = "default",
        metadata_filters: dict[str, object] | None = None,
    ) -> list[SearchHit]:
        with self._lock:
            self._sync_if_stale()
            scope_id = WikiCompiler.scope_id(kb_id=kb_id)
            allowed_doc_ids = {
                doc_id
                for doc_id, document in self.documents.items()
                if document.kb_id == kb_id
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
                                "wiki_kind": document.kind,
                                "wiki_status": document.metadata.get("status"),
                                "original_source_path": document.source_path,
                                "doc_type": document.metadata.get("doc_type"),
                                "source_key": document.metadata.get("source_key"),
                                "effective_date": document.metadata.get("effective_date"),
                                "version": document.metadata.get("version"),
                                "is_latest_version": document.metadata.get("is_latest_version"),
                                "superseded_by": document.metadata.get("superseded_by"),
                            },
                        ),
                        score=score,
                    )
                )
            return hits

    # ------------------------------------------------------------------ #
    # Per-page indexing                                                   #
    # ------------------------------------------------------------------ #

    def _index_page(self, path: Path) -> None:
        if path.parent == self.sources_dir:
            self._index_source_page(path)
        elif path.parent == self.topics_dir:
            self._index_topic_page(path)
        elif path.parent == self.indexes_dir:
            self._index_scoped_index_page(path)

    def _remove_page(self, path_str: str) -> None:
        doc_id = self._path_doc_ids.pop(path_str, None)
        if doc_id is None:
            return
        document = self.documents.pop(doc_id, None)
        if document is not None and document.kind == "source":
            self._kb_page_counts[document.kb_id] = (
                self._kb_page_counts.get(document.kb_id, 1) - 1
            )
            if self._kb_page_counts[document.kb_id] <= 0:
                self._kb_page_counts.pop(document.kb_id, None)
        self.index.remove_document(doc_id)

    def _register_document(self, path: Path, doc_id: str, document: WikiDocument) -> None:
        # Re-indexing the same page replaces the previous entry.
        previous_doc_id = self._path_doc_ids.get(str(path))
        if previous_doc_id is not None and previous_doc_id != doc_id:
            self._remove_page(str(path))
        # The per-KB quota counts corpus size (source pages); topic/index
        # aggregates scale with headings and KB count, not with documents.
        if doc_id not in self.documents and document.kind == "source":
            count = self._kb_page_counts.get(document.kb_id, 0) + 1
            if count > self._max_pages_per_kb:
                raise RetrievalError(
                    f"wiki page quota for kb_id {document.kb_id!r} reached "
                    f"({self._max_pages_per_kb} source pages); cannot index "
                    f"{path.name}. Raise RAG_WIKI_MAX_PAGES_PER_KB instead of "
                    "silently dropping pages."
                )
            self._kb_page_counts[document.kb_id] = count
        self._path_doc_ids[str(path)] = doc_id
        self.documents[doc_id] = document
        self.index.add_document(doc_id, document.body)

    def _index_source_page(self, page_path: Path) -> None:
        metadata, body = WikiCompiler.read_frontmatter(page_path)
        if not metadata:
            raise RetrievalError(
                f"wiki source page {page_path} has unreadable frontmatter; "
                "the page is corrupt and must be recompiled"
            )
        kb_id = str(metadata.get("kb_id", "default"))
        doc_id = f"wiki:source:{metadata.get('doc_id', page_path.stem)}"
        self._register_document(
            page_path,
            doc_id,
            WikiDocument(
                doc_id=str(metadata.get("doc_id", page_path.stem)),
                title=str(metadata.get("title", page_path.stem)),
                source_path=str(metadata.get("source_path", "")),
                kb_id=kb_id,
                page_path=str(page_path.relative_to(self.root_dir.parent)),
                body=body,
                kind="source",
                metadata=metadata,
            ),
        )

    def _index_topic_page(self, page_path: Path) -> None:
        metadata, body = WikiCompiler.read_frontmatter(page_path)
        kb_id = str(metadata.get("kb_id", "default"))
        topic_name = str(metadata.get("topic", page_path.stem))
        doc_id = f"wiki:topic:{page_path.stem}"
        self._register_document(
            page_path,
            doc_id,
            WikiDocument(
                doc_id=page_path.stem,
                title=topic_name,
                source_path=str(page_path.relative_to(self.root_dir.parent)),
                kb_id=kb_id,
                page_path=str(page_path.relative_to(self.root_dir.parent)),
                body=body,
                kind="topic",
                metadata=metadata,
            ),
        )

    def _index_scoped_index_page(self, index_path: Path) -> None:
        metadata, body = WikiCompiler.read_frontmatter(index_path)
        kb_id = metadata.get("kb_id")
        if not kb_id:
            # Legacy pre-frontmatter scoped index page: it cannot be
            # attributed to a KB. Skip it (conservative — it is a compiler
            # aggregate that the next ingest rewrites) rather than guess an
            # attribution and risk cross-KB leakage.
            logger.warning(
                "skipping legacy scoped index page without kb_id frontmatter: %s",
                index_path,
            )
            return
        scope_id = index_path.stem
        doc_id = f"wiki:index:{scope_id}"
        self._register_document(
            index_path,
            doc_id,
            WikiDocument(
                doc_id=doc_id,
                title=f"Wiki Index {scope_id}",
                source_path=str(index_path.relative_to(self.root_dir.parent)),
                kb_id=str(kb_id),
                page_path=str(index_path.relative_to(self.root_dir.parent)),
                body=body,
                kind="index",
                metadata={},
            ),
        )

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
