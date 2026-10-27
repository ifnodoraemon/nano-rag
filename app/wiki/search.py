from __future__ import annotations

import logging
import os
import threading
from dataclasses import replace
import time
from dataclasses import dataclass
from pathlib import Path

from app.agentic.scope import ScopePreference, document_matches
from app.core.exceptions import ConfigurationError, RetrievalError
from app.retrieval.bm25 import BM25Config, BM25Index
from app.retrieval.filters import match_metadata_filters
from app.retrieval.title_weight import resolve_title_weight
from app.schemas.chunk import Chunk
from app.retrieval.hits import SearchHit
from app.retrieval.tokenizer import resolve_tokenizer_mode
from app.wiki.compiler import WikiCompiler

logger = logging.getLogger(__name__)

# How much page text a hit carries. The read plan reads this as the document's
# manifest, so a page fed to the planner must not be cut mid-sentence: the clause that
# matters is often the one after the truncation point.
MAX_WIKI_CONTEXT_CHARS = 2200
# Capacity of the in-process BM25 index, in PAGES (a source page plus one page per
# article). Measured on 595 Chinese statutes: 24,705 pages, 363 MB resident, ~147 MB per
# 10,000 pages. Article-level indexing multiplies page count by roughly 40x, so the
# earlier default of 50,000 was exhausted by about 1,200 statutes - ingest then failed
# loudly (by design: capacity never drops documents silently), but at a size ordinary
# corpora reach. The default is now set from measured memory rather than guessed, so a
# deployment hitting it has a real memory problem rather than a stale constant.
DEFAULT_MAX_DOCUMENTS = 200_000
# Per-KB source page quota. Unrelated to memory: it counts SOURCE pages to bound one
# tenant's share of the index.
DEFAULT_MAX_PAGES_PER_KB = 20_000


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
        self.articles_dir = self.root_dir / "articles"
        self.topics_dir = self.root_dir / "topics"
        self.indexes_dir = self.root_dir / "indexes"
        # None means "let the corpus decide"; an explicit value is an operator override.
        self._title_weight_override = self._title_weight_override_value()
        self._title_diversity: float | None = None
        self.index = BM25Index(
            BM25Config(
                max_documents=self._max_documents(),
                tokenizer=self._tokenizer_mode(),
                k1=self._k1_config(),
                b=self._b_config(),
                # Placeholder until pages are indexed; _reweight_title() derives the
                # real value from the titles that actually exist.
                title_weight=self._title_weight_override or 0.0,
                variant=self._scoring_variant_config(),
            )
        )
        self.documents: dict[str, WikiDocument] = {}
        self._page_mtimes: dict[str, int] = {}
        self._path_doc_ids: dict[str, str] = {}
        # Page id -> the provenance to report on a hit for that page. For an article
        # page this is its PARENT document's frontmatter: versioning, doc_type and
        # source_key describe the document, not the one article the page holds.
        self._document_provenance: dict[str, dict[str, object]] = {}
        self._kb_page_counts: dict[str, int] = {}
        self._kb_doc_ids: dict[str, set[str]] = {}
        self._max_pages_per_kb = self._max_pages_per_kb_config()
        # Detecting on-disk changes needs one stat() per page, so a per-query
        # scan costs ~7.5us/page (measured): at the 50k-page capacity that is
        # ~370ms of pure syscall overhead per search, serialized behind the
        # index lock. Coalesce the scan to at most once per window; hits within
        # the window serve the already-synced index. Same-process ingest calls
        # sync() explicitly, so freshness is not delayed there.
        self._sync_interval_seconds = self._sync_interval_config()
        self._last_sync_monotonic: float | None = None
        self._last_dir_signature: tuple[tuple[str, int], ...] | None = None
        self._lock = threading.RLock()
        self.refresh()

    @staticmethod
    def _sync_interval_config() -> float:
        raw = os.getenv("RAG_WIKI_SYNC_INTERVAL_SECONDS", "1.0")
        try:
            value = float(raw)
        except ValueError as exc:
            raise RetrievalError(
                f"RAG_WIKI_SYNC_INTERVAL_SECONDS must be a number, got {raw!r}: {exc}"
            ) from exc
        if value < 0:
            raise RetrievalError(
                f"RAG_WIKI_SYNC_INTERVAL_SECONDS must be >= 0, got {raw!r}"
            )
        return value

    @staticmethod
    def _max_documents() -> int:
        try:
            return max(1, int(os.getenv("RAG_BM25_MAX_DOCUMENTS", str(DEFAULT_MAX_DOCUMENTS))))
        except ValueError as exc:
            raise RetrievalError(
                f"RAG_BM25_MAX_DOCUMENTS must be an integer: {exc}"
            ) from exc

    @staticmethod
    def _tokenizer_mode() -> str:
        return resolve_tokenizer_mode(os.getenv("RAG_BM25_TOKENIZER"))

    @staticmethod
    def _scoring_variant_config() -> str:
        """Scoring formula: ``bm25`` (default) or ``bm25x``.

        ``bm25x`` adds the entropy weight and query/document overlap terms from
        arXiv:2408.06643; both are lexical, so this stays embedding-free. Unknown
        values fail loudly rather than silently falling back to BM25.
        """
        raw = (os.getenv("RAG_BM25_VARIANT") or "bm25").strip().lower()
        if raw not in {"bm25", "bm25x"}:
            raise ConfigurationError(
                f"RAG_BM25_VARIANT must be 'bm25' or 'bm25x', got {raw!r}"
            )
        return raw

    @staticmethod
    def _title_weight_override_value() -> float | None:
        """Operator override for the title weight, or None to let the corpus decide."""
        raw = os.getenv("RAG_BM25_TITLE_WEIGHT")
        if raw is None or not raw.strip():
            return None
        value = _float_env("RAG_BM25_TITLE_WEIGHT", 0.0, minimum=0.0)
        return value

    @staticmethod
    def _k1_config() -> float:
        """BM25 term-frequency saturation.

        Exposed because it is a relevance lever that had never been measured;
        see scripts/sweep_bm25_params.py for the sweep that picks the default.
        """
        return _float_env("RAG_BM25_K1", 1.5, minimum=0.0)

    @staticmethod
    def _b_config() -> float:
        """BM25 length normalisation in [0, 1].

        0 ignores document length, 1 normalises fully. Documents of uneven
        length otherwise lose to short ones purely through score dilution.
        """
        return _float_env("RAG_BM25_B", 0.75, minimum=0.0, maximum=1.0)

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
        edits, and removals — mtime alone would miss deletions.

        Only ``*.md`` files that are not hidden are pages. Anything else in
        these directories (staging files from an interrupted atomic write,
        editor swap files, ``.DS_Store``) is not a page and must not be
        validated as one: a leftover staging file used to raise "unreadable
        frontmatter" on every search and on every process start.
        """
        snapshot: dict[str, int] = {}
        for directory in (
            self.sources_dir,
            self.articles_dir,
            self.topics_dir,
            self.indexes_dir,
        ):
            if not directory.exists():
                continue
            for path in directory.iterdir():
                if not path.is_file():
                    continue
                if path.suffix != ".md" or path.name.startswith("."):
                    continue
                try:
                    snapshot[str(path)] = path.stat().st_mtime_ns
                except OSError:
                    continue
        return snapshot

    def _dir_signature(self) -> tuple[tuple[str, int], ...]:
        """Cheap (3 stat) signal that pages were created or deleted.

        Directory mtime changes on create/unlink, so an ingest in another
        process is visible on the next search without stat-ing every page.
        Content-only overwrites keep the directory mtime; those are caught by
        the sync window (or by an explicit sync() from the writing process).
        """
        signature: list[tuple[str, int]] = []
        for directory in (
            self.sources_dir,
            self.articles_dir,
            self.topics_dir,
            self.indexes_dir,
        ):
            try:
                signature.append((str(directory), directory.stat().st_mtime_ns))
            except OSError:
                signature.append((str(directory), -1))
        return tuple(signature)

    def _pages_touched_on_disk(self) -> bool:
        return self._dir_signature() != self._last_dir_signature

    def _sync_if_stale(self, *, force: bool = False) -> None:
        # The in-memory BM25 index is process-local, but the wiki directory is
        # shared across processes: in the standard runtime the Celery ingest
        # worker writes pages while the app serves search. Re-index only the
        # delta since the last sync — otherwise the app stays blind to every
        # document ingested after it started (or burns the whole corpus
        # re-indexing on every search during ingest).
        if not force and not self._pages_touched_on_disk():
            last = self._last_sync_monotonic
            if last is not None and (time.monotonic() - last) < self._sync_interval_seconds:
                return
        self._last_sync_monotonic = time.monotonic()
        self._last_dir_signature = self._dir_signature()
        current = self._collect_page_mtimes()
        if current == self._page_mtimes:
            # Nothing changed, but the weight may still be the placeholder from
            # construction (e.g. a second process starting against an unchanged wiki).
            self._reweight_title()
            return
        removed = set(self._page_mtimes) - set(current)
        for path_str in removed:
            self._remove_page(path_str)
        for path_str, mtime in current.items():
            if self._page_mtimes.get(path_str) != mtime:
                self._index_page(Path(path_str))
        self._page_mtimes = current
        self._reweight_title()

    def _reweight_title(self) -> None:
        """Set the title weight from the corpus that was actually indexed.

        A constant encodes a guess about the corpus: a title-weighted field helps where
        titles are distinctive (a Chinese regulation is named after its subject) and hurts
        where they are taxonomy labels shared by many documents (US Code groups) - measured
        +2.6pt and -2.3..-5.3pt respectively. The distinguishing statistic is computed from
        the indexed titles themselves, so no per-corpus configuration is needed.

        An explicit RAG_BM25_TITLE_WEIGHT always wins; this is a default, not an authority.
        """
        if self._title_weight_override is not None:
            return
        titles = [document.title for document in self.documents.values()]
        weight, diversity = resolve_title_weight(titles)
        if weight != self.index.config.title_weight:
            # BM25Config is a frozen-ish dataclass; replace rather than mutate so the
            # weight change is visible in the config the index reports.
            self.index.config = replace(self.index.config, title_weight=weight)
        self._title_diversity = diversity

    def sync(self, *, force: bool = True) -> None:
        """Incrementally re-index on-disk page changes.

        Called by the ingest path after writing pages (same process) so a
        freshly ingested document is discoverable immediately instead of
        waiting out the sync window.
        """
        with self._lock:
            self._sync_if_stale(force=force)


    def refresh(self) -> None:
        """Full rebuild (startup, worker-side post-ingest verification)."""
        with self._lock:
            self.index.clear()
            self.documents.clear()
            self._page_mtimes.clear()
            self._path_doc_ids.clear()
            self._document_provenance.clear()
            self._kb_page_counts.clear()
            self._kb_doc_ids.clear()
            if not self.root_dir.exists():
                return
            # Record the SAME snapshot that was indexed. Re-collecting after
            # the loop would mark a page created in between as "already
            # indexed" and hide it from every future sync until it changed
            # again — a silently unretrievable document.
            snapshot = self._collect_page_mtimes()
            for path_str in sorted(snapshot):
                self._index_page(Path(path_str))
            self._page_mtimes = snapshot
            # A full rebuild must derive the title weight too: this is the path that
            # runs at startup, so skipping it would leave every process on the
            # constructor's placeholder until the first incremental sync.
            self._reweight_title()
            self._last_sync_monotonic = time.monotonic()
            self._last_dir_signature = self._dir_signature()

    def stats(self) -> dict[str, object]:
        """Storage summary for the debug endpoints."""
        with self._lock:
            # Ops views must reflect disk now, not the coalesced view.
            self._sync_if_stale(force=True)
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
                "tokenizer": self.index.config.tokenizer,
            }

    def tokenize(self, text: str) -> list[str]:
        """The index's own tokenizer, so generated text uses the same terms."""
        return self.index._tokenize(text)

    def document_frequencies(self) -> tuple[dict[str, int], int]:
        """``(term -> document frequency, document count)`` for Rocchio's cutoff."""
        return self.index.document_frequencies()

    def search(
        self,
        query: str,
        top_k: int,
        kb_id: str = "default",
        metadata_filters: dict[str, object] | None = None,
        *,
        term_weights: dict[str, float] | None = None,
        scope_preference: ScopePreference | None = None,
        scope_verdict: str | None = None,
        granularity: str = "page",
    ) -> list[SearchHit]:
        """Search the wiki.

        ``term_weights`` enables weighted (Rocchio/HyDE) queries. ``scope_preference``
        multiplies a hit's score according to whether the document's recorded scope
        matches the question's scope; ``None`` means no preference. Multipliers rather
        than filters, on purpose: a question about the general rule must not make a
        regional rule unfindable.

        ``granularity`` decides what the ranked list is a ranking OF:

        - ``page`` (default): the retrieval unit is the indexed page, so a long statute
          can appear several times - once per article page, plus its source page. That
          is the right list when one match means one candidate to read.
        - ``document``: ranks each document once, by its best-matching page, and
          reports that page as the hit. A document that matched at all therefore takes
          exactly one slot, instead of crowding its own neighbours out with its
          siblings (measured: repeated documents consumed 2-3 of the first twelve
          candidates for the obvious queries and for the hard ones alike).
        """
        with self._lock:
            self._sync_if_stale()
            scope_id = WikiCompiler.scope_id(kb_id=kb_id)
            index_doc_id = f"wiki:index:{scope_id}"
            # Per-KB doc id sets are maintained incrementally: rebuilding the
            # allowed set by scanning every document made every search O(corpus)
            # even when the index itself was untouched.
            kb_doc_ids: set[str] | frozenset[str] = self._kb_doc_ids.get(
                kb_id, frozenset()
            )
            allowed_doc_ids: set[str] | frozenset[str]
            if metadata_filters:
                allowed_doc_ids = {
                    doc_id
                    for doc_id in kb_doc_ids
                    if (document := self.documents.get(doc_id)) is not None
                    and match_metadata_filters(document.metadata, metadata_filters)
                    and (
                        document.kind in {"topic", "source"}
                        or doc_id == index_doc_id
                    )
                }
            else:
                allowed_doc_ids = kb_doc_ids
            if not allowed_doc_ids:
                return []
            ranking = self._rank_pages(
                query,
                allowed_doc_ids=allowed_doc_ids,
                top_k=top_k,
                term_weights=term_weights,
                scope_preference=scope_preference,
                scope_verdict=scope_verdict,
            )
            if granularity == "document":
                ranking = self._one_page_per_document(ranking)
            return [
                self._hit_for_page(page_id, score)
                for page_id, score in ranking[:top_k]
                if self.documents.get(page_id) is not None
            ]

    def _rank_pages(
        self,
        query: str,
        *,
        allowed_doc_ids: set[str] | frozenset[str],
        top_k: int,
        term_weights: dict[str, float] | None,
        scope_preference: ScopePreference | None,
        scope_verdict: str | None,
    ) -> list[tuple[str, float]]:
        """The page ranking, with the scope preference and page-kind bonus applied.

        The index is asked for more than ``top_k`` pages because collapse and the
        deterministic version filter run downstream: ranking exactly ``top_k`` pages
        and then dropping duplicates would return fewer documents than requested.
        """
        results = self.index.search_weighted(
            query,
            top_k=max(top_k * 4, 8),
            allowed_doc_ids=allowed_doc_ids,
            term_weights=term_weights,
        )

        def level_factor(doc_id: str) -> float:
            if scope_preference is None or scope_preference.is_neutral:
                return 1.0
            document = self.documents.get(doc_id)
            if document is None:
                return 1.0
            matched = document_matches(document.metadata, scope_verdict or "")
            return (
                scope_preference.matches
                if matched
                else scope_preference.others
            )

        return sorted(
            results,
            key=lambda item: (
                (item[1] + self._kind_bonus(self.documents.get(item[0])))
                * level_factor(item[0]),
                item[1],
            ),
            reverse=True,
        )

    def _one_page_per_document(
        self, ranking: list[tuple[str, float]]
    ) -> list[tuple[str, float]]:
        """Collapse a page ranking to one entry per document, best page first.

        Documents are keyed on ``(kb_id, doc_id)`` rather than on the page id, because
        that is the identity the deep read needs: an article page and a source page of
        the same document are two ways of finding one thing to read.
        """
        seen: set[tuple[str, str]] = set()
        collapsed: list[tuple[str, float]] = []
        for page_id, score in ranking:
            document = self.documents.get(page_id)
            if document is None:
                continue
            key = (document.kb_id, document.doc_id)
            if key in seen:
                continue
            seen.add(key)
            collapsed.append((page_id, score))
        return collapsed

    def _hit_for_page(self, page_id: str, score: float) -> SearchHit:
        """Turn an indexed page into the hit the agentic pipeline consumes."""
        document = self.documents[page_id]
        # Generation metadata describes the DOCUMENT; this page may be one article of
        # it (``document.metadata`` is the article page's own frontmatter there).
        provenance = self._document_provenance.get(page_id, document.metadata)
        return SearchHit(
            chunk=Chunk(
                chunk_id=page_id,
                doc_id=document.doc_id,
                chunk_index=0,
                text=self._limit_body(document.body),
                source_path=document.page_path,
                title=document.title,
                metadata={
                    "kb_id": document.kb_id,
                    "wiki_kind": document.kind,
                    "wiki_status": provenance.get("status"),
                    "wiki_page_path": document.page_path,
                    "original_source_path": document.source_path,
                    "doc_type": provenance.get("doc_type"),
                    "source_key": provenance.get("source_key"),
                    "effective_date": provenance.get("effective_date"),
                    "version": provenance.get("version"),
                    "is_latest_version": provenance.get("is_latest_version"),
                    "superseded_by": provenance.get("superseded_by"),
                },
            ),
            score=score,
        )

    # ------------------------------------------------------------------ #
    # Per-page indexing                                                   #
    # ------------------------------------------------------------------ #

    def _index_page(self, path: Path) -> None:
        if path.parent == self.sources_dir:
            self._index_source_page(path)
        elif path.parent == self.articles_dir:
            self._index_article_page(path)
        elif path.parent == self.topics_dir:
            self._index_topic_page(path)
        elif path.parent == self.indexes_dir:
            self._index_scoped_index_page(path)

    def _remove_page(self, path_str: str) -> None:
        doc_id = self._path_doc_ids.pop(path_str, None)
        if doc_id is None:
            return
        self._document_provenance.pop(doc_id, None)
        document = self.documents.pop(doc_id, None)
        if document is not None:
            kb_docs = self._kb_doc_ids.get(document.kb_id)
            if kb_docs is not None:
                kb_docs.discard(doc_id)
                if not kb_docs:
                    self._kb_doc_ids.pop(document.kb_id, None)
        if document is not None and document.kind == "source":
            self._kb_page_counts[document.kb_id] = (
                self._kb_page_counts.get(document.kb_id, 1) - 1
            )
            if self._kb_page_counts[document.kb_id] <= 0:
                self._kb_page_counts.pop(document.kb_id, None)
        self.index.remove_document(doc_id)

    def _register_document(
        self,
        path: Path,
        doc_id: str,
        document: WikiDocument,
        *,
        provenance: dict[str, object] | None = None,
    ) -> None:
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
        self._document_provenance[doc_id] = (
            dict(provenance) if provenance is not None else document.metadata
        )
        self._kb_doc_ids.setdefault(document.kb_id, set()).add(doc_id)
        # The page title carries the document's own summary, which is a much
        # denser signal per token than body text; it is indexed as a separate
        # field so it can be weighted without duplicating it into the body.
        self.index.add_document(doc_id, document.body, document.title)

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
        self._adopt_pending_articles(metadata)

    def _adopt_pending_articles(self, metadata: dict[str, object]) -> None:
        """Give already-indexed article pages of this document its provenance.

        Pages are indexed in path order and ``articles/`` sorts before ``sources/``, so
        in a full refresh every article page is seen before the source page that
        describes it. The article page cannot invent the document's versioning, and the
        document must not lose it, so the source page fills it in the moment it arrives.
        Without this an article hit reported no ``source_key``/``is_latest_version`` and
        slipped past the deterministic version filter.
        """
        parent_doc_id = str(metadata.get("doc_id", ""))
        if not parent_doc_id:
            return
        prefix = f"wiki:article:{parent_doc_id}:"
        for page_id in self.documents:
            if page_id.startswith(prefix):
                self._document_provenance[page_id] = metadata

    def _index_article_page(self, page_path: Path) -> None:
        """Index one article of a long document as its own retrieval unit.

        The doc_id is the PARENT document's, prefixed so BM25 keeps the two entries
        apart. Callers that resolve hits to documents read ``metadata['doc_id']``, so an
        article hit still points at the document the read plan needs.
        """
        metadata, body = WikiCompiler.read_frontmatter(page_path)
        if not metadata:
            raise RetrievalError(
                f"wiki article page {page_path} has unreadable frontmatter; "
                "the page is corrupt and must be recompiled"
            )
        parent_doc_id = str(metadata.get("doc_id", ""))
        if not parent_doc_id:
            raise RetrievalError(
                f"wiki article page {page_path} has no parent doc_id; the page is corrupt"
            )
        kb_id = str(metadata.get("kb_id", "default"))
        label = str(metadata.get("article_label", page_path.stem))
        self._register_document(
            page_path,
            f"wiki:article:{parent_doc_id}:{metadata.get('article_index', 0)}",
            WikiDocument(
                doc_id=parent_doc_id,
                title=f"{metadata.get('title', '')}",
                source_path=str(metadata.get("source_path", "")),
                kb_id=kb_id,
                page_path=str(page_path.relative_to(self.root_dir.parent)),
                body=body,
                kind="article",
                metadata={**metadata, "wiki_kind": "article", "article_label": label},
            ),
            provenance=self._parent_provenance(kb_id, parent_doc_id, metadata),
        )

    def _parent_provenance(
        self, kb_id: str, parent_doc_id: str, article_metadata: dict[str, object]
    ) -> dict[str, object]:
        """The parent document's frontmatter, if its source page is already indexed.

        An article page's own frontmatter carries the label and index, not the
        document's versioning (``source_key``, ``effective_date``, ``is_latest_version``).
        Those describe the DOCUMENT, and an article hit that reported them as absent
        would slip past the deterministic version filter. Pages are indexed in path
        order, so an article page can arrive before its source page: it then falls back
        to its own frontmatter, and nothing is invented - a field the article page does
        not carry stays absent here.
        """
        source_page = self.documents.get(f"wiki:source:{parent_doc_id}")
        if source_page is not None and source_page.kb_id == kb_id:
            return source_page.metadata
        return article_metadata

    def _index_topic_page(self, page_path: Path) -> None:
        metadata, body = WikiCompiler.read_frontmatter(page_path)
        if not metadata:
            # Same rule as source pages: a topic page whose frontmatter cannot
            # be read is corrupt. Indexing it with empty metadata would silently
            # attribute it to kb_id "default" and leak it into that scope.
            raise RetrievalError(
                f"wiki topic page {page_path} has unreadable frontmatter; "
                "the page is corrupt and must be recompiled"
            )
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


def _float_env(name: str, default: float, *, minimum: float, maximum: float | None = None) -> float:
    """Read a float from the environment, failing loudly on a bad value."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number, got {raw!r}") from exc
    if value < minimum or (maximum is not None and value > maximum):
        bound = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise ConfigurationError(f"{name} must be {bound}, got {value}")
    return value
