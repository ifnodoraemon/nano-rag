from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator
from uuid import uuid4

import yaml
from pydantic import ValidationError

from app.core.exceptions import ParsingError, RetrievalError
from app.retrieval.versioning import version_sort_key
from app.utils.coerce import as_int, as_list, as_str, as_str_list
from app.schemas.chunk import Chunk
from app.schemas.document import Document

MAX_SUMMARY_CHARS = 600
MAX_CHUNK_PREVIEW_CHARS = 180
MAX_INDEX_SUMMARY_CHARS = 120
MAX_TOPIC_ENTRIES = 8
MAX_TOPIC_FACTS = 8

FRONTMATTER_BOUNDARY = "---"
LEDGER_FILENAME = "versions.json"
LEDGER_FORMAT = 1


class WikiCompiler:
    """Compiles parsed artifacts into the wiki discovery layer.

    Concurrency and durability contract (32 ingest workers share this
    directory through a Docker volume):

    - every public mutation takes a cross-process ``flock`` on
      ``<root>/.lock`` — concurrent writers serialize instead of producing
      lost updates or torn files;
    - every file write is atomic (temp file + ``os.replace``), so a crash
      mid-write can never leave a half-written page that a reader would
      otherwise silently parse as ``{}`` and attribute to the wrong KB;
    - version-chain bookkeeping lives in a ledger JSON that is updated in
      O(affected documents) per ingest instead of rewriting every page (the
      old full rewrite also churned mtimes, which forced the query-side
      searcher into a full re-index on every search during ingest);
    - corrupt inputs raise — a bootstrap artifact or wiki page that fails to
      parse is a data-integrity failure, never silently skipped.
    """

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.sources_dir = self.root_dir / "sources"
        self.topics_dir = self.root_dir / "topics"
        self.indexes_dir = self.root_dir / "indexes"
        # One page per article of a long document. The document page is a summary
        # (~1,049 of 5,446 characters on the measured corpus); without these pages the
        # remaining 80% of every statute is invisible to the lexical index.
        self.articles_dir = self.root_dir / "articles"
        self._ensure_structure()
        self.cleanup_stale_staging()
        self._cleanup_legacy_staging()

    # ------------------------------------------------------------------ #
    # Public API (all take the cross-process lock)                        #
    # ------------------------------------------------------------------ #

    def upsert_document(
        self, document: Document, chunks: list[Chunk], *, record_log: bool = True
    ) -> Path:
        self.upsert_documents([(document, chunks)], record_log=record_log)
        return self.sources_dir / f"{document.doc_id}.md"

    def upsert_documents(
        self, items: list[tuple[Document, list[Chunk]]], *, record_log: bool = True
    ) -> None:
        """Compile a batch of documents with one lock acquisition.

        The page writes are O(1) per document; the version ledger is updated
        incrementally (only affected source_key groups); the index/topic
        aggregates are rebuilt once per batch — not once per file.
        """
        if not items:
            return
        self._ensure_structure()
        with self._cross_process_lock():
            written: list[tuple[Document, dict[str, object]]] = []
            ledger = self._load_or_rebuild_ledger()
            for document, chunks in items:
                metadata = self._write_source_page(document, chunks)
                self._ledger_upsert(ledger, document, metadata)
                written.append((document, metadata))
            self._apply_version_chain(ledger)
            self._write_index()
            self._save_ledger(ledger)
            if record_log:
                for document, metadata in written:
                    self._append_log("ingest", document, str(metadata["updated_at"]))

    def remove_document(self, doc_id: str) -> None:
        self._ensure_structure()
        with self._cross_process_lock():
            (self.sources_dir / f"{doc_id}.md").unlink(missing_ok=True)
            ledger = self._load_or_rebuild_ledger()
            self._ledger_remove(ledger, doc_id)
            self._apply_version_chain(ledger)
            self._write_index()
            self._save_ledger(ledger)
            self._append_log("delete", None, doc_id=doc_id)

    def bootstrap_from_parsed_dir(self, parsed_dir: Path) -> int:
        """Rebuild wiki pages from committed parsed artifacts on startup.

        The wiki directory is persisted (named volume) but can be empty after
        a fresh container or a rebuild; parsed artifacts are the durable source
        of truth. Corrupt artifacts raise ParsingError — a torn or invalid
        artifact must be fixed, not silently skipped.
        Returns the number of source pages (re)written.
        """
        self._ensure_structure()
        if not parsed_dir.exists():
            return 0
        parsed: list[tuple[Document, list[Chunk]]] = []
        corrupt: list[str] = []
        for artifact in sorted(parsed_dir.glob("*.json")):
            try:
                payload = json.loads(artifact.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                corrupt.append(f"{artifact.name}: {exc}")
                continue
            if not isinstance(payload, dict):
                corrupt.append(f"{artifact.name}: not a JSON object")
                continue
            try:
                document = Document.model_validate(payload.get("document", {}))
                raw_chunks = payload.get("chunks", [])
                chunks = (
                    [Chunk.model_validate(raw) for raw in raw_chunks if isinstance(raw, dict)]
                    if isinstance(raw_chunks, list)
                    else []
                )
            except ValidationError as exc:
                corrupt.append(f"{artifact.name}: {exc}")
                continue
            parsed.append((document, chunks))
        if corrupt:
            raise ParsingError(
                "bootstrap found corrupt parsed artifacts (data integrity "
                "failure, not skipped): " + "; ".join(corrupt)
            )
        if not parsed:
            return 0
        with self._cross_process_lock():
            for document, chunks in parsed:
                # Startup rebuild: keep identical pages byte-for-byte
                # unchanged. Rewriting them would churn mtimes, which is
                # exactly what forces the query-side searcher into a full
                # re-index on every container start.
                self._write_source_page(document, chunks, write_if_unchanged=True)
            ledger = self._rebuild_ledger_from_pages()
            self._apply_version_chain(ledger)
            self._write_index()
            self._save_ledger(ledger)
        return len(parsed)

    # ------------------------------------------------------------------ #
    # Cross-process locking and atomic writes                             #
    # ------------------------------------------------------------------ #

    @contextmanager
    def _cross_process_lock(self) -> Generator[None, None, None]:
        lock_path = self.root_dir / ".lock"
        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)

    def _tmp_dir(self) -> Path:
        """Staging directory for atomic writes.

        Deliberately OUTSIDE sources/topics/indexes: the query-side searcher
        indexes every file it finds in those directories, so a staging file
        left behind by a crash (worker SIGKILL, OOM, docker kill) used to be
        parsed as a page, raise "unreadable frontmatter", and take down both
        search and process startup until an operator deleted it by hand.
        """
        tmp_dir = self.root_dir / ".staging"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        return tmp_dir

    def _write_atomic(self, path: Path, content: str) -> None:
        tmp_path = self._tmp_dir() / f"{path.name}.{uuid4().hex}.tmp"
        try:
            tmp_path.write_text(content, encoding="utf-8")
            os.replace(str(tmp_path), str(path))
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    def _cleanup_legacy_staging(self) -> int:
        """Remove ``.*.tmp`` files left inside the page directories by builds
        that staged next to the target. They can never be pages (pages end in
        ``.md``), and the searcher no longer indexes them, but leaving them
        would leak whatever a crashed worker was writing."""
        removed = 0
        for directory in (self.sources_dir, self.topics_dir, self.indexes_dir):
            if not directory.exists():
                continue
            for path in directory.glob(".*.tmp"):
                try:
                    path.unlink(missing_ok=True)
                    removed += 1
                except OSError:
                    continue
        return removed

    def cleanup_stale_staging(self, *, max_age_seconds: float = 3600.0) -> int:
        """Drop abandoned staging files (a crash between write and rename).

        Returns the number of files removed. Staging is outside the indexed
        directories, so this is pure garbage collection — but leaving them to
        accumulate forever is still a leak.
        """
        tmp_dir = self.root_dir / ".staging"
        if not tmp_dir.exists():
            return 0
        cutoff = time.time() - max_age_seconds
        removed = 0
        for path in tmp_dir.glob("*.tmp"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink(missing_ok=True)
                    removed += 1
            except OSError:
                continue
        return removed

    # ------------------------------------------------------------------ #
    # Version ledger                                                      #
    # ------------------------------------------------------------------ #

    def _ledger_path(self) -> Path:
        return self.root_dir / LEDGER_FILENAME

    def _new_ledger(self) -> dict[str, object]:
        return {"format": LEDGER_FORMAT, "docs": {}, "groups": {}}

    def _load_or_rebuild_ledger(self) -> dict[str, object]:
        path = self._ledger_path()
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RetrievalError(
                    f"wiki version ledger {path} is corrupt: {exc}"
                ) from exc
            if (
                not isinstance(payload, dict)
                or payload.get("format") != LEDGER_FORMAT
                or not isinstance(payload.get("docs"), dict)
                or not isinstance(payload.get("groups"), dict)
            ):
                raise RetrievalError(
                    f"wiki version ledger {path} has an unexpected structure"
                )
            return payload
        return self._rebuild_ledger_from_pages()

    def _rebuild_ledger_from_pages(self) -> dict[str, object]:
        ledger = self._new_ledger()
        for page_path in sorted(self.sources_dir.glob("*.md")):
            metadata, _body = self._read_frontmatter(page_path)
            if not metadata:
                raise RetrievalError(
                    f"wiki source page {page_path} has unreadable frontmatter"
                )
            doc_id = str(metadata.get("doc_id") or page_path.stem)
            source_key = metadata.get("source_key")
            if not source_key:
                continue
            self._ledger_set_member(ledger, doc_id, str(source_key), metadata)
        return ledger

    def _group_key(self, kb_id: object, source_key: str) -> str:
        return f"{kb_id}\t{source_key}"

    def _ledger_set_member(
        self,
        ledger: dict[str, object],
        doc_id: str,
        source_key: str,
        metadata: dict[str, object],
    ) -> None:
        docs: dict[str, str] = ledger["docs"]  # type: ignore[assignment]
        groups: dict[str, dict[str, object]] = ledger["groups"]  # type: ignore[assignment]
        previous_group = docs.get(doc_id)
        if previous_group is not None and previous_group in groups:
            members: dict[str, dict[str, object]] = groups[previous_group].get("members", {})  # type: ignore[assignment]
            members.pop(doc_id, None)
            if not members:
                del groups[previous_group]
        key = self._group_key(metadata.get("kb_id", "default"), source_key)
        group = groups.setdefault(key, {"members": {}})
        members = group["members"]  # type: ignore[assignment]
        members[doc_id] = {
            "effective_date": metadata.get("effective_date"),
            "version": metadata.get("version"),
            "source_modified_at": metadata.get("source_modified_at"),
            "updated_at": metadata.get("updated_at"),
            "is_latest_version": metadata.get("is_latest_version", True),
            "superseded_by": metadata.get("superseded_by"),
        }
        docs[doc_id] = key

    def _ledger_upsert(
        self,
        ledger: dict[str, object],
        document: Document,
        metadata: dict[str, object],
    ) -> None:
        source_key = document.metadata.get("source_key")
        if not source_key:
            # Documents without a source_key have no version semantics; their
            # pages always claim is_latest_version=True by construction.
            return
        self._ledger_set_member(ledger, document.doc_id, str(source_key), metadata)

    def _ledger_remove(self, ledger: dict[str, object], doc_id: str) -> None:
        docs: dict[str, str] = ledger["docs"]  # type: ignore[assignment]
        groups: dict[str, dict[str, object]] = ledger["groups"]  # type: ignore[assignment]
        key = docs.pop(doc_id, None)
        if key is None or key not in groups:
            return
        members: dict[str, dict[str, object]] = groups[key].get("members", {})  # type: ignore[assignment]
        members.pop(doc_id, None)
        if not members:
            del groups[key]

    def _apply_version_chain(self, ledger: dict[str, object]) -> None:
        """Deterministically mark the latest version within each source_key
        group and rewrite only the pages whose ledger fields changed.

        The winner is chosen by (effective_date, version) ordering — the same
        rule as retrieval freshness — so the ledger never depends on LLM
        judgment. Non-winners get is_latest_version=false and superseded_by
        set to the winner's doc_id.
        """
        groups: dict[str, dict[str, object]] = ledger["groups"]  # type: ignore[assignment]
        for key, group in groups.items():
            members: dict[str, dict[str, object]] = group.get("members", {})  # type: ignore[assignment]
            ranked = sorted(
                members.items(),
                key=lambda item: version_sort_key(
                    item[1], score=self._chain_tiebreak(item[1])
                ),
                reverse=True,
            )
            if not ranked:
                continue
            winner_doc_id = ranked[0][0]
            for doc_id, member in ranked:
                desired_latest = doc_id == winner_doc_id
                desired_superseded = None if desired_latest else winner_doc_id
                if (
                    member.get("is_latest_version") == desired_latest
                    and member.get("superseded_by") == desired_superseded
                ):
                    continue
                member["is_latest_version"] = desired_latest
                member["superseded_by"] = desired_superseded
                self._rewrite_page_version_fields(doc_id, desired_latest, desired_superseded)

    def _rewrite_page_version_fields(
        self, doc_id: str, is_latest: bool, superseded_by: str | None
    ) -> None:
        page_path = self.sources_dir / f"{doc_id}.md"
        if not page_path.exists():
            return
        metadata, body = self._read_frontmatter(page_path)
        if not metadata:
            raise RetrievalError(
                f"wiki source page {page_path} has unreadable frontmatter"
            )
        metadata["is_latest_version"] = is_latest
        metadata["superseded_by"] = superseded_by
        self._write_page(page_path, metadata, body)

    def _save_ledger(self, ledger: dict[str, object]) -> None:
        self._write_atomic(
            self._ledger_path(),
            json.dumps(ledger, ensure_ascii=False, indent=2, sort_keys=True),
        )

    @staticmethod
    def _chain_tiebreak(metadata: dict[str, object]) -> tuple[float, str]:
        """Deterministic tie-break when date and version agree.

        Prefers the source file mtime (numeric or ISO-8601); falls back to the
        page's updated_at string.
        """
        modified = metadata.get("source_modified_at")
        modified_value = 0.0
        if isinstance(modified, (int, float)):
            modified_value = float(modified)
        elif isinstance(modified, str) and modified.strip():
            try:
                parsed = datetime.fromisoformat(modified.replace("Z", "+00:00"))
                modified_value = parsed.timestamp()
            except ValueError:
                modified_value = 0.0
        return (modified_value, str(metadata.get("updated_at") or ""))

    # ------------------------------------------------------------------ #
    # Page rendering                                                      #
    # ------------------------------------------------------------------ #

    def _ensure_structure(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.sources_dir.mkdir(parents=True, exist_ok=True)
        self.topics_dir.mkdir(parents=True, exist_ok=True)
        self.indexes_dir.mkdir(parents=True, exist_ok=True)
        self.articles_dir.mkdir(parents=True, exist_ok=True)
        schema_path = self.root_dir / "SCHEMA.md"
        if not schema_path.exists():
            schema_path.write_text(self._render_schema(), encoding="utf-8")
        index_path = self.root_dir / "index.md"
        if not index_path.exists():
            index_path.write_text("# Wiki Index\n\n", encoding="utf-8")
        log_path = self.root_dir / "log.md"
        if not log_path.exists():
            log_path.write_text("# Wiki Log\n\n", encoding="utf-8")

    def _build_page_metadata(
        self, document: Document, chunks: list[Chunk]
    ) -> dict[str, object]:
        headings = self._extract_headings(document.content)
        return {
            "doc_id": document.doc_id,
            "title": document.title,
            "source_path": document.source_path,
            "kb_id": document.metadata.get("kb_id", "default"),
            "doc_type": document.metadata.get("doc_type"),
            "source_key": document.metadata.get("source_key"),
            "effective_date": document.metadata.get("effective_date"),
            "version": document.metadata.get("version"),
            "owner": document.metadata.get("owner"),
            "department": document.metadata.get("department"),
            "headings": headings,
            "summary": self._extract_summary(document.content),
            "key_passages": [
                self._preview(chunk.text, MAX_CHUNK_PREVIEW_CHARS) for chunk in chunks[:8]
            ],
            "chunk_count": len(chunks),
            "content_hash": document.metadata.get("source_content_hash"),
            "source_modified_at": document.metadata.get("source_modified_at"),
            # Version-ledger fields, finalized by _apply_version_chain() so
            # that every page in a source_key group reflects the
            # deterministic pick.
            "is_latest_version": True,
            "superseded_by": None,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            # The questions this document answers, in a user's own words. Rendered into
            # the page body, which is what the BM25 discovery layer indexes.
            "synthetic_questions": as_str_list(
                document.metadata.get("synthetic_questions")
            ),
        }

    def _write_source_page(
        self,
        document: Document,
        chunks: list[Chunk],
        *,
        write_if_unchanged: bool = False,
    ) -> dict[str, object]:
        """Write one source page and return its metadata (no chain/index/log).

        ``write_if_unchanged`` (bootstrap only) preserves ``updated_at`` and
        skips the write when the rendered page is identical, so a restart does
        not churn mtimes for the whole corpus.
        """
        metadata = self._build_page_metadata(document, chunks)
        page_path = self.sources_dir / f"{document.doc_id}.md"
        if write_if_unchanged:
            existing, _body = self._read_frontmatter(page_path) if page_path.exists() else ({}, "")
            if (
                existing
                and existing.get("content_hash") == metadata.get("content_hash")
                and existing.get("source_key") == metadata.get("source_key")
                and existing.get("updated_at")
            ):
                metadata["updated_at"] = existing["updated_at"]
        body = self._render_source_body(document, chunks, metadata)
        if write_if_unchanged:
            self._write_page_unchanged(page_path, metadata, body)
        else:
            self._write_page(page_path, metadata, body)
        return metadata

    def _write_page_unchanged(
        self, path: Path, metadata: dict[str, object], body: str
    ) -> None:
        frontmatter = yaml.safe_dump(
            metadata,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        ).strip()
        content = "\n".join(
            [FRONTMATTER_BOUNDARY, frontmatter, FRONTMATTER_BOUNDARY, "", body, ""]
        )
        self._write_index_file(path, content)

    def _write_page(self, path: Path, metadata: dict[str, object], body: str) -> None:
        frontmatter = yaml.safe_dump(
            metadata,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        ).strip()
        self._write_atomic(
            path,
            "\n".join(
                [FRONTMATTER_BOUNDARY, frontmatter, FRONTMATTER_BOUNDARY, "", body, ""]
            ),
        )

    def _render_source_body(
        self, document: Document, chunks: list[Chunk], metadata: dict[str, object]
    ) -> str:
        headings = as_list(metadata.get("headings"))
        summary = str(metadata.get("summary", "")) or "No summary available."
        lines = [
            f"# {document.title}",
            "",
            "## Source",
            f"- doc_id: `{document.doc_id}`",
            f"- source_path: `{document.source_path}`",
            f"- kb_id: `{metadata['kb_id']}`",
            f"- chunk_count: `{metadata['chunk_count']}`",
            "",
            "## Summary",
            summary,
            "",
            "## Headings",
        ]
        if headings:
            lines.extend(f"- {heading}" for heading in headings)
        else:
            lines.append("- No explicit markdown headings detected.")
        lines.extend(["", "## Key Passages"])
        key_passages = as_list(metadata.get("key_passages"))
        if key_passages:
            for index, passage in enumerate(key_passages):
                chunk_id = chunks[index].chunk_id if index < len(chunks) else f"{document.doc_id}:{index}"
                lines.append(f"- `{chunk_id}` {passage}")
        else:
            lines.append("- No chunks available.")
        # The questions this document answers, in ordinary language. The BM25 discovery
        # layer indexes this body, so a user's everyday phrasing can match even when it
        # shares no vocabulary with the statute's own wording. Measured on Chinese
        # statutes: verbatim clause queries retrieve at hit@1 0.87, real user questions
        # at 0.12 - this is the section that closes part of that gap.
        questions = as_list(metadata.get("synthetic_questions"))
        if questions:
            lines.extend(["", "## Questions This Document Answers"])
            lines.extend(f"- {question}" for question in questions)
        return "\n".join(lines).strip()

    def write_article_pages(
        self,
        document: Document,
        articles: list[tuple[str, str]],
    ) -> int:
        """Write one wiki page per article; return how many were written.

        Article pages are index entries, not documents: each carries the parent
        ``doc_id`` so a hit resolves to the document the read plan and citations work
        with. Pages from a previous ingest of the same document are removed first, so a
        re-parse that yields fewer articles cannot leave stale entries matching queries.
        """
        kb_id = str(document.metadata.get("kb_id", "default"))
        for stale in self.articles_dir.glob(f"{document.doc_id}--*.md"):
            stale.unlink(missing_ok=True)
        written = 0
        for index, (label, text) in enumerate(articles):
            slug = hashlib.sha256(label.encode("utf-8")).hexdigest()[:10]
            path = self.articles_dir / f"{document.doc_id}--{index:04d}-{slug}.md"
            metadata = {
                "kind": "article",
                "doc_id": document.doc_id,
                "article_label": label,
                "article_index": index,
                "title": f"{document.title} {label}",
                "source_path": document.source_path,
                "kb_id": kb_id,
            }
            self._write_page(path, metadata, text)
            written += 1
        return written

    def _render_schema(self) -> str:
        return (
            "# Nano RAG Wiki Schema\n\n"
            "This directory is the compiled knowledge layer between raw source files and query-time retrieval.\n\n"
            "Structure:\n"
            "- `sources/`: one markdown page per ingested source document\n"
            "- `topics/`: synthesized topic pages aggregated from compiled sources\n"
            "- `index.md`: catalog of compiled pages\n"
            "- `log.md`: append-only ingest/delete timeline\n"
            f"- `{LEDGER_FILENAME}`: per-source_key version ledger (latest-wins bookkeeping)\n\n"
            "Conventions:\n"
            "- Raw sources remain immutable in `data/raw/`\n"
            "- Wiki pages are regenerated or updated by the ingestion pipeline\n"
            "- All writes are atomic and serialized by `.lock` across ingest workers\n"
            "- `index.md` should be readable first when exploring the wiki layer\n"
        )

    # ------------------------------------------------------------------ #
    # Index / topic aggregates (full pass, write-if-changed)              #
    # ------------------------------------------------------------------ #

    def _write_index(self) -> None:
        grouped: dict[str, list[tuple[dict[str, object], str]]] = {}
        scoped_groups: dict[str, list[tuple[dict[str, object], str]]] = {}
        for page_path in sorted(self.sources_dir.glob("*.md")):
            metadata, body = self._read_frontmatter(page_path)
            if not metadata:
                raise RetrievalError(
                    f"wiki source page {page_path} has unreadable frontmatter"
                )
            kb_id = str(metadata.get("kb_id", "default"))
            grouped.setdefault(kb_id, []).append((metadata, body))
            scoped_groups.setdefault(self.scope_id(kb_id), []).append((metadata, body))

        topic_pages = self._write_topic_pages(grouped)

        lines = [
            "# Wiki Index",
            "",
            "This file catalogs the compiled source pages maintained by the ingestion pipeline.",
            "",
        ]
        if not grouped:
            lines.append("No compiled sources yet.")
        else:
            for kb_id in sorted(grouped):
                lines.extend([f"## KB: {kb_id}", ""])
                kb_topics = [
                    page for page in topic_pages if str(page["kb_id"]) == kb_id
                ]
                if kb_topics:
                    lines.extend(["### Topics", ""])
                    for page in kb_topics:
                        lines.append(
                            f"- [{page['title']}](topics/{page['slug']}.md) — {page['summary']}"
                        )
                    lines.extend(["", "### Sources", ""])
                entries = sorted(
                    grouped[kb_id],
                    key=lambda item: str(item[0].get("title", "")).lower(),
                )
                for metadata, body in entries:
                    lines.append(self._render_index_entry(metadata, body))
                lines.append("")
        self._write_index_file(self.root_dir / "index.md", "\n".join(lines).rstrip() + "\n")
        self._write_scoped_indexes(scoped_groups)

    def _render_index_entry(self, metadata: dict[str, object], body: str) -> str:
        title = str(metadata.get("title", metadata.get("doc_id", "Untitled")))
        source_path = str(metadata.get("source_path", ""))
        chunk_count = as_int(metadata.get("chunk_count"))
        summary = str(metadata.get("summary", "")) or self._extract_summary(body)
        page_name = f"{metadata.get('doc_id')}.md"
        return (
            f"- [{title}](sources/{page_name}) — {self._preview(summary, MAX_INDEX_SUMMARY_CHARS)} "
            f"(chunks: {chunk_count}, source: `{source_path}`)"
        )

    def _write_scoped_indexes(
        self, scoped_groups: dict[str, list[tuple[dict[str, object], str]]]
    ) -> None:
        for existing in self.indexes_dir.glob("*.md"):
            if existing.stem not in scoped_groups:
                existing.unlink(missing_ok=True)
        for scope_id, entries in scoped_groups.items():
            # The real kb_id travels in the frontmatter so the searcher can
            # attribute the page without reversing the (possibly hashed)
            # filesystem-safe scope id.
            kb_id = str(entries[0][0].get("kb_id", "default"))
            lines = [
                FRONTMATTER_BOUNDARY,
                yaml.safe_dump(
                    {"kb_id": kb_id, "scope_id": scope_id},
                    allow_unicode=True,
                    sort_keys=False,
                    default_flow_style=False,
                ).strip(),
                FRONTMATTER_BOUNDARY,
                "",
                f"# Wiki Index: {scope_id}",
                "",
                "This file catalogs compiled source pages for a specific retrieval scope.",
                "",
            ]
            ordered_entries = sorted(
                entries,
                key=lambda item: str(item[0].get("title", "")).lower(),
            )
            for metadata, body in ordered_entries:
                lines.append(self._render_scoped_index_entry(metadata, body))
            self._write_index_file(
                self.indexes_dir / f"{scope_id}.md",
                "\n".join(lines).rstrip() + "\n",
            )

    def _render_scoped_index_entry(self, metadata: dict[str, object], body: str) -> str:
        title = str(metadata.get("title", metadata.get("doc_id", "Untitled")))
        source_path = str(metadata.get("source_path", ""))
        chunk_count = as_int(metadata.get("chunk_count"))
        summary = str(metadata.get("summary", "")) or self._extract_summary(body)
        page_name = f"{metadata.get('doc_id')}.md"
        return (
            f"- [{title}](../sources/{page_name}) — {self._preview(summary, MAX_INDEX_SUMMARY_CHARS)} "
            f"(chunks: {chunk_count}, source: `{source_path}`)"
        )

    def _write_index_file(self, path: Path, content: str) -> None:
        """Write-if-changed: identical content must not bump mtime, because
        the query-side searcher diff-syncs on mtime and a spurious rewrite
        would force a re-index of the page."""
        if path.exists() and path.read_text(encoding="utf-8") == content:
            return
        self._write_atomic(path, content)

    def _write_topic_pages(
        self, grouped: dict[str, list[tuple[dict[str, object], str]]]
    ) -> list[dict[str, str]]:
        grouped_topics: dict[tuple[str, str], list[dict[str, object]]] = {}
        for _kb_id, records in grouped.items():
            for metadata, body in records:
                kb_id = str(metadata.get("kb_id", "default"))
                doc_id = str(metadata.get("doc_id", "unknown"))
                title = str(metadata.get("title", doc_id))
                summary = str(metadata.get("summary", "")) or self._extract_summary(body)
                headings = as_list(metadata.get("headings"))
                key_passages = as_str_list(metadata.get("key_passages"))
                for topic_name in self._extract_topic_names(title, headings):
                    entry: dict[str, object] = {
                        "doc_id": doc_id,
                        "title": title,
                        "source_path": str(metadata.get("source_path", "")),
                        "summary": summary,
                        "key_passages": key_passages,
                        "doc_type": str(metadata.get("doc_type", "")),
                    }
                    grouped_topics.setdefault((kb_id, topic_name), []).append(entry)

        written_topics: list[dict[str, str]] = []
        wanted: set[str] = set()
        for (kb_id, topic_name), entries in sorted(
            grouped_topics.items(),
            key=lambda item: (item[0][0], item[0][1]),
        ):
            slug = self._topic_slug(kb_id, topic_name)
            wanted.add(f"{slug}.md")
            summary = self._preview(
                " ".join(
                    as_str(entry.get("summary"))
                    for entry in entries
                    if as_str(entry.get("summary"))
                ),
                MAX_SUMMARY_CHARS,
            )
            content = self._render_topic_page(
                topic_name=topic_name,
                kb_id=kb_id,
                entries=entries,
                summary=summary,
            )
            self._write_index_file(self.topics_dir / f"{slug}.md", content)
            written_topics.append(
                {
                    "kb_id": kb_id,
                    "slug": slug,
                    "title": topic_name,
                    "summary": self._preview(summary, MAX_INDEX_SUMMARY_CHARS),
                }
            )
        # Delete only true orphans: pages whose topic no longer exists. Pages
        # still wanted keep their mtime (write-if-changed above).
        for existing in self.topics_dir.glob("*.md"):
            if existing.name not in wanted:
                existing.unlink(missing_ok=True)
        return written_topics

    def _render_topic_page(
        self,
        topic_name: str,
        kb_id: str,
        entries: list[dict[str, object]],
        summary: str,
    ) -> str:
        facts = self._build_topic_facts(topic_name, entries)
        status = self._topic_status(entries, facts)
        frontmatter = yaml.safe_dump(
            {
                "topic": topic_name,
                "kb_id": kb_id,
                "doc_types": sorted(
                    {
                        as_str(entry.get("doc_type"))
                        for entry in entries
                        if as_str(entry.get("doc_type"))
                    }
                ),
                "source_count": len(entries),
                "status": status,
            },
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        ).strip()
        lines = [
            FRONTMATTER_BOUNDARY,
            frontmatter,
            FRONTMATTER_BOUNDARY,
            "",
            f"# {topic_name}",
            "",
            "## Status",
            f"- {status}",
            "",
            "## Summary",
            summary or "No summary available.",
            "",
            "## Key Facts",
        ]
        if facts:
            lines.extend(f"- {fact}" for fact in facts)
        else:
            lines.append("- No aggregated facts available yet.")
        lines.extend(
            [
                "",
                "## Sources",
            ]
        )
        for entry in entries[:MAX_TOPIC_ENTRIES]:
            lines.append(
                f"- [{as_str(entry.get('title'))}](../sources/{as_str(entry.get('doc_id'))}.md) — "
                f"{self._preview(as_str(entry.get('summary')), MAX_INDEX_SUMMARY_CHARS)} "
                f"(source: `{entry['source_path']}`)"
            )
        return "\n".join(lines).strip() + "\n"

    def _append_log(
        self,
        operation: str,
        document: Document | None,
        updated_at: str | None = None,
        *,
        doc_id: str | None = None,
    ) -> None:
        log_path = self.root_dir / "log.md"
        if operation == "ingest" and document is not None:
            entry = [
                f"## [{updated_at}] ingest | {document.title}",
                f"- doc_id: `{document.doc_id}`",
                f"- source_path: `{document.source_path}`",
                f"- wiki_page: [sources/{document.doc_id}.md](sources/{document.doc_id}.md)",
            ]
        elif operation == "delete" and doc_id is not None:
            entry = [
                f"## [{datetime.now(timezone.utc).isoformat()}] delete | {doc_id}",
                f"- doc_id: `{doc_id}`",
            ]
        else:
            return
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join([*entry, ""]))

    # ------------------------------------------------------------------ #
    # Scope ids                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def scope_id(kb_id: str) -> str:
        """Filesystem-safe, collision-free scope id for a kb_id.

        Filesystem-safe ids (ASCII alphanumerics, dot, underscore, dash) keep
        their readable form. Anything else — including pure-CJK ids that the
        previous sanitizer folded onto the shared ``"default"`` scope, merging
        unrelated KBs — is namespaced with a sha256 digest of the original
        id so distinct KBs can never collide.
        """
        raw = str(kb_id)
        sanitized = re.sub(r"[^a-zA-Z0-9._-]+", "-", raw).strip("-").rstrip("-")
        if sanitized and sanitized == raw:
            return sanitized
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
        prefix = sanitized[:24].rstrip("-")
        return f"kb-{prefix}-{digest}" if prefix else f"kb-{digest}"

    # ------------------------------------------------------------------ #
    # Frontmatter                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def read_frontmatter(path: Path) -> tuple[dict[str, object], str]:
        content = path.read_text(encoding="utf-8")
        return WikiCompiler._parse_frontmatter_content(content)

    def _read_frontmatter(self, path: Path) -> tuple[dict[str, object], str]:
        content = path.read_text(encoding="utf-8")
        return self._parse_frontmatter_content(content)

    @staticmethod
    def _parse_frontmatter_content(content: str) -> tuple[dict[str, object], str]:
        """Split a page into frontmatter and body using LINE boundaries.

        The delimiter must occupy its own line (the standard front-matter
        convention). Splitting on the raw "---" substring broke every page whose
        YAML contained that sequence inside a value - a markdown table separator
        (``| --- | --- |``) inside ``summary``/``key_passages`` was enough - and
        the truncated YAML then read as "unreadable frontmatter", which made
        bootstrap fail and the application refuse to start.
        """
        lines = content.splitlines()
        if not lines or lines[0].strip() != FRONTMATTER_BOUNDARY:
            return {}, content
        closing_index = next(
            (
                index
                for index, line in enumerate(lines[1:], start=1)
                if line.strip() == FRONTMATTER_BOUNDARY
            ),
            None,
        )
        if closing_index is None:
            return {}, content
        raw_metadata = "\n".join(lines[1:closing_index]).strip()
        body = "\n".join(lines[closing_index + 1 :]).strip()
        try:
            metadata = yaml.safe_load(raw_metadata) or {}
        except yaml.YAMLError:
            return {}, content
        if not isinstance(metadata, dict):
            return {}, content
        return metadata, body

    # ------------------------------------------------------------------ #
    # Text helpers                                                        #
    # ------------------------------------------------------------------ #

    def _extract_headings(self, text: str) -> list[str]:
        headings: list[str] = []
        for line in text.splitlines():
            match = re.match(r"^(#{1,6})\s+(.*\S)\s*$", line.strip())
            if match:
                headings.append(match.group(2).strip())
        return headings[:20]

    def _extract_topic_names(
        self, source_title: str, headings: object | None
    ) -> list[str]:
        topic_names = [
            str(heading).strip()
            for heading in (headings if isinstance(headings, list) else [])
            if str(heading).strip()
        ]
        if not topic_names and source_title and source_title not in topic_names:
            topic_names.append(source_title)
        unique_topics: list[str] = []
        seen: set[str] = set()
        for topic_name in topic_names:
            lowered = topic_name.lower().strip()
            if not lowered or lowered in seen:
                continue
            seen.add(lowered)
            unique_topics.append(topic_name.strip())
        return unique_topics[:8]

    def _topic_slug(self, kb_id: str, topic_name: str) -> str:
        scope = self.scope_id(kb_id=kb_id)
        raw = topic_name.lower().strip()
        topic_part = re.sub(r"[^a-z0-9._-]+", "-", raw).strip("-")
        digest = hashlib.sha256(topic_name.lower().encode("utf-8")).hexdigest()[:8]
        if not topic_part:
            # Fully non-ASCII topic names (e.g. CJK) would collapse to an
            # empty slug and collide across topics.
            topic_part = f"topic-{digest}"
        elif re.search(r"[^a-z0-9._ -]", raw):
            # Lossy sanitization (CJK mixed with ASCII): namespace with a
            # digest so distinct topics cannot collide.
            topic_part = f"{topic_part}-{digest}"
        return f"{scope}--{topic_part}"

    def _extract_summary(self, text: str) -> str:
        paragraphs: list[str] = []
        for block in text.split("\n\n"):
            candidate = block.strip()
            if not candidate or candidate.startswith("#"):
                continue
            candidate = re.sub(r"\s+", " ", candidate)
            if len(candidate) < 20:
                continue
            paragraphs.append(candidate)
            if len(" ".join(paragraphs)) >= MAX_SUMMARY_CHARS:
                break
        summary = " ".join(paragraphs)
        return self._preview(summary, MAX_SUMMARY_CHARS)

    def _build_topic_facts(
        self, topic_name: str, entries: list[dict[str, object]]
    ) -> list[str]:
        facts: list[str] = []
        seen: set[str] = set()
        for entry in entries:
            candidate_texts = [
                as_str(entry.get("summary")).strip(),
                *as_str_list(entry.get("key_passages")),
            ]
            for candidate in candidate_texts:
                if not candidate:
                    continue
                normalized = candidate.lower()
                if normalized in seen:
                    continue
                seen.add(normalized)
                facts.append(
                    f"{self._preview(candidate, MAX_CHUNK_PREVIEW_CHARS)} "
                    f"(source: [{entry['title']}](../sources/{entry['doc_id']}.md))"
                )
                if len(facts) >= MAX_TOPIC_FACTS:
                    return facts
        return facts

    def _topic_status(
        self,
        entries: list[dict[str, object]],
        facts: list[str],
    ) -> str:
        # Structural only: no semantic keyword or number-set heuristics.
        # Version conflicts between sources are expressed by the version
        # ledger (source_key groups, superseded_by), not guessed here.
        if len(entries) <= 1 or len(facts) <= 1:
            return "sparse"
        return "stable"

    def _preview(self, text: str, limit: int) -> str:
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized:
            return "No summary available."
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3].rstrip() + "..."



