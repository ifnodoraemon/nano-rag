from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import yaml

from app.schemas.chunk import Chunk
from app.schemas.document import Document

MAX_SUMMARY_CHARS = 600
MAX_CHUNK_PREVIEW_CHARS = 180
MAX_INDEX_SUMMARY_CHARS = 120
MAX_TOPIC_ENTRIES = 8
MAX_TOPIC_FACTS = 8
NEGATION_MARKERS = (" not ", " no ", " never ", " cannot ", " can't ", "禁止", "不得", "不能", "不可以")

FRONTMATTER_BOUNDARY = "---"


class WikiCompiler:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.sources_dir = self.root_dir / "sources"
        self.topics_dir = self.root_dir / "topics"
        self.indexes_dir = self.root_dir / "indexes"
        self._ensure_structure()

    def upsert_document(self, document: Document, chunks: list[Chunk]) -> Path:
        self._ensure_structure()
        page_path = self.sources_dir / f"{document.doc_id}.md"
        metadata = self._build_page_metadata(document, chunks)
        page_path.write_text(
            self._render_source_page(document, chunks, metadata), encoding="utf-8"
        )
        self._write_index()
        self._append_log(document, metadata["updated_at"])
        return page_path

    def remove_document(self, doc_id: str) -> None:
        self._ensure_structure()
        (self.sources_dir / f"{doc_id}.md").unlink(missing_ok=True)
        self._write_index()

    def _ensure_structure(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.sources_dir.mkdir(parents=True, exist_ok=True)
        self.topics_dir.mkdir(parents=True, exist_ok=True)
        self.indexes_dir.mkdir(parents=True, exist_ok=True)
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
            "tenant_id": document.metadata.get("tenant_id"),
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
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _render_source_page(
        self, document: Document, chunks: list[Chunk], metadata: dict[str, object]
    ) -> str:
        headings = metadata.get("headings", []) or []
        summary = str(metadata.get("summary", "")) or "No summary available."
        frontmatter = yaml.safe_dump(
            metadata,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        ).strip()
        lines = [
            FRONTMATTER_BOUNDARY,
            frontmatter,
            FRONTMATTER_BOUNDARY,
            "",
            f"# {document.title}",
            "",
            "## Source",
            f"- doc_id: `{document.doc_id}`",
            f"- source_path: `{document.source_path}`",
            f"- kb_id: `{metadata['kb_id']}`",
        ]
        tenant_id = metadata.get("tenant_id")
        if tenant_id:
            lines.append(f"- tenant_id: `{tenant_id}`")
        lines.extend(
            [
                f"- chunk_count: `{metadata['chunk_count']}`",
                "",
                "## Summary",
                summary,
                "",
                "## Headings",
            ]
        )
        if headings:
            lines.extend(f"- {heading}" for heading in headings)
        else:
            lines.append("- No explicit markdown headings detected.")
        lines.extend(["", "## Key Passages"])
        key_passages = metadata.get("key_passages", []) or []
        if key_passages:
            for index, passage in enumerate(key_passages):
                chunk_id = chunks[index].chunk_id if index < len(chunks) else f"{document.doc_id}:{index}"
                lines.append(f"- `{chunk_id}` {passage}")
        else:
            lines.append("- No chunks available.")
        return "\n".join(lines).strip() + "\n"

    def _render_schema(self) -> str:
        return (
            "# Nano RAG Wiki Schema\n\n"
            "This directory is the compiled knowledge layer between raw source files and query-time retrieval.\n\n"
            "Structure:\n"
            "- `sources/`: one markdown page per ingested source document\n"
            "- `topics/`: synthesized topic pages aggregated from compiled sources\n"
            "- `index.md`: catalog of compiled pages\n"
            "- `log.md`: append-only ingest timeline\n\n"
            "Conventions:\n"
            "- Raw sources remain immutable in `data/raw/`\n"
            "- Wiki pages are regenerated or updated by the ingestion pipeline\n"
            "- `index.md` should be readable first when exploring the wiki layer\n"
        )

    def _write_index(self) -> None:
        source_pages = sorted(self.sources_dir.glob("*.md"))
        grouped: dict[str, list[tuple[dict[str, object], str]]] = {}
        scoped_groups: dict[str, list[tuple[dict[str, object], str]]] = {}
        source_records: list[tuple[dict[str, object], str]] = []
        for page_path in source_pages:
            metadata, body = self._read_frontmatter(page_path)
            kb_id = str(metadata.get("kb_id", "default"))
            scope_id = self.scope_id(
                kb_id=kb_id,
                tenant_id=metadata.get("tenant_id"),
            )
            grouped.setdefault(kb_id, []).append((metadata, body))
            scoped_groups.setdefault(scope_id, []).append((metadata, body))
            source_records.append((metadata, body))

        topic_pages = self._write_topic_pages(source_records)

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
                    title = str(metadata.get("title", metadata.get("doc_id", "Untitled")))
                    source_path = str(metadata.get("source_path", ""))
                    chunk_count = int(metadata.get("chunk_count", 0))
                    summary = self._extract_summary(body)
                    page_name = f"{metadata.get('doc_id')}.md"
                    lines.append(
                        f"- [{title}](sources/{page_name}) — {self._preview(summary, MAX_INDEX_SUMMARY_CHARS)} "
                        f"(chunks: {chunk_count}, source: `{source_path}`)"
                    )
                lines.append("")
        (self.root_dir / "index.md").write_text(
            "\n".join(lines).rstrip() + "\n", encoding="utf-8"
        )
        self._write_scoped_indexes(scoped_groups)

    def _write_topic_pages(
        self, source_records: list[tuple[dict[str, object], str]]
    ) -> list[dict[str, str]]:
        grouped_topics: dict[tuple[str, str | None, str], list[dict[str, str]]] = {}
        for metadata, body in source_records:
            kb_id = str(metadata.get("kb_id", "default"))
            tenant_id = (
                str(metadata.get("tenant_id"))
                if metadata.get("tenant_id") not in (None, "")
                else None
            )
            doc_id = str(metadata.get("doc_id", "unknown"))
            title = str(metadata.get("title", doc_id))
            summary = str(metadata.get("summary", "")) or self._extract_summary(body)
            headings = metadata.get("headings", [])
            key_passages = [
                str(item).strip()
                for item in (metadata.get("key_passages", []) or [])
                if str(item).strip()
            ]
            for topic_name in self._extract_topic_names(title, headings):
                grouped_topics.setdefault((kb_id, tenant_id, topic_name), []).append(
                    {
                        "doc_id": doc_id,
                        "title": title,
                        "source_path": str(metadata.get("source_path", "")),
                        "summary": summary,
                        "key_passages": key_passages,
                        "doc_type": str(metadata.get("doc_type", "")),
                    }
                )

        for existing in self.topics_dir.glob("*.md"):
            existing.unlink(missing_ok=True)

        written_topics: list[dict[str, str]] = []
        for (kb_id, tenant_id, topic_name), entries in sorted(
            grouped_topics.items(),
            key=lambda item: (item[0][0], item[0][1] or "", item[0][2]),
        ):
            slug = self._topic_slug(kb_id, tenant_id, topic_name)
            summary = self._preview(
                " ".join(entry["summary"] for entry in entries if entry["summary"]),
                MAX_SUMMARY_CHARS,
            )
            (self.topics_dir / f"{slug}.md").write_text(
                self._render_topic_page(
                    topic_name=topic_name,
                    kb_id=kb_id,
                    tenant_id=tenant_id,
                    entries=entries,
                    summary=summary,
                ),
                encoding="utf-8",
            )
            written_topics.append(
                {
                    "kb_id": kb_id,
                    "tenant_id": tenant_id or "",
                    "slug": slug,
                    "title": topic_name,
                    "summary": self._preview(summary, MAX_INDEX_SUMMARY_CHARS),
                }
            )
        return written_topics

    def _render_topic_page(
        self,
        topic_name: str,
        kb_id: str,
        tenant_id: str | None,
        entries: list[dict[str, str]],
        summary: str,
    ) -> str:
        facts = self._build_topic_facts(topic_name, entries)
        conflicts = self._detect_topic_conflicts(topic_name, entries)
        status = self._topic_status(entries, facts, conflicts)
        frontmatter = yaml.safe_dump(
            {
                "topic": topic_name,
                "kb_id": kb_id,
                "tenant_id": tenant_id,
                "doc_types": sorted(
                    {
                        entry["doc_type"]
                        for entry in entries
                        if entry.get("doc_type")
                    }
                ),
                "source_count": len(entries),
                "status": status,
                "conflict_count": len(conflicts),
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
        lines.extend([
            "",
            "## Potential Conflicts",
        ])
        if conflicts:
            lines.extend(f"- {conflict}" for conflict in conflicts)
        else:
            lines.append("- No obvious conflicts detected.")
        lines.extend([
            "",
            "## Sources",
        ])
        for entry in entries[:MAX_TOPIC_ENTRIES]:
            lines.append(
                f"- [{entry['title']}](../sources/{entry['doc_id']}.md) — {self._preview(entry['summary'], MAX_INDEX_SUMMARY_CHARS)} "
                f"(source: `{entry['source_path']}`)"
            )
        return "\n".join(lines).strip() + "\n"

    def _append_log(self, document: Document, updated_at: str) -> None:
        log_path = self.root_dir / "log.md"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                "\n".join(
                    [
                        f"## [{updated_at}] ingest | {document.title}",
                        f"- doc_id: `{document.doc_id}`",
                        f"- source_path: `{document.source_path}`",
                        f"- wiki_page: [sources/{document.doc_id}.md](sources/{document.doc_id}.md)",
                        "",
                    ]
                )
            )

    def _write_scoped_indexes(
        self, scoped_groups: dict[str, list[tuple[dict[str, object], str]]]
    ) -> None:
        for existing in self.indexes_dir.glob("*.md"):
            if existing.stem not in scoped_groups:
                existing.unlink(missing_ok=True)
        for scope_id, entries in scoped_groups.items():
            lines = [
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
                title = str(metadata.get("title", metadata.get("doc_id", "Untitled")))
                source_path = str(metadata.get("source_path", ""))
                chunk_count = int(metadata.get("chunk_count", 0))
                summary = self._extract_summary(body)
                page_name = f"{metadata.get('doc_id')}.md"
                lines.append(
                    f"- [{title}](../sources/{page_name}) — {self._preview(summary, MAX_INDEX_SUMMARY_CHARS)} "
                    f"(chunks: {chunk_count}, source: `{source_path}`)"
                )
            (self.indexes_dir / f"{scope_id}.md").write_text(
                "\n".join(lines).rstrip() + "\n", encoding="utf-8"
            )

    @staticmethod
    def scope_id(kb_id: str, tenant_id: object | None = None) -> str:
        raw_scope = kb_id if tenant_id in (None, "", "null") else f"{kb_id}__{tenant_id}"
        return re.sub(r"[^a-zA-Z0-9._-]+", "-", str(raw_scope)).strip("-") or "default"

    @staticmethod
    def read_frontmatter(path: Path) -> tuple[dict[str, object], str]:
        content = path.read_text(encoding="utf-8")
        return WikiCompiler._parse_frontmatter_content(content)

    def _read_frontmatter(self, path: Path) -> tuple[dict[str, object], str]:
        content = path.read_text(encoding="utf-8")
        return self._parse_frontmatter_content(content)

    @staticmethod
    def _parse_frontmatter_content(content: str) -> tuple[dict[str, object], str]:
        if not content.startswith(f"{FRONTMATTER_BOUNDARY}\n"):
            return {}, content
        parts = content.split(FRONTMATTER_BOUNDARY, 2)
        if len(parts) < 3:
            return {}, content
        raw_metadata = parts[1].strip()
        body = parts[2].strip()
        try:
            metadata = yaml.safe_load(raw_metadata) or {}
        except yaml.YAMLError:
            metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
        return metadata, body

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

    def _topic_slug(self, kb_id: str, tenant_id: str | None, topic_name: str) -> str:
        scope = self.scope_id(kb_id=kb_id, tenant_id=tenant_id)
        topic_part = re.sub(r"[^a-zA-Z0-9._-]+", "-", topic_name.lower()).strip("-")
        return f"{scope}--{topic_part or 'topic'}"

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
        topic_terms = self._topic_terms(topic_name)
        for entry in entries:
            candidate_texts = [
                str(entry.get("summary", "")).strip(),
                *[
                    str(item).strip()
                    for item in (entry.get("key_passages", []) or [])
                    if str(item).strip()
                ],
            ]
            for candidate in candidate_texts:
                if not candidate:
                    continue
                if topic_terms and not any(term in candidate.lower() for term in topic_terms):
                    # Keep supporting statements broadly related to the topic, but avoid totally generic text.
                    if len(candidate) < 40:
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

    def _detect_topic_conflicts(
        self, topic_name: str, entries: list[dict[str, object]]
    ) -> list[str]:
        evidences: list[tuple[str, str]] = []
        for entry in entries:
            for candidate in [
                str(entry.get("summary", "")).strip(),
                *[
                    str(item).strip()
                    for item in (entry.get("key_passages", []) or [])
                    if str(item).strip()
                ],
            ]:
                if candidate:
                    evidences.append((str(entry["title"]), candidate))
        conflicts: list[str] = []
        for index, (left_title, left_text) in enumerate(evidences):
            for right_title, right_text in evidences[index + 1 :]:
                if left_title == right_title:
                    continue
                if self._is_conflict_pair(topic_name, left_text, right_text):
                    conflicts.append(
                        f"{self._preview(left_text, MAX_INDEX_SUMMARY_CHARS)} "
                        f"(source: [{left_title}](../sources/{self._find_doc_id(entries, left_title)}.md)) "
                        f"conflicts with "
                        f"{self._preview(right_text, MAX_INDEX_SUMMARY_CHARS)} "
                        f"(source: [{right_title}](../sources/{self._find_doc_id(entries, right_title)}.md))"
                    )
                if len(conflicts) >= 4:
                    return conflicts
        return conflicts

    def _topic_status(
        self,
        entries: list[dict[str, object]],
        facts: list[str],
        conflicts: list[str],
    ) -> str:
        if conflicts:
            return "conflicting"
        if len(entries) <= 1 or len(facts) <= 1:
            return "sparse"
        return "stable"

    def _is_conflict_pair(self, topic_name: str, left_text: str, right_text: str) -> bool:
        left_norm = f" {left_text.lower()} "
        right_norm = f" {right_text.lower()} "
        left_has_negation = any(marker in left_norm for marker in NEGATION_MARKERS)
        right_has_negation = any(marker in right_norm for marker in NEGATION_MARKERS)
        if left_has_negation != right_has_negation:
            return True
        left_numbers = set(re.findall(r"\b\d+\b", left_text))
        right_numbers = set(re.findall(r"\b\d+\b", right_text))
        if left_numbers and right_numbers and left_numbers != right_numbers:
            topic_terms = self._topic_terms(topic_name)
            if not topic_terms or any(term in left_norm or term in right_norm for term in topic_terms):
                return True
        return False

    def _topic_terms(self, topic_name: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-zA-Z0-9\u4e00-\u9fff]+", topic_name.lower())
            if len(token) >= 2
        }

    def _find_doc_id(self, entries: list[dict[str, object]], title: str) -> str:
        for entry in entries:
            if str(entry.get("title")) == title:
                return str(entry.get("doc_id"))
        return "unknown"

    def _preview(self, text: str, limit: int) -> str:
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized:
            return "No summary available."
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3].rstrip() + "..."
