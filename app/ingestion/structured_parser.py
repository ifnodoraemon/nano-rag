from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup

from app.core.exceptions import ParsingError
from app.model_client.document_parser import DocumentParserClient
from app.schemas.structured import (
    DocumentNode,
    NodeProvenance,
    NodeType,
    StructuredDocument,
    TableCell,
    TablePayload,
)


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
NUMBERED_HEADING_RE = re.compile(
    r"^((?:第[一二三四五六七八九十百\d]+[章节])|(?:\d+(?:\.\d+){0,5}))\s*[、.)]?\s*(.+)$"
)


@dataclass(frozen=True)
class ParsedBlock:
    node_type: NodeType
    text: str
    title: str | None
    level: int | None
    source_ref: str
    table: TablePayload | None = None


class StructuredDocumentParser:
    def __init__(self, document_parser: DocumentParserClient | None = None) -> None:
        self.document_parser = document_parser

    async def parse(
        self,
        path: Path,
        *,
        doc_id: str,
        kb_id: str,
        source_path: str,
    ) -> StructuredDocument:
        suffix = path.suffix.lower()
        if suffix in {".md", ".markdown", ".txt"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            blocks = self._parse_markdown_blocks(text)
            parser_name = "markdown-tree"
        elif suffix == ".html":
            html = path.read_text(encoding="utf-8", errors="ignore")
            text = BeautifulSoup(html, "html.parser").get_text("\n")
            blocks = self._parse_markdown_blocks(text)
            parser_name = "html-text-tree"
        else:
            text = await self._parse_with_multimodal_llm(path)
            blocks = self._parse_markdown_blocks(text)
            parser_name = "multimodal-llm-tree"

        if not blocks:
            raise ParsingError(f"Document parsing produced no structured nodes for {source_path}")

        root = DocumentNode(
            node_id=f"{doc_id}:root",
            doc_id=doc_id,
            kb_id=kb_id,
            node_type=NodeType.ROOT,
            title=path.stem,
            provenance=NodeProvenance(source_document_id=doc_id),
            metadata={"parser": parser_name},
        )
        stack: list[tuple[int, DocumentNode]] = [(0, root)]
        leaf_index = 0

        for block in blocks:
            if block.node_type == NodeType.SECTION:
                level = block.level or 1
                while len(stack) > 1 and stack[-1][0] >= level:
                    stack.pop()
                parent = stack[-1][1]
                node = self._build_node(
                    block,
                    doc_id=doc_id,
                    kb_id=kb_id,
                    parent=parent,
                    hierarchy_path=[*(parent.provenance.hierarchy_path), block.title or block.text],
                    leaf_index=leaf_index,
                )
                parent.children.append(node)
                stack.append((level, node))
                leaf_index += 1
                continue

            parent = stack[-1][1]
            node = self._build_node(
                block,
                doc_id=doc_id,
                kb_id=kb_id,
                parent=parent,
                hierarchy_path=parent.provenance.hierarchy_path,
                leaf_index=leaf_index,
            )
            parent.children.append(node)
            leaf_index += 1

        return StructuredDocument(
            doc_id=doc_id,
            kb_id=kb_id,
            source_path=source_path,
            title=path.stem,
            root=root,
            metadata={"parser": parser_name},
        )

    async def _parse_with_multimodal_llm(self, path: Path) -> str:
        parser_enabled = bool(getattr(self.document_parser, "enabled", True))
        if self.document_parser and parser_enabled and self.document_parser.supports(path):
            parsed = await self.document_parser.parse_file(path)
            if parsed.strip():
                return parsed
            raise ParsingError(
                f"Document parsing returned empty content for {path.name}."
            )
        raise ParsingError(
            f"{path.suffix or 'file'} parsing requires a configured multimodal document parser."
        )

    def _build_node(
        self,
        block: ParsedBlock,
        *,
        doc_id: str,
        kb_id: str,
        parent: DocumentNode,
        hierarchy_path: list[str],
        leaf_index: int,
    ) -> DocumentNode:
        source_ref = block.source_ref or f"node:{leaf_index}"
        node_id = self._stable_node_id(doc_id, source_ref)
        text = block.text.strip()
        return DocumentNode(
            node_id=node_id,
            doc_id=doc_id,
            kb_id=kb_id,
            node_type=block.node_type,
            text=text,
            title=block.title,
            parent_id=parent.node_id,
            provenance=NodeProvenance(
                source_document_id=doc_id,
                page_number=1,
                hierarchy_path=[item for item in hierarchy_path if item],
                source_ref=source_ref,
            ),
            table=block.table,
            metadata={
                "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "parser_confidence": 1.0,
            },
        )

    def _parse_markdown_blocks(self, text: str) -> list[ParsedBlock]:
        lines = text.splitlines()
        blocks: list[ParsedBlock] = []
        paragraph: list[str] = []
        paragraph_start = 0
        index = 0

        def flush_paragraph(until_line: int) -> None:
            nonlocal paragraph, paragraph_start
            content = "\n".join(paragraph).strip()
            if content:
                blocks.append(
                    ParsedBlock(
                        node_type=NodeType.PARAGRAPH,
                        text=content,
                        title=None,
                        level=None,
                        source_ref=f"lines:{paragraph_start + 1}-{until_line}",
                    )
                )
            paragraph = []

        while index < len(lines):
            line = lines[index]
            stripped = line.strip()
            heading = HEADING_RE.match(stripped)
            numbered = NUMBERED_HEADING_RE.match(stripped)
            if heading or (numbered and len(stripped) <= 120):
                flush_paragraph(index)
                if heading:
                    level = len(heading.group(1))
                    title = heading.group(2).strip()
                else:
                    marker = numbered.group(1)
                    title = f"{marker} {numbered.group(2).strip()}"
                    level = 1 if marker.startswith("第") or "." not in marker else min(marker.count(".") + 1, 6)
                blocks.append(
                    ParsedBlock(
                        node_type=NodeType.SECTION,
                        text=title,
                        title=title,
                        level=level,
                        source_ref=f"lines:{index + 1}-{index + 1}",
                    )
                )
                index += 1
                continue

            if self._is_table_line(stripped):
                flush_paragraph(index)
                start = index
                table_lines: list[str] = []
                while index < len(lines) and self._is_table_line(lines[index].strip()):
                    table_lines.append(lines[index].strip())
                    index += 1
                table = self._parse_table(table_lines)
                blocks.append(
                    ParsedBlock(
                        node_type=NodeType.TABLE,
                        text=table.narrative or "\n".join(table_lines),
                        title=table.caption,
                        level=None,
                        source_ref=f"lines:{start + 1}-{index}",
                        table=table,
                    )
                )
                continue

            if not stripped:
                flush_paragraph(index)
                index += 1
                continue

            if not paragraph:
                paragraph_start = index
            paragraph.append(line.rstrip())
            index += 1

        flush_paragraph(len(lines))
        return blocks

    def _parse_table(self, lines: list[str]) -> TablePayload:
        rows = [
            [cell.strip() for cell in line.strip().strip("|").split("|")]
            for line in lines
            if line.strip()
        ]
        if len(rows) >= 2 and all(set(cell) <= {":", "-"} for cell in rows[1] if cell):
            header_rows = [rows[0]]
            body_rows = rows[2:]
        else:
            header_rows = rows[:1]
            body_rows = rows[1:]
        all_rows = [*header_rows, *body_rows]
        max_cols = max((len(row) for row in all_rows), default=0)
        cells: list[TableCell] = []
        for row_index, row in enumerate(all_rows):
            for col_index, cell_text in enumerate(row):
                cells.append(
                    TableCell(
                        row=row_index,
                        col=col_index,
                        text=cell_text,
                        is_header=row_index < len(header_rows),
                    )
                )
        header_map = {
            cell.text: [cell.col]
            for cell in cells
            if cell.is_header and cell.text
        }
        narrative = self._table_narrative(header_map, body_rows)
        return TablePayload(
            rows=len(all_rows),
            cols=max_cols,
            cells=cells,
            header_map=header_map,
            narrative=narrative,
        )

    def _table_narrative(self, header_map: dict[str, list[int]], rows: list[list[str]]) -> str:
        headers = list(header_map)
        if not headers:
            return ""
        preview_rows = []
        for row in rows[:8]:
            pairs = [
                f"{headers[col]}={value}"
                for col, value in enumerate(row[: len(headers)])
                if value.strip()
            ]
            if pairs:
                preview_rows.append("; ".join(pairs))
        return "Table with columns: " + ", ".join(headers) + (
            "\n" + "\n".join(preview_rows) if preview_rows else ""
        )

    def _is_table_line(self, value: str) -> bool:
        return value.startswith("|") and value.endswith("|") and value.count("|") >= 3

    def _stable_node_id(self, doc_id: str, source_ref: str) -> str:
        digest = hashlib.sha256(f"{doc_id}:{source_ref}".encode("utf-8")).hexdigest()
        return f"{doc_id}:node:{digest[:16]}"
