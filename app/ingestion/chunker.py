from __future__ import annotations

from app.ingestion.metadata import (
    build_section_metadata,
    split_sections,
)
from app.schemas.chunk import Chunk


def _is_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 3


def _is_table_separator(line: str) -> bool:
    stripped = line.strip()
    if not _is_table_line(stripped):
        return False
    cells = [cell.strip() for cell in stripped.strip("|").split("|")]
    return bool(cells) and all(cell and set(cell) <= {":", "-"} for cell in cells)


def _split_table_block(block: str, chunk_size: int) -> list[str]:
    lines = [line.rstrip() for line in block.splitlines() if line.strip()]
    if len(lines) < 3 or not all(_is_table_line(line) for line in lines[:2]):
        return []
    header = lines[:2] if _is_table_separator(lines[1]) else lines[:1]
    rows = lines[len(header):]
    header_text = "\n".join(header)
    chunks: list[str] = []
    current = header_text
    for row in rows:
        candidate = f"{current}\n{row}" if current else row
        if len(candidate) <= chunk_size or current == header_text:
            current = candidate
            continue
        chunks.append(current.strip())
        current = f"{header_text}\n{row}"
    if current.strip():
        chunks.append(current.strip())
    return chunks


def _split_oversized_section(section: str, chunk_size: int, overlap: int) -> list[str]:
    table_chunks = _split_table_block(section, chunk_size)
    if table_chunks:
        return table_chunks
    chunks: list[str] = []
    start = 0
    step = chunk_size - overlap
    while start < len(section):
        chunks.append(section[start : start + chunk_size].strip())
        start += step
    return chunks


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if not text:
        return []
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    sections = [segment.strip() for segment in text.split("\n\n") if segment.strip()]
    chunks: list[str] = []
    buffer = ""

    for section in sections:
        candidate = f"{buffer}\n\n{section}".strip() if buffer else section
        if len(candidate) <= chunk_size:
            buffer = candidate
            continue
        if buffer:
            chunks.append(buffer)
        if len(section) <= chunk_size:
            buffer = section
            continue
        chunks.extend(_split_oversized_section(section, chunk_size, overlap))
        buffer = ""

    if buffer:
        chunks.append(buffer)
    return chunks


def build_chunks(
    doc_id: str,
    source_path: str,
    title: str,
    text: str,
    chunk_size: int,
    overlap: int,
    metadata: dict | None = None,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    chunk_index = 0
    for section_index, section in enumerate(split_sections(text, title)):
        section_chunks = split_text(section.text, chunk_size, overlap)
        if not section_chunks and section.text.strip():
            section_chunks = [section.text.strip()]
        parent_chunk_id = f"{doc_id}:parent:{section_index}"
        for child_index, chunk_text in enumerate(section_chunks):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}:{chunk_index}",
                    doc_id=doc_id,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    source_path=source_path,
                    title=" / ".join(section.path),
                    metadata=build_section_metadata(
                        metadata,
                        section,
                        parent_chunk_id=parent_chunk_id,
                        child_chunk_count=len(section_chunks),
                        child_chunk_index=child_index,
                    ),
                )
            )
            chunk_index += 1
    return chunks
