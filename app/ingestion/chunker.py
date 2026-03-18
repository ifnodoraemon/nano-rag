from __future__ import annotations

from app.schemas.chunk import Chunk


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
        start = 0
        step = chunk_size - overlap
        while start < len(section):
            chunks.append(section[start : start + chunk_size].strip())
            start += step
        buffer = ""

    if buffer:
        chunks.append(buffer)
    return chunks


def build_chunks(doc_id: str, source_path: str, title: str, text: str, chunk_size: int, overlap: int) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"{doc_id}:{index}",
            doc_id=doc_id,
            chunk_index=index,
            text=chunk_text,
            source_path=source_path,
            title=title,
            metadata={},
        )
        for index, chunk_text in enumerate(split_text(text, chunk_size, overlap))
    ]
