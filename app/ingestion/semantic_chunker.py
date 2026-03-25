from __future__ import annotations

import os
import re
from dataclasses import dataclass

from app.schemas.chunk import Chunk


@dataclass
class SemanticChunkerConfig:
    min_chunk_size: int = 100
    max_chunk_size: int = 1000
    overlap_sentences: int = 1
    respect_sentence_boundary: bool = True

    @classmethod
    def from_env(cls) -> "SemanticChunkerConfig":
        return cls(
            min_chunk_size=int(os.getenv("RAG_CHUNK_MIN_SIZE", "100")),
            max_chunk_size=int(os.getenv("RAG_CHUNK_MAX_SIZE", "1000")),
            overlap_sentences=int(os.getenv("RAG_CHUNK_OVERLAP_SENTENCES", "1")),
            respect_sentence_boundary=os.getenv(
                "RAG_CHUNK_RESPECT_SENTENCE", "true"
            ).lower()
            in ("true", "1", "yes"),
        )


def _split_sentences(text: str) -> list[str]:
    sentence_endings = r"(?<=[。！？.!?])\s*"
    sentences = re.split(sentence_endings, text)
    return [s.strip() for s in sentences if s.strip()]


def _estimate_tokens(text: str) -> int:
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    other_chars = len(text) - chinese_chars
    return int(chinese_chars + other_chars / 4)


class SemanticChunker:
    def __init__(self, config: SemanticChunkerConfig | None = None) -> None:
        self.config = config or SemanticChunkerConfig.from_env()

    def chunk(
        self,
        text: str,
        doc_id: str,
        source_path: str,
        title: str | None = None,
        metadata: dict | None = None,
    ) -> list[Chunk]:
        if not text.strip():
            return []
        sentences = _split_sentences(text)
        if not sentences:
            return []
        chunks: list[Chunk] = []
        current_sentences: list[str] = []
        current_size = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_size = _estimate_tokens(sentence)
            if (
                current_size + sentence_size > self.config.max_chunk_size
                and current_sentences
            ):
                chunk_text = "".join(current_sentences)
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}-{chunk_index}",
                        doc_id=doc_id,
                        chunk_index=chunk_index,
                        text=chunk_text,
                        source_path=source_path,
                        title=title,
                        metadata=metadata or {},
                    )
                )
                chunk_index += 1
                overlap_sentences = current_sentences[-self.config.overlap_sentences :]
                current_sentences = overlap_sentences
                current_size = sum(_estimate_tokens(s) for s in overlap_sentences)
            current_sentences.append(sentence)
            current_size += sentence_size

        if current_sentences:
            chunk_text = "".join(current_sentences)
            if (
                len(chunks) > 0
                and _estimate_tokens(chunk_text) < self.config.min_chunk_size
            ):
                last_chunk = chunks[-1]
                last_chunk.text += chunk_text
            else:
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}-{chunk_index}",
                        doc_id=doc_id,
                        chunk_index=chunk_index,
                        text=chunk_text,
                        source_path=source_path,
                        title=title,
                        metadata=metadata or {},
                    )
                )

        return chunks

    def chunk_by_paragraph(
        self,
        text: str,
        doc_id: str,
        source_path: str,
        title: str | None = None,
        metadata: dict | None = None,
    ) -> list[Chunk]:
        paragraphs = re.split(r"\n\s*\n", text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        chunks: list[Chunk] = []
        current_para: list[str] = []
        current_size = 0
        chunk_index = 0

        for para in paragraphs:
            para_size = _estimate_tokens(para)
            if current_size + para_size > self.config.max_chunk_size and current_para:
                chunk_text = "\n\n".join(current_para)
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}-{chunk_index}",
                        doc_id=doc_id,
                        chunk_index=chunk_index,
                        text=chunk_text,
                        source_path=source_path,
                        title=title,
                        metadata=metadata or {},
                    )
                )
                chunk_index += 1
                current_para = []
                current_size = 0
            current_para.append(para)
            current_size += para_size

        if current_para:
            chunk_text = "\n\n".join(current_para)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}-{chunk_index}",
                    doc_id=doc_id,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    source_path=source_path,
                    title=title,
                    metadata=metadata or {},
                )
            )

        return chunks
