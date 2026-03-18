from app.vectorstore.repository import SearchHit


def build_contexts(hits: list[SearchHit], limit: int) -> list[dict[str, object]]:
    contexts = []
    for hit in hits[:limit]:
        contexts.append(
            {
                "chunk_id": hit.chunk.chunk_id,
                "text": hit.chunk.text,
                "source": hit.chunk.source_path,
                "score": round(hit.score, 6),
            }
        )
    return contexts
