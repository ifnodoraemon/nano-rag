from app.schemas.chat import ChatResponse, Citation


class AnswerFormatter:
    def format(self, answer: str, contexts: list[dict[str, object]], trace_id: str | None) -> ChatResponse:
        citations_by_chunk: dict[str, Citation] = {}
        for context in contexts:
            chunk_id = str(context["chunk_id"])
            if chunk_id not in citations_by_chunk:
                citations_by_chunk[chunk_id] = Citation(
                    chunk_id=chunk_id,
                    source=str(context["source"]),
                    score=float(context["score"]),
                )
        normalized_answer = answer.strip()
        if citations_by_chunk and "[" not in normalized_answer:
            normalized_answer = f"{normalized_answer} [{next(iter(citations_by_chunk))}]"
        return ChatResponse(
            answer=normalized_answer,
            citations=list(citations_by_chunk.values()),
            contexts=contexts,
            trace_id=trace_id,
        )
