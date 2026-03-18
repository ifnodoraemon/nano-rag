from app.schemas.chat import ChatResponse, Citation


class AnswerFormatter:
    def format(self, answer: str, contexts: list[dict[str, object]], trace_id: str | None) -> ChatResponse:
        citations = [
            Citation(chunk_id=context["chunk_id"], source=str(context["source"]), score=float(context["score"]))
            for context in contexts
        ]
        return ChatResponse(answer=answer, citations=citations, contexts=contexts, trace_id=trace_id)
