from __future__ import annotations

from typing import TYPE_CHECKING

from app.schemas.chat import ChatRequest

if TYPE_CHECKING:
    from app.core.config import AppContainer


def _context_to_text(context: object) -> str:
    if isinstance(context, dict):
        text = context.get("text")
        if text is not None:
            return str(text)
    return str(context)


async def materialize_eval_records(
    container: AppContainer, records: list[dict]
) -> list[dict]:
    prepared_records: list[dict] = []
    for index, record in enumerate(records):
        prepared = dict(record)
        prepared.setdefault("sample_id", f"sample-{index + 1}")
        query = str(prepared.get("query", "")).strip()
        top_k = prepared.get("top_k")
        kb_id = str(prepared.get("kb_id", "default") or "default")
        tenant_id = prepared.get("tenant_id")
        session_id = prepared.get("session_id")
        answer = str(prepared.get("answer", "")).strip()
        retrieved_contexts = prepared.get("retrieved_contexts", []) or []

        if not query:
            prepared_records.append(prepared)
            continue

        if not answer:
            chat_response = await container.chat_pipeline.run(
                ChatRequest(
                    query=query,
                    top_k=top_k,
                    kb_id=kb_id,
                    tenant_id=tenant_id,
                    session_id=session_id,
                    sample_id=str(prepared["sample_id"]),
                )
            )
            prepared["answer"] = chat_response.answer
            if chat_response.trace_id:
                prepared["trace_id"] = chat_response.trace_id
                trace = container.trace_store.get(chat_response.trace_id)
                if trace is not None:
                    trace.sample_id = str(prepared["sample_id"])
            if not retrieved_contexts:
                prepared["retrieved_contexts"] = [
                    _context_to_text(context) for context in chat_response.contexts
                ]
                retrieved_contexts = prepared["retrieved_contexts"]

        if not retrieved_contexts:
            retrieval = await container.retrieval_pipeline.debug(
                query,
                top_k,
                kb_id=kb_id,
                tenant_id=tenant_id,
                session_id=session_id,
            )
            prepared["retrieved_contexts"] = [
                _context_to_text(context) for context in retrieval.contexts
            ]

        prepared_records.append(prepared)
    return prepared_records
