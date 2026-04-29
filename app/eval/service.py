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


def _count_conflicting_contexts(contexts: object) -> int:
    if not isinstance(contexts, list):
        return 0
    return sum(
        1
        for context in contexts
        if isinstance(context, dict) and context.get("wiki_status") == "conflicting"
    )


def _count_claim_type(claims: object, claim_type: str) -> int:
    if not isinstance(claims, list):
        return 0
    return sum(
        1
        for claim in claims
        if isinstance(claim, dict) and claim.get("claim_type") == claim_type
    )


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
        session_id = prepared.get("session_id")
        answer = str(prepared.get("answer", "")).strip()
        retrieved_contexts = prepared.get("retrieved_contexts", []) or []
        conflicting_context_count = int(prepared.get("conflicting_context_count", 0) or 0)
        conflict_claim_count = int(prepared.get("conflict_claim_count", 0) or 0)
        insufficiency_claim_count = int(
            prepared.get("insufficiency_claim_count", 0) or 0
        )

        if not query:
            prepared_records.append(prepared)
            continue

        if not answer:
            chat_response = await container.chat_pipeline.run(
                ChatRequest(
                    query=query,
                    top_k=top_k,
                    kb_id=kb_id,
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
                    conflicting_context_count = _count_conflicting_contexts(trace.contexts)
                    conflict_claim_count = _count_claim_type(
                        trace.supporting_claims, "conflict"
                    )
                    insufficiency_claim_count = _count_claim_type(
                        trace.supporting_claims, "insufficiency"
                    )
            if not retrieved_contexts:
                prepared["retrieved_contexts"] = [
                    _context_to_text(context) for context in chat_response.contexts
                ]
                retrieved_contexts = prepared["retrieved_contexts"]
                conflicting_context_count = _count_conflicting_contexts(
                    chat_response.contexts
                )
            conflict_claim_count = max(
                conflict_claim_count,
                _count_claim_type(
                    getattr(chat_response, "supporting_claims", []), "conflict"
                ),
            )
            insufficiency_claim_count = max(
                insufficiency_claim_count,
                _count_claim_type(
                    getattr(chat_response, "supporting_claims", []), "insufficiency"
                ),
            )

        if not retrieved_contexts:
            retrieval = await container.retrieval_pipeline.debug(
                query,
                top_k,
                kb_id=kb_id,
                session_id=session_id,
            )
            prepared["retrieved_contexts"] = [
                _context_to_text(context) for context in retrieval.contexts
            ]
            conflicting_context_count = _count_conflicting_contexts(retrieval.contexts)

        if not conflicting_context_count and prepared.get("trace_id"):
            trace = container.trace_store.get(str(prepared["trace_id"]))
            if trace is not None:
                conflicting_context_count = _count_conflicting_contexts(trace.contexts)
                conflict_claim_count = max(
                    conflict_claim_count,
                    _count_claim_type(trace.supporting_claims, "conflict"),
                )
                insufficiency_claim_count = max(
                    insufficiency_claim_count,
                    _count_claim_type(trace.supporting_claims, "insufficiency"),
                )

        prepared["conflicting_context_count"] = conflicting_context_count
        prepared["conflict_claim_count"] = conflict_claim_count
        prepared["insufficiency_claim_count"] = insufficiency_claim_count

        prepared_records.append(prepared)
    return prepared_records
