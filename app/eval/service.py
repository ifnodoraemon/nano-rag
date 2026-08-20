from __future__ import annotations

from typing import TYPE_CHECKING

from app.schemas.chat import ChatRequest

if TYPE_CHECKING:
    from app.core.config import AppContainer

import logging
logger = logging.getLogger(__name__)


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


def _claim_verification_stats(claims: object) -> dict[str, object]:
    if not isinstance(claims, list):
        return {
            "supporting_claim_count": 0,
            "verified_claim_count": 0,
            "unsupported_claim_count": 0,
            "claim_support_score_avg": 0.0,
            "missing_number_count": 0,
            "missing_term_count": 0,
        }
    normalized: list[dict[str, object]] = []
    for claim in claims:
        if isinstance(claim, dict):
            normalized.append(claim)
        elif hasattr(claim, "model_dump"):
            normalized.append(claim.model_dump())
    if not normalized:
        return {
            "supporting_claim_count": 0,
            "verified_claim_count": 0,
            "unsupported_claim_count": 0,
            "claim_support_score_avg": 0.0,
            "missing_number_count": 0,
            "missing_term_count": 0,
        }
    scored = [
        float(claim.get("support_score", 0.0) or 0.0)
        for claim in normalized
        if claim.get("support_score") is not None
    ]
    verified = sum(1 for claim in normalized if claim.get("verified") is True)
    unsupported = sum(1 for claim in normalized if claim.get("verified") is False)
    missing_numbers = sum(
        len(claim.get("missing_numbers", []) or [])
        for claim in normalized
        if isinstance(claim.get("missing_numbers", []), list)
    )
    missing_terms = sum(
        len(claim.get("missing_terms", []) or [])
        for claim in normalized
        if isinstance(claim.get("missing_terms", []), list)
    )
    return {
        "supporting_claim_count": len(normalized),
        "verified_claim_count": verified,
        "unsupported_claim_count": unsupported,
        "claim_support_score_avg": round(sum(scored) / len(scored), 4) if scored else 0.0,
        "missing_number_count": missing_numbers,
        "missing_term_count": missing_terms,
    }



async def materialize_eval_records(
    container: AppContainer, records: list[dict]
) -> list[dict]:
    prepared_records: list[dict] = []
    for index, record in enumerate(records):
        logger.info(f"Materializing record {index+1}/{len(records)}: {record.get('query')}")
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
        claim_stats = _claim_verification_stats(prepared.get("supporting_claims", []))

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
                    claim_stats = _claim_verification_stats(trace.supporting_claims)
                    prepared["supporting_claims"] = trace.supporting_claims
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
            if not prepared.get("supporting_claims"):
                prepared["supporting_claims"] = [
                    claim.model_dump() if hasattr(claim, "model_dump") else claim
                    for claim in getattr(chat_response, "supporting_claims", [])
                ]
            claim_stats = _claim_verification_stats(prepared.get("supporting_claims", []))

        if not retrieved_contexts:
            contexts, trace = await container.chat_pipeline.retrieve(
                ChatRequest(
                    query=query,
                    top_k=top_k,
                    kb_id=kb_id,
                    session_id=session_id,
                )
            )
            prepared["retrieved_contexts"] = [
                _context_to_text(context) for context in contexts
            ]
            conflicting_context_count = _count_conflicting_contexts(contexts)

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
                if not prepared.get("supporting_claims"):
                    prepared["supporting_claims"] = trace.supporting_claims
                claim_stats = _claim_verification_stats(prepared.get("supporting_claims", []))

        prepared["conflicting_context_count"] = conflicting_context_count
        prepared["conflict_claim_count"] = conflict_claim_count
        prepared["insufficiency_claim_count"] = insufficiency_claim_count
        prepared.update(claim_stats)

        prepared_records.append(prepared)
    return prepared_records
