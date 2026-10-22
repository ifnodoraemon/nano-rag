from __future__ import annotations

import asyncio
import os

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
    def _score_of(value: object) -> float | None:
        if value is None:
            return None
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    scored = [
        score
        for claim in normalized
        if (score := _score_of(claim.get("support_score"))) is not None
    ]
    verified = sum(1 for claim in normalized if claim.get("verified") is True)
    unsupported = sum(1 for claim in normalized if claim.get("verified") is False)
    def _count_items(value: object) -> int:
        """Count a list-ish field; a non-list (corrupt claim payload) is 0.

        ``len(claim.get(...) or [])`` type-checks as len(object) and would raise
        TypeError the moment a claim carried a scalar there.
        """
        if isinstance(value, list):
            return len(value)
        return 0

    missing_numbers = sum(
        _count_items(claim.get("missing_numbers")) for claim in normalized
    )
    missing_terms = sum(
        _count_items(claim.get("missing_terms")) for claim in normalized
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
    container: AppContainer, records: list[dict], *, concurrency: int | None = None
) -> list[dict]:
    """Run each dataset row through the live pipeline.

    Sequential by construction before this: every row waited for a full retrieval
    plus synthesis round trip, so a 60-row answer-level evaluation took tens of
    minutes of pure waiting. The rows are independent, so they are evaluated with
    bounded concurrency and reassembled in input order.
    """
    limit = concurrency if concurrency is not None else _eval_concurrency()
    if limit <= 1 or len(records) <= 1:
        return [await _materialize_one(container, index, record) for index, record in enumerate(records)]

    semaphore = asyncio.Semaphore(limit)

    async def bounded(index: int, record: dict) -> dict:
        async with semaphore:
            return await _materialize_one(container, index, record)

    return list(
        await asyncio.gather(
            *(bounded(index, record) for index, record in enumerate(records))
        )
    )


def _eval_concurrency() -> int:
    raw = os.getenv("RAG_EVAL_CONCURRENCY", "8")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"RAG_EVAL_CONCURRENCY must be an integer, got {raw!r}") from exc
    return max(1, value)


async def _materialize_one(container: AppContainer, index: int, record: dict) -> dict:
    # (single record; concurrency is handled by the caller)
    logger.info("Materializing record %d: %s", index + 1, record.get("query"))
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
        return prepared

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
                # The model's structured answerability verdict plus the cost and
                # latency of the request: the abstention, token and p95 metrics
                # read these, and nothing else in the row carries them.
                prepared["answerable"] = trace.answerable
                prepared["latency_seconds"] = trace.latency_seconds
                if trace.generation_usage:
                    prepared["generation_usage"] = trace.generation_usage
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
        contexts, _retrieval_trace = await container.chat_pipeline.retrieve(
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

    return prepared
