from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.core.config import AppContainer


@dataclass
class ReplayDiff:
    field: str
    original: object
    replayed: object


@dataclass
class ReplayResult:
    trace_id: str
    status: str
    diffs: list[ReplayDiff]
    original_contexts: list[dict]
    replayed_contexts: list[dict]
    original_answer: str | None
    replayed_answer: str | None


async def replay_trace(container: AppContainer, trace_id: str) -> ReplayResult:
    from app.schemas.chat import ChatRequest

    trace_record = container.trace_store.get(trace_id)
    if trace_record is None:
        return ReplayResult(
            trace_id=trace_id,
            status="trace_not_found",
            diffs=[],
            original_contexts=[],
            replayed_contexts=[],
            original_answer=None,
            replayed_answer=None,
        )

    query = trace_record.query
    if not query:
        return ReplayResult(
            trace_id=trace_id,
            status="trace_missing_query",
            diffs=[],
            original_contexts=trace_record.contexts,
            replayed_contexts=[],
            original_answer=trace_record.answer,
            replayed_answer=None,
        )

    kb_id = trace_record.kb_id or "default"
    tenant_id = trace_record.tenant_id
    session_id = trace_record.session_id
    metadata_filters = (
        trace_record.retrieval_params.get("metadata_filters")
        if trace_record.retrieval_params
        else None
    )

    chat_request = ChatRequest(
        query=query,
        kb_id=kb_id,
        tenant_id=tenant_id,
        session_id=session_id,
        metadata_filters=metadata_filters,
    )

    chat_response = await container.chat_pipeline.run(chat_request)

    diffs: list[ReplayDiff] = []

    original_chunk_ids = trace_record.contexts
    if isinstance(original_chunk_ids, list):
        original_ids = {
            ctx.get("chunk_id") for ctx in original_chunk_ids if isinstance(ctx, dict)
        }
    else:
        original_ids = set()

    replayed_ids = {
        ctx.get("chunk_id")
        for ctx in chat_response.contexts
        if isinstance(ctx, dict)
    }

    if original_ids != replayed_ids:
        diffs.append(
            ReplayDiff(
                field="context_chunk_ids",
                original=sorted(original_ids),
                replayed=sorted(replayed_ids),
            )
        )

    original_citation_ids = {
        c.get("chunk_id") for c in trace_record.citations if isinstance(c, dict)
    }
    replayed_citation_ids = {
        c.chunk_id for c in chat_response.citations
    }
    if original_citation_ids != replayed_citation_ids:
        diffs.append(
            ReplayDiff(
                field="citation_chunk_ids",
                original=sorted(original_citation_ids),
                replayed=sorted(replayed_citation_ids),
            )
        )

    if trace_record.answer and chat_response.answer:
        if trace_record.answer.strip() != chat_response.answer.strip():
            diffs.append(
                ReplayDiff(
                    field="answer",
                    original=trace_record.answer.strip()[:200],
                    replayed=chat_response.answer.strip()[:200],
                )
            )

    return ReplayResult(
        trace_id=trace_id,
        status="diff" if diffs else "identical",
        diffs=diffs,
        original_contexts=trace_record.contexts,
        replayed_contexts=chat_response.contexts,
        original_answer=trace_record.answer,
        replayed_answer=chat_response.answer,
    )
