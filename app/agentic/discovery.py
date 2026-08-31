from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

from app.core.exceptions import ModelOutputError, RetrievalError
from app.core.tracing import TraceSession
from app.retrieval.context_builder import build_contexts
from app.retrieval.filters import (
    infer_metadata_filters,
    merge_metadata_filters,
    sanitize_metadata_filters,
)
from app.retrieval.versioning import version_sort_key
from app.schemas.chat import ChatRequest
from app.schemas.chunk import Chunk
from app.retrieval.hits import SearchHit
from app.utils.json_utils import parse_json_object
from pydantic import ValidationError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.core.config import AppConfig
    from app.core.tracing import TraceStore, TracingManager
    from app.model_client.generation import GenerationClient
    from app.wiki.search import WikiSearcher

logger = logging.getLogger(__name__)

DEFAULT_DISCOVERY_TOP_K = 12
DEFAULT_MAX_READ_DOCS = 4
DEFAULT_MAX_READ_CHUNKS_PER_DOC = 24
DEFAULT_MAX_CONTEXT_TEXT_CHARS = 6000
# Topic/index pages consume ranking slots before the source-only filter, and
# stale versions of the same source_key can crowd their latest version out of
# a narrow top-K. Version selection therefore runs over an over-fetched
# candidate set so "latest wins per source_key" holds globally, not just
# within the first top-K hits.
DISCOVERY_OVERFETCH_MULTIPLIER = 4

READING_PLAN_SYSTEM_PROMPT = (
    "You are a RAG document reader planner. Given a question and a set of candidate "
    "documents (each with a manifest: summary, headings, key passages, and version "
    "metadata), decide which documents to read and which sections to focus on. "
    "Select only documents that can plausibly contain the answer. Use focus_sections "
    "to narrow reading to the relevant headings; leave it empty to read the whole "
    "document. Return only compact JSON."
)


class AgenticDiscovery:
    """Wiki-first retrieval for the agentic engine.

    A deterministic, metadata-driven pipeline:

      1. discovery  — BM25 over the document-level wiki manifest (first hop);
      2. version    — deterministic per-``source_key`` latest-version selection
      over an over-fetched candidate set (global latest-wins, not top-K-local);
      3. read plan  — an LLM structured decision on which docs/sections to read;
      4. deep read  — read the chosen sections from the durable parsed artifacts.

    Discovery and version selection are deterministic (testable, reproducible).
    The read plan is a single LLM call with a strict JSON contract: gateway
    failures and contract violations raise and fail the request visibly —
    there is no degraded read-all path.
    """

    def __init__(
        self,
        config: "AppConfig",
        wiki_searcher: "WikiSearcher",
        generation_client: "GenerationClient",
        trace_store: "TraceStore",
        tracing_manager: "TracingManager",
    ) -> None:
        self.config = config
        self.wiki_searcher = wiki_searcher
        self.generation_client = generation_client
        self.trace_store = trace_store
        self.tracing_manager = tracing_manager
        self.parsed_dir: Path = config.parsed_dir
        agent_settings = config.settings.get("agent", {})
        self.discovery_top_k = int(agent_settings.get("discovery_top_k", DEFAULT_DISCOVERY_TOP_K))
        self.max_read_docs = int(agent_settings.get("max_read_docs", DEFAULT_MAX_READ_DOCS))
        self.max_read_chunks_per_doc = int(
            agent_settings.get("max_read_chunks_per_doc", DEFAULT_MAX_READ_CHUNKS_PER_DOC)
        )
        self.max_context_text_chars = int(
            agent_settings.get("max_context_text_chars", DEFAULT_MAX_CONTEXT_TEXT_CHARS)
        )
        self.include_stale_default = bool(
            agent_settings.get("freshness_policy", {}).get("include_stale", False)
        )

    async def retrieve(
        self, payload: ChatRequest, query: str
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        kb_id = payload.kb_id or "default"
        started = perf_counter()
        with self.tracing_manager.span(
            "agentic.discovery",
            {
                "agentic.query": query,
                "agentic.kb_id": kb_id,
                "agentic.session_id": payload.session_id or "",
            },
        ):
            trace = TraceSession()
            filters = merge_metadata_filters(
                payload.metadata_filters, infer_metadata_filters(query)
            )
            public_filters = sanitize_metadata_filters(filters)

            discovered = await self._discover(query, kb_id, filters)
            candidates, version_filter = self._select_versions(discovered)
            plan = await self._plan_reads(query, candidates)
            contexts, read_doc_ids, read_chunk_ids = self._read_documents(plan, candidates)

            trace.record("query", query)
            trace.record("original_query", payload.query)
            trace.record("kb_id", kb_id)
            trace.record("session_id", payload.session_id)
            trace.record("sample_id", payload.sample_id)
            trace.record("retrieved_chunk_ids", read_chunk_ids)
            trace.record("contexts", contexts)
            trace.record(
                "retrieval_params",
                {
                    "engine": "agentic_wiki",
                    "discovery_top_k": self.discovery_top_k,
                    "discovery_hits": [
                        {
                            "doc_id": hit.chunk.doc_id,
                            "title": hit.chunk.title,
                            "score": round(hit.score, 6),
                            "source_key": (hit.chunk.metadata or {}).get("source_key"),
                            "version": (hit.chunk.metadata or {}).get("version"),
                            "effective_date": (hit.chunk.metadata or {}).get("effective_date"),
                            "is_latest_version": (hit.chunk.metadata or {}).get(
                                "is_latest_version"
                            ),
                        }
                        for hit in discovered
                    ],
                    "version_filter": version_filter,
                    "reading_plan": plan,
                    "read_doc_ids": read_doc_ids,
                    "read_chunk_ids": read_chunk_ids,
                    "metadata_filters": public_filters,
                },
            )
            trace.record(
                "step_latencies",
                {
                    "agentic_discovery_seconds": round(perf_counter() - started, 4)
                },
            )
            final_trace = trace.finish()
            self.trace_store.save_raw(final_trace)
            return contexts, final_trace

    async def _discover(
        self, query: str, kb_id: str, filters: dict[str, object] | None
    ) -> list[SearchHit]:
        fetch_k = self.discovery_top_k * DISCOVERY_OVERFETCH_MULTIPLIER
        # BM25 scoring (and a possible incremental sync) is CPU-bound; keep it
        # off the event loop so concurrent requests stay responsive.
        hits = await asyncio.to_thread(
            self.wiki_searcher.search,
            query,
            top_k=fetch_k,
            kb_id=kb_id,
            metadata_filters=filters,
        )
        # Only source pages map to a deep-readable parsed artifact; topic/index
        # pages are aggregates with no artifact of their own.
        return [
            hit
            for hit in hits
            if (hit.chunk.metadata or {}).get("wiki_kind") == "source"
        ]

    def _select_versions(
        self, discovered: list[SearchHit]
    ) -> tuple[list[SearchHit], dict[str, object]]:
        """Deterministically keep the latest version per source_key.

        Ranking uses the shared ``version_sort_key`` (effective_date, then version
        tuple, then discovery score) — the same rule the wiki version ledger
        uses. Never an LLM judgment. Runs over the whole over-fetched candidate
        set, so a stale version ranking above its latest sibling cannot hide it.
        """
        grouped: dict[tuple[str, str], list[SearchHit]] = {}
        ungrouped: list[SearchHit] = []
        for hit in discovered:
            metadata = hit.chunk.metadata or {}
            source_key = metadata.get("source_key")
            if source_key:
                key = (str(metadata.get("kb_id", "default")), str(source_key))
                grouped.setdefault(key, []).append(hit)
            else:
                ungrouped.append(hit)

        selected = list(ungrouped)
        filter_report: dict[str, object] = {
            "include_stale": self.include_stale_default,
            "groups": [],
        }
        for (kb_id, source_key), group in grouped.items():
            ranked = sorted(
                group,
                key=lambda hit: version_sort_key(
                    hit.chunk.metadata or {}, score=hit.score
                ),
                reverse=True,
            )
            winner = ranked[0]
            selected.append(winner)
            if self.include_stale_default:
                selected.extend(ranked[1:])
            dropped = [] if self.include_stale_default else [hit.chunk.doc_id for hit in ranked[1:]]
            filter_report["groups"].append(
                {
                    "kb_id": kb_id,
                    "source_key": source_key,
                    "winner": winner.chunk.doc_id,
                    "selected": [hit.chunk.doc_id for hit in ranked]
                    if self.include_stale_default
                    else [winner.chunk.doc_id],
                    "dropped": dropped,
                }
            )
        # Order the surviving candidates by discovery score so the LLM sees the
        # best evidence first. Stable: score ties keep BM25 discovery order.
        selected = sorted(selected, key=lambda hit: hit.score, reverse=True)
        return selected[: self.discovery_top_k], filter_report

    async def _plan_reads(
        self, query: str, candidates: list[SearchHit]
    ) -> dict[str, object]:
        if not candidates:
            return {"selected_docs": []}
        manifest = [
            {
                "doc_id": hit.chunk.doc_id,
                "title": hit.chunk.title,
                "source_path": (hit.chunk.metadata or {}).get(
                    "original_source_path"
                )
                or hit.chunk.source_path,
                "doc_type": (hit.chunk.metadata or {}).get("doc_type"),
                "version": (hit.chunk.metadata or {}).get("version"),
                "effective_date": (hit.chunk.metadata or {}).get("effective_date"),
                "is_latest_version": (hit.chunk.metadata or {}).get("is_latest_version"),
                "manifest": hit.chunk.text,
            }
            for hit in candidates
        ]
        schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "reading_plan",
                "schema": {
                    "type": "object",
                    "properties": {
                        "selected_docs": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "doc_id": {"type": "string"},
                                    "focus_sections": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "reason": {"type": "string"},
                                },
                                "required": ["doc_id", "focus_sections", "reason"],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["selected_docs"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }
        messages = [
            {"role": "system", "content": READING_PLAN_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Decide which candidate documents and sections to read to answer "
                    "the question. Return JSON with key selected_docs (array of "
                    '{doc_id, focus_sections, reason}).\n\n'
                    "Input JSON: "
                    + json.dumps(
                        {"question": query, "candidates": manifest}, ensure_ascii=False
                    )
                ),
            },
        ]
        with self.tracing_manager.span(
            "agentic.reading_plan", {"agentic.query": query}
        ):
            result = await self.generation_client.generate(messages, response_format=schema)
        parsed = parse_json_object(str(result.get("content") or ""))
        return self._parse_plan(parsed, candidates)

    def _parse_plan(
        self, parsed: dict[str, object], candidates: list[SearchHit]
    ) -> dict[str, object]:
        known = {hit.chunk.doc_id for hit in candidates}
        raw_docs = parsed.get("selected_docs")
        if not isinstance(raw_docs, list):
            raise ModelOutputError("reading_plan.selected_docs must be an array")
        entries: list[dict[str, object]] = []
        for raw in raw_docs:
            if not isinstance(raw, dict):
                raise ModelOutputError(
                    "reading_plan.selected_docs entries must be objects"
                )
            doc_id = str(raw.get("doc_id") or "")
            if doc_id not in known:
                raise ModelOutputError(
                    f"reading_plan referenced unknown doc_id {doc_id!r}; "
                    f"known candidates: {sorted(known)}"
                )
            focus = raw.get("focus_sections")
            if not isinstance(focus, list):
                raise ModelOutputError(
                    "reading_plan.focus_sections must be an array"
                )
            focus_sections = [
                str(item).strip()
                for item in focus
                if str(item).strip()
            ]
            entries.append(
                {
                    "doc_id": doc_id,
                    "focus_sections": focus_sections,
                    "reason": str(raw.get("reason") or ""),
                }
            )
        deduped: list[dict[str, object]] = []
        seen: set[str] = set()
        for entry in entries:
            if entry["doc_id"] in seen:
                continue
            seen.add(entry["doc_id"])
            deduped.append(entry)
        # An empty selection is a legitimate structured decision: none of the
        # candidates plausibly contains the answer, so nothing is read.
        return {"selected_docs": deduped[: self.max_read_docs]}

    def _read_documents(
        self, plan: dict[str, object], candidates: list[SearchHit]
    ) -> tuple[list[dict[str, object]], list[str], list[str]]:
        focus_by_doc: dict[str, set[str]] = {}
        for entry in plan.get("selected_docs", []):  # type: ignore[union-attr]
            if isinstance(entry, dict):
                focus_by_doc[str(entry.get("doc_id"))] = {
                    str(item)
                    for item in (entry.get("focus_sections") or [])
                    if str(item).strip()
                }

        metadata_by_doc = {
            hit.chunk.doc_id: dict(hit.chunk.metadata or {}) for hit in candidates
        }
        contexts: list[dict[str, object]] = []
        read_doc_ids: list[str] = []
        read_chunk_ids: list[str] = []
        for doc_id in list(focus_by_doc)[: self.max_read_docs]:
            metadata = metadata_by_doc.get(doc_id)
            if metadata is None:
                continue
            doc_contexts = self._contexts_for_document(doc_id, focus_by_doc[doc_id], metadata)
            if not doc_contexts:
                continue
            contexts.extend(doc_contexts)
            read_doc_ids.append(doc_id)
            read_chunk_ids.extend(str(item.get("chunk_id")) for item in doc_contexts)
        return contexts, read_doc_ids, read_chunk_ids

    def _contexts_for_document(
        self, doc_id: str, focus: set[str], metadata: dict[str, object]
    ) -> list[dict[str, object]]:
        artifact = self._load_artifact(doc_id)
        raw_chunks = artifact.get("chunks", [])
        selected = self._select_chunks(
            raw_chunks if isinstance(raw_chunks, list) else [], focus, doc_id
        )
        hits: list[SearchHit] = []
        for index, raw_chunk in enumerate(selected):
            try:
                chunk = Chunk.model_validate(raw_chunk)
            except (ValidationError, TypeError) as exc:
                raise RetrievalError(
                    f"parsed artifact {doc_id}.json contains a malformed chunk "
                    f"at index {index}: {exc}"
                ) from exc
            hits.append(SearchHit(chunk=chunk, score=0.0))
        if not hits:
            return []
        # query=None is deliberate: the read plan already scoped these chunks
        # to the question, so a query-term coverage promotion would only
        # reorder (not add) sections we deliberately chose — and it would let
        # low-relevance chunks with query-term overlap climb above the planned
        # focus.
        base_contexts = build_contexts(
            hits,
            limit=len(hits),
            query=None,
            max_text_chars=self.max_context_text_chars,
        )
        return [self._overlay_discovery_metadata(context, metadata) for context in base_contexts]

    def _overlay_discovery_metadata(
        self, context: dict[str, object], metadata: dict[str, object]
    ) -> dict[str, object]:
        is_latest = metadata.get("is_latest_version")
        overlay: dict[str, object] = {
            "wiki_kind": "source",
            "source_key": metadata.get("source_key"),
            "doc_type": metadata.get("doc_type"),
            "effective_date": metadata.get("effective_date"),
            "version": metadata.get("version"),
            "is_latest_version": is_latest,
            "superseded_by": metadata.get("superseded_by"),
        }
        if is_latest is True:
            overlay["freshness_tier"] = "primary"
            overlay["evidence_role"] = "primary"
        elif is_latest is False:
            overlay["freshness_tier"] = "stale"
            overlay["evidence_role"] = "supporting"
        merged = dict(context)
        for key, value in overlay.items():
            if value is not None:
                merged[key] = value
        return merged

    def _select_chunks(
        self, raw_chunks: list[object], focus: set[str], doc_id: str
    ) -> list[object]:
        chunks = [chunk for chunk in raw_chunks if isinstance(chunk, dict)]
        if not focus:
            return chunks[: self.max_read_chunks_per_doc]
        matched: list[object] = []
        matched_terms: set[str] = set()
        for chunk in chunks:
            metadata = chunk.get("metadata") or {}
            hierarchy = metadata.get("hierarchy_path") or []
            haystack = [str(item) for item in hierarchy] + [str(chunk.get("title") or "")]
            for term in focus:
                if term in matched_terms:
                    continue
                if _section_matches(term, haystack):
                    matched.append(chunk)
                    matched_terms.add(term)
                    break
        if not matched:
            # No fallback to a whole-document read: the plan named sections
            # that do not resolve, which is a contract violation between the
            # planner LLM and the artifact's heading structure. Fail visibly.
            raise ModelOutputError(
                f"reading plan focus sections {sorted(focus)} do not match any "
                f"section of document {doc_id}"
            )
        return matched[: self.max_read_chunks_per_doc]

    def _load_artifact(self, doc_id: str) -> dict[str, object]:
        path = self.parsed_dir / f"{doc_id}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RetrievalError(
                f"parsed artifact for discovered document {doc_id} is missing "
                f"or corrupt ({path}): {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise RetrievalError(
                f"parsed artifact for discovered document {doc_id} is not a "
                f"JSON object ({path})"
            )
        return payload


def _section_matches(focus_term: str, haystack: list[str]) -> bool:
    term = focus_term.strip().lower()
    if not term:
        return False
    for candidate in haystack:
        lowered = candidate.lower()
        if not lowered:
            continue
        if term == lowered or term in lowered or lowered in term:
            return True
    return False
