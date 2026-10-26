from __future__ import annotations

import os

import asyncio
import json
import logging
from pathlib import Path
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING

from app.core.exceptions import ConfigurationError, ModelOutputError, RetrievalError
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
from app.agentic.candidate_filter import filter_candidates
from app.agentic.hyde import build_hyde_weights
from app.agentic.llm_rerank import RerankDocument, rerank_documents
from app.agentic.query_expansion import expand_query, fuse_rankings
from app.utils.coerce import as_dict, as_list, as_str, as_str_list
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


@dataclass(frozen=True)
class DiscoveryCandidates:
    """Deterministic discovery output, before the LLM reading plan."""

    discovered: list[SearchHit]
    candidates: list[SearchHit]
    version_filter: dict[str, object]


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
        # Env overrides for the retrieval budget: these are the knobs that decide how
        # much of the ranked list is even considered, and they have to be sweepable
        # without editing a mounted config file.
        self.discovery_top_k = _positive_int(
            "RAG_DISCOVERY_TOP_K",
            int(agent_settings.get("discovery_top_k", DEFAULT_DISCOVERY_TOP_K)),
        )
        self.max_read_docs = _positive_int(
            "RAG_MAX_READ_DOCS",
            int(agent_settings.get("max_read_docs", DEFAULT_MAX_READ_DOCS)),
        )
        self.max_read_chunks_per_doc = int(
            agent_settings.get("max_read_chunks_per_doc", DEFAULT_MAX_READ_CHUNKS_PER_DOC)
        )
        self.max_context_text_chars = int(
            agent_settings.get("max_context_text_chars", DEFAULT_MAX_CONTEXT_TEXT_CHARS)
        )
        self.include_stale_default = bool(
            agent_settings.get("freshness_policy", {}).get("include_stale", False)
        )
        # Let the model take part in the FIRST hop. On by default: measured
        # +3.1pt recall@12 on two independent samples, both significant
        # (p=0.004 and p=0.003). It costs one extra model call per query; set
        # RAG_QUERY_EXPANSION_ENABLED=false for a strictly lexical baseline.
        self.query_expansion_enabled = _flag("RAG_QUERY_EXPANSION_ENABLED", True)
        self.query_expansion_variants = _positive_int(
            "RAG_QUERY_EXPANSION_VARIANTS", 3
        )
        # Reorder the first-hop candidates with the SAME generation model: no
        # separate reranker is introduced. Off by default until a measurement on
        # the deployment's own corpus justifies the extra call per query.
        self.llm_rerank_enabled = _flag("RAG_LLM_RERANK_ENABLED", False)
        # Remove candidates the model judges irrelevant before the read plan sees them.
        # BM25 scores cannot do this: measured on the Chinese-law corpus, top-6 scores
        # are indistinguishable whether or not the answer is in the pool.
        self.candidate_filter_enabled = _flag("RAG_CANDIDATE_FILTER_ENABLED", True)
        # HyDE + Rocchio feedback: highest value-per-cost step measured in the 2026
        # survey (docs/research/industry-rag-2026.md). Off by default so the A/B is
        # explicit; enable with RAG_HYDE_ENABLED=true.
        self.hyde_enabled = _flag("RAG_HYDE_ENABLED", False)
        self.llm_rerank_top_n = _positive_int("RAG_LLM_RERANK_TOP_N", 10)

    async def discovery_candidates(
        self,
        query: str,
        kb_id: str = "default",
        filters: dict[str, object] | None = None,
        *,
        depth: int = 1,
    ) -> "DiscoveryCandidates":
        """Run the discovery stages (LLM-free unless query expansion is enabled).

        Exposed so retrieval quality can be measured on its own: "did the wiki
        BM25 index surface the right document?" is a different question from
        "did the synthesized answer use it?", and only this stage is
        deterministic enough to compare across changes.
        """
        discovered = await self._discover(query, kb_id, filters, depth=depth)
        candidates, version_filter = self._select_versions(discovered)
        if self.llm_rerank_enabled:
            candidates = await self._rerank(query, candidates)
        if self.candidate_filter_enabled:
            candidates = await self._filter_candidates(query, candidates)
        return DiscoveryCandidates(
            discovered=discovered,
            candidates=candidates,
            version_filter=version_filter,
        )

    async def retrieve(
        self, payload: ChatRequest, query: str, *, depth: int = 1
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

            staged = await self.discovery_candidates(
                query, kb_id, filters, depth=depth
            )
            discovered = staged.discovered
            candidates = staged.candidates
            version_filter = staged.version_filter
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
        self,
        query: str,
        kb_id: str,
        filters: dict[str, object] | None,
        *,
        depth: int = 1,
    ) -> list[SearchHit]:
        # Widen the candidate pool instead of only rephrasing the query. Measured on
        # the Chinese-law corpus: hit@12 = 0.37 while hit@100 = 0.60, i.e. the document
        # that answers the question is often already ranked below the first twelve. A
        # corrective round that only changes the wording never looks there; growing the
        # depth does. Cost stays bounded because each round only reads the NEW
        # candidates - see _new_candidates below.
        fetch_k = (
            self.discovery_top_k
            * DISCOVERY_OVERFETCH_MULTIPLIER
            * max(1, depth)
        )
        if self.hyde_enabled:
            # HyDE + Rocchio: the model writes the passage that would answer the
            # question, and that passage becomes TERM WEIGHTS rather than being
            # appended to the query string. Measured (SIGIR'26, 14 datasets):
            # Recall@20 28.8 BM25 -> 29.5 concatenation -> 34.7 Rocchio feedback.
            hits = await self._discover_with_hyde(
                query, kb_id=kb_id, filters=filters, fetch_k=fetch_k
            )
            return [
                hit
                for hit in hits
                if (hit.chunk.metadata or {}).get("wiki_kind") == "source"
            ]

        if not self.query_expansion_enabled:
            # BM25 scoring (and a possible incremental sync) is CPU-bound; keep it
            # off the event loop so concurrent requests stay responsive.
            hits = await asyncio.to_thread(
                self.wiki_searcher.search,
                query,
                top_k=fetch_k,
                kb_id=kb_id,
                metadata_filters=filters,
            )
        else:
            hits = await self._discover_with_expansion(
                query, kb_id=kb_id, filters=filters, fetch_k=fetch_k
            )
        # Only source pages map to a deep-readable parsed artifact; topic/index
        # pages are aggregates with no artifact of their own.
        return [
            hit
            for hit in hits
            if (hit.chunk.metadata or {}).get("wiki_kind") == "source"
        ]

    async def _discover_with_hyde(
        self,
        query: str,
        *,
        kb_id: str,
        filters: dict[str, object] | None,
        fetch_k: int,
    ) -> list[SearchHit]:
        """One weighted BM25 query built from the query plus a hypothetical passage.

        The original query keeps weight 1.0 for every term, so feedback can only add
        signal - it can never erase what the user actually typed. RM3-style interpolation
        over separate result lists is deliberately NOT used here: the reported gain comes
        from term weighting inside one query, not from fusing several queries.
        """
        try:
            term_weights = await build_hyde_weights(
                query,
                self.generation_client,
                tokenize=self.wiki_searcher.tokenize,
                document_frequencies=self.wiki_searcher.document_frequencies()[0],
                n_docs=self.wiki_searcher.document_frequencies()[1],
            )
        except ModelOutputError:
            raise
        with self.tracing_manager.span(
            "agent.hyde", {"agent.query": query, "agent.weighted_terms": len(term_weights)}
        ):
            # BM25 scoring is CPU-bound; keep it off the event loop.
            weighted = await asyncio.to_thread(
                self.wiki_searcher.search,
                query,
                top_k=fetch_k,
                kb_id=kb_id,
                metadata_filters=filters,
                term_weights=term_weights,
            )
        if not self.query_expansion_enabled:
            return weighted
        # Both signals: the weighted query (semantic recall from the hypothetical
        # passage) fused with keyword variants (literal recall). Fuse rather than pick,
        # because they fail on different questions.
        expanded = await self._discover_with_expansion(
            query, kb_id=kb_id, filters=filters, fetch_k=fetch_k
        )
        by_id: dict[str, SearchHit] = {hit.chunk.doc_id: hit for hit in expanded}
        for hit in weighted:
            by_id.setdefault(hit.chunk.doc_id, hit)
        fused = fuse_rankings(
            [
                [hit.chunk.doc_id for hit in weighted],
                [hit.chunk.doc_id for hit in expanded],
            ]
        )
        merged = [
            SearchHit(chunk=by_id[doc_id].chunk, score=score) for doc_id, score in fused
        ]
        logger.info(
            "hyde+expansion: %d weighted + %d expanded -> %d fused",
            len(weighted),
            len(expanded),
            len(merged),
        )
        return merged[:fetch_k]

    async def _discover_with_expansion(
        self,
        query: str,
        *,
        kb_id: str,
        filters: dict[str, object] | None,
        fetch_k: int,
    ) -> list[SearchHit]:
        """Search the original query plus model-generated keyword variants, fused.

        The original query always participates: expansion can only add candidates,
        never remove the documents BM25 already found for what the user typed.
        A gateway or contract failure propagates - a request that silently lost its
        expansion would be a different request than the one that was configured.
        """
        with self.tracing_manager.span(
            "agent.query_expansion", {"agent.query": query}
        ):
            variants = await expand_query(
                query,
                self.generation_client,
                max_variants=self.query_expansion_variants,
            )
        queries = [query, *variants]
        rankings: list[list[str]] = []
        hits_by_id: dict[str, SearchHit] = {}
        for candidate_query in queries:
            # BM25 scoring (and a possible incremental sync) is CPU-bound; keep it
            # off the event loop so concurrent requests stay responsive.
            hits = await asyncio.to_thread(
                self.wiki_searcher.search,
                candidate_query,
                top_k=fetch_k,
                kb_id=kb_id,
                metadata_filters=filters,
            )
            # Only source pages map to a deep-readable parsed artifact; topic and
            # index pages are aggregates with no artifact of their own.
            source_hits = [
                hit
                for hit in hits
                if (hit.chunk.metadata or {}).get("wiki_kind") == "source"
            ]
            for hit in source_hits:
                hits_by_id.setdefault(hit.chunk.doc_id, hit)
            rankings.append([hit.chunk.doc_id for hit in source_hits])
        fused = fuse_rankings(rankings)
        logger.info(
            "query expansion: %d variants, fused %d documents",
            len(variants),
            len(fused),
        )
        merged: list[SearchHit] = []
        for doc_id, score in fused:
            hit = hits_by_id[doc_id]
            merged.append(
                SearchHit(chunk=hit.chunk, score=score)
            )
        return merged[:fetch_k]

    async def _filter_candidates(
        self, query: str, hits: list[SearchHit]
    ) -> list[SearchHit]:
        """Drop candidates the model says cannot answer the question.

        The read plan chooses what to deep-read from this list; twelve unrelated
        candidates make that choice harder than four plausible ones. Order is preserved
        for whatever survives, and an unusable verdict keeps the full list.
        """
        if len(hits) <= 1:
            return hits
        payload = [
            (str(hit.chunk.title or ""), str(hit.chunk.text or "")) for hit in hits
        ]
        with self.tracing_manager.span("agent.candidate_filter", {"agent.query": query}):
            kept = await filter_candidates(query, payload, self.generation_client)
        logger.info("candidate filtering: %d -> %d candidates", len(hits), len(kept))
        if not kept:
            # The model judged the entire pool weak: the answer was not recalled. Hand
            # the caller an empty candidate list so the read plan finds nothing and
            # verification reports insufficient evidence, instead of reading a document
            # everyone (human or model) knows is off-topic.
            return []
        return [hits[index] for index in kept]

    async def _rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        """Reorder candidates by model judgement, keeping every candidate.

        Runs after the version filter so stale versions never consume rerank
        attention, and before the read plan chooses what to read - the plan can
        only work with the order it is given.
        """
        if len(hits) <= 1:
            return hits
        by_doc_id = {hit.chunk.doc_id: hit for hit in hits}
        documents = [
            RerankDocument(
                doc_id=hit.chunk.doc_id,
                title=str(hit.chunk.title or ""),
                text=str(hit.chunk.text or ""),
            )
            for hit in hits
        ]
        with self.tracing_manager.span("agent.llm_rerank", {"agent.query": query}):
            ordered_ids = await rerank_documents(
                query,
                documents,
                self.generation_client,
                top_n=self.llm_rerank_top_n,
            )
        reordered: list[SearchHit] = []
        seen: set[str] = set()
        for doc_id in ordered_ids:
            hit = by_doc_id.get(doc_id)
            if hit is None or doc_id in seen:
                continue
            seen.add(doc_id)
            reordered.append(hit)
        # Defensive: the rerank must be a permutation of what it was given, but a
        # candidate must never disappear from the pool because of an ordering step.
        for hit in hits:
            if hit.chunk.doc_id not in seen:
                reordered.append(hit)
        return reordered

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
        groups_report: list[dict[str, object]] = []
        filter_report: dict[str, object] = {
            "include_stale": self.include_stale_default,
            "groups": groups_report,
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
            groups_report.append(
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
            entry_doc_id = as_str(entry.get("doc_id"))
            if entry_doc_id in seen:
                continue
            seen.add(entry_doc_id)
            deduped.append(entry)
        # An empty selection is a legitimate structured decision: none of the
        # candidates plausibly contains the answer, so nothing is read.
        return {"selected_docs": deduped[: self.max_read_docs]}

    def _read_documents(
        self, plan: dict[str, object], candidates: list[SearchHit]
    ) -> tuple[list[dict[str, object]], list[str], list[str]]:
        focus_by_doc: dict[str, set[str]] = {}
        for entry in as_list(plan.get("selected_docs")):
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
    ) -> list[dict[str, object]]:
        return select_chunks_for_focus(
            raw_chunks,
            focus,
            doc_id=doc_id,
            max_chunks=self.max_read_chunks_per_doc,
        )

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


def select_chunks_for_focus(
    raw_chunks: Sequence[object],
    focus: set[str],
    *,
    doc_id: str,
    max_chunks: int,
) -> list[dict[str, object]]:
    """Choose the chunks to deep-read for a reading plan's focus sections.

    Extracted from ``AgenticDiscovery`` so the deterministic part of the read
    plan is directly evaluable (``app/eval/read_plan.py``) without constructing a
    pipeline or calling a model.

    A focus term selects a **section**, not a single chunk: every chunk belonging
    to that section is read. Consuming a term after its first hit used to drop the
    rest of the section, so a document whose heading spanned several chunks
    answered "not found" for evidence that existed in a sibling chunk.
    """
    chunks: list[dict[str, object]] = [
        chunk for chunk in raw_chunks if isinstance(chunk, dict)
    ]
    if not focus:
        return chunks[:max_chunks]
    matched: list[dict[str, object]] = []
    for chunk in chunks:
        metadata = as_dict(chunk.get("metadata"))
        # Section matching runs over the heading path and the chunk title.
        haystack = [*as_str_list(metadata.get("hierarchy_path")), as_str(chunk.get("title"))]
        for term in focus:
            if _section_matches(term, haystack):
                matched.append(chunk)
                break
    if matched:
        return matched[:max_chunks]

    if not _has_heading_structure(chunks):
        # The document has no headings at all - a scanned PDF, an exported email,
        # a flat web page. A focus section cannot resolve against structure that
        # does not exist, so this is a property of the document, not a planner
        # contract violation: reading it (bounded) is the correct interpretation of
        # "read the relevant part of this document". Failing here made every
        # request against a heading-less corpus an error.
        logger.warning(
            "reading plan named sections %s but document %s has no heading "
            "structure; reading the document instead",
            sorted(focus),
            doc_id,
        )
        return chunks[:max_chunks]

    # The document DOES have headings and the plan still named none of them: the
    # planner and the artifact disagree, so fail visibly rather than silently
    # reading something else.
    raise ModelOutputError(
        f"reading plan focus sections {sorted(focus)} do not match any section of "
        f"document {doc_id} (the document has {_heading_count(chunks)} headed chunks)"
    )


def _has_heading_structure(chunks: Sequence[dict[str, object]]) -> bool:
    """True when at least one chunk carries a heading path."""
    for chunk in chunks:
        metadata = as_dict(chunk.get("metadata"))
        if as_str_list(metadata.get("hierarchy_path")):
            return True
    return False


def _heading_count(chunks: Sequence[dict[str, object]]) -> int:
    return sum(
        1
        for chunk in chunks
        if as_str_list(as_dict(chunk.get("metadata")).get("hierarchy_path"))
    )


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


_TRUE_TOKENS = frozenset({"1", "true", "yes", "on"})
_FALSE_TOKENS = frozenset({"0", "false", "no", "off"})


def _flag(name: str, default: bool) -> bool:
    """Boolean env parsing that refuses to guess (a typo must not disable a stage)."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    token = raw.strip().lower()
    if token in _TRUE_TOKENS:
        return True
    if token in _FALSE_TOKENS:
        return False
    raise ConfigurationError(
        f"{name} must be a boolean, got {raw!r}"
    )


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer, got {raw!r}") from exc
    if value < 1:
        raise ConfigurationError(f"{name} must be >= 1, got {value}")
    return value
