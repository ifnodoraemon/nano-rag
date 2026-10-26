from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING, TypedDict

from app.agentic.discovery import AgenticDiscovery
from app.agentic.evidence_view import EvidenceDriller
from app.core.exceptions import ConfigurationError, ModelOutputError
from app.generation.answer_formatter import AnswerFormatter
from app.generation.citation_check import check_citations
from app.generation.prompt_builder import PromptBuilder
from app.model_client.generation import GenerationClient
from app.retrieval.graph_expander import GraphExpander
from app.schemas.chat import ChatRequest, ChatResponse
from app.utils.json_utils import parse_json_object
from langgraph.graph import END, StateGraph

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from app.core.config import AppConfig
    from app.core.tracing import TraceStore, TracingManager
    from app.retrieval.graph_store import GraphStore
    from app.wiki.search import WikiSearcher


ANSWER_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "answer_structure",
        "schema": {
            "type": "object",
            "properties": {
                "is_answerable": {
                    "type": "boolean",
                    "description": "如果上下文中完全找不到问题核心实体，请设为 false",
                },
                "missing_entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "列出找不到的具体专有名词，如果找到了则为空数组",
                },
                "extracted_answer": {
                    "type": "string",
                    "description": "带引用的极简答案，例如：xxx为xxx [C1]。",
                },
                "supporting_claims": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "如: - [factual] <证据支撑点 1> [C#]",
                },
            },
            "required": [
                "is_answerable",
                "missing_entities",
                "extracted_answer",
                "supporting_claims",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

VERIFY_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "evidence_verification",
        "schema": {
            "type": "object",
            "properties": {
                "sufficient": {"type": "boolean"},
                "coverage_ratio": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
                "missing_terms": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "follow_up_queries": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "reason": {"type": "string"},
            },
            "required": [
                "sufficient",
                "coverage_ratio",
                "missing_terms",
                "follow_up_queries",
                "reason",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

# How much of the evidence the auditor and the synthesizer actually get to see.
# The defaults are sized for the corpus this was measured on: a statute is one
# chunk of 5k-9k characters, so the per-document ceiling is what keeps one whole
# law readable, and the total keeps twelve of them inside the model's window.
DEFAULT_EVIDENCE_TOTAL_CHARS = 36000
DEFAULT_EVIDENCE_PER_CONTEXT_CHARS = 6000
DEFAULT_EVIDENCE_MIN_PER_CONTEXT_CHARS = 1200
DEFAULT_EVIDENCE_MAX_CONTEXTS = 12


@dataclass(frozen=True)
class EvidenceCheck:
    sufficient: bool
    coverage_ratio: float
    missing_terms: list[str]
    follow_up_queries: list[str]
    has_contexts: bool
    has_conflicts: bool
    reason: str
    # The auditor's first verdict, before its own gaps were searched for in the
    # evidence it had already been shown. ``sufficient`` is the final one.
    initial_sufficient: bool = True
    # True when the auditor was shown every retrieved document in full: neither
    # the retrieved evidence nor the auditor's view was truncated. A "not found"
    # from an auditor that saw only excerpts is "not found in what I was shown",
    # which is a different claim; synthesis needs to know which one it got.
    evidence_complete: bool = True
    # Seconds spent in each auditor round trip, in order. One entry for the first
    # audit; two when the auditor's gaps were searched for in the evidence it
    # already had. Recorded because "the savings came from triggering fewer
    # corrective rounds" and "the savings were eaten by extra audit calls" are
    # otherwise indistinguishable from the outside.
    audit_seconds: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "sufficient": self.sufficient,
            "initial_sufficient": self.initial_sufficient,
            "coverage_ratio": self.coverage_ratio,
            "missing_terms": self.missing_terms,
            "follow_up_queries": self.follow_up_queries,
            "has_contexts": self.has_contexts,
            "has_conflicts": self.has_conflicts,
            "evidence_complete": self.evidence_complete,
            "reason": self.reason,
            "audit_seconds": self.audit_seconds,
            "audit_calls": len(self.audit_seconds),
            "redrill_improved": (
                len(self.audit_seconds) > 1 and not self.initial_sufficient
                and self.sufficient
            ),
        }


class AgentState(TypedDict, total=False):
    payload: ChatRequest
    started_at: float
    resolved_query: str
    subqueries: list[str]
    contexts: list[dict[str, object]]
    trace_id: str
    retrieval_queries: list[str]
    graph_expanded_node_ids: list[str]
    reading_plans: list[dict[str, object]]
    check: EvidenceCheck
    messages: list[dict[str, object]]
    response: ChatResponse
    generation_result: dict[str, object]
    generation_seconds: float


class AgenticReasoningService:
    def __init__(
        self,
        config: AppConfig,
        generation_client: GenerationClient,
        prompt_builder: PromptBuilder,
        answer_formatter: AnswerFormatter,
        trace_store: TraceStore,
        tracing_manager: TracingManager,
        graph_store: GraphStore | None = None,
        wiki_searcher: "WikiSearcher | None" = None,
    ) -> None:
        if wiki_searcher is None:
            raise ConfigurationError(
                "wiki_searcher is required: the agentic engine has no dense "
                "fallback. Set RAG_WIKI_ENABLED=true (the default) so the "
                "document-level discovery layer is built."
            )
        self.config = config
        self.generation_client = generation_client
        self.prompt_builder = prompt_builder
        self.answer_formatter = answer_formatter
        self.trace_store = trace_store
        self.tracing_manager = tracing_manager
        self.graph_expander = GraphExpander(config.parsed_dir, graph_store)
        # The wiki discovery layer is the only retrieval engine. It is a hard
        # dependency (not an optional fallback): RAG_WIKI_ENABLED=false is a
        # misconfiguration that surfaces at startup, not a silent dense reroute.
        self.agentic_discovery = AgenticDiscovery(
            config=config,
            wiki_searcher=wiki_searcher,
            generation_client=generation_client,
            trace_store=trace_store,
            tracing_manager=tracing_manager,
        )
        agent_settings = config.settings.get("agent", {})
        self.max_retrieval_loops = int(agent_settings.get("max_retrieval_loops", 2))
        self.max_subqueries = int(agent_settings.get("max_subqueries", 4))
        self.graph_expansion_mode = os.getenv("RAG_GRAPH_EXPANSION_MODE", "auto").lower()
        # Evidence visibility budget: how much of each retrieved document the
        # auditor and the synthesizer are shown. See app/agentic/evidence_view.py
        # for why a per-document allocation replaced a per-context character cap.
        self.evidence_driller = EvidenceDriller(
            max_contexts=int(
                os.getenv(
                    "RAG_EVIDENCE_MAX_CONTEXTS", str(DEFAULT_EVIDENCE_MAX_CONTEXTS)
                )
            ),
            total_chars=int(
                os.getenv("RAG_EVIDENCE_TOTAL_CHARS", str(DEFAULT_EVIDENCE_TOTAL_CHARS))
            ),
            max_per_context=int(
                os.getenv(
                    "RAG_EVIDENCE_PER_CONTEXT_CHARS",
                    str(DEFAULT_EVIDENCE_PER_CONTEXT_CHARS),
                )
            ),
            min_per_context=int(
                os.getenv(
                    "RAG_EVIDENCE_MIN_PER_CONTEXT_CHARS",
                    str(DEFAULT_EVIDENCE_MIN_PER_CONTEXT_CHARS),
                )
            ),
        )
        self.workflow = self._build_workflow()

    async def run(self, payload: ChatRequest) -> ChatResponse:
        with self.tracing_manager.span("agent.run", {"agent.query": payload.query}):
            state = await self.workflow.ainvoke(
                {"payload": payload, "started_at": perf_counter()}
            )
            return state["response"]

    def _build_workflow(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("intent_decomposition", self._intent_decomposition_node)
        workflow.add_node("initial_recall", self._initial_recall_node)
        workflow.add_node("verification", self._verification_node)
        workflow.add_node("corrective_recall", self._corrective_recall_node)
        workflow.add_node("answer_synthesis", self._answer_synthesis_node)
        workflow.set_entry_point("intent_decomposition")
        workflow.add_edge("intent_decomposition", "initial_recall")
        workflow.add_edge("initial_recall", "verification")
        workflow.add_conditional_edges(
            "verification",
            self._route_after_verification,
            {
                "corrective_recall": "corrective_recall",
                "answer_synthesis": "answer_synthesis",
            },
        )
        workflow.add_edge("corrective_recall", "verification")
        workflow.add_edge("answer_synthesis", END)
        return workflow.compile()

    def _query_of(self, state: AgentState) -> str:
        """The query actually used for retrieval/synthesis.

        This is the contextualized (self-contained) rewrite when chat history
        exists, otherwise the user's original wording. The original query is
        never mutated: it stays in ``payload.query`` so traces keep the exact
        user input.
        """
        resolved = state.get("resolved_query")
        if isinstance(resolved, str) and resolved.strip():
            return resolved
        return state["payload"].query

    async def _intent_decomposition_node(self, state: AgentState) -> AgentState:
        payload = state["payload"]
        query = payload.query
        history = self.trace_store.get_history(payload.session_id) if payload.session_id else []
        resolved = query
        if history:
            with self.tracing_manager.span("agent.contextualize", {"agent.query": query}):
                resolved = await self._contextualize(query, history)
        # The subquery decomposition runs alongside the first retrieval instead of
        # before it: see _initial_recall_node. Here it used to put one model round
        # trip in front of every request, and nothing on the first pass consumes
        # its result.
        return {"resolved_query": resolved}

    async def _initial_recall_node(self, state: AgentState) -> AgentState:
        payload = state["payload"]
        query = self._query_of(state)
        # Subqueries are consulted only by a corrective round (`_next_queries`),
        # which runs after this node has already produced contexts; the call that
        # produces them does not depend on those contexts. Running it concurrently
        # with the first retrieval measured 104.9s -> 80.5s over six real queries
        # (-23%) against the same gateway.
        decompose = asyncio.ensure_future(self._decompose(query))
        try:
            contexts, trace = await self._retrieve(payload, query)
        except BaseException:
            # A failed retrieval must not leave the decomposition running unawaited:
            # that surfaces as "coroutine was never awaited" and, worse, hides the
            # retrieval error that actually failed the request.
            decompose.cancel()
            raise
        subqueries = await decompose
        graph_contexts = await self._expand_graph_contexts(payload, contexts, trace)
        return {
            "trace_id": str(trace["trace_id"]),
            "retrieval_queries": [query],
            "subqueries": subqueries,
            "graph_expanded_node_ids": self._node_ids(graph_contexts),
            "reading_plans": self._reading_plans_from_trace(trace),
            "contexts": self._merge_contexts(contexts, graph_contexts),
        }

    async def _verification_node(self, state: AgentState) -> AgentState:
        query = self._query_of(state)
        with self.tracing_manager.span("agent.verification", {"agent.query": query}):
            check = await self._verify(
                query,
                state.get("subqueries", [query]),
                state.get("contexts", []),
            )
        return {"check": check}

    async def _corrective_recall_node(self, state: AgentState) -> AgentState:
        payload = state["payload"]
        query = self._next_query(state)
        if query is None:
            return {}
        # Each corrective round deepens the candidate pool as well as possibly
        # rewording the question. Measured on the Chinese-law corpus: the answering
        # statute sits below rank 12 for most questions (hit@12 0.37 vs hit@100 0.60),
        # so a round that only changes the wording cannot reach it. Rounds are bounded
        # by max_retrieval_loops, so the pool cannot grow without limit.
        depth = len(state.get("retrieval_queries", [])) + 1
        with self.tracing_manager.span(
            "agent.corrective_recall", {"agent.query": query, "agent.depth": depth}
        ):
            more_contexts, trace = await self._retrieve(payload, query, depth=depth)
            graph_contexts = await self._expand_graph_contexts(
                payload, more_contexts, trace
            )
            return {
                "retrieval_queries": [*state.get("retrieval_queries", []), query],
                "reading_plans": [
                    *state.get("reading_plans", []),
                    *self._reading_plans_from_trace(trace),
                ],
                "graph_expanded_node_ids": self._dedupe(
                    [
                        *state.get("graph_expanded_node_ids", []),
                        *self._node_ids(graph_contexts),
                    ]
                ),
                "contexts": self._merge_contexts(
                    state.get("contexts", []),
                    more_contexts,
                    graph_contexts,
                ),
            }

    async def _answer_synthesis_node(self, state: AgentState) -> AgentState:
        payload = state["payload"]
        query = self._query_of(state)
        check = state["check"]
        contexts = state.get("contexts", [])
        agent_state = self._public_agent_state(state, check)
        # Synthesis gets the evidence the auditor judged, not each context's first
        # N characters: handing synthesis text the auditor never saw (or vice
        # versa) is how a verdict and an answer come to disagree about the same
        # request. Verbatim spans only.
        messages = self.prompt_builder.build_messages(
            query,
            self._synthesis_contexts(query, contexts),
            agent_state=agent_state,
        )
        generation_started = perf_counter()
        with self.tracing_manager.span("agent.answer_synthesis", {"agent.query": query}):
            result = await self.generation_client.generate(
                messages, response_format=ANSWER_SCHEMA
            )
        generation_seconds = round(perf_counter() - generation_started, 4)
        answer, answerable = self._render_structured_answer(
            str(result.get("content") or "")
        )

        response = self.answer_formatter.format(
            answer=answer,
            contexts=contexts,
            trace_id=state.get("trace_id"),
        )
        response = response.model_copy(update={"usage": result.get("usage") or {}})
        self._update_trace(
            trace_id=state["trace_id"],
            payload=payload,
            contexts=contexts,
            messages=messages,
            response=response,
            result=result,
            agent_state=agent_state,
            generation_seconds=generation_seconds,
            end_to_end_seconds=round(perf_counter() - state["started_at"], 4),
            answerable=answerable,
            audit_seconds=list(check.audit_seconds),
        )
        return {
            "messages": messages,
            "generation_result": result,
            "generation_seconds": generation_seconds,
            "response": response,
        }

    def _render_structured_answer(self, content: str) -> tuple[str, bool]:
        """Convert the structured synthesis JSON into the formatter's text plan.

        Returns the text plus the model's own answerability verdict. That verdict
        is the structured signal the abstention metrics read: deciding "this was a
        refusal" by matching refusal wording would be a keyword list standing in
        for a semantic decision.

        No fallback: an unparseable or contract-violating payload raises
        ModelOutputError and fails the request visibly.
        """
        parsed = parse_json_object(content)
        is_answerable = parsed.get("is_answerable")
        if not isinstance(is_answerable, bool):
            raise ModelOutputError(
                "answer_structure.is_answerable must be a boolean"
            )
        if not is_answerable:
            raw_missing = parsed.get("missing_entities", [])
            if not isinstance(raw_missing, list):
                raise ModelOutputError(
                    "answer_structure.missing_entities must be an array"
                )
            missing = [str(item).strip() for item in raw_missing if str(item).strip()]
            if missing:
                ans = f"文档中未包含关于“{'、'.join(missing)}”的相关信息。"
            else:
                ans = "文档中未包含相关信息。"
            return (
                f"Final Answer:\n{ans}\n\n"
                "Supporting Claims:\n- [insufficiency] 上下文未覆盖问题核心实体。",
                False,
            )
        extracted = parsed.get("extracted_answer")
        if not isinstance(extracted, str) or not extracted.strip():
            raise ModelOutputError(
                "answer_structure.extracted_answer must be a non-empty string"
            )
        claims_raw = parsed.get("supporting_claims")
        if not isinstance(claims_raw, list):
            raise ModelOutputError(
                "answer_structure.supporting_claims must be an array"
            )
        claims = "\n".join(
            str(item) for item in claims_raw if str(item).strip()
        )
        return (
            f"Final Answer:\n{extracted}\n\n"
            f"Supporting Claims:\n{claims if claims else '- None'}",
            True,
        )

    def _route_after_verification(self, state: AgentState) -> str:
        check = state["check"]
        if check.sufficient:
            return "answer_synthesis"
        # `retrieval_queries` already contains the initial retrieval, so it must
        # be compared with >= : with ">" a configured budget of 2 performed 3
        # retrievals. The setting means "how many retrievals may run", not "how
        # many corrections after the first".
        if len(state.get("retrieval_queries", [])) >= self.max_retrieval_loops:
            return "answer_synthesis"
        if self._next_query(state) is None:
            return "answer_synthesis"
        return "corrective_recall"

    def _next_query(self, state: AgentState) -> str | None:
        query = self._query_of(state)
        check = state["check"]
        used = set(state.get("retrieval_queries", []))
        for candidate in self._next_queries(
            query,
            state.get("subqueries", [query]),
            check,
        ):
            if candidate not in used:
                return candidate
        return None

    def _public_agent_state(
        self, state: AgentState, check: EvidenceCheck
    ) -> dict[str, object]:
        return {
            "engine": "langgraph",
            "retrieval_engine": "agentic_wiki",
            "original_query": state["payload"].query,
            "resolved_query": self._query_of(state),
            "workflow_nodes": [
                "intent_decomposition",
                "initial_recall",
                "graph_expansion",
                "verification_loop",
                "answer_synthesis",
            ],
            "subqueries": state.get("subqueries", []),
            "retrieval_queries": state.get("retrieval_queries", []),
            "graph_expanded_node_ids": self._dedupe(
                state.get("graph_expanded_node_ids", [])
            ),
            "reading_plan": self._latest_reading_plan(state),
            "verification": check.as_dict(),
            "evidence_view": self._evidence_view_summary(state),
        }

    def _evidence_view_summary(self, state: AgentState) -> dict[str, object]:
        """How much of each retrieved document the model was actually shown.

        Recorded on the trace because "the evidence does not contain X" and "the
        excerpt I read does not contain X" were indistinguishable in this
        project's refusals: the same 42 rows read identical either way. This
        number is what makes the next diagnosis possible without re-running.
        """
        contexts = state.get("contexts", [])
        excerpts = self.evidence_driller.excerpt(contexts, watch_terms=[self._query_of(state)])
        total = sum(excerpt.total_chars for excerpt in excerpts)
        visible = sum(excerpt.visible_chars for excerpt in excerpts)
        return {
            "contexts": len(contexts),
            "contexts_shown": len(excerpts),
            "document_chars": total,
            "visible_chars": visible,
            "visible_ratio": round(visible / total, 4) if total else 0.0,
            "all_exhausted": bool(excerpts) and all(e.exhausted for e in excerpts),
        }

    async def _retrieve(
        self, payload: ChatRequest, query: str, *, depth: int = 1
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        return await self.agentic_discovery.retrieve(payload, query, depth=depth)

    async def retrieve(self, payload: ChatRequest) -> tuple[list[dict[str, object]], dict[str, object]]:
        """Run the discovery+deep-read engine for a query.

        Exposed for the /retrieve and /retrieve/debug endpoints and the eval
        service: they get the same contexts+trace the chat workflow consumes.
        """
        return await self.agentic_discovery.retrieve(payload, payload.query)

    def _reading_plans_from_trace(
        self, trace: dict[str, object]
    ) -> list[dict[str, object]]:
        retrieval_params = trace.get("retrieval_params")
        if not isinstance(retrieval_params, dict):
            return []
        plan = retrieval_params.get("reading_plan")
        return [plan] if isinstance(plan, dict) else []

    def _latest_reading_plan(
        self, state: AgentState
    ) -> dict[str, object] | None:
        plans = state.get("reading_plans", [])
        for plan in reversed(plans):
            if isinstance(plan, dict):
                return plan
        return None

    async def _expand_graph_contexts(
        self,
        payload: ChatRequest,
        contexts: list[dict[str, object]],
        trace: dict[str, object],
    ) -> list[dict[str, object]]:
        mode = self.graph_expansion_mode
        if mode in {"false", "0", "no", "off", "disabled"}:
            return []
        if mode not in {"always", "true", "1", "yes"}:
            retrieval_params = trace.get("retrieval_params")
            route = (
                retrieval_params.get("query_route")
                if isinstance(retrieval_params, dict)
                else None
            )
            if not (
                isinstance(route, dict)
                and (route.get("requires_graph") is True or route.get("route") == "graph")
            ):
                return []
        # The graph store (PostgresGraphStore) is synchronous psycopg; run it
        # off the event loop so a slow query or pool wait can't stall every
        # concurrent API request. Mirrors the ingest upsert's to_thread usage.
        return await asyncio.to_thread(
            self.graph_expander.expand,
            contexts,
            kb_id=payload.kb_id or "default",
        )

    async def _decompose(self, query: str) -> list[str]:
        result = await self.generation_client.generate(
            [
                {
                    "role": "system",
                    "content": (
                        "You are an AI-first RAG query planner. Return only compact JSON. "
                        "Do not explain."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Break the user question from the input JSON into the minimum set "
                        "of retrieval subqueries. Keep the original wording when useful.\n"
                        'Return JSON: {"subqueries": ["..."]}\n\n'
                        "Input JSON: "
                        f"{json.dumps({'question': query}, ensure_ascii=False)}"
                    ),
                },
            ]
        )
        payload = parse_json_object(str(result.get("content") or ""))
        raw_subqueries = payload.get("subqueries")
        if not isinstance(raw_subqueries, list):
            raise ModelOutputError("subqueries must be an array")
        subqueries = [
            str(item).strip()
            for item in raw_subqueries
            if str(item).strip()
        ]
        if not subqueries:
            raise ModelOutputError("subqueries must not be empty")
        if query not in subqueries:
            subqueries.insert(0, query)
        return self._dedupe(subqueries)[: self.max_subqueries]

    async def _contextualize(self, query: str, history: list) -> str:
        if not history:
            return query
        history_text = "\n".join(
            f"User: {r.query}\nAssistant: {r.answer[:200]}..."
            for r in history
        )
        result = await self.generation_client.generate(
            [
                {
                    "role": "system",
                    "content": "You are a query contextualizer. Rewrite the user's latest query to be fully self-contained, resolving any pronouns based on the chat history. Return ONLY the rewritten query text. Do not explain.",
                },
                {
                    "role": "user",
                    "content": f"Chat History:\n{history_text}\n\nLatest Query: {query}\n\nRewritten fully self-contained query:",
                },
            ]
        )
        content = str(result.get("content") or "").strip()
        if not content:
            raise ModelOutputError("query contextualizer returned an empty rewrite")
        return content

    async def _verify(
        self,
        query: str,
        subqueries: list[str],
        contexts: list[dict[str, object]],
    ) -> EvidenceCheck:
        if not contexts:
            return EvidenceCheck(
                sufficient=False,
                coverage_ratio=0.0,
                missing_terms=[],
                follow_up_queries=subqueries[1:] or [query],
                has_contexts=False,
                has_conflicts=False,
                reason="no_contexts",
                initial_sufficient=False,
            )
        started = perf_counter()
        first = await self._audit(
            query, subqueries, contexts, watch_terms=[query, *subqueries]
        )
        audit_seconds = [round(perf_counter() - started, 4)]
        if first[0]:
            return self._check_from(
                contexts,
                payload=first[1],
                sufficient=True,
                initial_sufficient=True,
                exhausted=first[2],
                audit_seconds=audit_seconds,
            )
        # A second pass used to run here: re-show the same documents with the
        # auditor's own missing_terms added to the watch terms, in case the
        # auditor had missed a passage it could have seen. Measured against the
        # 41 insufficient rows of the 270-question run, it exposed ZERO terms
        # the first view had not already exposed, because the first view is
        # already term-driven - a term it does not show is a term the retrieved
        # contexts do not contain, and re-reading them cannot conjure it. So the
        # pass cost an extra audit call and found nothing; the gaps it was meant
        # to catch are handled by the first view, and the gaps that remain are
        # genuinely absent evidence, which is what corrective retrieval is for.
        return self._check_from(
            contexts,
            payload=first[1],
            sufficient=False,
            initial_sufficient=False,
            exhausted=first[2],
            audit_seconds=audit_seconds,
        )

    def _check_from(
        self,
        contexts: list[dict[str, object]],
        *,
        payload: dict[str, object],
        sufficient: bool,
        initial_sufficient: bool,
        exhausted: bool,
        audit_seconds: list[float],
    ) -> EvidenceCheck:
        coverage_ratio = self._float_in_range(payload.get("coverage_ratio"), 0.0, 1.0)
        missing = self._string_list_field(payload, "missing_terms")
        follow_up_queries = self._string_list_field(payload, "follow_up_queries")
        return EvidenceCheck(
            sufficient=sufficient,
            coverage_ratio=round(coverage_ratio, 3),
            missing_terms=missing[:8],
            follow_up_queries=self._dedupe(follow_up_queries)[: self.max_subqueries],
            has_contexts=True,
            has_conflicts=self._has_conflicts(contexts),
            reason=str(
                payload.get("reason")
                or ("coverage_ok" if sufficient else "coverage_gap")
            ),
            initial_sufficient=initial_sufficient,
            evidence_complete=exhausted,
            audit_seconds=audit_seconds,
        )

    async def _audit(
        self,
        query: str,
        subqueries: list[str],
        contexts: list[dict[str, object]],
        *,
        watch_terms: list[str],
    ) -> tuple[bool, dict[str, object], bool]:
        """One auditor round trip. Returns (sufficient, payload, saw_everything)."""
        excerpts = self.evidence_driller.excerpt(contexts, watch_terms=watch_terms)
        evidence = [excerpt.as_dict_for_prompt() for excerpt in excerpts]
        exhausted = all(excerpt.exhausted for excerpt in excerpts) and (
            len(contexts) <= self.evidence_driller.max_contexts
        )
        auditor_input = {
            "question": query,
            "subqueries": subqueries,
            "evidence": evidence,
        }
        # Native JSON Schema (same contract as the agentic reading plan). The
        # defensive value coercion below still handles a model that returns
        # string-typed booleans / out-of-range numbers; anything unparseable
        # raises and fails the request — there is no degraded pass-through.
        result = await self.generation_client.generate(
            [
                {
                    "role": "system",
                    "content": (
                        "You are an evidence auditor for a RAG agent. Return only JSON. "
                        "Judge sufficiency using the provided evidence, not outside knowledge."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Decide whether the evidence can answer the user's question.\n\n"
                        "SUFFICIENT means: the evidence contains the information needed to "
                        "answer the question that was actually asked. It does NOT require "
                        "that the evidence cover every angle of the topic, nor that it "
                        "restate the question's wording. If a reasonable reader could "
                        "answer from it, it is sufficient. Example: for \"男的多少岁能结婚?\" "
                        "the passage \"结婚年龄，男不得早于二十周岁\" is sufficient, even "
                        "though it says nothing about the conditions or exceptions.\n\n"
                        "Judge only what the question asks for. A question about one fact "
                        "does not need the rest of the document. Do NOT lower the verdict "
                        "because the evidence is partial about the general subject - mark "
                        "it insufficient only when the specific thing asked is absent or "
                        "the evidence contradicts itself.\n\n"
                        "Answering the question is not your job: do not write the answer, "
                        "and do not judge whether it takes effort to assemble from several "
                        "passages. If the passages together state it, the evidence is "
                        "sufficient.\n\n"
                        "Each evidence item is selected passages from one document, not "
                        "necessarily its beginning. The visible_region_* fields locate them. "
                        "\"[...]\" means text was left out there. excerpt_exhausted=true "
                        "means the item is that whole document, so an absence in it is a real "
                        "absence; false means you are judging what you were shown.\n\n"
                        "If it is genuinely insufficient, propose targeted follow-up "
                        "retrieval queries.\n\n"
                        "Return JSON with keys: sufficient(boolean), coverage_ratio(number 0..1), "
                        "missing_terms(array of short strings), follow_up_queries(array), reason(string).\n\n"
                        "Input JSON: "
                        f"{json.dumps(auditor_input, ensure_ascii=False)}"
                    ),
                },
            ],
            response_format=VERIFY_SCHEMA,
        )
        payload = parse_json_object(str(result.get("content") or ""))
        sufficient = self._parse_bool(payload.get("sufficient"))
        return sufficient, payload, exhausted

    def _next_queries(
        self, original_query: str, subqueries: list[str], check: EvidenceCheck
    ) -> list[str]:
        return self._dedupe([*check.follow_up_queries, *subqueries[1:], original_query])

    def _merge_contexts(
        self, *context_groups: list[dict[str, object]]
    ) -> list[dict[str, object]]:
        merged: list[dict[str, object]] = []
        seen: set[str] = set()
        for contexts in context_groups:
            for context in contexts:
                key = str(context.get("node_id") or context.get("chunk_id") or id(context))
                if key in seen:
                    continue
                seen.add(key)
                merged.append(context)
        for index, context in enumerate(merged, start=1):
            context["citation_label"] = f"C{index}"
        return merged

    def _node_ids(self, contexts: list[dict[str, object]]) -> list[str]:
        return [
            str(context.get("node_id") or context.get("chunk_id"))
            for context in contexts
            if context.get("node_id") or context.get("chunk_id")
        ]

    def _has_conflicts(self, contexts: list[dict[str, object]]) -> bool:
        return any(
            context.get("wiki_status") == "conflicting"
            or context.get("evidence_role") == "conflicting"
            or context.get("claim_role") == "conflict"
            for context in contexts
        )

    def _synthesis_contexts(
        self, query: str, contexts: list[dict[str, object]]
    ) -> list[dict[str, object]]:
        """Render the audited excerpts as synthesis contexts.

        The prompt builder still needs an ``item["text"]`` per context, so each
        excerpt is written back as a copy of its context with ``text`` replaced
        by the same verbatim passages the auditor was shown. Offsets and the
        elision markers are kept in ``evidence_excerpt`` so the trace records
        what was visible. Nothing is re-ranked or re-selected here.
        """
        excerpts = self.evidence_driller.excerpt(contexts, watch_terms=[query])
        by_key: dict[str, dict[str, object]] = {}
        for context in contexts:
            for key in (
                context.get("chunk_id"),
                context.get("node_id"),
                context.get("citation_label"),
            ):
                if key is not None:
                    by_key.setdefault(str(key), context)
        rendered: list[dict[str, object]] = []
        for excerpt in excerpts:
            source = by_key.get(str(excerpt.node_id)) or by_key.get(str(excerpt.label))
            if source is None:
                continue
            rendered.append(
                {
                    **source,
                    "text": excerpt.render(),
                    "evidence_excerpt": {
                        "document_chars": excerpt.total_chars,
                        "visible_chars": excerpt.visible_chars,
                        "exhausted": excerpt.exhausted,
                        "passages": [passage.as_dict() for passage in excerpt.passages],
                    },
                }
            )
        return rendered

    def _string_list_field(self, payload: dict[str, object], key: str) -> list[str]:
        """Read a JSON array of strings, failing loudly on any other shape."""
        raw = payload.get(key, [])
        if not isinstance(raw, list):
            raise ModelOutputError(f"evidence verification {key} must be an array")
        return [str(item).strip() for item in raw if str(item).strip()]

    def _float_in_range(self, value: object, minimum: float, maximum: float) -> float:
        """Coerce a string/number the model may have emitted as JSON text.

        A non-numeric value is a contract violation, not a 0.0 measurement:
        silently defaulting made an unparseable audit look like "zero coverage"
        and quietly drove an extra retrieval loop.
        """
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise ModelOutputError(
                f"evidence verification coverage_ratio must be a number, got {value!r}"
            )
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ModelOutputError(
                f"evidence verification coverage_ratio must be a number, got {value!r}"
            ) from exc
        return max(minimum, min(maximum, number))

    def _parse_bool(self, value: object) -> bool:
        """Coerce a JSON-ish boolean; anything else is a contract violation."""
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            token = value.strip().casefold()
            if token in {"true", "1", "yes"}:
                return True
            if token in {"false", "0", "no"}:
                return False
        raise ModelOutputError(
            f"evidence verification sufficient must be a boolean, got {value!r}"
        )

    def _dedupe(self, items: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for item in items:
            normalized = item.strip()
            key = normalized.casefold()
            if not normalized or key in seen:
                continue
            seen.add(key)
            deduped.append(normalized)
        return deduped

    def _update_trace(
        self,
        trace_id: str,
        payload: ChatRequest,
        contexts: list[dict[str, object]],
        messages: list[dict[str, object]],
        response: ChatResponse,
        result: dict[str, object],
        agent_state: dict[str, object],
        generation_seconds: float,
        end_to_end_seconds: float,
        answerable: bool | None = None,
        audit_seconds: list[float] | None = None,
    ) -> None:
        record = self.trace_store.get(trace_id)
        if record is None:
            return
        citation_integrity = check_citations(
            response.answer, contexts
        ).as_dict()
        updated = record.model_copy(
            update={
                "answer": response.answer,
                "citations": [citation.model_dump() for citation in response.citations],
                "supporting_claims": [
                    claim.model_dump() for claim in response.supporting_claims
                ],
                "contexts": contexts,
                "model_alias": self.generation_client.alias,
                "kb_id": payload.kb_id or record.kb_id,
                "session_id": payload.session_id or record.session_id,
                "sample_id": payload.sample_id or record.sample_id,
                "prompt_version": str(self.config.settings["prompt"]["version"]),
                "prompt_messages": messages if self._store_prompt_messages() else [],
                "generation_finish_reason": (
                    str(result["finish_reason"])
                    if result.get("finish_reason") is not None
                    else None
                ),
                "generation_usage": result.get("usage") or {},
                "retrieval_params": {
                    **record.retrieval_params,
                    "agent": agent_state,
                },
                "step_latencies": {
                    **record.step_latencies,
                    "generation_seconds": generation_seconds,
                    "agent_end_to_end_seconds": end_to_end_seconds,
                    "audit_seconds": list(audit_seconds or []),
                    "audit_calls": len(audit_seconds or []),
                },
                "citation_integrity": citation_integrity,
                "answerable": answerable,
            }
        )
        self.trace_store.update(updated)

    def _store_prompt_messages(self) -> bool:
        return os.getenv("RAG_TRACE_STORE_PROMPTS", "false").lower() in {
            "true",
            "1",
            "yes",
        }
