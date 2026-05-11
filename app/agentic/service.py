from __future__ import annotations

import json
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, TypedDict

from app.generation.answer_formatter import AnswerFormatter
from app.generation.prompt_builder import PromptBuilder
from app.model_client.generation import GenerationClient
from app.retrieval.graph_expander import GraphExpander
from app.retrieval.pipeline import RetrievalPipeline
from app.schemas.chat import ChatRequest, ChatResponse
from langgraph.graph import END, StateGraph

if TYPE_CHECKING:
    from app.core.config import AppConfig
    from app.core.tracing import TraceStore, TracingManager
    from app.retrieval.graph_store import GraphStore


@dataclass(frozen=True)
class EvidenceCheck:
    sufficient: bool
    coverage_ratio: float
    missing_terms: list[str]
    follow_up_queries: list[str]
    has_contexts: bool
    has_conflicts: bool
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "sufficient": self.sufficient,
            "coverage_ratio": self.coverage_ratio,
            "missing_terms": self.missing_terms,
            "follow_up_queries": self.follow_up_queries,
            "has_contexts": self.has_contexts,
            "has_conflicts": self.has_conflicts,
            "reason": self.reason,
        }


class AgentState(TypedDict, total=False):
    payload: ChatRequest
    started_at: float
    subqueries: list[str]
    contexts: list[dict[str, object]]
    trace_id: str
    retrieval_queries: list[str]
    graph_expanded_node_ids: list[str]
    check: EvidenceCheck
    messages: list[dict[str, object]]
    response: ChatResponse
    generation_result: dict[str, object]
    generation_seconds: float


class AgenticReasoningService:
    def __init__(
        self,
        config: AppConfig,
        retrieval_pipeline: RetrievalPipeline,
        generation_client: GenerationClient,
        prompt_builder: PromptBuilder,
        answer_formatter: AnswerFormatter,
        trace_store: TraceStore,
        tracing_manager: TracingManager,
        graph_store: GraphStore | None = None,
    ) -> None:
        self.config = config
        self.retrieval_pipeline = retrieval_pipeline
        self.generation_client = generation_client
        self.prompt_builder = prompt_builder
        self.answer_formatter = answer_formatter
        self.trace_store = trace_store
        self.tracing_manager = tracing_manager
        self.graph_expander = GraphExpander(config.parsed_dir, graph_store)
        agent_settings = config.settings.get("agent", {})
        self.max_retrieval_loops = int(agent_settings.get("max_retrieval_loops", 2))
        self.max_subqueries = int(agent_settings.get("max_subqueries", 4))
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

    async def _intent_decomposition_node(self, state: AgentState) -> AgentState:
        payload = state["payload"]
        return {"subqueries": await self._decompose(payload.query)}

    async def _initial_recall_node(self, state: AgentState) -> AgentState:
        payload = state["payload"]
        contexts, trace = await self._retrieve(payload, payload.query)
        graph_contexts = self.graph_expander.expand(
            contexts,
            kb_id=payload.kb_id or "default",
        )
        return {
            "trace_id": str(trace["trace_id"]),
            "retrieval_queries": [payload.query],
            "graph_expanded_node_ids": self._node_ids(graph_contexts),
            "contexts": self._merge_contexts(contexts, graph_contexts),
        }

    async def _verification_node(self, state: AgentState) -> AgentState:
        payload = state["payload"]
        return {
            "check": await self._verify(
                payload.query,
                state.get("subqueries", [payload.query]),
                state.get("contexts", []),
            )
        }

    async def _corrective_recall_node(self, state: AgentState) -> AgentState:
        payload = state["payload"]
        query = self._next_query(state)
        if query is None:
            return {}
        more_contexts, _ = await self._retrieve(payload, query)
        graph_contexts = self.graph_expander.expand(
            more_contexts,
            kb_id=payload.kb_id or "default",
        )
        return {
            "retrieval_queries": [*state.get("retrieval_queries", []), query],
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
        check = state["check"]
        contexts = state.get("contexts", [])
        agent_state = self._public_agent_state(state, check)
        messages = self.prompt_builder.build_messages(
            payload.query,
            contexts,
            agent_state=agent_state,
        )
        generation_started = perf_counter()
        result = await self.generation_client.generate(messages)
        generation_seconds = round(perf_counter() - generation_started, 4)
        response = self.answer_formatter.format(
            answer=str(result["content"]),
            contexts=contexts,
            trace_id=state.get("trace_id"),
        )
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
        )
        return {
            "messages": messages,
            "generation_result": result,
            "generation_seconds": generation_seconds,
            "response": response,
        }

    def _route_after_verification(self, state: AgentState) -> str:
        check = state["check"]
        if check.sufficient:
            return "answer_synthesis"
        if len(state.get("retrieval_queries", [])) > self.max_retrieval_loops:
            return "answer_synthesis"
        if self._next_query(state) is None:
            return "answer_synthesis"
        return "corrective_recall"

    def _next_query(self, state: AgentState) -> str | None:
        payload = state["payload"]
        check = state["check"]
        used = set(state.get("retrieval_queries", []))
        for query in self._next_queries(
            payload.query,
            state.get("subqueries", [payload.query]),
            check,
        ):
            if query not in used:
                return query
        return None

    def _public_agent_state(
        self, state: AgentState, check: EvidenceCheck
    ) -> dict[str, object]:
        return {
            "engine": "langgraph",
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
            "verification": check.as_dict(),
        }

    async def _retrieve(
        self, payload: ChatRequest, query: str
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        return await self.retrieval_pipeline.run(
            query,
            payload.top_k,
            kb_id=payload.kb_id or "default",
            session_id=payload.session_id,
            sample_id=payload.sample_id,
            metadata_filters=payload.metadata_filters,
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
                        "Break the user question into the minimum set of retrieval subqueries. "
                        "Keep the original wording when useful.\n"
                        'Return JSON: {"subqueries": ["..."]}\n\n'
                        f"Question: {query}"
                    ),
                },
            ]
        )
        payload = self._json_object(str(result.get("content") or ""))
        raw_subqueries = payload.get("subqueries") if isinstance(payload, dict) else None
        subqueries = [
            str(item).strip()
            for item in raw_subqueries or []
            if str(item).strip()
        ]
        if query not in subqueries:
            subqueries.insert(0, query)
        return self._dedupe(subqueries)[: self.max_subqueries]

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
            )
        evidence = self._compact_contexts(contexts)
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
                        "Decide whether the evidence can close the user's question. "
                        "If not, propose targeted follow-up retrieval queries.\n"
                        "Return JSON with keys: sufficient(boolean), coverage_ratio(number 0..1), "
                        "missing_terms(array of short strings), follow_up_queries(array), reason(string).\n\n"
                        f"Question: {query}\n"
                        f"Subqueries: {json.dumps(subqueries, ensure_ascii=False)}\n"
                        f"Evidence: {json.dumps(evidence, ensure_ascii=False)}"
                    ),
                },
            ]
        )
        payload = self._json_object(str(result.get("content") or ""))
        sufficient = bool(payload.get("sufficient"))
        coverage_ratio = self._float_in_range(payload.get("coverage_ratio"), 0.0, 1.0)
        missing = [
            str(item).strip()
            for item in payload.get("missing_terms", [])
            if str(item).strip()
        ]
        follow_up_queries = [
            str(item).strip()
            for item in payload.get("follow_up_queries", [])
            if str(item).strip()
        ]
        return EvidenceCheck(
            sufficient=sufficient,
            coverage_ratio=round(coverage_ratio, 3),
            missing_terms=missing[:8],
            follow_up_queries=self._dedupe(follow_up_queries)[: self.max_subqueries],
            has_contexts=True,
            has_conflicts=self._has_conflicts(contexts),
            reason=str(payload.get("reason") or ("coverage_ok" if sufficient else "coverage_gap")),
        )

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
            for context in contexts
        )

    def _compact_contexts(
        self, contexts: list[dict[str, object]]
    ) -> list[dict[str, object]]:
        compacted: list[dict[str, object]] = []
        for context in contexts[:12]:
            compacted.append(
                {
                    "label": context.get("citation_label"),
                    "node_id": context.get("node_id") or context.get("chunk_id"),
                    "source": context.get("source"),
                    "title": context.get("title"),
                    "hierarchy_path": context.get("hierarchy_path"),
                    "page_number": context.get("page_number"),
                    "graph_relation": context.get("graph_relation"),
                    "text": str(context.get("text") or "")[:1600],
                }
            )
        return compacted

    def _json_object(self, content: str) -> dict[str, object]:
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end >= start:
            text = text[start : end + 1]
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return loaded if isinstance(loaded, dict) else {}

    def _float_in_range(self, value: object, minimum: float, maximum: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return minimum
        return max(minimum, min(maximum, number))

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
    ) -> None:
        record = self.trace_store.get(trace_id)
        if record is None:
            return
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
                "prompt_messages": messages,
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
                },
            }
        )
        self.trace_store.update(updated)
