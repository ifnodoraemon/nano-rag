from __future__ import annotations

import json
import logging
import os
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

logger = logging.getLogger(__name__)

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
    evidence_plans: list[dict[str, object]]
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
        self.graph_expansion_mode = os.getenv("RAG_GRAPH_EXPANSION_MODE", "auto").lower()
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
        query = payload.query
        history = self.trace_store.get_history(payload.session_id) if payload.session_id else []
        if history:
            query = await self._contextualize(query, history)
            payload.query = query
        return {"subqueries": await self._decompose(query), "payload": payload}

    async def _initial_recall_node(self, state: AgentState) -> AgentState:
        payload = state["payload"]
        contexts, trace = await self._retrieve(payload, payload.query)
        graph_contexts = self._expand_graph_contexts(payload, contexts, trace)
        return {
            "trace_id": str(trace["trace_id"]),
            "retrieval_queries": [payload.query],
            "graph_expanded_node_ids": self._node_ids(graph_contexts),
            "evidence_plans": self._evidence_plans_from_trace(trace),
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
        more_contexts, trace = await self._retrieve(payload, query)
        graph_contexts = self._expand_graph_contexts(payload, more_contexts, trace)
        return {
            "retrieval_queries": [*state.get("retrieval_queries", []), query],
            "evidence_plans": [
                *state.get("evidence_plans", []),
                *self._evidence_plans_from_trace(trace),
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
        check = state["check"]
        contexts = state.get("contexts", [])
        agent_state = self._public_agent_state(state, check)
        messages = self.prompt_builder.build_messages(
            payload.query,
            contexts,
            agent_state=agent_state,
        )
        generation_started = perf_counter()
        schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "answer_structure",
                "schema": {
                    "type": "object",
                    "properties": {
                        "is_answerable": {
                            "type": "boolean",
                            "description": "如果上下文中完全找不到问题核心实体，请设为 false"
                        },
                        "missing_entities": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "列出找不到的具体专有名词，如果找到了则为空数组"
                        },
                        "extracted_answer": {
                            "type": "string",
                            "description": "带引用的极简答案，例如：xxx为xxx [C1]。"
                        },
                        "supporting_claims": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "如: - [factual] <证据支撑点 1> [C#]"
                        }
                    },
                    "required": ["is_answerable", "missing_entities", "extracted_answer", "supporting_claims"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }
        
        result = await self.generation_client.generate(messages, response_format=schema)
        generation_seconds = round(perf_counter() - generation_started, 4)
        
        raw_content = str(result["content"]).strip()
        if raw_content.startswith("```json"):
            raw_content = raw_content[7:]
            if raw_content.endswith("```"):
                raw_content = raw_content[:-3]
            raw_content = raw_content.strip()
            
        try:
            parsed = json.loads(raw_content)
            is_answerable = parsed.get("is_answerable", True)
            if not is_answerable:
                missing = parsed.get("missing_entities", [])
                if missing:
                    ans = f"文档中未包含关于“{'、'.join(missing)}”的相关信息。"
                else:
                    ans = "文档中未包含相关信息。"
                fake_answer = f"Final Answer:\n{ans}\n\nSupporting Claims:\n- None"
            else:
                ans = parsed.get("extracted_answer", "")
                claims_list = parsed.get("supporting_claims", [])
                claims = "\n".join(claims_list) if claims_list else "- None"
                fake_answer = f"Final Answer:\n{ans}\n\nSupporting Claims:\n{claims}"
        except Exception as e:
            logger.warning(f"Failed to parse structured JSON: {e}, falling back to raw")
            fake_answer = f"Final Answer:\n{raw_content}\n\nSupporting Claims:\n- None"

        response = self.answer_formatter.format(
            answer=fake_answer,
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
            "evidence_plan": self._latest_evidence_plan(state),
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

    def _evidence_plans_from_trace(self, trace: dict[str, object]) -> list[dict[str, object]]:
        retrieval_params = trace.get("retrieval_params")
        if not isinstance(retrieval_params, dict):
            return []
        plan = retrieval_params.get("evidence_plan")
        return [plan] if isinstance(plan, dict) else []

    def _latest_evidence_plan(self, state: AgentState) -> dict[str, object] | None:
        plans = state.get("evidence_plans", [])
        for plan in reversed(plans):
            if isinstance(plan, dict):
                return plan
        return None

    def _expand_graph_contexts(
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
        return self.graph_expander.expand(
            contexts,
            kb_id=payload.kb_id or "default",
        )

    async def _decompose(self, query: str) -> list[str]:
        try:
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
        except Exception as exc:
            logger.warning("agent intent decomposition failed: %s", exc)
            return [query]
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

    async def _contextualize(self, query: str, history: list) -> str:
        if not history:
            return query
        try:
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
            return content if content else query
        except Exception as exc:
            logger.warning("query contextualization failed: %s", exc)
            return query

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
        auditor_input = {
            "question": query,
            "subqueries": subqueries,
            "evidence": evidence,
        }
        try:
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
                            "Input JSON: "
                            f"{json.dumps(auditor_input, ensure_ascii=False)}"
                        ),
                    },
                ]
            )
        except Exception as exc:
            logger.warning("agent evidence verification failed: %s", exc)
            return EvidenceCheck(
                sufficient=True,
                coverage_ratio=1.0,
                missing_terms=[],
                follow_up_queries=[],
                has_contexts=True,
                has_conflicts=self._has_conflicts(contexts),
                reason="verifier_unavailable",
            )
        payload = self._json_object(str(result.get("content") or ""))
        sufficient = self._parse_bool(payload.get("sufficient"))
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

    def _parse_bool(self, value: object) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().casefold() in {"true", "1", "yes"}
        if isinstance(value, int | float):
            return value != 0
        return False

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
                },
            }
        )
        self.trace_store.update(updated)

    def _store_prompt_messages(self) -> bool:
        return os.getenv("RAG_TRACE_STORE_PROMPTS", "false").lower() in {
            "true",
            "1",
            "yes",
        }
