from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING

from app.utils.text import parse_bool_env

if TYPE_CHECKING:
    from app.model_client.generation import GenerationClient
    from app.retrieval.query_router import QueryRoute

logger = logging.getLogger(__name__)

PLAN_RELATIONS = {"supports", "contradicts", "qualifies", "elaborates", "unrelated"}
PLAN_STRATEGIES = {"direct", "conditional", "conflict", "insufficient"}


@dataclass(frozen=True)
class EvidencePlannerConfig:
    enabled: bool = True
    max_contexts: int = 8
    max_context_chars: int = 1200
    pairwise: bool = True

    @classmethod
    def from_settings(cls, settings: dict[str, object] | None) -> "EvidencePlannerConfig":
        settings = settings or {}
        raw_enabled = os.getenv(
            "RAG_EVIDENCE_PLANNER_ENABLED",
            str(settings.get("enabled", "true")),
        )
        raw_pairwise = os.getenv(
            "RAG_EVIDENCE_PLANNER_PAIRWISE",
            str(settings.get("pairwise", "true")),
        )
        return cls(
            enabled=parse_bool_env(raw_enabled),
            max_contexts=_int_setting("RAG_EVIDENCE_PLANNER_MAX_CONTEXTS", settings, "max_contexts", 8),
            max_context_chars=_int_setting(
                "RAG_EVIDENCE_PLANNER_MAX_CONTEXT_CHARS",
                settings,
                "max_context_chars",
                1200,
            ),
            pairwise=parse_bool_env(raw_pairwise),
        )


class EvidencePlanner:
    def __init__(
        self,
        generation_client: GenerationClient | None = None,
        config: EvidencePlannerConfig | None = None,
    ) -> None:
        self.generation_client = generation_client
        self.config = config or EvidencePlannerConfig.from_settings(None)

    async def plan(
        self,
        query: str,
        contexts: list[dict[str, object]],
        query_route: QueryRoute | None = None,
    ) -> dict[str, object]:
        started = perf_counter()
        if not contexts:
            return self._empty_plan("no_contexts", started)
        fallback = self._metadata_plan(query, contexts, query_route, started)
        if not self.config.enabled or self.generation_client is None:
            return fallback
        try:
            result = await self.generation_client.generate(
                [{"role": "user", "content": self._prompt(query, contexts, query_route)}]
            )
            parsed = self._parse_plan(str(result.get("content") or ""), contexts, fallback)
            parsed["status"] = "planned"
            parsed["planner_seconds"] = round(perf_counter() - started, 4)
            return parsed
        except Exception as exc:
            logger.warning("evidence planning failed; using metadata-only plan: %s", exc)
            fallback["status"] = "degraded"
            fallback["error_type"] = exc.__class__.__name__
            fallback["planner_seconds"] = round(perf_counter() - started, 4)
            return fallback

    def annotate_contexts(
        self,
        contexts: list[dict[str, object]],
        plan: dict[str, object],
    ) -> list[dict[str, object]]:
        annotations = plan.get("context_annotations")
        if not isinstance(annotations, dict):
            return contexts
        annotated: list[dict[str, object]] = []
        for context in contexts:
            label = str(context.get("citation_label") or "")
            annotation = annotations.get(label) if label else None
            if not isinstance(annotation, dict):
                annotated.append(context)
                continue
            public_annotation = {
                key: value
                for key, value in annotation.items()
                if key in {"claim_role", "claim_scope", "certainty", "discourse_units"}
            }
            annotated.append({**context, **public_annotation})
        return annotated

    def _prompt(
        self,
        query: str,
        contexts: list[dict[str, object]],
        query_route: QueryRoute | None,
    ) -> str:
        evidence = []
        for context in contexts[: self.config.max_contexts]:
            evidence.append(
                {
                    "label": context.get("citation_label"),
                    "title": context.get("title"),
                    "source": context.get("source"),
                    "evidence_role": context.get("evidence_role"),
                    "freshness_tier": context.get("freshness_tier"),
                    "version": context.get("version"),
                    "text": str(context.get("text") or "")[: self.config.max_context_chars],
                }
            )
        payload = {
            "question": query,
            "query_route": query_route.as_dict() if query_route else None,
            "evidence": evidence,
        }
        return (
            "You are a discourse-aware evidence planner for a production RAG system. "
            "Return compact JSON only. Use only the supplied evidence labels.\n"
            "Schema: {"
            '"answer_strategy":"direct|conditional|conflict|insufficient",'
            '"primary_evidence":["C1"],'
            '"conditions":["short condition tied to label"],'
            '"relations":[{"source":"C1","target":"C2","relation":"supports|contradicts|qualifies|elaborates|unrelated","reason":"short"}],'
            '"context_annotations":{"C1":{"claim_role":"conclusion|condition|evidence|exception|background|conflict",'
            '"claim_scope":"short scope","certainty":"strong|weak|disputed",'
            '"discourse_units":[{"role":"conclusion|condition|evidence|exception|background","text":"short span"}]}},'
            '"outline":["step 1","step 2"]'
            "}.\n\n"
            f"Input JSON: {json.dumps(payload, ensure_ascii=False)}"
        )

    def _parse_plan(
        self,
        content: str,
        contexts: list[dict[str, object]],
        fallback: dict[str, object],
    ) -> dict[str, object]:
        payload = _json_object(content)
        if not payload:
            return fallback
        labels = _context_labels(contexts)
        strategy = str(payload.get("answer_strategy") or fallback["answer_strategy"]).strip().lower()
        if strategy not in PLAN_STRATEGIES:
            strategy = str(fallback["answer_strategy"])
        plan = {
            **fallback,
            "answer_strategy": strategy,
            "primary_evidence": _label_list(payload.get("primary_evidence"), labels)
            or fallback["primary_evidence"],
            "conditions": _string_list(payload.get("conditions"), 8),
            "relations": self._relations(payload.get("relations"), labels),
            "outline": _string_list(payload.get("outline"), 8) or fallback["outline"],
            "context_annotations": self._annotations(
                payload.get("context_annotations"),
                labels,
            )
            or fallback["context_annotations"],
        }
        if not plan["relations"] and fallback.get("relations"):
            plan["relations"] = fallback["relations"]
        return plan

    def _relations(self, value: object, labels: set[str]) -> list[dict[str, object]]:
        if not self.config.pairwise or not isinstance(value, list):
            return []
        relations: list[dict[str, object]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source") or "").strip()
            target = str(item.get("target") or "").strip()
            relation = str(item.get("relation") or "").strip().lower()
            if source not in labels or target not in labels or source == target:
                continue
            if relation not in PLAN_RELATIONS:
                relation = "elaborates"
            relations.append(
                {
                    "source": source,
                    "target": target,
                    "relation": relation,
                    "reason": str(item.get("reason") or "")[:160],
                }
            )
        return relations[:16]

    def _annotations(self, value: object, labels: set[str]) -> dict[str, object]:
        if not isinstance(value, dict):
            return {}
        annotations: dict[str, object] = {}
        for label, raw in value.items():
            label = str(label).strip()
            if label not in labels or not isinstance(raw, dict):
                continue
            annotations[label] = {
                "claim_role": str(raw.get("claim_role") or "evidence")[:40],
                "claim_scope": str(raw.get("claim_scope") or "")[:180],
                "certainty": str(raw.get("certainty") or "weak")[:40],
                "discourse_units": _discourse_units(raw.get("discourse_units")),
            }
        return annotations

    def _metadata_plan(
        self,
        query: str,
        contexts: list[dict[str, object]],
        query_route: QueryRoute | None,
        started: float,
    ) -> dict[str, object]:
        labels = _context_labels(contexts)
        primary = [
            str(context.get("citation_label"))
            for context in contexts
            if str(context.get("citation_label") or "") in labels
            and context.get("evidence_role") == "primary"
        ]
        if not primary:
            primary = [str(contexts[0].get("citation_label") or "C1")]
        annotations: dict[str, object] = {}
        conditions: list[str] = []
        conflict_labels: list[str] = []
        for context in contexts[: self.config.max_contexts]:
            label = str(context.get("citation_label") or "")
            if label not in labels:
                continue
            role = _claim_role(context)
            if role == "condition":
                scope = str(context.get("claim_scope") or "").strip()
                conditions.append(f"{label}: {scope}" if scope else label)
            if role == "conflict":
                conflict_labels.append(label)
            annotations[label] = {
                "claim_role": role,
                "claim_scope": str(context.get("claim_scope") or "")[:180],
                "certainty": "disputed" if role == "conflict" else "weak",
                "discourse_units": _metadata_units(context),
            }
        route = query_route.route if query_route else "fact"
        has_conflict = bool(conflict_labels) or route == "conflict"
        strategy = "conflict" if has_conflict else "conditional" if conditions else "direct"
        if route == "version" and len(contexts) > 1:
            strategy = "conditional"
        relations = []
        if has_conflict and len(contexts) > 1:
            target = conflict_labels[0] if conflict_labels else str(contexts[-1].get("citation_label") or "")
            for source in primary:
                if source and target and source != target:
                    relations.append(
                        {
                            "source": source,
                            "target": target,
                            "relation": "contradicts",
                            "reason": "conflicting evidence metadata or route",
                        }
                    )
        outline = _outline(strategy, primary, conditions, relations)
        return {
            "status": "metadata_only",
            "answer_strategy": strategy,
            "primary_evidence": primary[:4],
            "conditions": conditions[:6],
            "relations": relations[:8],
            "outline": outline,
            "context_annotations": annotations,
            "planner_seconds": round(perf_counter() - started, 4),
        }

    def _empty_plan(self, status: str, started: float) -> dict[str, object]:
        return {
            "status": status,
            "answer_strategy": "insufficient",
            "primary_evidence": [],
            "conditions": [],
            "relations": [],
            "outline": ["State that no retrieved evidence is available."],
            "context_annotations": {},
            "planner_seconds": round(perf_counter() - started, 4),
        }


def _claim_role(context: dict[str, object]) -> str:
    explicit = str(context.get("claim_role") or "").strip().lower()
    if explicit in {"conclusion", "condition", "evidence", "exception", "background", "conflict"}:
        return explicit
    if context.get("evidence_role") == "conflicting" or context.get("wiki_status") == "conflicting":
        return "conflict"
    if context.get("chunk_kind") == "table_row":
        return "evidence"
    return "conclusion" if context.get("evidence_role") == "primary" else "background"


def _metadata_units(context: dict[str, object]) -> list[dict[str, str]]:
    units = _discourse_units(context.get("discourse_units"))
    if units:
        return units
    text = str(context.get("text") or "").strip()
    if not text:
        return []
    return [{"role": _claim_role(context), "text": _preview(text, 160)}]


def _outline(
    strategy: str,
    primary: list[str],
    conditions: list[str],
    relations: list[dict[str, object]],
) -> list[str]:
    if strategy == "conflict":
        return [
            "Present the main evidence with citations.",
            "Describe conflicting or contrasting evidence explicitly.",
            "Give a qualified conclusion instead of an absolute answer.",
        ]
    if strategy == "conditional":
        return [
            "State the answer together with its applicable conditions.",
            "Tie each condition to its cited evidence.",
            "Avoid generalizing beyond the retrieved scope.",
        ]
    if not primary:
        return ["State that the evidence is insufficient."]
    if relations:
        return ["Synthesize related evidence before answering directly."]
    return ["Answer directly using the primary evidence labels."]


def _json_object(content: str) -> dict[str, object]:
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


def _context_labels(contexts: list[dict[str, object]]) -> set[str]:
    return {
        str(context.get("citation_label"))
        for context in contexts
        if str(context.get("citation_label") or "").strip()
    }


def _label_list(value: object, labels: set[str]) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        label = str(item).strip()
        if label in labels and label not in result:
            result.append(label)
    return result[:8]


def _string_list(value: object, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip()[:240] for item in value if str(item).strip()][:limit]


def _discourse_units(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    units: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "evidence")[:40]
        text = str(item.get("text") or "").strip()
        if text:
            units.append({"role": role, "text": text[:240]})
    return units[:6]


def _preview(value: str, limit: int) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _int_setting(
    env_name: str,
    settings: dict[str, object],
    key: str,
    default: int,
) -> int:
    raw = os.getenv(env_name, settings.get(key, default))
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return default
