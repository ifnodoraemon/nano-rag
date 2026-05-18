from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.model_client.generation import GenerationClient

logger = logging.getLogger(__name__)

ROUTES = {"fact", "table", "graph", "version", "conflict", "definition", "visual"}
CHUNK_KINDS = {
    "table_row",
    "table_summary",
    "clause",
    "definition",
    "media_object",
    "document_page",
    "document_attachment",
    "rendered_page_image",
    "embedded_image",
}

QUERY_ROUTER_PROMPT = """You are the query planner for a production RAG system.
Classify the user question into one route using only the question intent.
Return compact JSON only, with this schema:
{
  "route": "fact|table|graph|version|conflict|definition|visual",
  "reasons": ["short reason"],
  "preferred_chunk_kinds": ["table_row|table_summary|clause|definition|media_object|document_page|document_attachment|rendered_page_image|embedded_image"],
  "requires_current_version": true|false,
  "requires_graph": true|false
}

Routing semantics:
- fact: ordinary evidence lookup.
- table: answer likely depends on a row, column, numeric value, unit, parameter, or tabular record.
- graph: answer likely depends on entity relationships, multi-hop dependencies, paths, or provenance reasoning.
- version: answer depends on current/latest/effective version or temporal validity.
- conflict: answer compares disagreeing sources, versions, policies, or evidence.
- definition: answer asks for the meaning of a term or canonical definition.
- visual: answer depends on image/page layout, screenshots, charts, figures, signatures, stamps, scanned pages, or visual details.

Input JSON:
__INPUT_JSON__"""


@dataclass(frozen=True)
class QueryRoute:
    route: str = "fact"
    reasons: list[str] = field(default_factory=lambda: ["default_fact"])
    preferred_chunk_kinds: list[str] = field(default_factory=list)
    requires_current_version: bool = False
    requires_graph: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "route": self.route,
            "reasons": self.reasons,
            "preferred_chunk_kinds": self.preferred_chunk_kinds,
            "requires_current_version": self.requires_current_version,
            "requires_graph": self.requires_graph,
        }


@dataclass(frozen=True)
class QueryRouterConfig:
    enabled: bool = True

    @classmethod
    def from_env(cls) -> "QueryRouterConfig":
        raw = os.getenv("RAG_AI_QUERY_ROUTER_ENABLED", "true").lower()
        return cls(enabled=raw in {"true", "1", "yes"})


class QueryRouter:
    def __init__(
        self,
        generation_client: GenerationClient | None = None,
        config: QueryRouterConfig | None = None,
    ) -> None:
        self.generation_client = generation_client
        self.config = config or QueryRouterConfig.from_env()

    async def route(self, query: str) -> QueryRoute:
        if not self.config.enabled or self.generation_client is None:
            return _heuristic_route(query, reason="router_unavailable")
        try:
            result = await self.generation_client.generate(
                [
                    {
                        "role": "user",
                        "content": QUERY_ROUTER_PROMPT.replace(
                            "__INPUT_JSON__",
                            json.dumps({"question": query}, ensure_ascii=False),
                        ),
                    }
                ]
            )
            return self._parse_route(str(result.get("content") or ""))
        except Exception as exc:
            logger.warning("AI query routing failed: %s", exc)
            return _heuristic_route(query, reason="router_failed")

    def _parse_route(self, content: str) -> QueryRoute:
        payload = self._json_object(content)
        route = str(payload.get("route") or "fact").strip().lower()
        if route not in ROUTES:
            route = "fact"
        reasons = [
            str(item).strip()[:80]
            for item in payload.get("reasons", [])
            if str(item).strip()
        ][:5]
        preferred = [
            str(item).strip().lower()
            for item in payload.get("preferred_chunk_kinds", [])
            if str(item).strip().lower() in CHUNK_KINDS
        ][:4]
        return QueryRoute(
            route=route,
            reasons=reasons or ["ai_classified"],
            preferred_chunk_kinds=_dedupe(preferred),
            requires_current_version=_parse_bool(payload.get("requires_current_version")),
            requires_graph=_parse_bool(payload.get("requires_graph")),
        )

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


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().casefold() in {"true", "1", "yes"}
    if isinstance(value, int | float):
        return value != 0
    return False


TABLE_RE = re.compile(
    r"(table|row|column|price|amount|total|count|rate|ratio|score|metric|"
    r"表格|行|列|价格|金额|总计|数量|比例|分数|指标|参数|费用|税率)",
    re.IGNORECASE,
)
VERSION_RE = re.compile(
    r"(latest|current|effective|version|as of|now|today|"
    r"最新|当前|现行|生效|版本|截至|今天)",
    re.IGNORECASE,
)
CONFLICT_RE = re.compile(
    r"(conflict|contradict|inconsistent|compare|difference|disagree|"
    r"冲突|矛盾|不一致|比较|区别|差异|哪个为准)",
    re.IGNORECASE,
)
GRAPH_RE = re.compile(
    r"(relationship|dependency|depends on|path|impact|provenance|"
    r"关系|依赖|路径|影响|上下游|来源链路|血缘)",
    re.IGNORECASE,
)
DEFINITION_RE = re.compile(
    r"^(what is|define|meaning of|什么是|定义|解释)\b|[“\"']?[^？?]{1,40}[”\"']?[:：]?(?:是什么意思|的定义是什么)",
    re.IGNORECASE,
)
VISUAL_RE = re.compile(
    r"(image|picture|photo|figure|chart|diagram|screenshot|scan|scanned|layout|"
    r"signature|stamp|logo|visual|page view|"
    r"图片|照片|图像|图表|图示|截图|扫描|版面|页面|签名|签章|盖章|印章|logo|外观|视觉)",
    re.IGNORECASE,
)


def _heuristic_route(query: str, *, reason: str) -> QueryRoute:
    stripped = query.strip()
    if VISUAL_RE.search(stripped):
        return QueryRoute(
            route="visual",
            reasons=[reason, "heuristic_visual_terms"],
            preferred_chunk_kinds=[
                "rendered_page_image",
                "embedded_image",
                "media_object",
                "document_page",
                "document_attachment",
            ],
        )
    if CONFLICT_RE.search(stripped):
        return QueryRoute(
            route="conflict",
            reasons=[reason, "heuristic_conflict_terms"],
            preferred_chunk_kinds=["clause"],
            requires_current_version=True,
            requires_graph=True,
        )
    if VERSION_RE.search(stripped):
        return QueryRoute(
            route="version",
            reasons=[reason, "heuristic_version_terms"],
            preferred_chunk_kinds=["clause"],
            requires_current_version=True,
        )
    if TABLE_RE.search(stripped) or re.search(r"\d", stripped):
        return QueryRoute(
            route="table",
            reasons=[reason, "heuristic_table_or_numeric_terms"],
            preferred_chunk_kinds=["table_row", "table_summary"],
        )
    if GRAPH_RE.search(stripped):
        return QueryRoute(
            route="graph",
            reasons=[reason, "heuristic_graph_terms"],
            requires_graph=True,
        )
    if DEFINITION_RE.search(stripped):
        return QueryRoute(
            route="definition",
            reasons=[reason, "heuristic_definition_terms"],
            preferred_chunk_kinds=["definition"],
        )
    return QueryRoute(reasons=[reason, "heuristic_fact_default"])
