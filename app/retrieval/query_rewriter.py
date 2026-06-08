from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.model_client.generation import GenerationClient

logger = logging.getLogger(__name__)

LIST_ITEM_PREFIX_RE = re.compile(r"^\s*(?:[-*]|\d+[\.)])\s*")


@dataclass
class QueryExpansionPlan:
    rewritten_query: str | None
    retrieval_queries: list[str]
    hyde_query: str | None = None


@dataclass
class QueryRewriterConfig:
    enable_rewrite: bool = False
    enable_multi_query: bool = False
    multi_query_count: int = 3
    enable_hyde: bool = False
    enable_decomposition: bool = False

    @classmethod
    def from_env(cls) -> "QueryRewriterConfig":
        return cls(
            enable_rewrite=os.getenv("RAG_QUERY_REWRITE_ENABLED", "true").lower()
            in ("true", "1", "yes"),
            enable_multi_query=os.getenv("RAG_MULTI_QUERY_ENABLED", "true").lower()
            in ("true", "1", "yes"),
            multi_query_count=int(os.getenv("RAG_MULTI_QUERY_COUNT", "3")),
            enable_hyde=os.getenv("RAG_HYDE_ENABLED", "true").lower()
            in ("true", "1", "yes"),
            enable_decomposition=os.getenv("RAG_QUERY_DECOMPOSITION_ENABLED", "true").lower()
            in ("true", "1", "yes"),
        )


QUERY_REWRITE_PROMPT = """You are a query optimization assistant. Rewrite the query from the input JSON to be more effective for document retrieval. Make it more specific and searchable while preserving the original intent.

Input JSON:
{input_json}

Rewritten query:"""

MULTI_QUERY_PROMPT = """You are a query expansion assistant. Generate {count} different versions of the query from the input JSON that would help find relevant documents from different angles. Each version should use different phrasing while maintaining the same intent.

Input JSON:
{input_json}

Generate {count} queries, one per line:
1."""

HYDE_PROMPT = """You are a hypothetical document generator. Given the query from the input JSON, generate a hypothetical document that would perfectly answer this query. This document will be used to find similar real documents.

Input JSON:
{input_json}

Generate a brief hypothetical document that answers this query:"""

DECOMPOSITION_PROMPT = """You are a query decomposition assistant. If the input query contains multiple independent questions or intents, break them down into separate, self-contained queries. If it is a single question, return just that question. Each sub-query must be fully self-contained without pronouns.

Input JSON:
{input_json}

Generate independent queries, one per line:
1."""


class QueryRewriter:
    def __init__(
        self,
        generation_client: GenerationClient | None = None,
        config: QueryRewriterConfig | None = None,
    ) -> None:
        self.generation_client = generation_client
        self.config = config or QueryRewriterConfig.from_env()

    async def rewrite(self, query: str) -> str:
        if not self.generation_client or not self.config.enable_rewrite:
            return query
        try:
            prompt = QUERY_REWRITE_PROMPT.format(input_json=_query_input_json(query))
            result = await self.generation_client.generate(
                [{"role": "user", "content": prompt}]
            )
            rewritten = result.get("content", "").strip()
            return rewritten if rewritten else query
        except Exception as exc:
            logger.warning("query rewrite failed: %s", exc)
            return query

    async def generate_multi_queries(self, query: str) -> list[str]:
        if not self.generation_client or not self.config.enable_multi_query:
            return [query]
        try:
            prompt = MULTI_QUERY_PROMPT.format(
                input_json=_query_input_json(query), count=self.config.multi_query_count
            )
            result = await self.generation_client.generate(
                [{"role": "user", "content": prompt}]
            )
            content = result.get("content", "").strip()
            queries = [query]
            for line in content.split("\n"):
                line = line.strip()
                if line and len(line) > 5:
                    cleaned = LIST_ITEM_PREFIX_RE.sub("", line, count=1).strip()
                    if cleaned and cleaned.lower() != query.lower():
                        queries.append(cleaned)
            return queries[: self.config.multi_query_count + 1]
        except Exception as exc:
            logger.warning("multi-query generation failed: %s", exc)
            return [query]

    async def generate_hyde(self, query: str) -> str:
        if not self.generation_client or not self.config.enable_hyde:
            return query
        try:
            prompt = HYDE_PROMPT.format(input_json=_query_input_json(query))
            result = await self.generation_client.generate(
                [{"role": "user", "content": prompt}]
            )
            hyde_doc = result.get("content", "").strip()
            return hyde_doc if hyde_doc else query
        except Exception as exc:
            logger.warning("hyde generation failed: %s", exc)
            return query

    async def decompose(self, query: str) -> list[str]:
        if not self.generation_client or not self.config.enable_decomposition:
            return [query]
        try:
            prompt = DECOMPOSITION_PROMPT.format(input_json=_query_input_json(query))
            result = await self.generation_client.generate(
                [{"role": "user", "content": prompt}]
            )
            content = result.get("content", "").strip()
            queries = []
            for line in content.split("\n"):
                line = line.strip()
                if line and len(line) > 3:
                    cleaned = LIST_ITEM_PREFIX_RE.sub("", line, count=1).strip()
                    if cleaned:
                        queries.append(cleaned)
            return queries if queries else [query]
        except Exception as exc:
            logger.warning("query decomposition failed: %s", exc)
            return [query]

    async def build_plan(self, query: str) -> QueryExpansionPlan:
        decomposed = await self.decompose(query)
        retrieval_queries: list[str] = []
        rewritten_query = None
        hyde_query = None

        for sub_query in decomposed:
            rewritten = await self.rewrite(sub_query)
            if not rewritten_query and rewritten != query:
                rewritten_query = rewritten
            
            sub_retrieval_queries = [rewritten]
            if self.config.enable_multi_query:
                sub_retrieval_queries = await self.generate_multi_queries(rewritten)
            
            for q in sub_retrieval_queries:
                retrieval_queries.append(q)

            if self.config.enable_hyde and not hyde_query:
                generated_hyde = await self.generate_hyde(rewritten)
                if generated_hyde and generated_hyde.strip() and generated_hyde.strip() != rewritten:
                    hyde_query = generated_hyde.strip()

        unique_queries: list[str] = []
        seen: set[str] = set()
        for candidate in retrieval_queries:
            normalized = candidate.strip()
            if not normalized:
                continue
            lowered = normalized.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            unique_queries.append(normalized)
        if not unique_queries:
            unique_queries = [query]
        return QueryExpansionPlan(
            rewritten_query=rewritten_query,
            retrieval_queries=unique_queries,
            hyde_query=hyde_query,
        )


def _query_input_json(query: str) -> str:
    return json.dumps({"query": query}, ensure_ascii=False)
