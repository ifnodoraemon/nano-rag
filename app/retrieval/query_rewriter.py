from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.model_client.generation import GenerationClient

logger = logging.getLogger(__name__)


@dataclass
class QueryRewriterConfig:
    enable_rewrite: bool = False
    enable_multi_query: bool = False
    multi_query_count: int = 3
    enable_hyde: bool = False

    @classmethod
    def from_env(cls) -> "QueryRewriterConfig":
        return cls(
            enable_rewrite=os.getenv("RAG_QUERY_REWRITE_ENABLED", "false").lower()
            in ("true", "1", "yes"),
            enable_multi_query=os.getenv("RAG_MULTI_QUERY_ENABLED", "false").lower()
            in ("true", "1", "yes"),
            multi_query_count=int(os.getenv("RAG_MULTI_QUERY_COUNT", "3")),
            enable_hyde=os.getenv("RAG_HYDE_ENABLED", "false").lower()
            in ("true", "1", "yes"),
        )


QUERY_REWRITE_PROMPT = """You are a query optimization assistant. Rewrite the following query to be more effective for document retrieval. Make it more specific and searchable while preserving the original intent.

Original query: {query}

Rewritten query:"""

MULTI_QUERY_PROMPT = """You are a query expansion assistant. Generate {count} different versions of the following query that would help find relevant documents from different angles. Each version should use different keywords or phrasing while maintaining the same intent.

Original query: {query}

Generate {count} queries, one per line:
1."""

HYDE_PROMPT = """You are a hypothetical document generator. Given a query, generate a hypothetical document that would perfectly answer this query. This document will be used to find similar real documents.

Query: {query}

Generate a brief hypothetical document that answers this query:"""


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
            prompt = QUERY_REWRITE_PROMPT.format(query=query)
            result = await self.generation_client.generate(
                [{"role": "user", "content": prompt}]
            )
            rewritten = result.get("content", "").strip()
            return rewritten if rewritten else query
        except Exception:
            logger.debug("query rewrite failed, returning original query")
            return query

    async def generate_multi_queries(self, query: str) -> list[str]:
        if not self.generation_client or not self.config.enable_multi_query:
            return [query]
        try:
            prompt = MULTI_QUERY_PROMPT.format(
                query=query, count=self.config.multi_query_count
            )
            result = await self.generation_client.generate(
                [{"role": "user", "content": prompt}]
            )
            content = result.get("content", "").strip()
            queries = [query]
            for line in content.split("\n"):
                line = line.strip()
                if line and len(line) > 5:
                    cleaned = line
                    for prefix in ["1.", "2.", "3.", "4.", "5.", "-", "*"]:
                        if cleaned.startswith(prefix):
                            cleaned = cleaned[len(prefix) :].strip()
                            break
                    if cleaned and cleaned.lower() != query.lower():
                        queries.append(cleaned)
            return queries[: self.config.multi_query_count + 1]
        except Exception:
            logger.debug("multi-query generation failed, returning original query")
            return [query]

    async def generate_hyde(self, query: str) -> str:
        if not self.generation_client or not self.config.enable_hyde:
            return query
        try:
            prompt = HYDE_PROMPT.format(query=query)
            result = await self.generation_client.generate(
                [{"role": "user", "content": prompt}]
            )
            hyde_doc = result.get("content", "").strip()
            return hyde_doc if hyde_doc else query
        except Exception:
            logger.debug("hyde generation failed, returning original query")
            return query
