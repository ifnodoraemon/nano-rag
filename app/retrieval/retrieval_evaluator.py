from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.utils.constants import MAX_DOC_PREVIEW_LENGTH, MIN_RELEVANCE_DOC_LENGTH

if TYPE_CHECKING:
    from app.model_client.generation import GenerationClient

logger = logging.getLogger(__name__)


@dataclass
class RetrievalEvaluatorConfig:
    enable_evaluation: bool = False
    relevance_threshold: float = 0.5

    @classmethod
    def from_env(cls) -> "RetrievalEvaluatorConfig":
        return cls(
            enable_evaluation=os.getenv("RAG_RETRIEVAL_EVAL_ENABLED", "false").lower()
            in ("true", "1", "yes"),
            relevance_threshold=float(
                os.getenv("RAG_RETRIEVAL_RELEVANCE_THRESHOLD", "0.5")
            ),
        )


RELEVANCE_EVAL_PROMPT = """You are a document relevance evaluator. Given a query and a document, determine if the document is relevant to answering the query.

Query: {query}

Document: {document}

Instructions:
1. Read the query and document carefully
2. Determine if the document contains information that could help answer the query
3. Rate the relevance from 0 to 1, where:
   - 1.0 = Highly relevant, directly answers the query
   - 0.7 = Relevant, contains useful information
   - 0.3 = Partially relevant, some related information
   - 0.0 = Not relevant

Output only a single number between 0 and 1:"""


class RetrievalEvaluator:
    def __init__(
        self,
        generation_client: GenerationClient | None = None,
        config: RetrievalEvaluatorConfig | None = None,
    ) -> None:
        self.generation_client = generation_client
        self.config = config or RetrievalEvaluatorConfig.from_env()

    async def evaluate_relevance(self, query: str, document: str) -> float:
        if not self.generation_client or not self.config.enable_evaluation:
            return 1.0
        if not document or len(document.strip()) < MIN_RELEVANCE_DOC_LENGTH:
            return 0.0
        try:
            prompt = RELEVANCE_EVAL_PROMPT.format(
                query=query, document=document[:MAX_DOC_PREVIEW_LENGTH]
            )
            result = await self.generation_client.generate(
                [{"role": "user", "content": prompt}]
            )
            content = result.get("content", "").strip()
            for line in content.split("\n"):
                line = line.strip()
                if line and line[0].isdigit():
                    try:
                        score = float(line.split()[0])
                        return max(0.0, min(1.0, score))
                    except ValueError:
                        continue
            return 0.5
        except Exception:
            logger.debug("retrieval evaluation failed, returning default score")
            return 0.5

    async def filter_relevant(
        self, query: str, documents: list[str], top_k: int
    ) -> list[tuple[str, float]]:
        if not self.generation_client or not self.config.enable_evaluation:
            return [(doc, 1.0) for doc in documents[:top_k]]
        results = await asyncio.gather(
            *[self.evaluate_relevance(query, doc) for doc in documents]
        )
        scored = list(zip(documents, results))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            (doc, score)
            for doc, score in scored
            if score >= self.config.relevance_threshold
        ][:top_k]
