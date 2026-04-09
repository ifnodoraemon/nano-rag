from __future__ import annotations

from datetime import date, datetime

from app.model_client.rerank import RerankClient
from app.vectorstore.repository import SearchHit

DEFAULT_METADATA_WEIGHTS = {
    "stable_topic": 0.35,
    "stable_source": 0.15,
    "conflicting_penalty": 0.25,
    "section_match": 0.1,
    "doc_type_match": 0.08,
    "effective_date_recency": 0.12,
}


class RetrievalReranker:
    def __init__(
        self,
        client: RerankClient,
        metadata_weights: dict[str, float] | None = None,
    ) -> None:
        self.client = client
        self.metadata_weights = {
            **DEFAULT_METADATA_WEIGHTS,
            **(metadata_weights or {}),
        }

    async def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        results = await self.client.rerank(query, [hit.chunk.text for hit in hits], top_k)
        reranked: list[SearchHit] = []
        for result in results:
            hit = hits[result.index]
            reranked.append(
                SearchHit(
                    chunk=hit.chunk,
                    score=result.score + self._metadata_adjustment(query, hit),
                )
            )
        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked

    def _metadata_adjustment(self, query: str, hit: SearchHit) -> float:
        metadata = hit.chunk.metadata or {}
        adjustment = 0.0

        wiki_kind = metadata.get("wiki_kind")
        wiki_status = metadata.get("wiki_status")
        if wiki_status == "stable" and wiki_kind == "topic":
            adjustment += self.metadata_weights["stable_topic"]
        elif wiki_status == "stable" and wiki_kind == "source":
            adjustment += self.metadata_weights["stable_source"]
        if wiki_status == "conflicting":
            adjustment -= self.metadata_weights["conflicting_penalty"]

        query_tokens = _tokenize(query)
        if query_tokens:
            section_tokens = _tokenize(_flatten_text(metadata.get("section_path")))
            if query_tokens & section_tokens:
                adjustment += self.metadata_weights["section_match"]

            doc_type_tokens = _tokenize(metadata.get("doc_type"))
            if query_tokens & doc_type_tokens:
                adjustment += self.metadata_weights["doc_type_match"]

        effective_date = _parse_date(metadata.get("effective_date"))
        if effective_date:
            adjustment += self.metadata_weights["effective_date_recency"] * _recency_ratio(
                effective_date
            )

        return round(adjustment, 6)


def _flatten_text(value: object) -> str:
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    if isinstance(value, str):
        return value
    return ""


def _tokenize(value: object) -> set[str]:
    text = _flatten_text(value).lower()
    return {token for token in text.replace("-", " ").replace("/", " ").split() if token}


def _parse_date(value: object) -> date | None:
    if not isinstance(value, str) or not value.strip():
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(value.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _recency_ratio(effective_date: date) -> float:
    age_days = max((date.today() - effective_date).days, 0)
    if age_days <= 30:
        return 1.0
    if age_days >= 365:
        return 0.0
    return round(1.0 - ((age_days - 30) / 335), 4)
