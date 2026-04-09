import pytest

from app.retrieval.reranker import RetrievalReranker
from app.schemas.chunk import Chunk
from app.vectorstore.repository import SearchHit


class FixedRerankClient:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores

    async def rerank(self, query: str, documents: list[str], top_k: int):  # noqa: ARG002
        return [
            type(
                "RerankResult",
                (),
                {"index": index, "score": score, "document": documents[index]},
            )
            for index, score in enumerate(self.scores[:top_k])
        ]


def _hit(
    chunk_id: str,
    text: str,
    *,
    metadata: dict | None = None,
) -> SearchHit:
    return SearchHit(
        chunk=Chunk(
            chunk_id=chunk_id,
            doc_id="doc",
            chunk_index=0,
            text=text,
            source_path="/tmp/test.md",
            title="Test",
            metadata=metadata or {},
        ),
        score=0.0,
    )


@pytest.mark.asyncio
async def test_reranker_boosts_stable_topic_hits_with_matching_section() -> None:
    reranker = RetrievalReranker(FixedRerankClient([1.0, 0.8]))
    hits = [
        _hit("raw-1", "Generic expense guidance."),
        _hit(
            "wiki:topic:leave",
            "Carryover policy summary.",
            metadata={
                "wiki_kind": "topic",
                "wiki_status": "stable",
                "section_path": ["Employee Handbook", "Leave Policy"],
            },
        ),
    ]

    reranked = await reranker.rerank("leave policy", hits, 2)

    assert reranked[0].chunk.chunk_id == "wiki:topic:leave"


@pytest.mark.asyncio
async def test_reranker_penalizes_conflicting_wiki_hits() -> None:
    reranker = RetrievalReranker(FixedRerankClient([1.0, 0.9]))
    hits = [
        _hit(
            "wiki:topic:leave",
            "Carryover policy summary.",
            metadata={
                "wiki_kind": "topic",
                "wiki_status": "conflicting",
                "section_path": ["Employee Handbook", "Leave Policy"],
            },
        ),
        _hit("raw-1", "Expense reimbursements must be filed within 30 days."),
    ]

    reranked = await reranker.rerank("leave policy", hits, 2)

    assert reranked[0].chunk.chunk_id == "raw-1"


@pytest.mark.asyncio
async def test_reranker_boosts_recent_effective_date_and_doc_type_match() -> None:
    reranker = RetrievalReranker(FixedRerankClient([1.0, 1.0]))
    hits = [
        _hit(
            "policy-old",
            "Legacy policy text.",
            metadata={"doc_type": "policy", "effective_date": "2024-01-01"},
        ),
        _hit(
            "policy-new",
            "Current policy text.",
            metadata={"doc_type": "policy", "effective_date": "9999-01-01"},
        ),
    ]

    reranked = await reranker.rerank("policy", hits, 2)

    assert reranked[0].chunk.chunk_id == "policy-new"
