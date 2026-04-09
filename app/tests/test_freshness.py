from app.retrieval.freshness import FreshnessPolicy, prioritize_fresh_hits
from app.schemas.chunk import Chunk
from app.vectorstore.repository import SearchHit


def _hit(
    chunk_id: str,
    score: float,
    *,
    title: str = "Policy",
    source_path: str = "/tmp/policy.md",
    metadata: dict | None = None,
) -> SearchHit:
    return SearchHit(
        chunk=Chunk(
            chunk_id=chunk_id,
            doc_id="doc",
            chunk_index=0,
            text=chunk_id,
            source_path=source_path,
            title=title,
            metadata=metadata or {},
        ),
        score=score,
    )


def test_prioritize_fresh_hits_prefers_newer_effective_date_within_same_section() -> None:
    hits = [
        _hit(
            "old",
            2.0,
            metadata={
                "source_key": "leave policy",
                "section_path_text": "Handbook > Carryover",
                "effective_date": "2025-01-01",
            },
        ),
        _hit(
            "new",
            1.0,
            metadata={
                "source_key": "leave policy",
                "section_path_text": "Handbook > Carryover",
                "effective_date": "2026-01-01",
            },
        ),
    ]

    ranked = prioritize_fresh_hits(hits, FreshnessPolicy(enabled=True, allow_stale_fallback=False))

    assert [hit.chunk.chunk_id for hit in ranked] == ["new"]
    assert ranked[0].chunk.metadata["freshness_tier"] == "primary"
    assert ranked[0].chunk.metadata["is_latest_version"] is True


def test_prioritize_fresh_hits_prefers_higher_version_when_dates_match() -> None:
    hits = [
        _hit(
            "v1",
            3.0,
            metadata={
                "source_key": "expense policy",
                "section_path_text": "Guide > Reimbursement",
                "effective_date": "2026-01-01",
                "version": "v1.2",
            },
        ),
        _hit(
            "v2",
            1.0,
            metadata={
                "source_key": "expense policy",
                "section_path_text": "Guide > Reimbursement",
                "effective_date": "2026-01-01",
                "version": "v2.0",
            },
        ),
    ]

    ranked = prioritize_fresh_hits(hits, FreshnessPolicy(enabled=True, allow_stale_fallback=False))

    assert [hit.chunk.chunk_id for hit in ranked] == ["v2"]
