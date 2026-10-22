"""Model-backed listwise rerank: strict permutation contract, nothing lost."""
from __future__ import annotations

import pytest

from app.agentic.llm_rerank import RerankDocument, rerank_documents
from app.core.exceptions import ModelOutputError


class _Client:
    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls = 0
        self.last = None

    async def generate(self, messages, **kwargs):  # noqa: ANN001, ANN003
        self.calls += 1
        self.last = messages
        return {"content": self.payload}


def _docs(count: int) -> list[RerankDocument]:
    return [
        RerankDocument(doc_id=f"d{index}", title=f"title {index}", text=f"text {index}")
        for index in range(count)
    ]


@pytest.mark.asyncio
async def test_rerank_reorders_by_the_model_order() -> None:
    client = _Client('{"order": [2, 0, 1]}')
    order = await rerank_documents("q", _docs(3), client, top_n=3)  # type: ignore[arg-type]
    assert order == ["d2", "d0", "d1"]


@pytest.mark.asyncio
async def test_only_the_head_is_sent_and_the_tail_is_preserved() -> None:
    """Sending every candidate would cost tokens for documents never reached."""
    client = _Client('{"order": [1, 0]}')
    order = await rerank_documents("q", _docs(5), client, top_n=2)  # type: ignore[arg-type]
    assert order == ["d1", "d0", "d2", "d3", "d4"]


@pytest.mark.asyncio
async def test_a_partial_order_fails_loudly() -> None:
    client = _Client('{"order": [0]}')
    with pytest.raises(ModelOutputError, match="of 3 candidate indices"):
        await rerank_documents("q", _docs(3), client, top_n=3)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_a_duplicate_index_fails_loudly() -> None:
    client = _Client('{"order": [0, 0, 1]}')
    with pytest.raises(ModelOutputError, match="repeated index"):
        await rerank_documents("q", _docs(3), client, top_n=3)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_an_out_of_range_index_fails_loudly() -> None:
    client = _Client('{"order": [0, 1, 9]}')
    with pytest.raises(ModelOutputError, match="out of range"):
        await rerank_documents("q", _docs(3), client, top_n=3)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_a_non_integer_index_fails_loudly() -> None:
    client = _Client('{"order": ["0", 1, 2]}')
    with pytest.raises(ModelOutputError, match="non-integer"):
        await rerank_documents("q", _docs(3), client, top_n=3)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_prose_output_fails_loudly() -> None:
    client = _Client("I think document 2 is best.")
    with pytest.raises(ModelOutputError):
        await rerank_documents("q", _docs(3), client, top_n=3)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_a_missing_order_key_fails_loudly() -> None:
    client = _Client('{"other": []}')
    with pytest.raises(ModelOutputError, match="no ordering array"):
        await rerank_documents("q", _docs(3), client, top_n=3)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_a_renamed_key_is_accepted_with_a_warning() -> None:
    """A gateway that ignores response_format may rename the key."""
    client = _Client('{"ranking": [1, 0]}')
    order = await rerank_documents("q", _docs(2), client, top_n=2)  # type: ignore[arg-type]
    assert order == ["d1", "d0"]


@pytest.mark.asyncio
async def test_a_single_candidate_skips_the_model() -> None:
    client = _Client('{"order": [0]}')
    order = await rerank_documents("q", _docs(1), client, top_n=10)  # type: ignore[arg-type]
    assert order == ["d0"]
    assert client.calls == 0


@pytest.mark.asyncio
async def test_the_prompt_carries_the_query_and_indices() -> None:
    client = _Client('{"order": [0, 1]}')
    await rerank_documents("差旅报销多久", _docs(2), client, top_n=2)  # type: ignore[arg-type]
    rendered = str(client.last)
    assert "差旅报销多久" in rendered
    assert "[0]" in rendered and "[1]" in rendered
