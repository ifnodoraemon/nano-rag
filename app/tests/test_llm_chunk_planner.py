"""LLM chunk planning: the model judges boundaries, the code guarantees fidelity."""
from __future__ import annotations

import pytest

from app.core.exceptions import ModelOutputError
from app.ingestion.llm_chunk_planner import (
    locate_boundaries,
    plan_chunk_boundaries,
    slice_by_boundaries,
)

DOC = (
    "第一条 员工应遵守考勤制度。\n"
    "第二条 出差前须提交申请，经主管批准。\n"
    "第三条 报销单应附票据。"
)


class _Client:
    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls = 0

    async def generate(self, messages, **kwargs):  # noqa: ANN001, ANN003
        self.calls += 1
        return {"content": self.payload}


def test_boundaries_slice_the_original_verbatim():
    anchors = [("第一条 员工应遵守考勤制度。", "考勤"), ("第二条 出差前须提交申请，经主管批准。", "出差"), ("第三条 报销单应附票据。", "报销")]
    boundaries = locate_boundaries(DOC, anchors)
    pieces = slice_by_boundaries(DOC, boundaries)
    assert [text for text, _ in pieces] == [a for a, _ in anchors]
    # Nothing lost: every piece is a contiguous slice of the original document.
    for text, _ in pieces:
        assert text in DOC
    assert [heading for _, heading in pieces] == ["考勤", "出差", "报销"]


def test_a_paraphrased_anchor_is_rejected():
    """The model rewriting instead of quoting must never produce a chunk."""
    anchors = [("第一条 员工应当遵守考勤制度（改写）", "")]
    with pytest.raises(ModelOutputError, match="anchored|not found|paraphrased"):
        locate_boundaries(DOC, anchors)


def test_an_out_of_order_anchor_is_rejected():
    """A later anchor first means the model reordered the document."""
    anchors = [("第三条 报销单应附票据。", ""), ("第一条 员工应遵守考勤制度。", "")]
    with pytest.raises(ModelOutputError, match="not found after offset|out of order"):
        locate_boundaries(DOC, anchors)


def test_a_document_not_starting_at_offset_zero_gets_a_first_boundary():
    anchors = [("第二条 出差前须提交申请，经主管批准。", "出差")]
    boundaries = locate_boundaries(DOC, anchors)
    assert boundaries[0].start == 0
    pieces = slice_by_boundaries(DOC, boundaries)
    assert pieces[0][0].startswith("第一条")


@pytest.mark.asyncio
async def test_planning_returns_verbatim_chunks():
    payload = (
        '{"chunks": ['
        '{"starts_with": "第一条 员工应遵守考勤制度。", "heading": "考勤"},'
        '{"starts_with": "第二条 出差前须提交申请，经主管批准。", "heading": "出差"}'
        "]}"
    )
    client = _Client(payload)
    chunks = await plan_chunk_boundaries(DOC, client, max_chars=40)  # type: ignore[arg-type]
    assert len(chunks) == 2
    assert "".join(text for text, _ in chunks).startswith("第一条")
    # Content is a contiguous slice of the original.
    assert all(piece in DOC for piece, _ in chunks)


@pytest.mark.asyncio
async def test_a_short_document_skips_the_model():
    client = _Client("{}")
    chunks = await plan_chunk_boundaries("短文本", client, max_chars=1000)  # type: ignore[arg-type]
    assert chunks == [("短文本", "")]
    assert client.calls == 0


@pytest.mark.asyncio
async def test_a_missing_chunk_list_fails_loudly():
    client = _Client('{"other": 1}')
    with pytest.raises(ModelOutputError, match="no chunk list"):
        await plan_chunk_boundaries(DOC * 10, client, max_chars=50)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_an_entry_without_an_anchor_fails_loudly():
    client = _Client('{"chunks": [{"heading": "x"}]}')
    with pytest.raises(ModelOutputError, match="no verbatim anchor"):
        await plan_chunk_boundaries(DOC * 10, client, max_chars=50)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_prose_output_fails_loudly():
    client = _Client("我把文档切成了三段。")
    with pytest.raises(ModelOutputError):
        await plan_chunk_boundaries(DOC * 10, client, max_chars=50)  # type: ignore[arg-type]
