"""Article-level indexing: the document page is a summary, articles are the content.

Measured on 595 Chinese statutes (5,446 chars average): the indexed wiki page body is
1,049 characters, about 15% of the source. The other 80% - every article that is not in
the summary - is invisible to the lexical index unless it is indexed on its own.
"""
from __future__ import annotations

import pytest

from app.core.exceptions import ModelOutputError
from app.wiki.article_index import (
    MIN_ARTICLE_CHARS,
    _body_offset,
    _find_label,
    split_into_articles,
)

STATUTE = """某某条例

目 录

第一章  总则

第二章  组织机构

第一章  总则

第一条 为了规范某某活动，根据有关法律，制定本条例。

第二条 本条例适用于本行政区域内的某某活动。

第三条 某某活动实行属地管理原则，县级以上人民政府负责组织实施。

第二章  组织机构

第四条 县级以上人民政府应当设立某某工作机构，配备专职人员。

第五条 某某工作机构履行下列职责：制定规划、组织实施、监督检查。
"""


class _Client:
    def __init__(self, labels: list[str]) -> None:
        self.labels = labels

    async def generate(self, messages, **kwargs):  # noqa: ANN001, ANN003
        import json

        return {"content": json.dumps({"articles": self.labels}, ensure_ascii=False)}


@pytest.mark.asyncio
async def test_articles_are_sliced_from_the_original_text():
    labels = ["第一条", "第二条", "第三条", "第四条", "第五条"]
    articles = await split_into_articles(STATUTE, _Client(labels), title="某某条例")  # type: ignore[arg-type]
    assert [a.label for a in articles] == labels
    for article in articles:
        assert article.text in STATUTE, "article text must be a slice of the source"
    assert "制定本条例" in articles[0].text


@pytest.mark.asyncio
async def test_the_table_of_contents_does_not_shift_the_cuts():
    """The contents page repeats chapter headings; locating from the front lands there."""
    labels = ["第一条", "第二条", "第三条", "第四条", "第五条"]
    articles = await split_into_articles(STATUTE, _Client(labels), title="某某条例")  # type: ignore[arg-type]
    assert articles[0].text.startswith("第一条")
    assert "目 录" not in articles[0].text


def test_body_offset_skips_a_contents_page():
    offset = _body_offset(STATUTE)
    assert STATUTE[offset:].startswith("第一条")


def test_body_offset_is_zero_without_a_contents_page():
    plain = "第一条 内容内容内容内容内容。\n\n第二条 内容内容内容内容内容。"
    assert _body_offset(plain) == 0


def test_label_lookup_tolerates_nbsp_and_whitespace():
    text = "第十七条\xa0 国防动员潜力统计调查可以采用普查。"
    assert _find_label(text, "第十七条", 0) == 0


def test_label_lookup_respects_the_start_offset():
    text = "第一条 甲。第一条 乙。"
    first = _find_label(text, "第一条", 0)
    second = _find_label(text, "第一条", first + 1)
    assert first is not None and second is not None and second > first


@pytest.mark.asyncio
async def test_a_label_the_document_does_not_contain_raises():
    """Inventing an article would insert a unit the document never had."""
    with pytest.raises(ModelOutputError, match="not found"):
        await split_into_articles(STATUTE, _Client(["第九十九条"]), title="某某条例")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_an_empty_label_list_raises():
    with pytest.raises(ModelOutputError, match="no articles"):
        await split_into_articles(STATUTE, _Client([]), title="某某条例")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_a_short_but_real_article_is_kept():
    """'第二条 ...适用本条例' is 34 chars and defines scope - dropping it is invisible."""
    assert MIN_ARTICLE_CHARS < 34
    labels = ["第一条", "第二条"]
    articles = await split_into_articles(STATUTE, _Client(labels), title="某某条例")  # type: ignore[arg-type]
    assert any("适用" in a.text for a in articles)


@pytest.mark.asyncio
async def test_a_very_short_document_yields_no_articles():
    articles = await split_into_articles("太短", _Client(["第一条"]), title="x")  # type: ignore[arg-type]
    assert articles == []
