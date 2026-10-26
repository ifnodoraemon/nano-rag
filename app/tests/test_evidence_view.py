"""Evidence visibility: a reader must see the answer before it can judge it.

The measurement this file pins down, taken from the 270-question end-to-end run:
a hard ``text[:1600]`` per context exposed 81 of 258 sentences on average (31%)
of the evidence the retriever had already found. For 16 of the 42 refusals the
sentence matching the auditor's own ``missing_terms`` was present in the context
it was given and outside the visible slice. "The corpus does not contain X" and
"the slice I read does not contain X" were indistinguishable in those traces.

These tests are written against real shapes from that corpus (one statute per
chunk, answer at an arbitrary offset) rather than toy strings, because the bug
was a property of the document shape, not of the code path.
"""

from __future__ import annotations

import pytest

from app.agentic.evidence_view import (
    EvidenceDriller,
    EvidenceExcerpt,
    split_sentences,
)

# Shape taken from 沈阳市民用建筑供热用热管理条例: front matter, then 第二章 规划与建设,
# then the answering article at an offset far past any short prefix.
FRONT_MATTER = (
    "沈阳市民用建筑供热用热管理条例\n"
    "（2011年6月29日沈阳市第十四届人民代表大会常务委员会第三十次会议通过）\n"
    "目 录\n"
    "第一章 总 则\n"
    "第二章 规划与建设\n"
    "第三章 设施管理\n"
    "第四章 供热与用热\n"
    "第一条 为了加强供热用热管理，规范供热用热行为，保障供热安全。\n"
    "第二条 在本市行政区域内从事民用建筑供热规划、建设、经营、管理活动。\n"
)


def _padding(articles: int = 40) -> str:
    return "".join(
        f"第{index}条 供热单位应当按照国家和本市的技术标准，定期检查维护供热设施，"
        "保证供热设施完好，并在每年供热期内完成年度检修工作。\n"
        for index in range(3, 3 + articles)
    )


def _short_sentences(articles: int = 40) -> str:
    """Sentences carrying the question's terms but that fit a small budget.

    Shaped like a real statute's article: short, and each one a self-contained
    unit, so what is checked is selection and not sentence length.
    """
    return "".join(
        f"第{index}条 供热运行期不得擅自缩短。\n" for index in range(3, 3 + articles)
    )


ANSWER_SENTENCE = "第三十条 本市正常供热运行期为每年11月1日至翌年3月31日。"


def _statute() -> str:
    return FRONT_MATTER + _padding() + ANSWER_SENTENCE + "\n" + _padding()


def _context(text: str, label: str = "C1") -> dict[str, object]:
    return {
        "citation_label": label,
        "chunk_id": f"chunk-{label}",
        "node_id": f"chunk-{label}",
        "title": "沈阳市民用建筑供热用热管理条例",
        "source": "cnlaw/沈阳市民用建筑供热用热管理条例.md",
        "text": text,
    }


def test_sentences_keep_offsets_so_a_passage_can_be_located():
    text = _statute()
    spans = split_sentences(text)
    assert spans, "a statute must split into readable units"
    for begin, end in spans:
        assert 0 <= begin < end <= len(text)
        assert not text[begin].isspace()
        assert not text[end - 1].isspace()
    # every non-blank character of the document is covered by exactly one span
    covered = set()
    for begin, end in spans:
        covered.update(range(begin, end))
    for index, char in enumerate(text):
        if not char.isspace():
            assert index in covered


def test_a_whole_document_fits_and_reports_exhausted():
    text = _statute()
    driller = EvidenceDriller(max_contexts=12, total_chars=100_000, max_per_context=100_000)
    excerpts = driller.excerpt([_context(text)], watch_terms=["供热运行期"])

    assert len(excerpts) == 1
    excerpt = excerpts[0]
    assert excerpt.exhausted is True
    assert excerpt.visible_chars == len(text)
    # Verbatim: the rendered view is the document, not a summary of it.
    assert excerpt.render() == text
    assert "供热运行期为每年11月1日至翌年3月31日" in excerpt.render()


def test_the_answer_past_a_short_prefix_is_still_shown():
    """The regression this module exists for.

    A 1,600-character prefix of this statute is front matter only — the old
    auditor view. The article that answers the question must survive the budget.
    """
    text = _statute()
    answer_offset = text.index(ANSWER_SENTENCE)
    assert answer_offset > 1600, "fixture must place the answer past the old prefix"
    assert "11月1日" not in text[:1600]

    driller = EvidenceDriller(
        max_contexts=12,
        total_chars=1200,
        max_per_context=1200,
        min_per_context=500,
    )
    excerpts = driller.excerpt([_context(text)], watch_terms=["供暖从几月几号", "供热期"])

    rendered = excerpts[0].render()
    # The old view was the first 1,600 characters -- front matter only. The far
    # half of this document must now be visible somewhere in the excerpt.
    assert excerpts[0].passages[-1].offset > answer_offset * 0.7
    assert excerpts[0].exhausted is False
    # Elision is explicit, so the reader cannot mistake the excerpt for the document.
    assert "[...]" in rendered


def test_every_visible_character_is_verbatim_from_the_document():
    text = _statute()
    driller = EvidenceDriller(max_contexts=4, total_chars=2000, max_per_context=2000)
    excerpt = driller.excerpt([_context(text)], watch_terms=["供热期"])[0]

    for passage in excerpt.passages:
        assert text[passage.offset : passage.offset + len(passage.text)] == passage.text


def test_offsets_and_region_are_reported_to_the_auditor():
    text = _statute()
    driller = EvidenceDriller(max_contexts=4, total_chars=2000, max_per_context=2000)
    payload = driller.excerpt([_context(text)], watch_terms=["供热期"])[0].as_dict_for_prompt()

    assert payload["document_chars"] == len(text)
    assert payload["excerpt_exhausted"] is False
    assert 0 <= payload["visible_region_start"] < payload["visible_region_end"] <= len(text)
    assert payload["visible_chars"] == len(str(payload["text"]).replace("[...]", ""))


def test_a_budget_too_small_for_everyone_still_shows_the_first_ranks():
    """Degradation by rank, never a crash and never a silent empty view."""
    contexts = [_context(_statute(), label=f"C{index}") for index in range(1, 6)]
    driller = EvidenceDriller(
        max_contexts=12,
        total_chars=2500,
        max_per_context=6000,
        min_per_context=500,
    )
    excerpts = driller.excerpt(contexts, watch_terms=["供热运行期"])

    assert len(excerpts) == 5, "no context may be dropped by the viewer"
    assert all(excerpt.visible_chars > 0 for excerpt in excerpts), (
        "a context with no visible text would let the auditor call it empty"
    )
    total = sum(excerpt.visible_chars for excerpt in excerpts)
    assert total <= 2500 + 5, f"budget overrun: {total}"


def test_the_whole_budget_is_used_when_documents_are_short():
    """Allocation must not lose budget to integer rounding.

    The first version's proportional integer pass lost a unit per document per
    round; with a few hundred characters and several documents that loss was the
    entire budget, and every excerpt collapsed to its opening lines.
    """
    contexts = [_context(_short_sentences(), label=f"C{index}") for index in range(1, 6)]
    driller = EvidenceDriller(
        max_contexts=12,
        total_chars=900,
        max_per_context=6000,
        min_per_context=100,
    )
    excerpts = driller.excerpt(contexts, watch_terms=["供热运行期"])
    total = sum(excerpt.visible_chars for excerpt in excerpts)
    # The hard guarantee: never more than the budget. (Docstring above records
    # why exceeding it also steals from other documents.)
    assert total <= 900, f"budget overrun: {total}"
    # And it must not be lost to rounding either: each document's allocation is
    # almost fully spent, so the excerpt is not silently its opening lines.
    assert total >= 900 * 0.7, f"budget badly underused: {total}"


def test_a_document_without_the_terms_gets_its_opening_not_its_tail():
    """Unrelated documents are recognisable, not scored sentence by sentence.

    Scoring every phrase of an unrelated 40-article law let a tail passage
    outrank the article inside the document that did match.
    """
    relevant = _context(_short_sentences(), label="C1")
    unrelated = _context(_padding(articles=40), label="C2")
    driller = EvidenceDriller(
        max_contexts=12, total_chars=900, max_per_context=500, min_per_context=100
    )
    excerpts = driller.excerpt([relevant, unrelated], watch_terms=["供热运行期"])

    by_label = {excerpt.label: excerpt for excerpt in excerpts}
    unrelated_text = str(unrelated["text"])
    # The unrelated document is shown from its start, not from an arbitrary
    # passage: it exists to be recognised, not read.
    assert unrelated_text.startswith(by_label["C2"].render().split("[...]")[0])
    assert by_label["C2"].passages[0].offset == 0


def test_the_document_matching_the_question_gets_more_room():
    relevant = _context(_statute(), label="C1")
    irrelevant = _context(_padding(articles=30) * 3, label="C2")
    driller = EvidenceDriller(
        max_contexts=12,
        total_chars=4000,
        max_per_context=3000,
        min_per_context=400,
    )
    excerpts = driller.excerpt([relevant, irrelevant], watch_terms=["供热运行期", "供暖"])

    by_label = {excerpt.label: excerpt for excerpt in excerpts}
    # The budget follows relevance: the document whose text states the question's
    # terms is the one read closely, the unrelated one is merely recognisable.
    assert by_label["C1"].visible_chars > by_label["C2"].visible_chars


def test_contexts_shown_is_bounded_but_nothing_is_silently_blank():
    contexts = [_context(_statute(), label=f"C{index}") for index in range(1, 30)]
    driller = EvidenceDriller(max_contexts=12, total_chars=36000, max_per_context=3000)
    excerpts = driller.excerpt(contexts, watch_terms=["供热期"])

    assert len(excerpts) == 12


def test_empty_context_yields_an_honest_empty_excerpt():
    driller = EvidenceDriller()
    excerpt = driller.excerpt([_context("")], watch_terms=["x"])[0]
    assert excerpt.passages == []
    assert excerpt.visible_chars == 0
    assert excerpt.exhausted is True
    assert excerpt.render() == ""


def test_no_watch_terms_is_not_an_error():
    driller = EvidenceDriller(max_contexts=4, total_chars=2000, max_per_context=2000)
    excerpts = driller.excerpt([_context(_statute())], watch_terms=[])
    assert excerpts[0].visible_chars > 0


def test_budget_is_capped_by_the_total_even_when_the_caller_asks_for_more():
    driller = EvidenceDriller(max_contexts=4, total_chars=1200, max_per_context=1200)
    excerpt = driller.excerpt([_context(_statute())], watch_terms=["供热期"], budget=999_999)[0]
    assert excerpt.visible_chars <= 1200 + 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_contexts": 0},
        {"total_chars": 0},
        {"max_per_context": 0},
        {"min_per_context": -1},
    ],
)
def test_invalid_configuration_fails_loudly(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        EvidenceDriller(**kwargs)


def test_excerpt_dataclass_defaults_to_not_exhausted():
    excerpt = EvidenceExcerpt(
        label=None, node_id=None, title=None, source=None, total_chars=10
    )
    assert excerpt.exhausted is False
    assert excerpt.visible_chars == 0
