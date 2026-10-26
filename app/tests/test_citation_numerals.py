"""A source that writes 一百八十 supports an answer that writes 180.

Found on a real regression: "工地围挡一般路段至少得多高？" answered "1.8米（180厘米）[C1]"
against a statute reading "一般路段围挡高度不得低于1.8米" and "一百八十厘米". The answer was
right; the citation checker called it unsupported because "180" never appeared literally.
"""
from __future__ import annotations

import pytest

from app.generation.citation_check import chinese_to_number, check_citations


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("一", 1),
        ("二", 2),
        ("两", 2),
        ("十", 10),
        ("十五", 15),
        ("二十", 20),
        ("一百八十", 180),
        ("三千五百", 3500),
        ("一万", 10000),
    ],
)
def test_chinese_numerals_parse(raw, expected):
    assert chinese_to_number(raw) == expected


def test_an_ambiguous_string_returns_none_instead_of_a_guess():
    """A wrong parse would manufacture support that does not exist."""
    assert chinese_to_number("甲乙") is None
    assert chinese_to_number("") is None


def test_arabic_forms_of_the_same_quantity_agree():
    contexts = [
        {
            "citation_label": "C1",
            "text": "一般路段围挡高度不得低于1.8米；城区主要路段不得低于2.5米。",
        }
    ]
    report = check_citations("一般路段围挡高度不得低于 1.8 米 [C1]。", contexts)
    assert report.ungrounded_number_count == 0


def test_a_chinese_numeral_in_the_source_supports_the_arabic_claim():
    contexts = [
        {"citation_label": "C1", "text": "一般路段围挡高度不得低于一百八十厘米。"}
    ]
    report = check_citations("一般路段围挡高度不得低于 180 厘米 [C1]。", contexts)
    assert report.ungrounded_number_count == 0, (
        "一百八十 and 180 are the same claim; failing this reports a false support failure"
    )
