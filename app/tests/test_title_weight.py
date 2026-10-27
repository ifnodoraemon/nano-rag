"""Title weight is derived from the corpus, not configured as a constant.

Measured across two corpora: a title-weighted field helps where titles describe their
document (Chinese regulations: hit@1 0.6111 -> 0.6370 at the derived weight) and buys
almost nothing where titles are taxonomy labels shared by many documents (US Code groups:
hit@1 0.3571 either way). A single constant would be wrong for one of them.
"""
from __future__ import annotations


from app.retrieval.title_weight import (
    MAX_TITLE_WEIGHT,
    MIN_USABLE_DIVERSITY,
    leading_segment,
    resolve_title_weight,
    title_diversity,
    weight_for_diversity,
)

CN_TITLES = [
    "重庆市国防动员条例",
    "宁夏回族自治区人才市场条例",
    "湖北省渔港渔船管理条例",
]

US_TITLES = [
    "BANKS AND BANKING — CHECK TRUNCATION",
    "BANKS AND BANKING — WALL STREET REFORM",
    "ARMED FORCES — PERSONNEL GENERALLY > SEPARATION",
    "ARMED FORCES — PERSONNEL GENERALLY > DISCHARGE",
]


def test_a_leading_segment_is_the_general_part():
    assert leading_segment("BANKS AND BANKING — CHECK TRUNCATION") == "banks and banking"
    assert leading_segment("ARMED FORCES — PERSONNEL > SEPARATION") == "armed forces"
    assert leading_segment("重庆市国防动员条例") == "重庆市国防动员条例"


def test_distinct_titles_score_high_diversity():
    assert title_diversity(CN_TITLES) == 1.0


def test_repeated_taxonomy_labels_score_low_diversity():
    # Two distinct leading segments across four documents.
    assert title_diversity(US_TITLES) <= 0.5


def test_a_corpus_with_no_titles_reports_unknown_not_zero():
    """Silence is not evidence that titles are useless."""
    assert title_diversity([]) is None
    assert title_diversity(["", "   "]) is None


def test_weight_scales_with_diversity_and_is_bounded():
    assert weight_for_diversity(None) == 0.0
    assert weight_for_diversity(0.0) == 0.0
    assert weight_for_diversity(MIN_USABLE_DIVERSITY) == 0.0
    assert weight_for_diversity(1.0) == MAX_TITLE_WEIGHT
    mid = weight_for_diversity(0.6)
    assert 0.0 < mid < MAX_TITLE_WEIGHT


def test_an_explicit_override_beats_the_measurement():
    weight, diversity = resolve_title_weight(CN_TITLES, override=0.0)
    assert weight == 0.0, "the operator's value wins"
    assert diversity == 1.0, "the measurement is still reported"


def test_the_two_corpora_shape_gets_opposite_verdicts():
    cn_weight, _ = resolve_title_weight(CN_TITLES)
    us_weight, _ = resolve_title_weight(US_TITLES * 10)  # 40 docs, 2 segments
    assert cn_weight > us_weight
    assert us_weight == 0.0, "taxonomy labels carry no discriminating signal"
