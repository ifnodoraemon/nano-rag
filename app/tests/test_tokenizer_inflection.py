"""Inflection folding: one word root, one index term.

Measured on a US Code corpus: 23% of word roots appear in more than one form
("find"/"finding"/"finds"), and each form was a separate index term, so a query using one
form could not match a document using another. Folding is suffix-based and applies to any
alphabetic script that inflects at a word's end; it never touches a CJK token and never
introduces vocabulary - it only recognises two spellings as the same word.
"""
from __future__ import annotations

from app.retrieval.tokenizer import tokenize


def test_plural_and_verb_forms_fold_together():
    """Every form of a root must reach the same term, or a query using one misses the
    other - measured at 23% of roots on a US Code corpus."""
    assert tokenize("dispute disputes disputed") == ["disput", "disput", "disput"]
    assert tokenize("find finds finding") == ["find", "find", "find"]
    assert tokenize("bank banks banking")[0] == tokenize("banks")[0]


def test_y_ending_plurals_restore_the_y():
    assert tokenize("policies") == ["policy"]
    assert tokenize("applied") == ["apply"]


def test_a_doubled_consonant_is_undone():
    assert tokenize("stopping") == ["stop"]
    assert tokenize("planned") == ["plan"]


def test_short_words_are_left_alone():
    """'is' must not become 'i'."""
    assert tokenize("is it as") == ["is", "it", "as"]


def test_a_stem_without_an_inflection_is_left_alone():
    assert tokenize("string thing king") == ["string", "thing", "king"]


def test_an_e_dropping_gerund_folds_too_and_that_is_accepted():
    """A documented trade-off, not an accident.

    "writing" folds to "writ" rather than staying whole, because telling it apart from a
    real "writ" needs a dictionary of e-dropping stems - and a word list is what this
    module refuses to carry. Folding is the safer error: "write" and "writing" still
    match each other, and the collision with a genuine "writ" is rare.
    """
    assert tokenize("writing") == tokenize("writes")


def test_cjk_tokens_are_untouched():
    """The folding rule must not change Chinese indexing at all."""
    tokens = tokenize("银行应当依法办理")
    assert "银行" in tokens and "应当" in tokens
    assert all(t for t in tokens)


def test_numbers_and_punctuation_survive():
    tokens = tokenize("Section 12(b)")
    assert "12" in tokens and "b" in tokens
