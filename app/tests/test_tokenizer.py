"""Tokenization modes for the BM25 discovery index.

The tokenizer decides what a term is, which is the biggest relevance lever for a
lexical index over Chinese. These pin the three modes, the shared non-CJK
behaviour (so an A/B measures only the CJK difference), and the fail-loud rules.
"""

from __future__ import annotations

import pytest

from app.core.exceptions import ConfigurationError
from app.retrieval.bm25 import BM25Config, BM25Index
from app.retrieval.tokenizer import DEFAULT_TOKENIZER, tokenize


def test_default_mode_is_unchanged() -> None:
    # The shipped default must stay cjk_ngram: switching it would silently
    # change every existing index.
    assert DEFAULT_TOKENIZER == "cjk_ngram"


def test_unknown_mode_raises() -> None:
    with pytest.raises(ConfigurationError):
        from app.retrieval.tokenizer import resolve_tokenizer_mode

        resolve_tokenizer_mode("jieba2")


def test_cjk_ngram_emits_characters_and_bigrams() -> None:
    tokens = tokenize("检索增强", "cjk_ngram")
    assert set(tokens) == {"检", "索", "增", "强", "检索", "索增", "增强"}


def test_cjk_bigram_emits_bigrams_only() -> None:
    tokens = tokenize("检索增强", "cjk_bigram")
    assert tokens == ["检索", "索增", "增强"]


def test_bigram_mode_keeps_single_character_runs() -> None:
    # A one-character run has no bigram; dropping it would make a single
    # character unqueryable.
    assert tokenize("税", "cjk_bigram") == ["税"]


def test_jieba_emits_dictionary_words() -> None:
    tokens = tokenize("检索增强生成", "jieba")
    assert "检索" in tokens
    # jieba's search mode also yields the sub-words it is composed of, so the
    # full compound word is itself available as a term.
    assert "增强" in tokens


def test_non_cjk_tokens_are_identical_across_modes() -> None:
    # The A/B must measure CJK segmentation only.
    text = "RAG 检索 2024 v2.0 beta-test"
    baseline = [t for t in tokenize(text, "cjk_ngram") if not _is_cjk_token(t)]
    for mode in ("cjk_bigram", "jieba"):
        other = [t for t in tokenize(text, mode) if not _is_cjk_token(t)]
        assert other == baseline


def test_empty_text_has_no_tokens() -> None:
    for mode in ("cjk_ngram", "cjk_bigram", "jieba"):
        assert tokenize("", mode) == []


def test_tokenizer_mode_is_part_of_the_index_config() -> None:
    config = BM25Config(tokenizer="cjk_bigram")
    assert config.tokenizer == "cjk_bigram"


def test_index_uses_the_configured_tokenizer() -> None:
    """The mode must reach the index, not just the config object.

    Single-character queries are the clearest probe: cjk_ngram indexes one
    token per character, so a single character matches; cjk_bigram indexes only
    bigrams, so it cannot.
    """
    document = "检索增强生成方法"
    ngram = BM25Index(BM25Config(tokenizer="cjk_ngram"))
    bigram = BM25Index(BM25Config(tokenizer="cjk_bigram"))
    ngram.add_document("d1", document)
    bigram.add_document("d1", document)

    assert [doc_id for doc_id, _ in ngram.search("检", top_k=5)] == ["d1"]
    assert bigram.search("检", top_k=5) == []


def test_jieba_missing_dependency_fails_loud(monkeypatch) -> None:
    # Selecting a tokenizer whose dependency is absent must raise, not fall
    # back to a different segmentation.
    import app.retrieval.tokenizer as tokenizer_module

    monkeypatch.setattr(tokenizer_module, "_JIEBA", None)
    monkeypatch.setitem(__import__("sys").modules, "jieba", None)
    with pytest.raises(ConfigurationError):
        tokenizer_module._jieba_module()


def _is_cjk_token(token: str) -> bool:
    return all("\u4e00" <= char <= "\u9fff" for char in token)
