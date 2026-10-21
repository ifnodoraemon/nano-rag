"""Tokenization for the BM25 discovery index.

Chinese has no whitespace between words, so the tokenizer decides what a "term"
is. The choice materially changes ranking:

- ``cjk_ngram`` (default) emits one token per character *and* one per character
  bigram. High recall, low precision: every common character is a term, so two
  unrelated documents share many tokens and IDF discriminates poorly.
- ``cjk_bigram`` emits bigrams only. Drops the single-character noise floor, so
  common characters stop creating spurious matches. Measured best separation on
  this repo's Chinese corpus (see README). Known limitation: a one-character
  query cannot match a multi-character document, because there is no shared
  bigram.
- ``jieba`` emits dictionary words (jieba's search mode: words plus their
  sub-words). Precise terms, best IDF contrast, at the cost of depending on a
  segmentation dictionary and of not matching a word boundary the dictionary
  disagrees with.

Non-CJK text is tokenized identically in every mode, so an A/B measures the CJK
difference and nothing else.

This is a *lexical* tokenizer, not a semantic keyword list: it decides where
word boundaries are, never what a word means.
"""

from __future__ import annotations

import re
import threading

from app.core.exceptions import ConfigurationError

TOKENIZER_MODES = ("cjk_ngram", "cjk_bigram", "jieba")
DEFAULT_TOKENIZER = "cjk_ngram"

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]+")
_NON_CJK_TOKEN_RE = re.compile(r"\b\w+\b|[^\w\s]")

# jieba's first call builds the prefix dictionary (~0.5s). Cache the module and
# guard initialisation so concurrent ingest threads do not each pay for it.
_JIEBA_LOCK = threading.Lock()
_JIEBA = None


def resolve_tokenizer_mode(raw: str | None) -> str:
    """Validate a tokenizer mode name, failing loudly on a typo."""
    mode = (raw or DEFAULT_TOKENIZER).strip().lower()
    if mode not in TOKENIZER_MODES:
        raise ConfigurationError(
            f"unknown BM25 tokenizer {raw!r}; expected one of "
            f"{', '.join(TOKENIZER_MODES)}"
        )
    return mode


def _jieba_module():
    global _JIEBA
    with _JIEBA_LOCK:
        if _JIEBA is None:
            try:
                import jieba
            except ImportError as exc:  # pragma: no cover - dependency is pinned
                raise ConfigurationError(
                    "RAG_BM25_TOKENIZER=jieba requires the 'jieba' package; "
                    "install it or use cjk_ngram/cjk_bigram"
                ) from exc
            jieba.initialize()
            _JIEBA = jieba
    return _JIEBA


def _cjk_tokens(run: str, mode: str) -> list[str]:
    if mode == "cjk_ngram":
        tokens = list(run)
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
        return tokens
    if mode == "cjk_bigram":
        if len(run) == 1:
            # A one-character run has no bigram; keep it, otherwise a query on a
            # single character could never match.
            return list(run)
        return [run[i : i + 2] for i in range(len(run) - 1)]
    jieba = _jieba_module()
    return [
        token for token in jieba.lcut_for_search(run) if token and token.strip()
    ]


def _non_cjk_tokens(segment: str) -> list[str]:
    return [
        token
        for token in _NON_CJK_TOKEN_RE.findall(segment)
        if len(token) > 1 or token.isalpha() or token.isdigit()
    ]


def tokenize(text: str, mode: str = DEFAULT_TOKENIZER) -> list[str]:
    """Split text into index terms for the given mode."""
    normalized = str(text or "").lower()
    if not normalized:
        return []
    tokens: list[str] = []
    last = 0
    for match in _CJK_RE.finditer(normalized):
        if match.start() > last:
            tokens.extend(_non_cjk_tokens(normalized[last : match.start()]))
        tokens.extend(_cjk_tokens(match.group(), mode))
        last = match.end()
    if last < len(normalized):
        tokens.extend(_non_cjk_tokens(normalized[last:]))
    return tokens
