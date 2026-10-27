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
    tokens = [
        token
        for token in _NON_CJK_TOKEN_RE.findall(segment)
        if len(token) > 1 or token.isalpha() or token.isdigit()
    ]
    if not _PLURAL_FOLDING:
        return tokens
    # Inflection fragments a term: measured on a US Code corpus, 23% of word roots
    # appear in more than one form ("find" / "finding" / "finds"), and each form is a
    # separate index term, so a query using one form misses documents using another.
    # Folding is suffix-based and applies to any alphabetic script that inflects at the
    # end of a word; it never changes a CJK token and never introduces vocabulary.
    return [_fold_inflection(token) for token in tokens]


# Suffixes stripped when indexing Latin-script words. Deliberately shallow - a full
# stemmer would conflate unrelated words ("university" / "universal") and the index must
# not invent equivalences. Ordered longest-first so "ies" is not reduced to "ie".
_INFLECTIONS = ("ingly", "edly", "ing", "ied", "ies", "ed", "es", "s")
# A stem shorter than this is left alone: "is" -> "i" would be worse than useless.
_MIN_STEM = 4

_PLURAL_FOLDING = True



def _fold_inflection(token: str) -> str:
    """Strip a common inflection suffix from an alphabetic token, conservatively.

    Deliberately shallow. A real stemmer (Porter/Snowball) would do this properly, but
    those are language-specific and this index serves any Latin-script corpus; a small
    suffix rule covers the common cases and, more importantly, cannot conflate unrelated
    words the way an aggressive stemmer can ("university" / "universal").
    """
    # The word must be long enough that stripping a suffix still leaves a stem: the
    # shortest realistic case is a 5-letter plural ("banks" -> "bank").
    if not token.isalpha() or len(token) <= _MIN_STEM:
        return token
    # A vowel-final word and its inflections must reach the same term: "dispute" and
    # "disputes" fold to "disput", so the base form has to as well. Without this the
    # base and the plural live in different index terms - the exact failure this rule
    # exists to prevent.
    if token.endswith("e") and len(token) >= _MIN_STEM + 1:
        trimmed = token[:-1]
        if not trimmed.endswith("e"):
            return trimmed
    for suffix in _INFLECTIONS:
        if not token.endswith(suffix):
            continue
        stem = token[: -len(suffix)]
        if len(stem) < _MIN_STEM:
            continue
        if suffix in ("ies", "ied"):
            return stem + "y"
        # A doubled final consonant marks a regular inflection: "stopp(ing)" -> "stop",
        # "runn(ing)" -> "run"; "ll" and "ss" are already the word's own ending.
        if len(stem) > 2 and stem[-1] == stem[-2] and stem[-1] not in "lsz":
            return stem[:-1]
        # A stem ending in a vowel lost its silent e: "write" + "ing" = "writing", the
        # visible stem being "writ". Keeping such tokens whole would need a dictionary of
        # which stems are e-dropping, and a word list is exactly what this module refuses
        # to carry. Folding them instead is the safer error: "writing" and "write" both
        # reach "writ", so they still match each other, and the collision with a genuine
        # "writ" is rare enough to cost far less than missing the inflection entirely.
        if suffix == "ing" and stem[-1] in "aeiou":
            return token
        return stem
    return token


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
