from __future__ import annotations

import math
import re
import threading
from collections import Counter
from dataclasses import dataclass

from app.core.exceptions import RetrievalError


@dataclass
class BM25Config:
    k1: float = 1.5
    b: float = 0.75
    max_documents: int = 50000


class BM25Index:
    """Inverted-index BM25.

    Design points (high concurrency, high performance):

    - ``_postings`` maps term -> {doc_id: tf}, so a query only touches the
      posting lists of its query terms instead of scoring every document
      (the previous implementation was O(N · |query|) per search).
    - Document text is NOT retained: only lengths and term frequencies are
      kept, halving resident memory next to the wiki page bodies.
    - Reaching ``max_documents`` raises instead of silently dropping the
      document: an invisible capacity ceiling is exactly the kind of silent
      degradation this project forbids.
    - A reentrant lock guards mutation + search so concurrent callers (event
      loop via to_thread, ingest-side refresh) never observe a half-built
      index. Searchers built into a fresh instance and swapped atomically
      keep read latency stable while a rebuild is in flight.
    """

    def __init__(self, config: BM25Config | None = None) -> None:
        self.config = config or BM25Config()
        self._doc_lengths: dict[str, int] = {}
        # term -> {doc_id: term frequency}
        self._postings: dict[str, dict[str, int]] = {}
        self._total_length = 0
        self._lock = threading.RLock()

    _CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]+")

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        tokens: list[str] = []
        last = 0
        for match in self._CJK_RE.finditer(text):
            if match.start() > last:
                segment = text[last : match.start()]
                word_tokens = re.findall(r"\b\w+\b|[^\w\s]", segment)
                tokens.extend(
                    t for t in word_tokens if len(t) > 1 or t.isalpha() or t.isdigit()
                )
            cjk_chars = [c for c in match.group() if c.strip()]
            tokens.extend(cjk_chars)
            for i in range(len(cjk_chars) - 1):
                tokens.append(cjk_chars[i] + cjk_chars[i + 1])
            last = match.end()
        if last < len(text):
            segment = text[last:]
            word_tokens = re.findall(r"\b\w+\b|[^\w\s]", segment)
            tokens.extend(
                t for t in word_tokens if len(t) > 1 or t.isalpha() or t.isdigit()
            )
        return tokens

    @property
    def document_count(self) -> int:
        with self._lock:
            return len(self._doc_lengths)

    def add_document(self, doc_id: str, text: str) -> None:
        with self._lock:
            if doc_id in self._doc_lengths:
                self._remove_locked(doc_id)
            if len(self._doc_lengths) >= self.config.max_documents:
                raise RetrievalError(
                    f"BM25 index capacity reached ({self.config.max_documents} "
                    f"documents); cannot add {doc_id!r}. Raise the capacity "
                    "configuration instead of silently dropping documents."
                )
            tokens = self._tokenize(text)
            self._doc_lengths[doc_id] = len(tokens)
            self._total_length += len(tokens)
            for term, tf in Counter(tokens).items():
                postings = self._postings.get(term)
                if postings is None:
                    postings = {}
                    self._postings[term] = postings
                postings[doc_id] = tf

    def remove_document(self, doc_id: str) -> None:
        with self._lock:
            self._remove_locked(doc_id)

    def _remove_locked(self, doc_id: str) -> None:
        length = self._doc_lengths.pop(doc_id, None)
        if length is None:
            return
        self._total_length -= length
        empty_terms: list[str] = []
        for term, postings in self._postings.items():
            if postings.pop(doc_id, None) is not None and not postings:
                empty_terms.append(term)
        for term in empty_terms:
            del self._postings[term]

    def search(
        self,
        query: str,
        top_k: int,
        allowed_doc_ids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        with self._lock:
            if not self._doc_lengths:
                return []
            query_tokens = self._tokenize(query)
            if not query_tokens:
                return []
            n_docs = len(self._doc_lengths)
            avg_length = self._total_length / n_docs if n_docs else 1.0

            # Gather candidates from the posting lists of the query terms
            # only — never a full scan over all documents.
            candidates: set[str] = set()
            term_freqs: dict[str, dict[str, int]] = {}
            for term in set(query_tokens):
                postings = self._postings.get(term)
                if not postings:
                    continue
                term_freqs[term] = postings
                if allowed_doc_ids is None:
                    candidates.update(postings)
                else:
                    candidates.update(postings.keys() & allowed_doc_ids)
            if not candidates:
                return []

            scores: list[tuple[str, float]] = []
            for doc_id in candidates:
                doc_length = self._doc_lengths.get(doc_id)
                if not doc_length:
                    continue
                score = 0.0
                for term in query_tokens:
                    postings = term_freqs.get(term)
                    if postings is None:
                        continue
                    tf = postings.get(doc_id)
                    if not tf:
                        continue
                    df = len(postings)
                    idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
                    numerator = tf * (self.config.k1 + 1)
                    denominator = tf + self.config.k1 * (
                        1 - self.config.b + self.config.b * doc_length / avg_length
                    )
                    score += idf * numerator / denominator
                if score > 0:
                    scores.append((doc_id, score))
            scores.sort(key=lambda item: item[1], reverse=True)
            return scores[:top_k]

    def clear(self) -> None:
        with self._lock:
            self._doc_lengths.clear()
            self._postings.clear()
            self._total_length = 0
