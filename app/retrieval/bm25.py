from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass


@dataclass
class BM25Config:
    k1: float = 1.5
    b: float = 0.75
    max_documents: int = 50000


class BM25Index:
    def __init__(self, config: BM25Config | None = None) -> None:
        self.config = config or BM25Config()
        self._documents: dict[str, str] = {}
        self._doc_lengths: dict[str, int] = {}
        self._term_freqs: dict[str, dict[str, int]] = {}
        self._doc_freqs: dict[str, int] = Counter()
        self._avg_doc_length: float = 0.0
        self._is_built: bool = False

    _CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]+")

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        tokens: list[str] = []
        last = 0
        for match in self._CJK_RE.finditer(text):
            if match.start() > last:
                segment = text[last : match.start()]
                word_tokens = re.findall(r"\b\w+\b|[^\w\s]", segment)
                tokens.extend(t for t in word_tokens if len(t) > 1 or t.isalpha())
            cjk_chars = [c for c in match.group() if c.strip()]
            tokens.extend(cjk_chars)
            for i in range(len(cjk_chars) - 1):
                tokens.append(cjk_chars[i] + cjk_chars[i + 1])
            last = match.end()
        if last < len(text):
            segment = text[last:]
            word_tokens = re.findall(r"\b\w+\b|[^\w\s]", segment)
            tokens.extend(t for t in word_tokens if len(t) > 1 or t.isalpha())
        return tokens

    def add_document(self, doc_id: str, text: str) -> None:
        if len(self._documents) >= self.config.max_documents:
            return
        self._documents[doc_id] = text
        tokens = self._tokenize(text)
        self._doc_lengths[doc_id] = len(tokens)
        self._term_freqs[doc_id] = Counter(tokens)
        self._is_built = False

    def remove_document(self, doc_id: str) -> None:
        self._documents.pop(doc_id, None)
        self._doc_lengths.pop(doc_id, None)
        self._term_freqs.pop(doc_id, None)
        self._is_built = False

    def build(self) -> None:
        if self._is_built:
            return
        self._doc_freqs = Counter()
        for doc_id, term_freq in self._term_freqs.items():
            for term in term_freq:
                self._doc_freqs[term] += 1
        total_length = sum(self._doc_lengths.values())
        self._avg_doc_length = (
            total_length / len(self._doc_lengths) if self._doc_lengths else 1.0
        )
        self._is_built = True

    def _score_document(self, query_tokens: list[str], doc_id: str) -> float:
        if not self._is_built:
            self.build()
        if doc_id not in self._term_freqs:
            return 0.0
        term_freq = self._term_freqs[doc_id]
        doc_length = self._doc_lengths.get(doc_id, 0)
        score = 0.0
        N = len(self._documents)
        for term in query_tokens:
            if term not in term_freq:
                continue
            tf = term_freq[term]
            df = self._doc_freqs.get(term, 0)
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
            numerator = tf * (self.config.k1 + 1)
            denominator = tf + self.config.k1 * (
                1 - self.config.b + self.config.b * doc_length / self._avg_doc_length
            )
            score += idf * numerator / denominator
        return score

    def search(
        self,
        query: str,
        top_k: int,
        allowed_doc_ids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        if not self._documents:
            return []
        if not self._is_built:
            self.build()
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        scores = [
            (doc_id, self._score_document(query_tokens, doc_id))
            for doc_id in self._documents
            if allowed_doc_ids is None or doc_id in allowed_doc_ids
        ]
        scores = [(doc_id, score) for doc_id, score in scores if score > 0]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def clear(self) -> None:
        self._documents.clear()
        self._doc_lengths.clear()
        self._term_freqs.clear()
        self._doc_freqs = Counter()
        self._avg_doc_length = 0.0
        self._is_built = False
