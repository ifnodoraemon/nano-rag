from __future__ import annotations

import math
import threading
from collections import Counter
from dataclasses import dataclass

from app.core.exceptions import RetrievalError
from app.retrieval.tokenizer import DEFAULT_TOKENIZER, tokenize


@dataclass
class BM25Config:
    k1: float = 1.5
    b: float = 0.75
    max_documents: int = 50000
    # See app/retrieval/tokenizer.py: the CJK segmentation strategy decides
    # what a "term" is, which is the single biggest relevance lever for a
    # lexical index over Chinese text.
    tokenizer: str = DEFAULT_TOKENIZER
    # How much a term in the document TITLE counts relative to the same term in
    # the body. 0 disables the title field entirely (the historical behaviour).
    # A wiki page's title is a human-written summary of the whole document, so it
    # carries far more signal per token than body text; on enterprise corpora the
    # query terms often appear only there.
    title_weight: float = 0.0
    # Scoring formula. "bm25" is the classic Okapi BM25. "bm25x" implements the
    # entropy-weighted similarity extension from Li et al., "BMX: Entropy-weighted
    # Similarity and Semantic-enhanced Lexical Search" (arXiv:2408.06643), which
    # adds two lexical-only terms: an entropy weight per query term and an
    # explicit query/document overlap ratio. Both are computable from the inverted
    # index, so no model is involved.
    variant: str = "bm25"


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
        # Title field, kept separate so it can be scored with its own weight and
        # its own length normalisation.
        self._title_postings: dict[str, dict[str, int]] = {}
        self._title_lengths: dict[str, int] = {}
        self._total_title_length = 0
        self._total_length = 0
        self._lock = threading.RLock()

    def _term_entropy(self, postings: dict[str, int], n_docs: int) -> float:
        """Normalised entropy of a term's tf distribution across the documents.

        A term spread evenly across many documents concentrates little
        information; a term that is rare or highly skewed is informative. The
        paper maps tf through a sigmoid first so that very high frequencies do not
        dominate the distribution.
        """
        if not postings or n_docs <= 0:
            return 0.0
        probabilities: list[float] = []
        for tf in postings.values():
            # sigmoid(tf): tf 0..5+ maps smoothly to 0.5..~0.99
            try:
                p = 1.0 / (1.0 + math.exp(-float(tf)))
            except OverflowError:  # pragma: no cover - tf would have to be absurd
                p = 1.0
            probabilities.append(p)
        total = sum(probabilities)
        if total <= 0:
            return 0.0
        entropy = 0.0
        for p in probabilities:
            share = p / total
            if share > 0:
                entropy -= share * math.log(share)
        return entropy

    def _tokenize(self, text: str) -> list[str]:
        return tokenize(text, self.config.tokenizer)

    @property
    def document_count(self) -> int:
        with self._lock:
            return len(self._doc_lengths)

    def add_document(self, doc_id: str, text: str, title: str = "") -> None:
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

            title_tokens = self._tokenize(title) if title else []
            self._title_lengths[doc_id] = len(title_tokens)
            self._total_title_length += len(title_tokens)
            for term, tf in Counter(title_tokens).items():
                title_postings = self._title_postings.get(term)
                if title_postings is None:
                    title_postings = {}
                    self._title_postings[term] = title_postings
                title_postings[doc_id] = tf

    def remove_document(self, doc_id: str) -> None:
        with self._lock:
            self._remove_locked(doc_id)

    def _remove_locked(self, doc_id: str) -> None:
        length = self._doc_lengths.pop(doc_id, None)
        if length is None:
            return
        self._total_length -= length
        self._total_title_length -= self._title_lengths.pop(doc_id, 0)
        for postings, target in (
            (self._postings, self._postings),
            (self._title_postings, self._title_postings),
        ):
            empty_terms: list[str] = []
            for term, term_postings in postings.items():
                if term_postings.pop(doc_id, None) is not None and not term_postings:
                    empty_terms.append(term)
            for term in empty_terms:
                del target[term]

    def document_frequencies(self) -> tuple[dict[str, int], int]:
        """Term -> document frequency, and the corpus document count.

        Exposed so pseudo-relevance feedback can drop terms that occur in most documents
        (they discriminate nothing); computed from the existing postings, no extra index.
        """
        with self._lock:
            return (
                {term: len(postings) for term, postings in self._postings.items()},
                len(self._doc_lengths),
            )

    def search(
        self,
        query: str,
        top_k: int,
        allowed_doc_ids: set[str] | frozenset[str] | None = None,
    ) -> list[tuple[str, float]]:
        return self.search_weighted(
            query, top_k, allowed_doc_ids=allowed_doc_ids
        )

    def search_weighted(
        self,
        query: str,
        top_k: int,
        allowed_doc_ids: set[str] | frozenset[str] | None = None,
        *,
        term_weights: dict[str, float] | None = None,
    ) -> list[tuple[str, float]]:
        """BM25 search where query terms can carry weights.

        Weighted queries are what make pseudo-relevance feedback possible without a
        second model: HyDE-generated text is expanded into a weighted term set
        (Rocchio) instead of being appended to the query string, which measurably adds
        noise rather than signal - SIGIR'26 measured Recall@20 going 28.8 (BM25) ->
        29.5 (naive concatenation) -> 34.7 (Rocchio feedback) on 14 datasets.

        ``term_weights`` maps a term to a non-negative multiplier applied to that
        term's BM25 contribution; absent terms weigh 1.0.
        """
        with self._lock:
            if not self._doc_lengths:
                return []
            query_tokens = self._tokenize(query)
            if term_weights:
                # Terms introduced by feedback are already tokenised by their producer;
                # only the original query is tokenised here.
                merged: dict[str, float] = {}
                for token in query_tokens:
                    merged[token] = merged.get(token, 0.0) + 1.0
                for term, weight in term_weights.items():
                    if weight > 0:
                        merged[term] = merged.get(term, 0.0) + weight
                weights = merged
            else:
                weights = {token: 1.0 for token in query_tokens}
            if not weights:
                return []
            query_tokens = list(weights)
            n_docs = len(self._doc_lengths)
            avg_length = self._total_length / n_docs if n_docs else 1.0
            use_title = self.config.title_weight > 0.0 and bool(self._title_postings)
            avg_title_length = (
                self._total_title_length / n_docs if n_docs and use_title else 1.0
            )

            # Gather candidates from the posting lists of the query terms
            # only — never a full scan over all documents.
            candidates: set[str] = set()
            term_freqs: dict[str, dict[str, int]] = {}
            title_freqs: dict[str, dict[str, int]] = {}
            for term in set(query_tokens):
                postings = self._postings.get(term)
                if postings:
                    term_freqs[term] = postings
                    if allowed_doc_ids is None:
                        candidates.update(postings)
                    else:
                        candidates.update(postings.keys() & allowed_doc_ids)
                if not use_title:
                    continue
                # A term that appears ONLY in a title still makes the document a
                # candidate: skipping it would hide documents whose body is terse
                # but whose title states exactly what the query asks.
                title_postings = self._title_postings.get(term)
                if title_postings:
                    title_freqs[term] = title_postings
                    if allowed_doc_ids is None:
                        candidates.update(title_postings)
                    else:
                        candidates.update(title_postings.keys() & allowed_doc_ids)
            if not candidates:
                return []

            use_bm25x = self.config.variant == "bm25x"
            bm25x_alpha = 0.0
            bm25x_beta = 0.0
            bm25x_mean_entropy = 0.0
            bm25x_normalised: dict[str, float] = {}
            if use_bm25x:
                # alpha / beta follow the paper's data-derived defaults; they depend
                # only on corpus statistics, so no tuning is required to try it.
                alpha = max(0.5, min(1.5, avg_length / 100.0))
                beta = 1.0 / math.log(1.0 + n_docs) if n_docs > 0 else 0.0
                entropies = {
                    term: self._term_entropy(postings, n_docs)
                    for term, postings in term_freqs.items()
                }
                peak = max(entropies.values()) if entropies else 0.0
                normalised = {
                    term: (value / peak if peak > 0 else 0.0)
                    for term, value in entropies.items()
                }
                mean_entropy = (
                    sum(normalised.values()) / len(normalised) if normalised else 0.0
                )
                bm25x_alpha = alpha
                bm25x_beta = beta
                bm25x_mean_entropy = mean_entropy
                bm25x_normalised = normalised

            scores: list[tuple[str, float]] = []
            for doc_id in candidates:
                doc_length = self._doc_lengths.get(doc_id)
                if not doc_length:
                    continue
                score = 0.0
                for term in query_tokens:
                    term_weight = weights.get(term, 1.0)
                    postings = term_freqs.get(term)
                    title_postings = title_freqs.get(term)
                    body_tf = postings.get(doc_id, 0) if postings else 0
                    title_tf = title_postings.get(doc_id, 0) if title_postings else 0
                    if not body_tf and not title_tf:
                        continue
                    # IDF comes from the body field's document frequency so that a
                    # rare term is rare for the same reason whether or not it also
                    # appears in a title; title-only terms fall back to their own df.
                    df = len(postings) if postings else len(title_postings or {})
                    idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
                    term_score = 0.0
                    if use_bm25x:
                        numerator = body_tf * (bm25x_alpha + 1.0)
                        denominator = (
                            body_tf
                            + bm25x_alpha
                            * (doc_length / avg_length if avg_length else 1.0)
                            + bm25x_alpha * bm25x_mean_entropy
                        )
                        term_score += idf * numerator / denominator
                    elif body_tf:
                        numerator = body_tf * (self.config.k1 + 1)
                        denominator = body_tf + self.config.k1 * (
                            1 - self.config.b + self.config.b * doc_length / avg_length
                        )
                        term_score += idf * numerator / denominator
                    if title_tf:
                        title_length = self._title_lengths.get(doc_id, 0) or 1
                        numerator = title_tf * (self.config.k1 + 1)
                        denominator = title_tf + self.config.k1 * (
                            1 - self.config.b
                            + self.config.b * title_length / avg_title_length
                        )
                        term_score += (
                            self.config.title_weight * idf * numerator / denominator
                        )
                    score += term_weight * term_score
                if use_bm25x:
                    # The query/document overlap ratio: how many DISTINCT query
                    # terms this document covers. Classic BM25 has no such term, so
                    # a document matching one query term three times can outrank one
                    # matching five different terms - the failure mode that hurts
                    # long natural-language queries.
                    matched = sum(
                        1
                        for term in set(query_tokens)
                        if (term_freqs.get(term) or {}).get(doc_id)
                        or (title_freqs.get(term) or {}).get(doc_id)
                    )
                    overlap = matched / len(set(query_tokens)) if query_tokens else 0.0
                    entropy_bonus = 0.0
                    for term in set(query_tokens):
                        weight = bm25x_normalised.get(term, 0.0)
                        if weight:
                            entropy_bonus += bm25x_beta * weight * overlap
                    score += entropy_bonus
                if score > 0:
                    scores.append((doc_id, score))
            scores.sort(key=lambda item: (-item[1], item[0]))
            return scores[:top_k]

    def clear(self) -> None:
        with self._lock:
            self._doc_lengths.clear()
            self._postings.clear()
            self._total_length = 0
