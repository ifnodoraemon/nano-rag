from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class HybridSearchConfig:
    vector_weight: float = 0.7
    bm25_weight: float = 0.3
    rrf_k: int = 60

    @classmethod
    def from_env(cls) -> "HybridSearchConfig":
        return cls(
            vector_weight=float(os.getenv("RAG_HYBRID_VECTOR_WEIGHT", "0.7")),
            bm25_weight=float(os.getenv("RAG_HYBRID_BM25_WEIGHT", "0.3")),
            rrf_k=int(os.getenv("RAG_HYBRID_RRF_K", "60")),
        )

    def __post_init__(self) -> None:
        if not (0 <= self.vector_weight <= 1):
            raise ValueError("vector_weight must be between 0 and 1")
        if not (0 <= self.bm25_weight <= 1):
            raise ValueError("bm25_weight must be between 0 and 1")
        if self.rrf_k < 1:
            raise ValueError("rrf_k must be positive")


def reciprocal_rank_fusion(
    vector_results: list[tuple[str, float]],
    bm25_results: list[tuple[str, float]],
    config: HybridSearchConfig,
) -> list[tuple[str, float]]:
    rrf_scores: dict[str, float] = {}
    for rank, (doc_id, _) in enumerate(vector_results, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + config.vector_weight / (
            config.rrf_k + rank
        )
    for rank, (doc_id, _) in enumerate(bm25_results, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + config.bm25_weight / (
            config.rrf_k + rank
        )
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results


def weighted_score_fusion(
    vector_results: list[tuple[str, float]],
    bm25_results: list[tuple[str, float]],
    config: HybridSearchConfig,
) -> list[tuple[str, float]]:
    vector_dict = dict(vector_results)
    bm25_dict = dict(bm25_results)
    all_doc_ids = set(vector_dict) | set(bm25_dict)
    max_vector = max(vector_dict.values()) if vector_dict else 1.0
    max_bm25 = max(bm25_dict.values()) if bm25_dict else 1.0
    if max_vector == 0:
        max_vector = 1.0
    if max_bm25 == 0:
        max_bm25 = 1.0
    fused_scores: list[tuple[str, float]] = []
    for doc_id in all_doc_ids:
        v_score = vector_dict.get(doc_id, 0.0) / max_vector
        b_score = bm25_dict.get(doc_id, 0.0) / max_bm25
        combined = config.vector_weight * v_score + config.bm25_weight * b_score
        fused_scores.append((doc_id, combined))
    fused_scores.sort(key=lambda x: x[1], reverse=True)
    return fused_scores
