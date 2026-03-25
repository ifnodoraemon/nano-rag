import pytest

from app.retrieval.hybrid_fusion import (
    HybridSearchConfig,
    reciprocal_rank_fusion,
    weighted_score_fusion,
)


def test_hybrid_search_config_defaults() -> None:
    config = HybridSearchConfig.from_env()
    assert config.vector_weight == 0.7
    assert config.bm25_weight == 0.3
    assert config.rrf_k == 60


def test_hybrid_search_config_invalid_weights() -> None:
    with pytest.raises(ValueError):
        HybridSearchConfig(vector_weight=1.5, bm25_weight=0.3)
    with pytest.raises(ValueError):
        HybridSearchConfig(vector_weight=0.7, bm25_weight=-0.1)


def test_hybrid_search_config_invalid_rrf_k() -> None:
    with pytest.raises(ValueError):
        HybridSearchConfig(vector_weight=0.7, bm25_weight=0.3, rrf_k=0)


def test_reciprocal_rank_fusion() -> None:
    config = HybridSearchConfig(vector_weight=0.7, bm25_weight=0.3, rrf_k=60)
    vector_results = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
    bm25_results = [("doc2", 1.0), ("doc4", 0.9), ("doc1", 0.8)]
    fused = reciprocal_rank_fusion(vector_results, bm25_results, config)
    assert len(fused) == 4
    doc_ids = {doc_id for doc_id, _ in fused}
    assert doc_ids == {"doc1", "doc2", "doc3", "doc4"}


def test_weighted_score_fusion() -> None:
    config = HybridSearchConfig(vector_weight=0.7, bm25_weight=0.3, rrf_k=60)
    vector_results = [("doc1", 0.9), ("doc2", 0.8)]
    bm25_results = [("doc2", 1.0), ("doc3", 0.9)]
    fused = weighted_score_fusion(vector_results, bm25_results, config)
    assert len(fused) == 3
    doc_ids = {doc_id for doc_id, _ in fused}
    assert doc_ids == {"doc1", "doc2", "doc3"}


def test_reciprocal_rank_fusion_empty_inputs() -> None:
    config = HybridSearchConfig()
    assert reciprocal_rank_fusion([], [], config) == []
    assert len(reciprocal_rank_fusion([("doc1", 0.9)], [], config)) == 1
    assert len(reciprocal_rank_fusion([], [("doc1", 0.9)], config)) == 1
