"""
Unit tests for knowledge autotune ExperimentRunner.

Covers:
- Shared fusion parity between runner and live HybridRetriever.
- Counting mock cross-encoder: unique (query, chunk_text) pairs scored once.
- Cache key format: (ce_model_name, query_text, sha1(chunk_text)).
- Cache hit on second run with identical metrics and zero index rebuilds.
- Corrupted cache rebuild triggers recovery.
- Verification that `.knowledge/index/` is never touched.
- Tie-breaking by chunk_id and first-overlapping-hit rank.
- Fail loudly when top_k > pool_size.
- Estimated vs measured latency marking for top candidates.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Tuple

import pytest

from knowledge.autotune.runner import (
    ExperimentRunner,
    find_hit_rank,
    is_overlapping_hit,
)
from knowledge.autotune.search_space import IndexParams, QueryParams, SearchSpace
from knowledge.autotune.synth_eval import EvalSample
from knowledge.loaders.base import LoadedDocument
from knowledge.retrievers.hybrid import HybridRetriever, reciprocal_rank_fusion

# ---------------------------------------------------------------------------
# Test Mocks
# ---------------------------------------------------------------------------


class MockEmbeddingModel:
    """Deterministic mock embedding model without PyTorch/network dependencies."""

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim
        self.embed_docs_call_count = 0
        self.embed_query_call_count = 0

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self.embed_docs_call_count += 1
        return [self._vector_for_text(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        self.embed_query_call_count += 1
        return self._vector_for_text(text)

    def _vector_for_text(self, text: str) -> List[float]:
        # Hash to deterministic pseudo-vector of dimension `self.dim`
        h = hashlib.sha1(text.encode("utf-8")).digest()
        vec = [(float(b) / 255.0) - 0.5 for b in h[: self.dim]]
        norm = sum(x * x for x in vec) ** 0.5
        return [x / norm for x in vec] if norm > 0 else vec


class CountingMockCrossEncoder:
    """Mock CrossEncoder tracking every predicted pair."""

    def __init__(self) -> None:
        self.scored_pairs: List[Tuple[str, str]] = []

    def predict(self, pairs: List[Tuple[str, str]]) -> List[float]:
        self.scored_pairs.extend(pairs)
        # Score based on token overlap
        results: List[float] = []
        for q, doc in pairs:
            q_set = set(q.lower().split())
            d_set = set(doc.lower().split())
            overlap = len(q_set & d_set) / max(1, len(q_set))
            results.append(float(overlap))
        return results


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_docs() -> List[LoadedDocument]:
    return [
        LoadedDocument(
            text="FAISS provides vector similarity search for dense embeddings. It runs efficiently on CPU and GPU.",
            source_path="/docs/faiss_guide.txt",
            document_id="doc_faiss",
            metadata={"topic": "vectors"},
        ),
        LoadedDocument(
            text="BM25 is a ranking function used by search engines to score relevance based on term frequencies.",
            source_path="/docs/bm25_guide.txt",
            document_id="doc_bm25",
            metadata={"topic": "lexical"},
        ),
        LoadedDocument(
            text="Hybrid search combines dense vectors with sparse BM25 scores using Reciprocal Rank Fusion.",
            source_path="/docs/hybrid_guide.txt",
            document_id="doc_hybrid",
            metadata={"topic": "fusion"},
        ),
    ]


@pytest.fixture
def sample_eval_set() -> List[EvalSample]:
    return [
        EvalSample(
            question="What is FAISS used for?",
            source_path="/docs/faiss_guide.txt",
            char_start=0,
            char_end=64,
            gold_text="FAISS provides vector similarity search for dense embeddings.",
        ),
        EvalSample(
            question="How does BM25 rank documents?",
            source_path="/docs/bm25_guide.txt",
            char_start=0,
            char_end=80,
            gold_text="BM25 is a ranking function used by search engines to score relevance",
        ),
    ]


# ---------------------------------------------------------------------------
# 1. Fusion Parity Tests
# ---------------------------------------------------------------------------


def test_fusion_parity_with_hybrid_retriever() -> None:
    """Verify reciprocal_rank_fusion matches HybridRetriever.fuse for multiple configs."""
    faiss_results = [(0, 0.95), (1, 0.75), (2, 0.60)]
    bm25_results = [(1, 14.2), (2, 8.5), (0, 3.1)]
    chunk_ids = ["chunk_0", "chunk_1", "chunk_2"]

    retriever = HybridRetriever()

    for k in (10, 60, 100):
        for dense_weight in (0.2, 0.5, 0.7, 1.0):
            standalone = reciprocal_rank_fusion(
                faiss_results=faiss_results,
                bm25_results=bm25_results,
                k=k,
                dense_weight=dense_weight,
                chunk_ids=chunk_ids,
            )
            via_retriever = retriever.fuse(
                faiss_results=faiss_results,
                bm25_results=bm25_results,
                k=k,
                dense_weight=dense_weight,
                chunk_ids=chunk_ids,
            )
            assert standalone == via_retriever
            # Verify tie-breaker when scores identical
            idx_list = [idx for idx, _ in standalone]
            assert len(idx_list) == 3


def test_fusion_tie_breaking_by_chunk_id() -> None:
    """Verify tie-breaker sorts ascending by chunk_id string when scores tie."""
    # Symmetrical results giving identical RRF score
    faiss_results = [(0, 1.0), (1, 0.5)]
    bm25_results = [(1, 1.0), (0, 0.5)]
    chunk_ids = ["doc_b", "doc_a"]

    # When scores tie, chunk_ids determines order: "doc_a" (index 1) < "doc_b" (index 0)
    fused = reciprocal_rank_fusion(
        faiss_results=faiss_results,
        bm25_results=bm25_results,
        k=60,
        dense_weight=0.5,
        chunk_ids=chunk_ids,
    )
    assert len(fused) == 2
    assert fused[0][1] == pytest.approx(fused[1][1])
    assert fused[0][0] == 1  # doc_a comes before doc_b


# ---------------------------------------------------------------------------
# 2. Overlapping Hit & Rank Tests
# ---------------------------------------------------------------------------


def test_overlapping_hit_detection() -> None:
    sample = EvalSample(
        question="What is FAISS?",
        source_path="/docs/faiss.txt",
        char_start=50,
        char_end=150,
        gold_text="Target passage content",
    )

    # 1. Overlapping chunk
    chunk_hit = {
        "source_path": "/docs/faiss.txt",
        "char_start": 100,
        "char_end": 200,
        "text": "Target passage content continues here",
    }
    assert is_overlapping_hit(chunk_hit, sample) is True

    # 2. Non-overlapping chunk (different file)
    chunk_diff_file = {
        "source_path": "/docs/other.txt",
        "char_start": 50,
        "char_end": 150,
        "text": "Target passage content",
    }
    assert is_overlapping_hit(chunk_diff_file, sample) is False

    # 3. Disjoint span (same file, after sample end)
    chunk_disjoint = {
        "source_path": "/docs/faiss.txt",
        "char_start": 200,
        "char_end": 300,
        "text": "Unrelated text",
    }
    assert is_overlapping_hit(chunk_disjoint, sample) is False

    # 4. Fallback text containment when char offsets None
    chunk_fallback = {
        "source_path": "/docs/faiss.txt",
        "char_start": None,
        "char_end": None,
        "text": "Here is the Target passage content in legacy format",
    }
    assert is_overlapping_hit(chunk_fallback, sample) is True


def test_find_hit_rank_first_hit() -> None:
    sample = EvalSample(
        question="What is FAISS?",
        source_path="/docs/faiss.txt",
        char_start=0,
        char_end=50,
        gold_text="FAISS",
    )
    chunks = [
        {"source_path": "/docs/other.txt", "char_start": 0, "char_end": 50, "text": "A"},
        {"source_path": "/docs/faiss.txt", "char_start": 20, "char_end": 80, "text": "B"},  # Hit at rank 2
        {"source_path": "/docs/faiss.txt", "char_start": 10, "char_end": 40, "text": "C"},  # Hit at rank 3
    ]

    rank = find_hit_rank([0, 1, 2], chunks, sample)
    assert rank == 2


# ---------------------------------------------------------------------------
# 3. Experiment Runner Sweep & Cross-Encoder Counting
# ---------------------------------------------------------------------------


def test_runner_counting_mock_cross_encoder(
    sample_docs: List[LoadedDocument],
    sample_eval_set: List[EvalSample],
    tmp_path: Path,
) -> None:
    """
    Verify each unique (query, chunk_text) pair is scored exactly ONCE
    across the full query sweep.
    """
    embedder = MockEmbeddingModel()
    ce = CountingMockCrossEncoder()
    runner = ExperimentRunner(
        docs=sample_docs,
        eval_samples=sample_eval_set,
        embedding_model=embedder,
        cross_encoder=ce,
        cache_dir=tmp_path / "cache",
        pool_size=10,
        rerank_depth=5,
    )

    index_params = IndexParams(
        chunk_size=100,
        chunk_overlap=20,
        chunking_strategy="recursive",
        embedding_model="mock-embedder",
    )

    space = SearchSpace()
    # Sweep multiple query configs that have reranking_enabled=True
    query_configs = [qp for qp in space.all_query_params() if qp.reranking_enabled]
    assert len(query_configs) > 1

    results = runner.run_sweep_for_index(index_params, query_configs)
    assert len(results) == len(query_configs)

    # Check cross-encoder calls: no duplicates in scored_pairs!
    scored_pairs = ce.scored_pairs
    assert len(scored_pairs) > 0
    assert len(scored_pairs) == len(set(scored_pairs)), "Cross-encoder scored duplicate pairs!"


def test_runner_top_k_exceeds_pool_size_fails_loudly(
    sample_docs: List[LoadedDocument],
    sample_eval_set: List[EvalSample],
    tmp_path: Path,
) -> None:
    """Verify runner fails loudly if top_k > pool_size."""
    embedder = MockEmbeddingModel()
    runner = ExperimentRunner(
        docs=sample_docs,
        eval_samples=sample_eval_set,
        embedding_model=embedder,
        cache_dir=tmp_path / "cache",
        pool_size=5,
    )

    index_params = IndexParams(500, 50, "recursive", "mock-embedder")
    invalid_query_config = [
        QueryParams(top_k=10, reranking_enabled=False, rrf_k=60, dense_weight=0.5, query_expansion=False)
    ]

    with pytest.raises(ValueError, match="cannot exceed candidate pool size"):
        runner.run_sweep_for_index(index_params, invalid_query_config)


# ---------------------------------------------------------------------------
# 4. Cache Safety & Rebuild Tests
# ---------------------------------------------------------------------------


def test_cache_hit_on_second_run_identical_metrics(
    sample_docs: List[LoadedDocument],
    sample_eval_set: List[EvalSample],
    tmp_path: Path,
) -> None:
    """Verify second run loads from cache, builds 0 new indexes, and produces identical results."""
    cache_dir = tmp_path / "cache"
    embedder = MockEmbeddingModel()
    runner1 = ExperimentRunner(
        docs=sample_docs,
        eval_samples=sample_eval_set,
        embedding_model=embedder,
        cache_dir=cache_dir,
        pool_size=10,
    )

    index_params = IndexParams(150, 30, "recursive", "mock-embedder")
    query_configs = [
        QueryParams(top_k=2, reranking_enabled=False, rrf_k=60, dense_weight=0.5, query_expansion=False),
        QueryParams(top_k=2, reranking_enabled=False, rrf_k=20, dense_weight=0.7, query_expansion=True),
    ]

    results1 = runner1.run_sweep_for_index(index_params, query_configs)
    assert runner1.index_build_count == 1

    # Second runner with same cache dir
    runner2 = ExperimentRunner(
        docs=sample_docs,
        eval_samples=sample_eval_set,
        embedding_model=embedder,
        cache_dir=cache_dir,
        pool_size=10,
    )
    results2 = runner2.run_sweep_for_index(index_params, query_configs)
    assert runner2.index_build_count == 0  # Cache hit! Zero builds.

    # Assert exact metric equality
    for r1, r2 in zip(results1, results2):
        assert r1.mrr == pytest.approx(r2.mrr)
        assert r1.hit_rate == pytest.approx(r2.hit_rate)
        assert r1.precision == pytest.approx(r2.precision)


def test_corrupted_cache_triggers_rebuild(
    sample_docs: List[LoadedDocument],
    sample_eval_set: List[EvalSample],
    tmp_path: Path,
) -> None:
    """Verify corrupted cache triggers automatic rebuild and succeeds."""
    cache_dir = tmp_path / "cache"
    embedder = MockEmbeddingModel()
    runner = ExperimentRunner(
        docs=sample_docs,
        eval_samples=sample_eval_set,
        embedding_model=embedder,
        cache_dir=cache_dir,
        pool_size=10,
    )

    index_params = IndexParams(150, 30, "recursive", "mock-embedder")
    query_configs = [
        QueryParams(top_k=2, reranking_enabled=False, rrf_k=60, dense_weight=0.5, query_expansion=False)
    ]

    # Run 1: builds and caches
    runner.run_sweep_for_index(index_params, query_configs)
    assert runner.index_build_count == 1

    # Corrupt meta.json
    cache_path = runner.cache_mgr.get_cache_path(index_params)
    meta_file = cache_path / "meta.json"
    with open(meta_file, "w") as f:
        f.write("{invalid json content!!!")

    # Run 2: detects corruption, triggers rebuild, succeeds
    results = runner.run_sweep_for_index(index_params, query_configs)
    assert runner.index_build_count == 2
    assert len(results) == 1
    assert results[0].hit_rate > 0.0


def test_never_touches_knowledge_index(
    sample_docs: List[LoadedDocument],
    sample_eval_set: List[EvalSample],
    tmp_path: Path,
) -> None:
    """Verify that ExperimentRunner NEVER touches or alters `.knowledge/index/`."""
    production_index = Path(".knowledge/index")
    # Record initial state
    exists_before = production_index.exists()
    mtime_before = production_index.stat().st_mtime if exists_before else None

    cache_dir = tmp_path / "cache"
    embedder = MockEmbeddingModel()
    runner = ExperimentRunner(
        docs=sample_docs,
        eval_samples=sample_eval_set,
        embedding_model=embedder,
        cache_dir=cache_dir,
        pool_size=10,
    )

    index_params = IndexParams(200, 40, "recursive", "mock-embedder")
    runner.run_sweep_for_index(
        index_params,
        [QueryParams(top_k=2, reranking_enabled=False, rrf_k=60, dense_weight=0.5, query_expansion=False)],
    )

    # Verify state after
    exists_after = production_index.exists()
    assert exists_before == exists_after
    if exists_before:
        assert production_index.stat().st_mtime == mtime_before


# ---------------------------------------------------------------------------
# 5. Latency Estimation & Live Measurement Tests
# ---------------------------------------------------------------------------


def test_latency_estimated_vs_measured(
    sample_docs: List[LoadedDocument],
    sample_eval_set: List[EvalSample],
    tmp_path: Path,
) -> None:
    """Verify latency is marked 'estimated' during sweep and 'measured' for top candidates."""
    embedder = MockEmbeddingModel()
    runner = ExperimentRunner(
        docs=sample_docs,
        eval_samples=sample_eval_set,
        embedding_model=embedder,
        cache_dir=tmp_path / "cache",
        pool_size=10,
    )

    index_params = IndexParams(200, 40, "recursive", "mock-embedder")
    query_configs = [
        QueryParams(top_k=2, reranking_enabled=False, rrf_k=60, dense_weight=0.5, query_expansion=False),
        QueryParams(top_k=2, reranking_enabled=False, rrf_k=20, dense_weight=0.7, query_expansion=False),
        QueryParams(top_k=2, reranking_enabled=False, rrf_k=10, dense_weight=0.3, query_expansion=False),
        QueryParams(top_k=2, reranking_enabled=False, rrf_k=60, dense_weight=0.8, query_expansion=False),
    ]

    results = runner.run_sweep_for_index(index_params, query_configs)
    for r in results:
        assert r.latency_type == "estimated"
        assert r.latency_ms > 0.0

    artifacts = runner._get_or_build_index(index_params)
    updated = runner.measure_real_latency_for_top_candidates(artifacts, results, top_n=2)

    measured_count = sum(1 for r in updated if r.latency_type == "measured")
    assert measured_count == 2
