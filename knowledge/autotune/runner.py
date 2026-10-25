"""
Experiment runner for knowledge autotune.

Executes a two-tier evaluation sweep:
1. Index-level build tier (chunking + embedding + FAISS + BM25), cached on disk
   at `.knowledge/autotune/cache/<cache_key>/`.
2. Query-level offline sweep tier (dense_weight, rrf_k, reranking_enabled, top_k)
   computed offline against cached top-50 candidate pools without rerunning retrieval.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Set, Tuple

from knowledge.autotune.search_space import CandidateConfig, IndexParams, QueryParams
from knowledge.autotune.synth_eval import EvalSample
from knowledge.chunkers.base import DocumentChunk
from knowledge.chunkers.fixed_size import FixedSizeChunker
from knowledge.chunkers.recursive import RecursiveChunker
from knowledge.chunkers.sentence import SentenceChunker
from knowledge.loaders.base import LoadedDocument
from knowledge.retrievers.bm25 import BM25Retriever
from knowledge.retrievers.hybrid import reciprocal_rank_fusion
from knowledge.vectorstores.faiss_store import FAISSStore

logger = logging.getLogger(__name__)

EXTRACTOR_HASH = "v1_char_offset_extractor"


# ---------------------------------------------------------------------------
# Protocols (Injectable interfaces for testing without real models or network)
# ---------------------------------------------------------------------------


class EmbeddingModelInterface(Protocol):
    """Protocol satisfied by BaseEmbeddingModel, SentenceTransformer, or mocks."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        ...

    def embed_query(self, text: str) -> List[float]:
        ...


class CrossEncoderInterface(Protocol):
    """Protocol satisfied by CrossEncoder or counting mock."""

    def predict(self, pairs: List[Tuple[str, str]]) -> Any:
        ...


class QueryExpanderInterface(Protocol):
    """Protocol for optional query expansion."""

    def expand(self, query: str) -> str:
        ...


class DefaultQueryExpander:
    """Deterministic default query expander appending keywords."""

    def expand(self, query: str) -> str:
        return f"{query} details and context"


# ---------------------------------------------------------------------------
# Result Data Structures
# ---------------------------------------------------------------------------


@dataclass
class TrialResult:
    """Result of evaluating a single CandidateConfig across the eval set."""

    index_params: IndexParams
    query_params: QueryParams
    mrr: float
    hit_rate: float
    precision: float
    latency_ms: float
    latency_type: str = "estimated"  # "estimated" or "measured"
    stage_latencies: Dict[str, float] = field(default_factory=dict)
    eval_sample_results: List[Dict[str, Any]] = field(default_factory=list)
    pool_size: int = 50
    truncated: bool = False

    @property
    def candidate(self) -> CandidateConfig:
        return CandidateConfig(index_params=self.index_params, query_params=self.query_params)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": asdict(self.index_params),
            "query": asdict(self.query_params),
            "mrr": self.mrr,
            "hit_rate": self.hit_rate,
            "precision": self.precision,
            "latency_ms": self.latency_ms,
            "latency_type": self.latency_type,
            "stage_latencies": self.stage_latencies,
            "pool_size": self.pool_size,
            "truncated": self.truncated,
        }


# ---------------------------------------------------------------------------
# Pure Helper Functions
# ---------------------------------------------------------------------------


def compute_params_hash(params: IndexParams) -> str:
    """Compute a canonical SHA-1 hash for IndexParams."""
    canonical = json.dumps(
        {
            "chunk_overlap": params.chunk_overlap,
            "chunk_size": params.chunk_size,
            "chunking_strategy": params.chunking_strategy,
            "embedding_model": params.embedding_model,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


def compute_doc_set_hash(docs: Sequence[LoadedDocument]) -> str:
    """Compute a deterministic SHA-1 hash over loaded document texts and IDs."""
    hasher = hashlib.sha1()
    for doc in sorted(docs, key=lambda d: d.document_id):
        hasher.update(doc.document_id.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(doc.text.strip().encode("utf-8"))
        hasher.update(b"\x00")
    return hasher.hexdigest()


def is_hit(chunk: Dict[str, Any], sample: EvalSample) -> bool:
    """
    Return True if the retrieved chunk matches the gold eval sample.

    Criterion:
    The chunk must originate from the same source file, AND:
      (chunk covers >= 50% of gold span) OR (>= 50% of chunk lies inside gold span).

    Falls back to text containment if character offsets are missing.
    """
    c_source = str(chunk.get("source_path", ""))
    s_source = str(sample.source_path)
    if c_source and s_source and c_source != s_source:
        return False

    c_start = chunk.get("char_start")
    c_end = chunk.get("char_end")
    s_start = sample.char_start
    s_end = sample.char_end

    if (
        c_start is not None
        and c_end is not None
        and s_start is not None
        and s_end is not None
    ):
        chunk_len = max(1, c_end - c_start)
        span_len = max(1, s_end - s_start)
        overlap_start = max(c_start, s_start)
        overlap_end = min(c_end, s_end)
        overlap_len = max(0, overlap_end - overlap_start)

        covers_gold = bool((overlap_len / span_len) >= 0.5)
        chunk_in_gold = bool((overlap_len / chunk_len) >= 0.5)
        return covers_gold or chunk_in_gold

    # Fallback when offsets are absent
    c_text = chunk.get("text", "")
    s_text = sample.gold_text
    if s_text and c_text:
        return bool(s_text in c_text or c_text in s_text)
    return False


is_overlapping_hit = is_hit


def find_hit_rank(
    ranked_indices: Sequence[int],
    chunks: Sequence[Dict[str, Any]],
    sample: EvalSample,
) -> Optional[int]:
    """Return 1-indexed hit rank of the first overlapping chunk, or None."""
    for rank, idx in enumerate(ranked_indices, start=1):
        if 0 <= idx < len(chunks) and is_overlapping_hit(chunks[idx], sample):
            return rank
    return None


def get_chunker_instance(strategy: str) -> Any:
    """Instantiate a chunker corresponding to chunking_strategy."""
    strat = strategy.lower().strip()
    if strat == "recursive":
        return RecursiveChunker()
    elif strat in ("fixed", "fixed_size"):
        return FixedSizeChunker()
    elif strat == "sentence":
        return SentenceChunker()
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy!r}")


# ---------------------------------------------------------------------------
# Index Artifacts & Cache
# ---------------------------------------------------------------------------


@dataclass
class IndexArtifacts:
    """In-memory representation of an index built for an IndexParams setting."""

    index_params: IndexParams
    chunks: List[Dict[str, Any]]
    chunk_ids: List[str]
    faiss_store: FAISSStore
    bm25: BM25Retriever
    stage_latencies: Dict[str, float]


class IndexCacheManager:
    """Manages on-disk caching of built indexes with corruption detection."""

    def __init__(self, cache_root: Path) -> None:
        self.cache_root = cache_root
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def get_cache_path(self, index_params: IndexParams) -> Path:
        return self.cache_root / index_params.cache_key()

    def load(
        self,
        index_params: IndexParams,
        expected_doc_hash: str,
    ) -> Optional[IndexArtifacts]:
        """Load index from cache if valid; return None on mismatch or corruption."""
        cache_dir = self.get_cache_path(index_params)
        meta_file = cache_dir / "meta.json"
        chunks_file = cache_dir / "chunks.json"
        faiss_file = cache_dir / "index.faiss"

        if not (meta_file.exists() and chunks_file.exists() and faiss_file.exists()):
            return None

        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)

            if meta.get("doc_set_hash") != expected_doc_hash:
                return None
            if meta.get("extractor_hash") != EXTRACTOR_HASH:
                return None
            if meta.get("embedding_model") != index_params.embedding_model:
                return None
            if meta.get("params_hash") != compute_params_hash(index_params):
                return None

            with open(chunks_file, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            faiss_store = FAISSStore()
            faiss_store.load(str(faiss_file))

            bm25 = BM25Retriever()
            bm25.build([c.get("text", "") for c in chunks])

            chunk_ids = [c.get("chunk_id", str(i)) for i, c in enumerate(chunks)]
            stage_latencies = meta.get("stage_latencies", {})

            return IndexArtifacts(
                index_params=index_params,
                chunks=chunks,
                chunk_ids=chunk_ids,
                faiss_store=faiss_store,
                bm25=bm25,
                stage_latencies=stage_latencies,
            )
        except Exception:
            # Corruption detected: return None to trigger rebuild
            return None

    def save(
        self,
        artifacts: IndexArtifacts,
        doc_set_hash: str,
    ) -> None:
        """
        Atomically save index artifacts to cache.
        Builds in a temp dir then renames atomically to avoid partial writes.
        """
        dest_dir = self.get_cache_path(artifacts.index_params)
        tmp_dir = self.cache_root / f".tmp_{artifacts.index_params.cache_key()}_{uuid.uuid4().hex[:8]}"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. chunks.json
            with open(tmp_dir / "chunks.json", "w", encoding="utf-8") as f:
                json.dump(artifacts.chunks, f, indent=2)

            # 2. index.faiss
            artifacts.faiss_store.save(str(tmp_dir / "index.faiss"))

            # 3. meta.json
            meta = {
                "cache_key": artifacts.index_params.cache_key(),
                "doc_set_hash": doc_set_hash,
                "extractor_hash": EXTRACTOR_HASH,
                "embedding_model": artifacts.index_params.embedding_model,
                "params_hash": compute_params_hash(artifacts.index_params),
                "params": asdict(artifacts.index_params),
                "stage_latencies": artifacts.stage_latencies,
            }
            with open(tmp_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            # Atomic replace
            if dest_dir.exists():
                shutil.rmtree(dest_dir)
            os.replace(str(tmp_dir), str(dest_dir))
        except Exception:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            raise


# ---------------------------------------------------------------------------
# Experiment Runner
# ---------------------------------------------------------------------------


class ExperimentRunner:
    """
    Two-tier experiment runner with candidate pooling and offline query sweeps.
    """

    def __init__(
        self,
        docs: Sequence[LoadedDocument],
        eval_samples: Sequence[EvalSample],
        embedding_model: EmbeddingModelInterface,
        cross_encoder: Optional[CrossEncoderInterface] = None,
        query_expander: Optional[QueryExpanderInterface] = None,
        cache_dir: Optional[Path] = None,
        pool_size: int = 50,
        rerank_depth: int = 20,
        max_union_size: int = 100,
        ce_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self.docs = list(docs)
        self.eval_samples = list(eval_samples)
        self.embedding_model = embedding_model
        self.cross_encoder = cross_encoder
        self.query_expander = query_expander or DefaultQueryExpander()
        self.pool_size = pool_size
        self.rerank_depth = rerank_depth
        self.max_union_size = max_union_size
        self.ce_model_name = ce_model_name

        self.cache_dir = cache_dir or Path(".knowledge/autotune/cache")
        self.cache_mgr = IndexCacheManager(self.cache_dir)
        self.doc_set_hash = compute_doc_set_hash(self.docs)

        self.ce_cache_path = self.cache_dir / "ce_cache.json"
        # Caches across runs
        # Key: (ce_model_name, query_text, sha1(chunk_text)) -> score
        self._ce_cache: Dict[Tuple[str, str, str], float] = {}
        self._load_ce_cache()
        # Key: original_query -> expanded_query
        self._expansion_cache: Dict[str, str] = {}
        # Telemetry for tests
        self.index_build_count = 0
        self.has_truncated = False

    def _load_ce_cache(self) -> None:
        """Load persisted cross-encoder scores from disk for resume support."""
        if self.ce_cache_path.exists():
            try:
                with open(self.ce_cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for k, score in data.items():
                    parts = k.split("\t", 2)
                    if len(parts) == 3:
                        self._ce_cache[(parts[0], parts[1], parts[2])] = float(score)
            except Exception:
                pass

    def _save_ce_cache(self) -> None:
        """Atomically persist cross-encoder cache to disk."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            data = {
                f"{k[0]}\t{k[1]}\t{k[2]}": v
                for k, v in self._ce_cache.items()
            }
            tmp_file = self.cache_dir / f".tmp_ce_cache_{uuid.uuid4().hex[:8]}.json"
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(str(tmp_file), str(self.ce_cache_path))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_sweep_for_index(
        self,
        index_params: IndexParams,
        query_configs: Sequence[QueryParams],
    ) -> List[TrialResult]:
        """
        Build (or load) index for index_params, retrieve candidate pools,
        pre-score cross-encoder union, and sweep query-time configs offline.
        """
        # Validate that no query config requests top_k > pool_size
        for qp in query_configs:
            if qp.top_k > self.pool_size:
                raise ValueError(
                    f"top_k ({qp.top_k}) cannot exceed candidate pool size ({self.pool_size})"
                )

        # 1. Build or load index artifacts
        artifacts = self._get_or_build_index(index_params)

        # 2. Retrieve top-50 pools per (index, query_expansion) setting
        pools_by_expansion = self._build_candidate_pools(artifacts)

        # 3. Pre-score cross-encoder union across all query configs
        self._prescore_cross_encoder_union(artifacts, pools_by_expansion, query_configs)

        # 4. Sweep all query configs offline
        results: List[TrialResult] = []
        for qp in query_configs:
            res = self._evaluate_query_config_offline(artifacts, pools_by_expansion, qp)
            results.append(res)

        return results

    def measure_real_latency_for_top_candidates(
        self,
        artifacts: IndexArtifacts,
        results: List[TrialResult],
        top_n: int = 3,
    ) -> List[TrialResult]:
        """
        Measure actual live pipeline latency for top N candidates + winner,
        updating latency_ms and setting latency_type = 'measured'.
        """
        if not results:
            return results

        # Rank by composite MRR
        sorted_indices = sorted(range(len(results)), key=lambda i: results[i].mrr, reverse=True)
        top_indices = sorted_indices[:top_n]

        for idx in top_indices:
            trial = results[idx]
            qp = trial.query_params
            measured_ms = self._measure_live_latency(artifacts, qp)
            trial.latency_ms = measured_ms
            trial.latency_type = "measured"

        return results

    # ------------------------------------------------------------------
    # Index Building & Retrieval
    # ------------------------------------------------------------------

    def _get_or_build_index(self, index_params: IndexParams) -> IndexArtifacts:
        """Load index from cache or build from scratch and save to cache."""
        cached = self.cache_mgr.load(index_params, self.doc_set_hash)
        if cached is not None:
            return cached

        self.index_build_count += 1
        artifacts = self._build_index_from_scratch(index_params)
        self.cache_mgr.save(artifacts, self.doc_set_hash)
        return artifacts

    def _build_index_from_scratch(self, index_params: IndexParams) -> IndexArtifacts:
        """Chunk documents, embed, and build FAISS + BM25 indices."""
        chunker = get_chunker_instance(index_params.chunking_strategy)
        chunks_objs: List[DocumentChunk] = chunker.chunk(
            self.docs,
            chunk_size=index_params.chunk_size,
            chunk_overlap=index_params.chunk_overlap,
        )

        chunks: List[Dict[str, Any]] = [c.model_dump() for c in chunks_objs]
        chunk_ids: List[str] = [c["chunk_id"] for c in chunks]
        texts = [c["text"] for c in chunks]

        # Embed texts
        t0 = time.perf_counter()
        embeddings = self.embedding_model.embed_documents(texts)
        embed_time_ms = (time.perf_counter() - t0) * 1000.0

        # Build FAISS
        faiss_store = FAISSStore()
        if embeddings:
            faiss_store.add(embeddings)

        # Build BM25
        t0 = time.perf_counter()
        bm25 = BM25Retriever()
        bm25.build(texts)
        bm25_build_time_ms = (time.perf_counter() - t0) * 1000.0

        # Measure single-stage benchmark costs on sample query
        stage_latencies = self._benchmark_stage_costs(faiss_store, bm25)
        stage_latencies["embed_total_ms"] = embed_time_ms
        stage_latencies["bm25_build_ms"] = bm25_build_time_ms

        return IndexArtifacts(
            index_params=index_params,
            chunks=chunks,
            chunk_ids=chunk_ids,
            faiss_store=faiss_store,
            bm25=bm25,
            stage_latencies=stage_latencies,
        )

    def _benchmark_stage_costs(
        self,
        faiss_store: FAISSStore,
        bm25: BM25Retriever,
    ) -> Dict[str, float]:
        """Benchmark representative per-stage latencies for p50 estimation."""
        if not self.eval_samples:
            return {
                "embed_query": 2.0,
                "dense_search": 0.5,
                "bm25_search": 0.5,
                "fusion": 0.1,
                "rerank_per_item": 0.5,
                "query_expansion": 5.0,
            }

        benchmark_q = self.eval_samples[0].question

        # 1. embed query
        t0 = time.perf_counter()
        q_emb = self.embedding_model.embed_query(benchmark_q)
        embed_q_ms = max(0.01, (time.perf_counter() - t0) * 1000.0)

        # 2. dense search
        t0 = time.perf_counter()
        if faiss_store.index is not None:
            faiss_store.search(q_emb, top_k=min(self.pool_size, faiss_store.index.ntotal))
        dense_ms = max(0.01, (time.perf_counter() - t0) * 1000.0)

        # 3. bm25 search
        t0 = time.perf_counter()
        bm25.search(benchmark_q, top_k=self.pool_size)
        bm25_ms = max(0.01, (time.perf_counter() - t0) * 1000.0)

        # 4. fusion
        dummy_dense = [(i, float(i)) for i in range(min(self.pool_size, 50))]
        dummy_bm25 = [(i, float(i)) for i in range(min(self.pool_size, 50))]
        t0 = time.perf_counter()
        reciprocal_rank_fusion(dummy_dense, dummy_bm25, k=60, dense_weight=0.5)
        fusion_ms = max(0.01, (time.perf_counter() - t0) * 1000.0)

        return {
            "embed_query": embed_q_ms,
            "dense_search": dense_ms,
            "bm25_search": bm25_ms,
            "fusion": fusion_ms,
            "rerank_per_item": 0.8,
            "query_expansion": 4.0,
        }

    # ------------------------------------------------------------------
    # Candidate Pools & Pre-scoring
    # ------------------------------------------------------------------

    def _get_query_text(self, question: str, expansion: bool) -> str:
        """Return raw question or cached expanded query."""
        if not expansion:
            return question
        if question not in self._expansion_cache:
            self._expansion_cache[question] = self.query_expander.expand(question)
        return self._expansion_cache[question]

    def _build_candidate_pools(
        self,
        artifacts: IndexArtifacts,
    ) -> Dict[bool, List[Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]]]:
        """
        Retrieve top-50 dense and sparse candidate pools for each eval sample.
        Keyed by query_expansion bool: {False: [...], True: [...]}.
        """
        pools_by_expansion: Dict[bool, List[Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]]] = {
            False: [],
            True: [],
        }

        total_chunks = len(artifacts.chunks)
        effective_pool = min(self.pool_size, total_chunks) if total_chunks > 0 else 0

        for expansion in (False, True):
            pools: List[Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]] = []
            for sample in self.eval_samples:
                q_text = self._get_query_text(sample.question, expansion)

                # Dense search
                if effective_pool > 0 and artifacts.faiss_store.index is not None:
                    q_emb = self.embedding_model.embed_query(q_text)
                    scores, indices = artifacts.faiss_store.search(q_emb, top_k=effective_pool)
                    dense_results: List[Tuple[int, float]] = [
                        (int(idx), float(score))
                        for idx, score in zip(indices[0], scores[0])
                        if idx != -1
                    ]
                else:
                    dense_results = []

                # Sparse search
                if effective_pool > 0:
                    bm25_results = artifacts.bm25.search(q_text, top_k=effective_pool)
                else:
                    bm25_results = []

                pools.append((dense_results, bm25_results))
            pools_by_expansion[expansion] = pools

        return pools_by_expansion

    def _prescore_cross_encoder_union(
        self,
        artifacts: IndexArtifacts,
        pools_by_expansion: Dict[bool, List[Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]]],
        query_configs: Sequence[QueryParams],
    ) -> None:
        """
        Pre-score candidate chunks across all fusion configs for configs where
        reranking_enabled is True. Collects unique unscored pairs and scores
        them once in batches with the cross-encoder.
        """
        if self.cross_encoder is None:
            return

        reranking_configs = [qp for qp in query_configs if qp.reranking_enabled]
        if not reranking_configs:
            return

        pairs_to_score: List[Tuple[str, str]] = []
        keys_to_score: List[Tuple[str, str, str]] = []
        seen_keys: Set[Tuple[str, str, str]] = set()

        for sample_idx, sample in enumerate(self.eval_samples):
            # Form candidate union across all reranking configs for this sample
            union_indices: Set[int] = set()

            for qp in reranking_configs:
                dense_res, bm25_res = pools_by_expansion[qp.query_expansion][sample_idx]
                fused = reciprocal_rank_fusion(
                    faiss_results=dense_res,
                    bm25_results=bm25_res,
                    k=qp.rrf_k,
                    dense_weight=qp.dense_weight,
                    chunk_ids=artifacts.chunk_ids,
                )
                top_rerank_candidates = [idx for idx, _ in fused[: self.rerank_depth]]
                union_indices.update(top_rerank_candidates)

            # Cap the union size with deterministic sorting
            sorted_union = sorted(union_indices)
            if len(sorted_union) > self.max_union_size:
                self.has_truncated = True
                logger.warning(
                    "Union cap exceeded: %d candidates truncated to %d for query %r",
                    len(sorted_union),
                    self.max_union_size,
                    sample.question,
                )
            capped_indices = sorted_union[: self.max_union_size]

            # Query text for reranker
            q_text = sample.question

            for idx in capped_indices:
                if 0 <= idx < len(artifacts.chunks):
                    chunk_text = artifacts.chunks[idx].get("text", "")
                    text_hash = hashlib.sha1(chunk_text.encode("utf-8")).hexdigest()
                    cache_key = (self.ce_model_name, q_text, text_hash)

                    if cache_key not in self._ce_cache and cache_key not in seen_keys:
                        seen_keys.add(cache_key)
                        keys_to_score.append(cache_key)
                        pairs_to_score.append((q_text, chunk_text))

        # Batch-predict unscored pairs
        if pairs_to_score:
            scores = self.cross_encoder.predict(pairs_to_score)
            for key, score in zip(keys_to_score, scores):
                self._ce_cache[key] = float(score)
            self._save_ce_cache()

    # ------------------------------------------------------------------
    # Offline Query Config Evaluation
    # ------------------------------------------------------------------

    def _evaluate_query_config_offline(
        self,
        artifacts: IndexArtifacts,
        pools_by_expansion: Dict[bool, List[Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]]],
        qp: QueryParams,
    ) -> TrialResult:
        """Evaluate a single QueryParams offline against candidate pools."""
        pools = pools_by_expansion[qp.query_expansion]
        sample_results: List[Dict[str, Any]] = []

        mrr_sum = 0.0
        hit_sum = 0.0
        prec_sum = 0.0

        for sample_idx, sample in enumerate(self.eval_samples):
            dense_res, bm25_res = pools[sample_idx]
            fused = reciprocal_rank_fusion(
                faiss_results=dense_res,
                bm25_results=bm25_res,
                k=qp.rrf_k,
                dense_weight=qp.dense_weight,
                chunk_ids=artifacts.chunk_ids,
            )

            if not qp.reranking_enabled or self.cross_encoder is None:
                final_indices = [idx for idx, _ in fused[: qp.top_k]]
            else:
                candidates_to_rerank = fused[: self.rerank_depth]
                q_text = sample.question
                scored_candidates: List[Tuple[int, float]] = []

                for idx, _ in candidates_to_rerank:
                    if 0 <= idx < len(artifacts.chunks):
                        c_text = artifacts.chunks[idx].get("text", "")
                        thash = hashlib.sha1(c_text.encode("utf-8")).hexdigest()
                        ckey = (self.ce_model_name, q_text, thash)
                        score = self._ce_cache.get(ckey, 0.0)
                        scored_candidates.append((idx, score))

                # Sort by rerank score descending with tie-break on chunk_id
                def _rerank_sort_key(item: Tuple[int, float]) -> Tuple[float, str]:
                    idx, score = item
                    cid = artifacts.chunk_ids[idx] if 0 <= idx < len(artifacts.chunk_ids) else str(idx)
                    return (-score, cid)

                scored_candidates.sort(key=_rerank_sort_key)
                final_indices = [idx for idx, _ in scored_candidates[: qp.top_k]]

            hit_rank = find_hit_rank(final_indices, artifacts.chunks, sample)
            s_mrr = 1.0 / hit_rank if hit_rank is not None else 0.0
            s_hit = 1.0 if hit_rank is not None else 0.0

            # Count all overlapping hits in final_indices for precision@k
            num_hits = sum(
                1
                for idx in final_indices
                if 0 <= idx < len(artifacts.chunks)
                and is_overlapping_hit(artifacts.chunks[idx], sample)
            )
            s_prec = num_hits / qp.top_k if qp.top_k > 0 else 0.0

            mrr_sum += s_mrr
            hit_sum += s_hit
            prec_sum += s_prec

            sample_results.append(
                {
                    "question": sample.question,
                    "hit_rank": hit_rank,
                    "mrr": s_mrr,
                    "hit": s_hit,
                    "precision": s_prec,
                    "final_indices": final_indices,
                }
            )

        n_samples = max(1, len(self.eval_samples))
        mean_mrr = mrr_sum / n_samples
        mean_hit = hit_sum / n_samples
        mean_prec = prec_sum / n_samples

        # Estimate p50 latency from per-stage benchmark costs
        st = artifacts.stage_latencies
        est_latency = (
            st.get("embed_query", 1.5)
            + st.get("dense_search", 0.5)
            + st.get("bm25_search", 0.5)
            + st.get("fusion", 0.1)
            + (self.rerank_depth * st.get("rerank_per_item", 0.8) if qp.reranking_enabled else 0.0)
            + (st.get("query_expansion", 4.0) if qp.query_expansion else 0.0)
        )

        return TrialResult(
            index_params=artifacts.index_params,
            query_params=qp,
            mrr=mean_mrr,
            hit_rate=mean_hit,
            precision=mean_prec,
            latency_ms=est_latency,
            latency_type="estimated",
            stage_latencies=dict(artifacts.stage_latencies),
            eval_sample_results=sample_results,
            pool_size=self.pool_size,
            truncated=self.has_truncated,
        )

    # ------------------------------------------------------------------
    # Live Latency Measurement
    # ------------------------------------------------------------------

    def _measure_live_latency(
        self,
        artifacts: IndexArtifacts,
        qp: QueryParams,
    ) -> float:
        """Measure actual end-to-end elapsed time on sample query."""
        if not self.eval_samples:
            return 10.0

        sample_q = self.eval_samples[0].question
        t0 = time.perf_counter()

        # Expansion
        q_text = self._get_query_text(sample_q, qp.query_expansion)

        # Dense
        if artifacts.faiss_store.index is not None and artifacts.faiss_store.index.ntotal > 0:
            q_emb = self.embedding_model.embed_query(q_text)
            scores, indices = artifacts.faiss_store.search(q_emb, top_k=self.pool_size)
            dense_res = [
                (int(idx), float(score))
                for idx, score in zip(indices[0], scores[0])
                if idx != -1
            ]
        else:
            dense_res = []

        # Sparse
        bm25_res = artifacts.bm25.search(q_text, top_k=self.pool_size)

        # Fusion
        fused = reciprocal_rank_fusion(
            faiss_results=dense_res,
            bm25_results=bm25_res,
            k=qp.rrf_k,
            dense_weight=qp.dense_weight,
            chunk_ids=artifacts.chunk_ids,
        )

        # Rerank
        if qp.reranking_enabled and self.cross_encoder is not None:
            candidates = fused[: self.rerank_depth]
            pairs = [
                (sample_q, artifacts.chunks[idx]["text"])
                for idx, _ in candidates
                if 0 <= idx < len(artifacts.chunks)
            ]
            if pairs:
                self.cross_encoder.predict(pairs)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return max(0.01, elapsed_ms)
