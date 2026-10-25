from collections import defaultdict
from typing import Dict, List, Optional, Tuple


def reciprocal_rank_fusion(
    faiss_results: List[Tuple[int, float]],
    bm25_results: List[Tuple[int, float]],
    k: int = 60,
    dense_weight: float = 0.5,
    chunk_ids: Optional[List[str]] = None,
) -> List[Tuple[int, float]]:
    """
    Reciprocal Rank Fusion with dense_weight and deterministic tie-breaking.

    Args:
        faiss_results: Dense retrieval results as list of (doc_index, score).
        bm25_results: Sparse retrieval results as list of (doc_index, score).
        k: RRF smoothing constant (default: 60).
        dense_weight: Dense score weight in [0.0, 1.0]. Sparse weight is (1.0 - dense_weight).
        chunk_ids: Optional list mapping doc_index to string chunk_id for tie-breaking.

    Returns:
        List of (doc_index, fused_score) sorted descending by score with ties broken by chunk_id.
    """
    scores: Dict[int, float] = defaultdict(float)
    sparse_weight = 1.0 - dense_weight

    for rank, (idx, _) in enumerate(faiss_results):
        scores[idx] += dense_weight * (1.0 / (k + rank + 1))

    for rank, (idx, _) in enumerate(bm25_results):
        scores[idx] += sparse_weight * (1.0 / (k + rank + 1))

    def _sort_key(item: Tuple[int, float]) -> Tuple[float, str]:
        idx, score = item
        cid = chunk_ids[idx] if chunk_ids and 0 <= idx < len(chunk_ids) else str(idx)
        return (-score, cid)

    return sorted(scores.items(), key=_sort_key)


class HybridRetriever:

    def fuse(
        self,
        faiss_results: List[Tuple[int, float]],
        bm25_results: List[Tuple[int, float]],
        k: int = 60,
        dense_weight: float = 0.5,
        chunk_ids: Optional[List[str]] = None,
    ) -> List[Tuple[int, float]]:
        """
        Reciprocal Rank Fusion with optional dense_weight and tie-breaking.
        """
        return reciprocal_rank_fusion(
            faiss_results=faiss_results,
            bm25_results=bm25_results,
            k=k,
            dense_weight=dense_weight,
            chunk_ids=chunk_ids,
        )
