"""
Search space definition for knowledge autotune.

Two-tier design
---------------
``IndexParams``  — index-level knobs that require a full FAISS rebuild per
                   unique combination (chunk_size, chunk_overlap, strategy, model).
``QueryParams``  — query-time knobs evaluated without rebuilding the index
                   (top_k, reranking, rrf_k, dense_weight, query_expansion).

The ``budget`` parameter limits the number of *index builds*, not the total
experiment count.  For each built index, every valid ``QueryParams`` combination
is swept exhaustively.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Iterator, List, Tuple

# ---------------------------------------------------------------------------
# Embedding allowlist
# ---------------------------------------------------------------------------

EMBEDDING_ALLOWLIST: Tuple[str, ...] = (
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "multi-qa-MiniLM-L6-cos-v1",
)
"""Models that autotune may load when ``--sweep-embeddings`` is active.
All three are small enough to run on CPU without a GPU.
The runner will emit a warning and skip any model that fails to load.
"""

# ---------------------------------------------------------------------------
# IndexParams
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IndexParams:
    """Index-level configuration.

    Each unique ``IndexParams`` triggers one embedding + FAISS build.
    ``budget`` caps how many of these builds occur per autotune run.
    """

    chunk_size: int
    chunk_overlap: int
    chunking_strategy: str   # "recursive" | "sentence" | "fixed"
    embedding_model: str

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def is_valid(self) -> bool:
        """Return True iff all field values are mutually consistent."""
        return (
            self.chunk_size > 0
            and 0 <= self.chunk_overlap < self.chunk_size
            and self.chunking_strategy in ("recursive", "sentence", "fixed")
            and bool(self.embedding_model)
        )

    # ------------------------------------------------------------------
    # Cache key — used by ExperimentRunner for on-disk index cache
    # ------------------------------------------------------------------

    def cache_key(self) -> str:
        """Return a stable, filesystem-safe string for on-disk cache lookup.

        The cache lives at ``.knowledge/autotune/cache/<cache_key>/`` and
        contains ``faiss.index`` + ``chunks.json`` so interrupted runs can
        resume without rebuilding.

        Example::

            cs500_co50_strecursive_emall-MiniLM-L6-v2
        """
        safe_model = self.embedding_model.replace("/", "-")
        return (
            f"cs{self.chunk_size}"
            f"_co{self.chunk_overlap}"
            f"_st{self.chunking_strategy}"
            f"_em{safe_model}"
        )


# ---------------------------------------------------------------------------
# QueryParams
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueryParams:
    """Query-time configuration evaluated without rebuilding the index.

    ``rrf_k``       — constant *k* in the RRF formula ``1 / (k + rank)``.
    ``dense_weight``— fraction of the fusion score assigned to dense results;
                      sparse contribution is ``1 - dense_weight``.
                      Setting ``dense_weight=1.0`` is pure dense retrieval;
                      ``0.0`` is pure sparse (BM25).
    ``query_expansion`` — whether to expand the user query before retrieval.
    """

    top_k: int
    reranking_enabled: bool
    rrf_k: int            # must be > 0
    dense_weight: float   # in [0.0, 1.0]
    query_expansion: bool

    def is_valid(self) -> bool:
        """Return True iff all field values are mutually consistent."""
        return (
            self.top_k > 0
            and self.rrf_k > 0
            and 0.0 <= self.dense_weight <= 1.0
        )


# ---------------------------------------------------------------------------
# CandidateConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateConfig:
    """A fully-specified autotune configuration: one IndexParams + one QueryParams."""

    index_params: IndexParams
    query_params: QueryParams

    def is_valid(self) -> bool:
        """Return True iff both sub-configs are individually valid."""
        return self.index_params.is_valid() and self.query_params.is_valid()

    def to_appconfig_dict(self) -> dict:
        """Return a dict whose keys match ``AppConfig`` field names.

        Used by ``reporter.py`` to write ``config.autotuned.yaml`` and by the
        ``--apply`` path that calls ``ConfigManager.update()``.

        Note: ``rrf_k``, ``dense_weight``, and ``query_expansion`` are
        autotune-specific extras written to ``config.autotuned.yaml`` but not
        (yet) present in the base ``AppConfig`` schema.
        """
        return {
            # IndexParams → AppConfig fields
            "embedding_model": self.index_params.embedding_model,
            "chunk_size": self.index_params.chunk_size,
            "chunk_overlap": self.index_params.chunk_overlap,
            "chunking_strategy": self.index_params.chunking_strategy,
            # QueryParams → AppConfig fields
            "top_k": self.query_params.top_k,
            "reranking_enabled": self.query_params.reranking_enabled,
        }

    def to_full_dict(self) -> dict:
        """Return a dict with *all* fields including autotune-specific extras.

        Written verbatim into ``config.autotuned.yaml`` for full
        reproducibility.
        """
        base = self.to_appconfig_dict()
        base.update(
            {
                "autotune_rrf_k": self.query_params.rrf_k,
                "autotune_dense_weight": self.query_params.dense_weight,
                "autotune_query_expansion": self.query_params.query_expansion,
            }
        )
        return base


# ---------------------------------------------------------------------------
# SearchSpace
# ---------------------------------------------------------------------------


@dataclass
class SearchSpace:
    """Defines the axes over which autotune searches.

    Index-level axes determine how many FAISS builds occur (capped by budget).
    Query-time axes are swept *exhaustively* for each built index.

    All tuple fields are immutable sequences so the dataclass can be compared
    and copied cheaply.  Pass custom tuples to narrow or widen the search:

    .. code-block:: python

        space = SearchSpace(
            chunk_sizes=(256, 512),
            embedding_models=EMBEDDING_ALLOWLIST,   # --sweep-embeddings
        )
    """

    # -- Index-level axes --------------------------------------------------
    chunk_sizes: Tuple[int, ...] = (300, 500, 800)
    chunk_overlaps: Tuple[int, ...] = (0, 50, 100)
    chunking_strategies: Tuple[str, ...] = ("recursive", "sentence", "fixed")
    embedding_models: Tuple[str, ...] = ("all-MiniLM-L6-v2",)

    # -- Query-time axes ---------------------------------------------------
    top_k_values: Tuple[int, ...] = (3, 5, 10)
    reranking_options: Tuple[bool, ...] = (True, False)
    rrf_k_values: Tuple[int, ...] = (20, 60)
    dense_weights: Tuple[float, ...] = (0.5, 0.7, 1.0)
    query_expansion_options: Tuple[bool, ...] = (True, False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def all_index_params(self) -> List[IndexParams]:
        """Return all *valid* ``IndexParams`` from the Cartesian product of
        the index-level axes."""
        return [
            p
            for p in (
                IndexParams(
                    chunk_size=cs,
                    chunk_overlap=co,
                    chunking_strategy=strat,
                    embedding_model=em,
                )
                for cs, co, strat, em in itertools.product(
                    self.chunk_sizes,
                    self.chunk_overlaps,
                    self.chunking_strategies,
                    self.embedding_models,
                )
            )
            if p.is_valid()
        ]

    def all_query_params(self) -> List[QueryParams]:
        """Return all *valid* ``QueryParams`` from the Cartesian product of
        the query-time axes."""
        return [
            p
            for p in (
                QueryParams(
                    top_k=top_k,
                    reranking_enabled=rerank,
                    rrf_k=rrf_k,
                    dense_weight=dw,
                    query_expansion=qe,
                )
                for top_k, rerank, rrf_k, dw, qe in itertools.product(
                    self.top_k_values,
                    self.reranking_options,
                    self.rrf_k_values,
                    self.dense_weights,
                    self.query_expansion_options,
                )
            )
            if p.is_valid()
        ]

    def budget_index_params(self, budget: int, seed: int = 42) -> List[IndexParams]:
        """Return up to *budget* ``IndexParams``, shuffled deterministically.

        Shuffling ensures random coverage when ``budget`` < total index configs,
        rather than always evaluating the same corner of the grid.

        Args:
            budget: Maximum number of index builds (0 = unlimited).
            seed:   RNG seed for reproducible ordering.

        Returns:
            Deterministically ordered list of ``IndexParams`` of length
            ``min(budget, len(all_index_params()))`` (or all when budget=0).

        Raises:
            ValueError: If ``budget`` is negative.
        """
        if budget < 0:
            raise ValueError(f"budget must be >= 0, got {budget!r}")
        all_params = self.all_index_params()
        rng = random.Random(seed)
        rng.shuffle(all_params)
        return all_params if budget == 0 else all_params[:budget]

    def iter_candidates(
        self, budget: int = 8, seed: int = 42
    ) -> Iterator[Tuple[IndexParams, List[QueryParams]]]:
        """Yield ``(IndexParams, [all QueryParams])`` for each index build.

        The outer loop drives one expensive FAISS build per ``IndexParams``.
        The inner list is the complete set of cheap query-time experiments.

        Typical caller pattern::

            for index_params, query_configs in space.iter_candidates(budget=8):
                index = runner.build_or_load_index(index_params)  # once
                for qp in query_configs:                           # many
                    result = runner.evaluate(index, qp, eval_set)

        Args:
            budget: Maximum index builds (0 = unlimited).
            seed:   RNG seed forwarded to ``budget_index_params``.
        """
        query_configs = self.all_query_params()
        for ip in self.budget_index_params(budget, seed):
            yield ip, query_configs

    def total_experiments(self, budget: int = 8, seed: int = 42) -> int:
        """Return the total number of ``(IndexParams × QueryParams)``
        experiments that ``iter_candidates`` would run.

        Useful for progress reporting before the run starts.
        """
        n_index = len(self.budget_index_params(budget, seed))
        n_query = len(self.all_query_params())
        return n_index * n_query
