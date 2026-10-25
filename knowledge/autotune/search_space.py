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

Runner design contract (for commit 74)
---------------------------------------
The query-time sweep must NOT rerun retrieval per QueryParams.  Instead:
  - Retrieve a top-50 candidate pool **once** per (index, query_expansion) pair.
  - Recompute ``rrf_k``/``dense_weight``/``top_k`` offline from stored ranks.
  - Cache cross-encoder scores keyed by ``(query_text, chunk_id)`` so reranking
    is free after the first call.
``QueryParams`` is structurally compatible with this: changing ``query_expansion``
is the only knob that requires a fresh retrieval pass.
"""

from __future__ import annotations

import hashlib
import itertools
import json as _json
import random
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Embedding allowlist
# ---------------------------------------------------------------------------

EMBEDDING_ALLOWLIST: Tuple[str, ...] = (
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "multi-qa-MiniLM-L6-cos-v1",
)
"""Models eligible for ``--sweep-embeddings``.  All three run on CPU without a
GPU.  The runner skips any model that raises on load and logs a warning."""


# ---------------------------------------------------------------------------
# IndexParams
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IndexParams:
    """Index-level configuration — one FAISS build per unique instance.

    ``budget`` (on ``SearchSpace``) caps how many distinct ``IndexParams``
    are built per autotune run.
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
        """Return a stable, filesystem-safe key for on-disk cache lookup.

        Format: ``<readable-prefix>_<sha1[:8]>``

        The readable prefix aids human browsing; the SHA-1 suffix prevents
        collisions such as ``org/model`` vs ``org-model`` (both normalise to
        the same prefix but have different hashes).

        The canonical JSON fed to SHA-1 uses sorted keys and no whitespace so
        the hash is byte-identical across Python versions and platforms.

        Cache directory: ``.knowledge/autotune/cache/<cache_key>/``

        Example::

            cs500_co50_rec_all-MiniLM_a3f8c21d
        """
        canonical = _json.dumps(
            {
                "chunk_overlap": self.chunk_overlap,
                "chunk_size": self.chunk_size,
                "chunking_strategy": self.chunking_strategy,
                "embedding_model": self.embedding_model,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        sha = hashlib.sha1(canonical.encode()).hexdigest()[:8]
        # Readable prefix: strategy abbreviated to 3 chars, model to basename
        model_tag = self.embedding_model.split("/")[-1][:20]
        prefix = (
            f"cs{self.chunk_size}"
            f"_co{self.chunk_overlap}"
            f"_{self.chunking_strategy[:3]}"
            f"_{model_tag}"
        )
        return f"{prefix}_{sha}"


# ---------------------------------------------------------------------------
# QueryParams
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueryParams:
    """Query-time configuration evaluated without rebuilding the index.

    ``rrf_k``        — constant *k* in ``1 / (k + rank)``; higher = smoother.
    ``dense_weight`` — fraction of the fusion score for dense results; sparse
                       contribution is ``1 − dense_weight``.  ``1.0`` = pure
                       dense; ``0.0`` = pure BM25.
    ``query_expansion`` — expand the query before retrieval.  **Changing this
                       is the only QueryParam that requires a fresh retrieval
                       pass** (different pool); all other knobs are applied
                       offline to stored ranks.
    """

    top_k: int
    reranking_enabled: bool
    rrf_k: int            # > 0
    dense_weight: float   # [0.0, 1.0]
    query_expansion: bool

    def is_valid(self) -> bool:
        """Return True iff all field values are within legal ranges."""
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
    """A fully-specified autotune configuration: one ``IndexParams`` + one
    ``QueryParams``."""

    index_params: IndexParams
    query_params: QueryParams

    def is_valid(self) -> bool:
        return self.index_params.is_valid() and self.query_params.is_valid()

    def to_appconfig_dict(self) -> Dict[str, object]:
        """Return a dict whose keys map 1-to-1 with ``AppConfig`` field names.

        Safe to pass directly to ``AppConfig(**d)`` or ``ConfigManager.update(d)``.
        Does **not** include autotune-specific extras (``rrf_k`` etc.) because
        those have no corresponding ``AppConfig`` field.
        """
        return {
            "embedding_model": self.index_params.embedding_model,
            "chunk_size": self.index_params.chunk_size,
            "chunk_overlap": self.index_params.chunk_overlap,
            "chunking_strategy": self.index_params.chunking_strategy,
            "top_k": self.query_params.top_k,
            "reranking_enabled": self.query_params.reranking_enabled,
        }

    def to_full_dict(self) -> Dict[str, object]:
        """Return the full ``config.autotuned.yaml`` structure.

        AppConfig-compatible fields live at the top level.
        Autotune-specific extras (not in ``AppConfig``) are nested under the
        ``autotune:`` key so that ``AppConfig(**to_full_dict())`` still loads
        cleanly (Pydantic v2 ignores unknown top-level keys by default, and
        ``autotune:`` is a single unknown dict-valued key).
        """
        result: Dict[str, object] = dict(self.to_appconfig_dict())
        result["autotune"] = {
            "rrf_k": self.query_params.rrf_k,
            "dense_weight": self.query_params.dense_weight,
            "query_expansion": self.query_params.query_expansion,
        }
        return result


# ---------------------------------------------------------------------------
# SearchSpace
# ---------------------------------------------------------------------------


@dataclass
class SearchSpace:
    """Defines the axes over which autotune searches.

    Index-level axes determine the number of FAISS builds (capped by budget).
    Query-time axes are swept exhaustively for each built index.

    Example — enable ``--sweep-embeddings``::

        space = SearchSpace(embedding_models=EMBEDDING_ALLOWLIST)

    Example — narrow to a single config for debugging::

        space = SearchSpace(
            chunk_sizes=(512,),
            chunk_overlaps=(0,),
            chunking_strategies=("recursive",),
            top_k_values=(5,),
            reranking_options=(False,),
            rrf_k_values=(60,),
            dense_weights=(0.7,),
            query_expansion_options=(False,),
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
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> List[str]:
        """Return a list of error messages for the search space configuration.

        An empty list means the space is valid and ready to iterate.

        Example::

            errors = space.validate()
            if errors:
                raise ValueError("\\n".join(errors))
        """
        errors: List[str] = []
        for attr in ("chunk_sizes", "chunk_overlaps", "chunking_strategies",
                      "embedding_models", "top_k_values", "rrf_k_values",
                      "dense_weights", "query_expansion_options", "reranking_options"):
            if not getattr(self, attr):
                errors.append(f"{attr} must not be empty")
        for strat in self.chunking_strategies:
            if strat not in ("recursive", "sentence", "fixed"):
                errors.append(f"Unknown chunking_strategy: {strat!r}")
        for dw in self.dense_weights:
            if not 0.0 <= dw <= 1.0:
                errors.append(f"dense_weight {dw} is outside [0.0, 1.0]")
        for rk in self.rrf_k_values:
            if rk <= 0:
                errors.append(f"rrf_k {rk} must be > 0")
        for tk in self.top_k_values:
            if tk <= 0:
                errors.append(f"top_k {tk} must be > 0")
        return errors

    # ------------------------------------------------------------------
    # Candidate enumeration
    # ------------------------------------------------------------------

    def all_index_params(self) -> List[IndexParams]:
        """Return all *valid* ``IndexParams`` from the Cartesian product of
        index-level axes."""
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
        query-time axes."""
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

    def neighbors(self, params: IndexParams) -> List[IndexParams]:
        """Return all valid ``IndexParams`` that differ from *params* by
        exactly one axis value.

        Useful for local exploration around the baseline config.  Neighbours
        are limited to values present in the search space axes.
        """
        result: List[IndexParams] = []
        all_set = set(self.all_index_params())

        # Each axis block: swap one field at a time
        for cs in self.chunk_sizes:
            if cs == params.chunk_size:
                continue
            candidate = IndexParams(cs, params.chunk_overlap, params.chunking_strategy, params.embedding_model)
            if candidate in all_set:
                result.append(candidate)

        for co in self.chunk_overlaps:
            if co == params.chunk_overlap:
                continue
            candidate = IndexParams(params.chunk_size, co, params.chunking_strategy, params.embedding_model)
            if candidate in all_set:
                result.append(candidate)

        for strat in self.chunking_strategies:
            if strat == params.chunking_strategy:
                continue
            candidate = IndexParams(params.chunk_size, params.chunk_overlap, strat, params.embedding_model)
            if candidate in all_set:
                result.append(candidate)

        for em in self.embedding_models:
            if em == params.embedding_model:
                continue
            candidate = IndexParams(params.chunk_size, params.chunk_overlap, params.chunking_strategy, em)
            if candidate in all_set:
                result.append(candidate)

        return result

    def seeded_sample(
        self,
        n: int,
        seed: int,
        exclude: Optional[List[IndexParams]] = None,
    ) -> List[IndexParams]:
        """Return up to *n* IndexParams from the full grid, excluding *exclude*,
        in a deterministically shuffled order.

        This is the random-fill step used internally by ``budget_index_params``.
        Calling it directly is useful for testing or custom selection logic.
        """
        exclude_set = set(exclude) if exclude else set()
        pool = [p for p in self.all_index_params() if p not in exclude_set]
        rng = random.Random(seed)
        rng.shuffle(pool)
        return pool[:n]

    def budget_index_params(
        self,
        budget: int,
        seed: int = 42,
        baseline: Optional[IndexParams] = None,
    ) -> List[IndexParams]:
        """Select up to *budget* IndexParams using a priority ordering:

        1. **Baseline** — always first (the currently live config).
        2. **Neighbors** of baseline — one-axis changes, deterministically
           shuffled.
        3. **Axis coverage** — for each axis in the space, ensure at least one
           config containing each axis *value* appears in the selection.
        4. **Random fill** — seeded shuffle of remaining valid configs.

        All selections are deduplicated and the ordering is stable for a given
        ``(budget, seed, baseline)``.

        Args:
            budget: Maximum index builds.  ``0`` returns all valid configs
                    (use with caution — print estimated builds first in the
                    CLI before proceeding).
            seed:   RNG seed; forwarded to the random-fill step.
            baseline: If provided and valid, always placed first.

        Raises:
            ValueError: If *budget* is negative.
        """
        if budget < 0:
            raise ValueError(f"budget must be >= 0, got {budget!r}")

        all_params = self.all_index_params()
        all_set = set(all_params)
        rng = random.Random(seed)

        selected: List[IndexParams] = []
        seen: set[IndexParams] = set()

        def _add(p: IndexParams) -> bool:
            if p in seen or p not in all_set:
                return False
            selected.append(p)
            seen.add(p)
            return True

        def _budget_reached() -> bool:
            return budget > 0 and len(selected) >= budget

        # 1. Baseline
        if baseline is not None and baseline.is_valid():
            _add(baseline)
        if _budget_reached():
            return selected

        # 2. Neighbors of baseline (deterministically shuffled)
        if baseline is not None:
            nbrs = self.neighbors(baseline)
            rng.shuffle(nbrs)
            for p in nbrs:
                _add(p)
                if _budget_reached():
                    return selected

        # 3. Axis coverage — one representative per (axis, value) pair
        axes_values: List[Tuple[str, Tuple[object, ...]]] = [
            ("chunk_size",        self.chunk_sizes),
            ("chunk_overlap",     self.chunk_overlaps),
            ("chunking_strategy", self.chunking_strategies),
            ("embedding_model",   self.embedding_models),
        ]
        for field_name, values in axes_values:
            for v in values:
                for p in all_params:
                    if getattr(p, field_name) == v and p not in seen:
                        _add(p)
                        break
                if _budget_reached():
                    return selected

        # 4. Random fill from remaining
        remaining = [p for p in all_params if p not in seen]
        rng.shuffle(remaining)
        for p in remaining:
            _add(p)
            if _budget_reached():
                return selected

        return selected

    # ------------------------------------------------------------------
    # Iteration helpers
    # ------------------------------------------------------------------

    def iter_candidates(
        self,
        budget: int = 8,
        seed: int = 42,
        baseline: Optional[IndexParams] = None,
    ) -> Iterator[Tuple[IndexParams, List[QueryParams]]]:
        """Yield ``(IndexParams, all_query_params)`` for each index build.

        The baseline (if supplied) is always the first ``IndexParams`` yielded.

        Typical caller pattern::

            for ip, query_configs in space.iter_candidates(budget=8, baseline=base):
                index = runner.build_or_load_index(ip)   # once per ip
                for qp in query_configs:                  # cheap
                    result = runner.evaluate(index, qp, eval_set)
        """
        query_configs = self.all_query_params()
        for ip in self.budget_index_params(budget, seed, baseline):
            yield ip, query_configs

    def total_experiments(
        self,
        budget: int = 8,
        seed: int = 42,
        baseline: Optional[IndexParams] = None,
    ) -> int:
        """Return the total ``IndexParams × QueryParams`` experiments that
        ``iter_candidates`` would run.  Useful for pre-run progress reporting.

        When ``budget=0`` this can be very large — the CLI should print this
        number and require ``--yes`` before proceeding.
        """
        n_index = len(self.budget_index_params(budget, seed, baseline))
        n_query = len(self.all_query_params())
        return n_index * n_query

    def estimate_index_builds(
        self,
        budget: int = 8,
        baseline: Optional[IndexParams] = None,
    ) -> int:
        """Return the number of index builds (not total experiments).

        Intended for the ``--budget 0`` guard: print this value before
        running and require explicit confirmation.
        """
        if budget == 0:
            return len(self.all_index_params())
        # With baseline the count might be < budget if total space is smaller
        return min(budget, len(self.all_index_params()))
