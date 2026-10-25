"""
Unit tests for knowledge.autotune.search_space.

No Ollama, no network, no disk I/O.  Pure dataclass / logic tests.
"""

import pytest

from knowledge.autotune.search_space import (
    EMBEDDING_ALLOWLIST,
    CandidateConfig,
    IndexParams,
    QueryParams,
    SearchSpace,
)

# ---------------------------------------------------------------------------
# IndexParams
# ---------------------------------------------------------------------------


class TestIndexParams:
    """Validity, cache key, immutability, and hashability."""

    def _make(self, **kw) -> IndexParams:  # type: ignore[no-untyped-def]
        defaults = dict(
            chunk_size=500,
            chunk_overlap=50,
            chunking_strategy="recursive",
            embedding_model="all-MiniLM-L6-v2",
        )
        defaults.update(kw)
        return IndexParams(**defaults)

    # -- is_valid ----------------------------------------------------------

    def test_valid_basic(self) -> None:
        assert self._make().is_valid()

    def test_invalid_overlap_equals_size(self) -> None:
        assert not self._make(chunk_size=300, chunk_overlap=300).is_valid()

    def test_invalid_overlap_exceeds_size(self) -> None:
        assert not self._make(chunk_size=300, chunk_overlap=400).is_valid()

    def test_invalid_zero_chunk_size(self) -> None:
        assert not self._make(chunk_size=0, chunk_overlap=0).is_valid()

    def test_valid_zero_overlap(self) -> None:
        assert self._make(chunk_size=300, chunk_overlap=0).is_valid()

    def test_valid_overlap_one_less_than_size(self) -> None:
        assert self._make(chunk_size=100, chunk_overlap=99).is_valid()

    def test_invalid_unknown_strategy(self) -> None:
        assert not self._make(chunking_strategy="sliding_window").is_valid()

    def test_valid_all_three_strategies(self) -> None:
        for strat in ("recursive", "sentence", "fixed"):
            assert self._make(chunking_strategy=strat).is_valid(), strat

    def test_invalid_empty_embedding_model(self) -> None:
        assert not self._make(embedding_model="").is_valid()

    # -- cache_key ---------------------------------------------------------

    def test_cache_key_no_slashes(self) -> None:
        p = self._make(embedding_model="sentence-transformers/all-MiniLM-L6-v2")
        assert "/" not in p.cache_key()

    def test_cache_key_format(self) -> None:
        p = IndexParams(
            chunk_size=500,
            chunk_overlap=50,
            chunking_strategy="recursive",
            embedding_model="all-MiniLM-L6-v2",
        )
        assert p.cache_key() == "cs500_co50_strecursive_emall-MiniLM-L6-v2"

    def test_cache_key_deterministic(self) -> None:
        p = self._make(chunk_size=300, chunk_overlap=0, chunking_strategy="fixed")
        assert p.cache_key() == p.cache_key()

    def test_cache_key_differs_between_configs(self) -> None:
        p1 = self._make(chunk_size=300)
        p2 = self._make(chunk_size=500)
        assert p1.cache_key() != p2.cache_key()

    # -- immutability & hashing --------------------------------------------

    def test_frozen_raises_on_assignment(self) -> None:
        p = self._make()
        with pytest.raises((AttributeError, TypeError)):
            p.chunk_size = 999  # type: ignore[misc]

    def test_equal_instances_same_hash(self) -> None:
        p1 = self._make()
        p2 = self._make()
        assert p1 == p2
        assert hash(p1) == hash(p2)

    def test_usable_as_dict_key(self) -> None:
        p = self._make()
        d = {p: "cached"}
        assert d[p] == "cached"

    def test_usable_in_set(self) -> None:
        p1 = self._make()
        p2 = self._make()
        assert len({p1, p2}) == 1


# ---------------------------------------------------------------------------
# QueryParams
# ---------------------------------------------------------------------------


class TestQueryParams:
    """Validity and immutability of query-time parameters."""

    def _make(self, **kw) -> QueryParams:  # type: ignore[no-untyped-def]
        defaults = dict(
            top_k=5,
            reranking_enabled=False,
            rrf_k=60,
            dense_weight=0.7,
            query_expansion=False,
        )
        defaults.update(kw)
        return QueryParams(**defaults)

    def test_valid_basic(self) -> None:
        assert self._make().is_valid()

    def test_invalid_zero_top_k(self) -> None:
        assert not self._make(top_k=0).is_valid()

    def test_invalid_negative_top_k(self) -> None:
        assert not self._make(top_k=-1).is_valid()

    def test_invalid_zero_rrf_k(self) -> None:
        assert not self._make(rrf_k=0).is_valid()

    def test_invalid_negative_rrf_k(self) -> None:
        assert not self._make(rrf_k=-10).is_valid()

    def test_invalid_dense_weight_above_1(self) -> None:
        assert not self._make(dense_weight=1.01).is_valid()

    def test_invalid_dense_weight_below_0(self) -> None:
        assert not self._make(dense_weight=-0.001).is_valid()

    def test_boundary_dense_weight_zero(self) -> None:
        assert self._make(dense_weight=0.0).is_valid()

    def test_boundary_dense_weight_one(self) -> None:
        assert self._make(dense_weight=1.0).is_valid()

    def test_reranking_true_valid(self) -> None:
        assert self._make(reranking_enabled=True).is_valid()

    def test_query_expansion_true_valid(self) -> None:
        assert self._make(query_expansion=True).is_valid()

    def test_frozen_raises_on_assignment(self) -> None:
        p = self._make()
        with pytest.raises((AttributeError, TypeError)):
            p.top_k = 99  # type: ignore[misc]

    def test_hashable(self) -> None:
        p1 = self._make()
        p2 = self._make()
        assert hash(p1) == hash(p2)
        assert len({p1, p2}) == 1


# ---------------------------------------------------------------------------
# SearchSpace — index and query param generation
# ---------------------------------------------------------------------------


class TestSearchSpaceIndexParams:
    """Tests for all_index_params and budget_index_params."""

    def setup_method(self) -> None:
        self.space = SearchSpace()

    def test_all_index_params_all_valid(self) -> None:
        assert all(p.is_valid() for p in self.space.all_index_params())

    def test_all_index_params_count_default(self) -> None:
        # 3 sizes × 3 overlaps × 3 strategies × 1 model = 27
        # all overlaps (0, 50, 100) are < min size (300), so all 27 are valid
        params = self.space.all_index_params()
        assert len(params) == 27

    def test_all_index_params_no_duplicates(self) -> None:
        params = self.space.all_index_params()
        assert len(params) == len(set(params))

    def test_custom_axes_reflected(self) -> None:
        space = SearchSpace(chunk_sizes=(256,), chunk_overlaps=(0,), chunking_strategies=("fixed",))
        params = space.all_index_params()
        assert len(params) == 1
        assert params[0].chunk_size == 256

    def test_invalid_overlap_excluded_from_all(self) -> None:
        # If we add an overlap larger than all sizes it should be filtered
        space = SearchSpace(chunk_sizes=(100,), chunk_overlaps=(200,))
        assert len(space.all_index_params()) == 0

    def test_budget_honors_limit(self) -> None:
        assert len(self.space.budget_index_params(budget=5)) == 5

    def test_budget_zero_returns_all(self) -> None:
        all_p = self.space.all_index_params()
        budgeted = self.space.budget_index_params(budget=0)
        assert len(budgeted) == len(all_p)

    def test_budget_larger_than_total_returns_all(self) -> None:
        total = len(self.space.all_index_params())
        assert len(self.space.budget_index_params(budget=total + 100)) == total

    def test_budget_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="budget"):
            self.space.budget_index_params(budget=-1)

    def test_budget_deterministic_same_seed(self) -> None:
        a = self.space.budget_index_params(budget=8, seed=42)
        b = self.space.budget_index_params(budget=8, seed=42)
        assert a == b

    def test_budget_different_seeds_differ(self) -> None:
        a = self.space.budget_index_params(budget=8, seed=1)
        b = self.space.budget_index_params(budget=8, seed=99)
        # With 27 configs shuffled by different seeds, very unlikely to match
        assert a != b

    def test_budget_all_results_valid(self) -> None:
        params = self.space.budget_index_params(budget=10)
        assert all(p.is_valid() for p in params)


class TestSearchSpaceQueryParams:
    """Tests for all_query_params."""

    def setup_method(self) -> None:
        self.space = SearchSpace()

    def test_all_query_params_all_valid(self) -> None:
        assert all(p.is_valid() for p in self.space.all_query_params())

    def test_all_query_params_count_default(self) -> None:
        # 3 top_k × 2 reranking × 2 rrf_k × 3 dense_weights × 2 query_expansion = 72
        params = self.space.all_query_params()
        assert len(params) == 72

    def test_all_query_params_no_duplicates(self) -> None:
        params = self.space.all_query_params()
        assert len(params) == len(set(params))

    def test_custom_query_axes(self) -> None:
        space = SearchSpace(
            top_k_values=(5,),
            reranking_options=(False,),
            rrf_k_values=(60,),
            dense_weights=(0.7,),
            query_expansion_options=(False,),
        )
        assert len(space.all_query_params()) == 1

    def test_dense_weight_boundary_values_included(self) -> None:
        space = SearchSpace(dense_weights=(0.0, 1.0))
        params = space.all_query_params()
        weights = {p.dense_weight for p in params}
        assert 0.0 in weights and 1.0 in weights


# ---------------------------------------------------------------------------
# SearchSpace — iter_candidates and total_experiments
# ---------------------------------------------------------------------------


class TestSearchSpaceIterCandidates:

    def setup_method(self) -> None:
        self.space = SearchSpace()

    def test_iter_candidates_length(self) -> None:
        results = list(self.space.iter_candidates(budget=3, seed=42))
        assert len(results) == 3

    def test_iter_candidates_types(self) -> None:
        for ip, qps in self.space.iter_candidates(budget=2, seed=42):
            assert isinstance(ip, IndexParams)
            assert isinstance(qps, list)
            assert all(isinstance(qp, QueryParams) for qp in qps)

    def test_iter_candidates_index_params_valid(self) -> None:
        for ip, _ in self.space.iter_candidates(budget=5, seed=42):
            assert ip.is_valid()

    def test_iter_candidates_query_params_valid(self) -> None:
        for _, qps in self.space.iter_candidates(budget=3, seed=42):
            assert all(qp.is_valid() for qp in qps)

    def test_iter_candidates_query_set_identical_across_indexes(self) -> None:
        """Every index gets the same exhaustive QueryParams list."""
        results = list(self.space.iter_candidates(budget=3, seed=42))
        q_sets = [frozenset(qps) for _, qps in results]
        assert all(s == q_sets[0] for s in q_sets)

    def test_iter_candidates_query_params_count(self) -> None:
        n_query = len(self.space.all_query_params())
        for _, qps in self.space.iter_candidates(budget=2, seed=42):
            assert len(qps) == n_query

    def test_iter_candidates_deterministic(self) -> None:
        a = [(ip, list(qps)) for ip, qps in self.space.iter_candidates(budget=4, seed=7)]
        b = [(ip, list(qps)) for ip, qps in self.space.iter_candidates(budget=4, seed=7)]
        assert a == b

    def test_total_experiments_formula(self) -> None:
        budget = 3
        total = self.space.total_experiments(budget=budget, seed=42)
        n_query = len(self.space.all_query_params())
        assert total == budget * n_query

    def test_total_experiments_zero_budget_returns_all(self) -> None:
        n_index = len(self.space.all_index_params())
        n_query = len(self.space.all_query_params())
        assert self.space.total_experiments(budget=0) == n_index * n_query


# ---------------------------------------------------------------------------
# Embedding allowlist and --sweep-embeddings integration
# ---------------------------------------------------------------------------


class TestEmbeddingAllowlist:

    def test_allowlist_has_at_least_one_entry(self) -> None:
        assert len(EMBEDDING_ALLOWLIST) >= 1

    def test_default_model_in_allowlist(self) -> None:
        assert "all-MiniLM-L6-v2" in EMBEDDING_ALLOWLIST

    def test_allowlist_has_exactly_three_entries(self) -> None:
        assert len(EMBEDDING_ALLOWLIST) == 3

    def test_sweep_embeddings_expands_index_space(self) -> None:
        default_space = SearchSpace()
        sweep_space = SearchSpace(embedding_models=EMBEDDING_ALLOWLIST)
        # With 3 models instead of 1, index space should be 3× larger
        default_count = len(default_space.all_index_params())
        sweep_count = len(sweep_space.all_index_params())
        assert sweep_count == default_count * len(EMBEDDING_ALLOWLIST)

    def test_models_in_index_params_match_allowlist(self) -> None:
        space = SearchSpace(embedding_models=EMBEDDING_ALLOWLIST)
        models_seen = {p.embedding_model for p in space.all_index_params()}
        assert models_seen == set(EMBEDDING_ALLOWLIST)


# ---------------------------------------------------------------------------
# CandidateConfig
# ---------------------------------------------------------------------------


class TestCandidateConfig:

    def _make_ip(self) -> IndexParams:
        return IndexParams(
            chunk_size=500,
            chunk_overlap=50,
            chunking_strategy="recursive",
            embedding_model="all-MiniLM-L6-v2",
        )

    def _make_qp(self) -> QueryParams:
        return QueryParams(
            top_k=5,
            reranking_enabled=False,
            rrf_k=60,
            dense_weight=0.7,
            query_expansion=False,
        )

    def test_is_valid_when_both_valid(self) -> None:
        cfg = CandidateConfig(self._make_ip(), self._make_qp())
        assert cfg.is_valid()

    def test_invalid_when_index_invalid(self) -> None:
        bad_ip = IndexParams(chunk_size=0, chunk_overlap=0, chunking_strategy="recursive", embedding_model="m")
        cfg = CandidateConfig(bad_ip, self._make_qp())
        assert not cfg.is_valid()

    def test_invalid_when_query_invalid(self) -> None:
        bad_qp = QueryParams(top_k=0, reranking_enabled=False, rrf_k=60, dense_weight=0.7, query_expansion=False)
        cfg = CandidateConfig(self._make_ip(), bad_qp)
        assert not cfg.is_valid()

    def test_to_appconfig_dict_contains_appconfig_keys(self) -> None:
        cfg = CandidateConfig(self._make_ip(), self._make_qp())
        d = cfg.to_appconfig_dict()
        expected_keys = {
            "embedding_model", "chunk_size", "chunk_overlap",
            "chunking_strategy", "top_k", "reranking_enabled",
        }
        assert expected_keys == set(d.keys())

    def test_to_appconfig_dict_values(self) -> None:
        cfg = CandidateConfig(self._make_ip(), self._make_qp())
        d = cfg.to_appconfig_dict()
        assert d["chunk_size"] == 500
        assert d["chunk_overlap"] == 50
        assert d["top_k"] == 5
        assert d["reranking_enabled"] is False
        assert d["embedding_model"] == "all-MiniLM-L6-v2"

    def test_to_full_dict_includes_autotune_extras(self) -> None:
        cfg = CandidateConfig(self._make_ip(), self._make_qp())
        d = cfg.to_full_dict()
        assert "autotune_rrf_k" in d
        assert "autotune_dense_weight" in d
        assert "autotune_query_expansion" in d
        assert d["autotune_rrf_k"] == 60
        assert d["autotune_dense_weight"] == 0.7
        assert d["autotune_query_expansion"] is False

    def test_frozen(self) -> None:
        cfg = CandidateConfig(self._make_ip(), self._make_qp())
        with pytest.raises((AttributeError, TypeError)):
            cfg.index_params = self._make_ip()  # type: ignore[misc]

    def test_hashable(self) -> None:
        cfg1 = CandidateConfig(self._make_ip(), self._make_qp())
        cfg2 = CandidateConfig(self._make_ip(), self._make_qp())
        assert cfg1 == cfg2
        assert hash(cfg1) == hash(cfg2)
        assert len({cfg1, cfg2}) == 1
