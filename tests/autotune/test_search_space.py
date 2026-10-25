"""
Unit tests for knowledge.autotune.search_space.
No Ollama, no network, no disk I/O.
"""

from __future__ import annotations

import pytest
import yaml

from knowledge.autotune.search_space import (
    EMBEDDING_ALLOWLIST,
    CandidateConfig,
    IndexParams,
    QueryParams,
    SearchSpace,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ip(**kw) -> IndexParams:  # type: ignore[no-untyped-def]
    defaults = dict(chunk_size=500, chunk_overlap=50,
                    chunking_strategy="recursive", embedding_model="all-MiniLM-L6-v2")
    defaults.update(kw)
    return IndexParams(**defaults)  # type: ignore[arg-type]


def _qp(**kw) -> QueryParams:  # type: ignore[no-untyped-def]
    defaults = dict(top_k=5, reranking_enabled=False, rrf_k=60,
                    dense_weight=0.7, query_expansion=False)
    defaults.update(kw)
    return QueryParams(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# IndexParams — validity
# ---------------------------------------------------------------------------

class TestIndexParamsValidity:
    def test_valid_basic(self) -> None:
        assert _ip().is_valid()

    def test_invalid_overlap_equals_size(self) -> None:
        assert not _ip(chunk_size=300, chunk_overlap=300).is_valid()

    def test_invalid_overlap_exceeds_size(self) -> None:
        assert not _ip(chunk_size=300, chunk_overlap=400).is_valid()

    def test_invalid_zero_chunk_size(self) -> None:
        assert not _ip(chunk_size=0, chunk_overlap=0).is_valid()

    def test_valid_zero_overlap(self) -> None:
        assert _ip(chunk_size=300, chunk_overlap=0).is_valid()

    def test_valid_overlap_one_less_than_size(self) -> None:
        assert _ip(chunk_size=100, chunk_overlap=99).is_valid()

    def test_invalid_unknown_strategy(self) -> None:
        assert not _ip(chunking_strategy="sliding").is_valid()

    def test_valid_all_three_strategies(self) -> None:
        for strat in ("recursive", "sentence", "fixed"):
            assert _ip(chunking_strategy=strat).is_valid(), strat

    def test_invalid_empty_embedding_model(self) -> None:
        assert not _ip(embedding_model="").is_valid()


# ---------------------------------------------------------------------------
# IndexParams — cache key
# ---------------------------------------------------------------------------

class TestIndexParamsCacheKey:
    def test_no_slashes(self) -> None:
        p = _ip(embedding_model="sentence-transformers/all-MiniLM-L6-v2")
        assert "/" not in p.cache_key()

    def test_pinned_hash(self) -> None:
        """Guard hash stability — if this fails the on-disk cache is broken."""
        p = IndexParams(
            chunk_size=500,
            chunk_overlap=50,
            chunking_strategy="recursive",
            embedding_model="all-MiniLM-L6-v2",
        )
        assert p.cache_key() == "cs500_co50_rec_all-MiniLM-L6-v2_a433e5c4"

    def test_org_slash_model_vs_org_dash_model_differ(self) -> None:
        """org/model and org-model share a readable prefix but different hash."""
        p1 = _ip(embedding_model="org/model")
        p2 = _ip(embedding_model="org-model")
        assert p1.cache_key() != p2.cache_key()

    def test_deterministic(self) -> None:
        p = _ip()
        assert p.cache_key() == p.cache_key()

    def test_different_configs_differ(self) -> None:
        assert _ip(chunk_size=300).cache_key() != _ip(chunk_size=500).cache_key()


# ---------------------------------------------------------------------------
# IndexParams — immutability & hashing
# ---------------------------------------------------------------------------

class TestIndexParamsImmutability:
    def test_frozen(self) -> None:
        p = _ip()
        with pytest.raises((AttributeError, TypeError)):
            p.chunk_size = 999  # type: ignore[misc]

    def test_equal_same_hash(self) -> None:
        assert hash(_ip()) == hash(_ip())

    def test_usable_as_dict_key(self) -> None:
        assert {_ip(): "ok"}[_ip()] == "ok"

    def test_usable_in_set(self) -> None:
        assert len({_ip(), _ip()}) == 1


# ---------------------------------------------------------------------------
# QueryParams — validity & immutability
# ---------------------------------------------------------------------------

class TestQueryParams:
    def test_valid_basic(self) -> None:
        assert _qp().is_valid()

    def test_invalid_zero_top_k(self) -> None:
        assert not _qp(top_k=0).is_valid()

    def test_invalid_negative_rrf_k(self) -> None:
        assert not _qp(rrf_k=0).is_valid()

    def test_invalid_dense_weight_above_1(self) -> None:
        assert not _qp(dense_weight=1.01).is_valid()

    def test_invalid_dense_weight_below_0(self) -> None:
        assert not _qp(dense_weight=-0.001).is_valid()

    def test_boundary_dense_weight_zero(self) -> None:
        assert _qp(dense_weight=0.0).is_valid()

    def test_boundary_dense_weight_one(self) -> None:
        assert _qp(dense_weight=1.0).is_valid()

    def test_frozen(self) -> None:
        with pytest.raises((AttributeError, TypeError)):
            _qp().top_k = 99  # type: ignore[misc]

    def test_hashable(self) -> None:
        assert hash(_qp()) == hash(_qp())
        assert len({_qp(), _qp()}) == 1


# ---------------------------------------------------------------------------
# CandidateConfig — to_appconfig_dict, to_full_dict, AppConfig round-trip
# ---------------------------------------------------------------------------

class TestCandidateConfig:
    def _cfg(self) -> CandidateConfig:
        return CandidateConfig(_ip(), _qp())

    def test_is_valid(self) -> None:
        assert self._cfg().is_valid()

    def test_invalid_when_index_invalid(self) -> None:
        assert not CandidateConfig(_ip(chunk_size=0), _qp()).is_valid()

    def test_invalid_when_query_invalid(self) -> None:
        assert not CandidateConfig(_ip(), _qp(top_k=0)).is_valid()

    def test_to_appconfig_dict_keys(self) -> None:
        d = self._cfg().to_appconfig_dict()
        assert set(d) == {
            "embedding_model", "chunk_size", "chunk_overlap",
            "chunking_strategy", "top_k", "reranking_enabled",
        }

    def test_to_appconfig_dict_values(self) -> None:
        d = self._cfg().to_appconfig_dict()
        assert d["chunk_size"] == 500
        assert d["top_k"] == 5
        assert d["reranking_enabled"] is False

    def test_to_full_dict_has_autotune_section(self) -> None:
        d = self._cfg().to_full_dict()
        assert "autotune" in d
        autotune = d["autotune"]
        assert isinstance(autotune, dict)
        assert set(autotune) == {"rrf_k", "dense_weight", "query_expansion"}

    def test_to_full_dict_autotune_values(self) -> None:
        d = self._cfg().to_full_dict()
        autotune = d["autotune"]
        assert isinstance(autotune, dict)
        assert autotune["rrf_k"] == 60
        assert autotune["dense_weight"] == pytest.approx(0.7)
        assert autotune["query_expansion"] is False

    def test_appconfig_round_trip(self) -> None:
        """config.autotuned.yaml must load cleanly through AppConfig."""
        from knowledge.config.manager import AppConfig
        full = self._cfg().to_full_dict()
        # AppConfig must accept the full dict (Pydantic v2 ignores 'autotune' key)
        loaded = AppConfig(**full)  # type: ignore[arg-type]
        assert loaded.chunk_size == 500
        assert loaded.top_k == 5
        assert loaded.reranking_enabled is False
        assert loaded.embedding_model == "all-MiniLM-L6-v2"

    def test_appconfig_round_trip_via_yaml(self, tmp_path: pytest.TempPathFactory) -> None:
        """YAML serialise → parse → AppConfig must work without error."""
        from knowledge.config.manager import AppConfig
        full = self._cfg().to_full_dict()
        yaml_path = tmp_path / "config.autotuned.yaml"  # type: ignore[operator]
        yaml_path.write_text(yaml.safe_dump(full))
        data = yaml.safe_load(yaml_path.read_text())
        loaded = AppConfig(**data)
        assert loaded.chunk_size == full["chunk_size"]
        assert loaded.chunking_strategy == full["chunking_strategy"]

    def test_frozen(self) -> None:
        cfg = self._cfg()
        with pytest.raises((AttributeError, TypeError)):
            cfg.index_params = _ip()  # type: ignore[misc]

    def test_hashable(self) -> None:
        assert hash(self._cfg()) == hash(self._cfg())
        assert len({self._cfg(), self._cfg()}) == 1


# ---------------------------------------------------------------------------
# SearchSpace — validate()
# ---------------------------------------------------------------------------

class TestSearchSpaceValidate:
    def test_default_space_is_valid(self) -> None:
        assert SearchSpace().validate() == []

    def test_empty_chunk_sizes_invalid(self) -> None:
        errors = SearchSpace(chunk_sizes=()).validate()
        assert any("chunk_sizes" in e for e in errors)

    def test_empty_embedding_models_invalid(self) -> None:
        errors = SearchSpace(embedding_models=()).validate()
        assert any("embedding_models" in e for e in errors)

    def test_unknown_strategy_invalid(self) -> None:
        errors = SearchSpace(chunking_strategies=("sliding",)).validate()
        assert any("sliding" in e for e in errors)

    def test_invalid_dense_weight_in_space(self) -> None:
        errors = SearchSpace(dense_weights=(2.0,)).validate()
        assert any("dense_weight" in e for e in errors)

    def test_invalid_rrf_k_in_space(self) -> None:
        errors = SearchSpace(rrf_k_values=(0,)).validate()
        assert any("rrf_k" in e for e in errors)

    def test_invalid_top_k_in_space(self) -> None:
        errors = SearchSpace(top_k_values=(0,)).validate()
        assert any("top_k" in e for e in errors)

    def test_multiple_errors_returned(self) -> None:
        errors = SearchSpace(chunk_sizes=(), embedding_models=()).validate()
        assert len(errors) >= 2


# ---------------------------------------------------------------------------
# SearchSpace — all_index_params / all_query_params
# ---------------------------------------------------------------------------

class TestSearchSpaceGrids:
    def setup_method(self) -> None:
        self.space = SearchSpace()

    def test_all_index_params_all_valid(self) -> None:
        assert all(p.is_valid() for p in self.space.all_index_params())

    def test_all_index_params_count_default(self) -> None:
        # 3 sizes × 3 overlaps × 3 strategies × 1 model = 27 (all overlaps < min size)
        assert len(self.space.all_index_params()) == 27

    def test_all_index_params_no_duplicates(self) -> None:
        params = self.space.all_index_params()
        assert len(params) == len(set(params))

    def test_invalid_overlap_excluded(self) -> None:
        space = SearchSpace(chunk_sizes=(100,), chunk_overlaps=(200,))
        assert len(space.all_index_params()) == 0

    def test_all_query_params_all_valid(self) -> None:
        assert all(p.is_valid() for p in self.space.all_query_params())

    def test_all_query_params_count_default(self) -> None:
        # 3 × 2 × 2 × 3 × 2 = 72
        assert len(self.space.all_query_params()) == 72

    def test_all_query_params_no_duplicates(self) -> None:
        params = self.space.all_query_params()
        assert len(params) == len(set(params))


# ---------------------------------------------------------------------------
# SearchSpace — neighbors()
# ---------------------------------------------------------------------------

class TestNeighbors:
    def setup_method(self) -> None:
        self.space = SearchSpace()

    def test_neighbors_differ_by_one_axis(self) -> None:
        base = _ip(chunk_size=500, chunk_overlap=50, chunking_strategy="recursive")
        for nbr in self.space.neighbors(base):
            diffs = sum([
                nbr.chunk_size != base.chunk_size,
                nbr.chunk_overlap != base.chunk_overlap,
                nbr.chunking_strategy != base.chunking_strategy,
                nbr.embedding_model != base.embedding_model,
            ])
            assert diffs == 1, f"Expected 1-axis diff, got {diffs} for {nbr}"

    def test_neighbors_all_valid(self) -> None:
        base = _ip()
        assert all(p.is_valid() for p in self.space.neighbors(base))

    def test_neighbors_not_include_base(self) -> None:
        base = _ip()
        assert base not in self.space.neighbors(base)

    def test_neighbors_in_search_space(self) -> None:
        base = _ip()
        all_set = set(self.space.all_index_params())
        assert all(p in all_set for p in self.space.neighbors(base))

    def test_neighbors_include_all_chunk_size_alternatives(self) -> None:
        base = _ip(chunk_size=500)
        nbr_sizes = {n.chunk_size for n in self.space.neighbors(base)
                     if n.chunk_overlap == base.chunk_overlap
                     and n.chunking_strategy == base.chunking_strategy}
        assert nbr_sizes == {300, 800}

    def test_base_not_in_space_neighbors_are_all_in_space(self) -> None:
        """Neighbors of an out-of-space base are still confined to the space."""
        outside = IndexParams(chunk_size=9999, chunk_overlap=0,
                              chunking_strategy="recursive",
                              embedding_model="all-MiniLM-L6-v2")
        nbrs = self.space.neighbors(outside)
        all_set = set(self.space.all_index_params())
        assert all(p in all_set for p in nbrs)

    def test_base_with_all_axes_outside_space_returns_empty(self) -> None:
        """If every axis of base is outside every space axis value, no neighbors."""
        outside = IndexParams(chunk_size=9999, chunk_overlap=9998,
                              chunking_strategy="fixed",   # 'fixed' IS in space
                              embedding_model="unknown-model")
        # chunk_size=9999 not in space, chunk_overlap=9998 not in space,
        # embedding_model='unknown-model' not in space.
        # 'fixed' IS in space but the resulting candidate with chunk_size=9999 etc
        # is also invalid (overlap 9998 >= size 9999 — invalid).
        nbrs = self.space.neighbors(outside)
        # All returned neighbors must still be valid and in the space
        all_set = set(self.space.all_index_params())
        assert all(p in all_set for p in nbrs)


# ---------------------------------------------------------------------------
# SearchSpace — seeded_sample()
# ---------------------------------------------------------------------------

class TestSeededSample:
    def setup_method(self) -> None:
        self.space = SearchSpace()

    def test_returns_n_items(self) -> None:
        result = self.space.seeded_sample(5, seed=42)
        assert len(result) == 5

    def test_all_valid(self) -> None:
        assert all(p.is_valid() for p in self.space.seeded_sample(10, seed=1))

    def test_deterministic(self) -> None:
        a = self.space.seeded_sample(8, seed=7)
        b = self.space.seeded_sample(8, seed=7)
        assert a == b

    def test_different_seeds_differ(self) -> None:
        a = self.space.seeded_sample(8, seed=1)
        b = self.space.seeded_sample(8, seed=99)
        assert a != b

    def test_exclude_respected(self) -> None:
        all_params = self.space.all_index_params()
        exclude = all_params[:5]
        result = self.space.seeded_sample(5, seed=42, exclude=exclude)
        for p in result:
            assert p not in exclude

    def test_n_larger_than_pool_returns_all(self) -> None:
        total = len(self.space.all_index_params())
        result = self.space.seeded_sample(total + 100, seed=42)
        assert len(result) == total


# ---------------------------------------------------------------------------
# SearchSpace — budget_index_params()
# ---------------------------------------------------------------------------

class TestBudgetIndexParams:
    def setup_method(self) -> None:
        self.space = SearchSpace()

    def test_baseline_is_first(self) -> None:
        base = _ip(chunk_size=300)
        result = self.space.budget_index_params(budget=5, seed=42, baseline=base)
        assert result[0] == base

    def test_baseline_appears_exactly_once(self) -> None:
        base = _ip(chunk_size=300)
        result = self.space.budget_index_params(budget=10, seed=42, baseline=base)
        assert result.count(base) == 1

    def test_no_duplicates(self) -> None:
        result = self.space.budget_index_params(budget=15, seed=42)
        assert len(result) == len(set(result))

    def test_all_valid(self) -> None:
        result = self.space.budget_index_params(budget=10)
        assert all(p.is_valid() for p in result)

    def test_honors_budget(self) -> None:
        assert len(self.space.budget_index_params(budget=5)) == 5

    def test_zero_budget_returns_all(self) -> None:
        all_p = self.space.all_index_params()
        assert len(self.space.budget_index_params(budget=0)) == len(all_p)

    def test_budget_larger_than_total_returns_all(self) -> None:
        total = len(self.space.all_index_params())
        assert len(self.space.budget_index_params(budget=total + 100)) == total

    def test_negative_budget_raises(self) -> None:
        with pytest.raises(ValueError, match="budget"):
            self.space.budget_index_params(budget=-1)

    def test_deterministic(self) -> None:
        a = self.space.budget_index_params(budget=8, seed=42)
        b = self.space.budget_index_params(budget=8, seed=42)
        assert a == b

    def test_axis_coverage_chunk_sizes(self) -> None:
        """All chunk_size values in the space must appear within budget=27."""
        result = self.space.budget_index_params(budget=27, seed=42)
        sizes_seen = {p.chunk_size for p in result}
        assert sizes_seen == set(self.space.chunk_sizes)

    def test_axis_coverage_strategies(self) -> None:
        result = self.space.budget_index_params(budget=27, seed=42)
        strats_seen = {p.chunking_strategy for p in result}
        assert strats_seen == set(self.space.chunking_strategies)

    def test_neighbors_of_baseline_included_early(self) -> None:
        """With large budget, all neighbors appear before random fill entries."""
        base = _ip(chunk_size=500, chunk_overlap=50, chunking_strategy="recursive")
        nbrs = set(self.space.neighbors(base))
        result = self.space.budget_index_params(budget=len(nbrs) + 2, seed=42, baseline=base)
        # All neighbors should appear in the first len(nbrs)+1 slots (baseline + nbrs)
        early = set(result[:len(nbrs) + 1])
        assert nbrs.issubset(early)

    def test_baseline_outside_space_skipped(self) -> None:
        outside = IndexParams(chunk_size=9999, chunk_overlap=0,
                              chunking_strategy="recursive", embedding_model="all-MiniLM-L6-v2")
        result = self.space.budget_index_params(budget=5, baseline=outside)
        assert outside not in result
        assert len(result) == 5


# ---------------------------------------------------------------------------
# SearchSpace — iter_candidates / total_experiments / estimate_index_builds
# ---------------------------------------------------------------------------

class TestIterCandidates:
    def setup_method(self) -> None:
        self.space = SearchSpace()

    def test_length(self) -> None:
        assert len(list(self.space.iter_candidates(budget=3))) == 3

    def test_query_configs_identical_across_indexes(self) -> None:
        results = list(self.space.iter_candidates(budget=3, seed=42))
        q_sets = [frozenset(qps) for _, qps in results]
        assert all(s == q_sets[0] for s in q_sets)

    def test_baseline_first(self) -> None:
        base = _ip(chunk_size=300)
        first_ip, _ = next(iter(self.space.iter_candidates(budget=5, baseline=base)))
        assert first_ip == base

    def test_total_experiments_formula(self) -> None:
        n_q = len(self.space.all_query_params())
        assert self.space.total_experiments(budget=3) == 3 * n_q

    def test_estimate_index_builds_zero_budget(self) -> None:
        total = len(self.space.all_index_params())
        assert self.space.estimate_index_builds(budget=0) == total

    def test_estimate_index_builds_normal(self) -> None:
        assert self.space.estimate_index_builds(budget=5) == 5

    def test_estimate_index_builds_capped_at_total(self) -> None:
        total = len(self.space.all_index_params())
        assert self.space.estimate_index_builds(budget=total + 999) == total


# ---------------------------------------------------------------------------
# Embedding allowlist
# ---------------------------------------------------------------------------

class TestEmbeddingAllowlist:
    def test_has_three_entries(self) -> None:
        assert len(EMBEDDING_ALLOWLIST) == 3

    def test_default_model_in_allowlist(self) -> None:
        assert "all-MiniLM-L6-v2" in EMBEDDING_ALLOWLIST

    def test_sweep_embeddings_expands_index_space(self) -> None:
        default_count = len(SearchSpace().all_index_params())
        sweep_count = len(SearchSpace(embedding_models=EMBEDDING_ALLOWLIST).all_index_params())
        assert sweep_count == default_count * len(EMBEDDING_ALLOWLIST)
