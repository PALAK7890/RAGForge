"""
Unit tests for knowledge autotune scorer, bootstrap stats, latency guards, and Pareto front.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict

import pytest

from knowledge.autotune.runner import TrialResult
from knowledge.autotune.scorer import (
    BootstrapResult,
    apply_holdout_latency_gate,
    bootstrap_ci,
    compare_latencies,
    compare_scores,
    judge_trials,
    paired_bootstrap,
    pareto_front,
    score_experiment,
)
from knowledge.autotune.search_space import IndexParams, QueryParams
from knowledge.autotune.synth_eval import EvalSample

# ---------------------------------------------------------------------------
# Helpers & Mocks
# ---------------------------------------------------------------------------


def _dummy_trial(
    mrr: float = 0.8,
    hit_rate: float = 0.9,
    latency_ms: float = 15.0,
    latency_type: str = "estimated",
    num_samples: int = 20,
    top_k: int = 5,
) -> TrialResult:
    sample_results = [
        {"question": f"q_{i}", "mrr": mrr, "hit": hit_rate, "precision": hit_rate}
        for i in range(num_samples)
    ]
    return TrialResult(
        index_params=IndexParams(500, 50, "recursive", "mock-model"),
        query_params=QueryParams(top_k, False, 60, 0.5, False),
        mrr=mrr,
        hit_rate=hit_rate,
        precision=hit_rate,
        latency_ms=latency_ms,
        latency_type=latency_type,
        eval_sample_results=sample_results,
    )


class MockJudge:
    """Mock LLM judge for faithfulness and relevance scoring."""

    def __init__(self, faith: float = 0.85, rel: float = 0.90) -> None:
        self.faith = faith
        self.rel = rel
        self.call_count = 0

    def evaluate_response(
        self,
        query: str,
        context: str,
        answer: str,
    ) -> Dict[str, float]:
        self.call_count += 1
        return {"faithfulness": self.faith, "relevance": self.rel}


# ---------------------------------------------------------------------------
# 1. Presets, Weight Renormalization, & Score Type Guards
# ---------------------------------------------------------------------------


def test_score_experiment_retrieval_only_renormalization() -> None:
    """Verify weight renormalization when faithfulness and relevance are missing."""
    trial = _dummy_trial(mrr=0.8, hit_rate=1.0, latency_ms=20.0)
    scored = score_experiment(trial, preset="balanced")

    assert scored.score_type == "retrieval-only"
    assert scored.faithfulness is None
    assert scored.relevance is None

    # Balanced preset raw weights are mrr: 0.3, hit_rate: 0.3.
    # Renormalized: each becomes 0.5. Quality score should be 0.5*0.8 + 0.5*1.0 = 0.9.
    assert scored.quality_score == pytest.approx(0.9)
    assert scored.composite_score <= scored.quality_score


def test_score_experiment_judged() -> None:
    """Verify composite scoring with LLM judge scores included."""
    trial = _dummy_trial(mrr=0.8, hit_rate=1.0, latency_ms=10.0)
    scored = score_experiment(
        trial,
        preset="quality",
        faithfulness=0.9,
        relevance=0.95,
        target_latency_ms=20.0,
    )

    assert scored.score_type == "judged"
    assert scored.faithfulness == pytest.approx(0.9)
    assert scored.relevance == pytest.approx(0.95)
    # Quality preset has 0 latency weight
    assert scored.latency_penalty == 0.0
    assert scored.composite_score == pytest.approx(scored.quality_score)


def test_compare_scores_mixing_retrieval_and_judged_raises() -> None:
    """Verify comparing retrieval-only score with judged score raises ValueError."""
    trial = _dummy_trial()
    retrieval_only = score_experiment(trial, preset="balanced")
    judged = score_experiment(trial, preset="balanced", faithfulness=0.85, relevance=0.85)

    with pytest.raises(ValueError, match="Cannot compare 'retrieval-only' score with 'judged' score"):
        compare_scores(retrieval_only, judged)


# ---------------------------------------------------------------------------
# 2. Latency Comparison & Holdout Gate
# ---------------------------------------------------------------------------


def test_compare_latencies_mixing_estimated_and_measured_raises() -> None:
    """Verify comparing estimated with measured latency raises ValueError."""
    t_est = _dummy_trial(latency_ms=10.0, latency_type="estimated")
    t_meas = _dummy_trial(latency_ms=12.0, latency_type="measured")

    with pytest.raises(ValueError, match="Cannot compare 'estimated' latency with 'measured' latency"):
        compare_latencies(t_est, t_meas)


def test_apply_holdout_latency_gate() -> None:
    """Verify holdout latency gate enforces <= 50% increase on measured numbers."""
    baseline = _dummy_trial(latency_ms=20.0, latency_type="measured")
    fast_cand = _dummy_trial(latency_ms=25.0, latency_type="measured")
    slow_cand = _dummy_trial(latency_ms=35.0, latency_type="measured")

    # 25ms is a 25% increase -> accepted (<= 50%)
    assert apply_holdout_latency_gate(fast_cand, baseline, max_latency_increase_ratio=0.5) is True
    # 35ms is a 75% increase -> rejected (> 50%)
    assert apply_holdout_latency_gate(slow_cand, baseline, max_latency_increase_ratio=0.5) is False

    # Estimated latency must raise
    est_cand = _dummy_trial(latency_ms=22.0, latency_type="estimated")
    with pytest.raises(ValueError, match="Candidate latency must be 'measured'"):
        apply_holdout_latency_gate(est_cand, baseline)


# ---------------------------------------------------------------------------
# 3. Bootstrap CI & Paired Bootstrap Tests
# ---------------------------------------------------------------------------


def test_bootstrap_ci_width_shrinks_with_more_data() -> None:
    """Verify bootstrap confidence interval width narrows as sample size increases."""
    rng = random.Random(42)
    scores_small = [rng.gauss(0.7, 0.15) for _ in range(50)]
    scores_large = [rng.gauss(0.7, 0.15) for _ in range(1000)]

    ci_low_small, ci_high_small = bootstrap_ci(scores_small, num_samples=1000, seed=42)
    ci_low_large, ci_high_large = bootstrap_ci(scores_large, num_samples=1000, seed=42)

    width_small = ci_high_small - ci_low_small
    width_large = ci_high_large - ci_low_large

    assert width_large < width_small * 0.5, "CI width did not shrink significantly with 20x data"


def test_paired_bootstrap_detects_planted_improvement() -> None:
    """Verify paired bootstrap marks a planted improvement as statistically significant."""
    rng = random.Random(123)
    base_scores = [rng.uniform(0.4, 0.7) for _ in range(100)]
    improved_scores = [s + 0.15 for s in base_scores]  # Uniform +0.15 boost

    result: BootstrapResult = paired_bootstrap(base_scores, improved_scores, num_samples=1000, seed=42)

    assert result.mean_diff == pytest.approx(0.15, abs=0.01)
    assert result.ci_low > 0.0  # CI strictly excludes zero
    assert result.is_significant is True
    assert result.p_value < 0.05


def test_paired_bootstrap_does_not_flag_identical_runs() -> None:
    """Verify paired bootstrap does NOT flag identical runs as significant."""
    scores = [0.75 for _ in range(100)]

    result: BootstrapResult = paired_bootstrap(scores, scores, num_samples=1000, seed=42)

    assert result.mean_diff == pytest.approx(0.0)
    assert result.ci_low == pytest.approx(0.0)
    assert result.ci_high == pytest.approx(0.0)
    assert result.is_significant is False
    assert result.p_value == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 4. LLM Judge Subsampling & Caching
# ---------------------------------------------------------------------------


def test_judge_trials_subsample_and_cache(tmp_path: Path) -> None:
    """
    Verify judge_trials:
    - Only evaluates top-N trials (e.g. top 2 of 4).
    - Caps evaluation items to max_items (e.g. 3 of 10).
    - Caches judgments to disk so repeat calls make zero new judge calls.
    """
    eval_samples = [
        EvalSample(
            question=f"Question {i}?",
            source_path=f"/doc_{i}.txt",
            char_start=0,
            char_end=50,
            gold_text=f"Gold text for question {i}",
        )
        for i in range(10)
    ]

    trials = [
        _dummy_trial(mrr=0.4, top_k=3, num_samples=10),
        _dummy_trial(mrr=0.9, top_k=5, num_samples=10),
        _dummy_trial(mrr=0.7, top_k=7, num_samples=10),
        _dummy_trial(mrr=0.5, top_k=10, num_samples=10),
    ]

    cache_file = tmp_path / "judge_cache.json"
    mock_judge = MockJudge(faith=0.88, rel=0.92)

    scored = judge_trials(
        trials=trials,
        eval_samples=eval_samples,
        judge=mock_judge,
        top_n=2,
        max_items=3,
        cache_file=cache_file,
    )

    # Evaluates top-2 trials on 3 items = 6 judge calls
    assert len(scored) == 2
    assert mock_judge.call_count == 6
    assert cache_file.exists()
    for s in scored:
        assert s.score_type == "judged"
        assert s.faithfulness == pytest.approx(0.88)

    # Second call with new judge instance using same cache
    mock_judge_2 = MockJudge()
    scored_2 = judge_trials(
        trials=trials,
        eval_samples=eval_samples,
        judge=mock_judge_2,
        top_n=2,
        max_items=3,
        cache_file=cache_file,
    )
    # Zero new calls made!
    assert len(scored_2) == 2
    assert mock_judge_2.call_count == 0


# ---------------------------------------------------------------------------
# 5. Pareto Front Analysis
# ---------------------------------------------------------------------------


def test_pareto_front_hand_made_example() -> None:
    """
    Verify Pareto front correctly identifies non-dominated candidates:
      A: quality=0.90, latency=30 (Pareto: highest quality)
      B: quality=0.85, latency=15 (Pareto: balanced)
      C: quality=0.70, latency=10 (Pareto: lowest latency)
      D: quality=0.80, latency=25 (Dominated by B: 0.85 > 0.80 and 15 < 25)
      E: quality=0.70, latency=35 (Dominated by A, B, C, D)
    """
    candidates = [
        {"id": "A", "q": 0.90, "lat": 30.0},
        {"id": "B", "q": 0.85, "lat": 15.0},
        {"id": "C", "q": 0.70, "lat": 10.0},
        {"id": "D", "q": 0.80, "lat": 25.0},
        {"id": "E", "q": 0.70, "lat": 35.0},
    ]

    front = pareto_front(
        candidates,
        quality_fn=lambda c: float(c["q"]),  # type: ignore[arg-type]
        latency_fn=lambda c: float(c["lat"]),  # type: ignore[arg-type]
    )

    front_ids = [c["id"] for c in front]
    assert front_ids == ["A", "B", "C"]
    assert "D" not in front_ids
    assert "E" not in front_ids
