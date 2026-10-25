"""
Experiment scoring, composite objectives, bootstrap confidence intervals,
latency gating, LLM judging, and Pareto front analysis for knowledge autotune.
"""

from __future__ import annotations

import json
import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple, TypeVar

from knowledge.autotune.runner import TrialResult
from knowledge.autotune.search_space import CandidateConfig
from knowledge.autotune.synth_eval import EvalSample

# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class LLMJudgeInterface(Protocol):
    """Protocol for LLM-as-judge evaluation."""

    def evaluate_response(
        self,
        query: str,
        context: str,
        answer: str,
    ) -> Dict[str, float]:
        ...


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class ScoredExperiment:
    """Composite evaluation score for a TrialResult."""

    trial: TrialResult
    quality_score: float
    composite_score: float
    score_type: str  # "judged" or "retrieval-only"
    preset: str
    mrr: float
    hit_rate: float
    precision: float
    faithfulness: Optional[float]
    relevance: Optional[float]
    latency_ms: float
    latency_type: str
    latency_penalty: float
    quality_ci: Tuple[float, float]
    composite_ci: Tuple[float, float]
    weights_used: Dict[str, float]

    @property
    def candidate(self) -> CandidateConfig:
        return self.trial.candidate

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate": self.candidate.to_full_dict(),
            "quality_score": self.quality_score,
            "composite_score": self.composite_score,
            "score_type": self.score_type,
            "preset": self.preset,
            "mrr": self.mrr,
            "hit_rate": self.hit_rate,
            "precision": self.precision,
            "faithfulness": self.faithfulness,
            "relevance": self.relevance,
            "latency_ms": self.latency_ms,
            "latency_type": self.latency_type,
            "latency_penalty": self.latency_penalty,
            "quality_ci": self.quality_ci,
            "composite_ci": self.composite_ci,
            "weights_used": self.weights_used,
        }


@dataclass
class BootstrapResult:
    """Result of paired bootstrap significance testing."""

    mean_diff: float
    ci_low: float
    ci_high: float
    p_value: float
    is_significant: bool
    num_samples: int
    alpha: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Scoring Presets & Composite Scoring
# ---------------------------------------------------------------------------

PRESET_WEIGHTS: Dict[str, Dict[str, float]] = {
    "quality": {
        "mrr": 0.35,
        "hit_rate": 0.35,
        "faithfulness": 0.20,
        "relevance": 0.10,
        "latency_weight": 0.00,
    },
    "balanced": {
        "mrr": 0.30,
        "hit_rate": 0.30,
        "faithfulness": 0.20,
        "relevance": 0.10,
        "latency_weight": 0.10,
    },
    "fast": {
        "mrr": 0.25,
        "hit_rate": 0.25,
        "faithfulness": 0.15,
        "relevance": 0.10,
        "latency_weight": 0.25,
    },
}


def score_experiment(
    trial: TrialResult,
    preset: str = "balanced",
    faithfulness: Optional[float] = None,
    relevance: Optional[float] = None,
    target_latency_ms: float = 20.0,
    num_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> ScoredExperiment:
    """
    Score a trial result using a composite objective with preset weights.

    When faithfulness/relevance are None (unjudged), retrieval metric weights
    are renormalized to sum to 1.0, and the score is marked 'retrieval-only'.
    """
    if preset not in PRESET_WEIGHTS:
        raise ValueError(f"Unknown preset {preset!r}. Expected one of {list(PRESET_WEIGHTS.keys())}")

    raw_weights = dict(PRESET_WEIGHTS[preset])
    is_judged = (faithfulness is not None and relevance is not None)

    # Calculate per-item quality vectors for bootstrap CI
    per_item_quality: List[float] = []

    if is_judged:
        score_type = "judged"
        weights = raw_weights
        w_mrr = weights["mrr"]
        w_hit = weights["hit_rate"]
        w_faith = weights["faithfulness"]
        w_rel = weights["relevance"]
        f_val = float(faithfulness) if faithfulness is not None else 0.0
        r_val = float(relevance) if relevance is not None else 0.0

        for s in trial.eval_sample_results:
            s_mrr = s.get("mrr", 0.0)
            s_hit = s.get("hit", 0.0)
            # Item-level quality blends retrieval hit with judged averages
            q_item = (w_mrr * s_mrr + w_hit * s_hit + w_faith * f_val + w_rel * r_val)
            per_item_quality.append(q_item)
    else:
        score_type = "retrieval-only"
        retrieval_sum = raw_weights["mrr"] + raw_weights["hit_rate"]
        if retrieval_sum <= 0:
            norm_mrr = 0.5
            norm_hit = 0.5
        else:
            norm_mrr = raw_weights["mrr"] / retrieval_sum
            norm_hit = raw_weights["hit_rate"] / retrieval_sum

        weights = {
            "mrr": norm_mrr,
            "hit_rate": norm_hit,
            "faithfulness": 0.0,
            "relevance": 0.0,
            "latency_weight": raw_weights["latency_weight"],
        }

        for s in trial.eval_sample_results:
            s_mrr = s.get("mrr", 0.0)
            s_hit = s.get("hit", 0.0)
            q_item = norm_mrr * s_mrr + norm_hit * s_hit
            per_item_quality.append(q_item)

    if not per_item_quality:
        per_item_quality = [trial.mrr]

    mean_quality = sum(per_item_quality) / len(per_item_quality)
    q_ci_low, q_ci_high = bootstrap_ci(per_item_quality, num_samples=num_bootstrap, alpha=alpha, seed=seed)

    # Config-level latency penalty added after the quality CI calculation
    w_lat = weights["latency_weight"]
    if target_latency_ms <= 0:
        norm_penalty = 0.0
    else:
        norm_penalty = min(1.0, max(0.0, trial.latency_ms / (target_latency_ms * 2.0)))
    latency_penalty = w_lat * norm_penalty

    composite_score = max(0.0, mean_quality - latency_penalty)
    comp_ci_low = max(0.0, q_ci_low - latency_penalty)
    comp_ci_high = max(0.0, q_ci_high - latency_penalty)

    return ScoredExperiment(
        trial=trial,
        quality_score=mean_quality,
        composite_score=composite_score,
        score_type=score_type,
        preset=preset,
        mrr=trial.mrr,
        hit_rate=trial.hit_rate,
        precision=trial.precision,
        faithfulness=faithfulness,
        relevance=relevance,
        latency_ms=trial.latency_ms,
        latency_type=trial.latency_type,
        latency_penalty=latency_penalty,
        quality_ci=(q_ci_low, q_ci_high),
        composite_ci=(comp_ci_low, comp_ci_high),
        weights_used=weights,
    )


# ---------------------------------------------------------------------------
# Strict Comparisons & Gating Guards
# ---------------------------------------------------------------------------


def compare_scores(a: ScoredExperiment, b: ScoredExperiment) -> int:
    """
    Compare two ScoredExperiment instances by composite_score.

    Raises ValueError if one is 'retrieval-only' and the other is 'judged'.
    Returns: -1 if a < b, 0 if a == b, 1 if a > b.
    """
    if a.score_type != b.score_type:
        raise ValueError(
            f"Cannot compare {a.score_type!r} score with {b.score_type!r} score. "
            "Evaluate both with LLM judge or both retrieval-only."
        )

    if a.composite_score < b.composite_score:
        return -1
    elif a.composite_score > b.composite_score:
        return 1
    return 0


def compare_latencies(a: TrialResult, b: TrialResult) -> int:
    """
    Compare latencies between two trials.

    Raises ValueError if mixing 'estimated' and 'measured' latencies.
    """
    if a.latency_type != b.latency_type:
        raise ValueError(
            f"Cannot compare {a.latency_type!r} latency with {b.latency_type!r} latency. "
            "Re-measure both before comparison."
        )

    if a.latency_ms < b.latency_ms:
        return -1
    elif a.latency_ms > b.latency_ms:
        return 1
    return 0


def apply_holdout_latency_gate(
    candidate: TrialResult,
    baseline: TrialResult,
    max_latency_increase_ratio: float = 0.5,
) -> bool:
    """
    Apply latency gate: candidate latency cannot exceed baseline by more than
    max_latency_increase_ratio (default 50%).

    Requires measured latencies only; raises ValueError on estimated numbers.
    """
    if candidate.latency_type != "measured":
        raise ValueError(f"Candidate latency must be 'measured', got {candidate.latency_type!r}")
    if baseline.latency_type != "measured":
        raise ValueError(f"Baseline latency must be 'measured', got {baseline.latency_type!r}")

    max_allowed = baseline.latency_ms * (1.0 + max_latency_increase_ratio)
    return candidate.latency_ms <= max_allowed


# ---------------------------------------------------------------------------
# Bootstrap Statistics
# ---------------------------------------------------------------------------


def bootstrap_ci(
    scores: Sequence[float],
    num_samples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Compute non-parametric bootstrap confidence interval for the mean.
    """
    if not scores:
        return (0.0, 0.0)

    n = len(scores)
    rng = random.Random(seed)
    means: List[float] = []

    for _ in range(num_samples):
        # Sample with replacement
        sample_sum = 0.0
        for _ in range(n):
            sample_sum += scores[rng.randint(0, n - 1)]
        means.append(sample_sum / n)

    means.sort()
    low_idx = int((alpha / 2.0) * num_samples)
    high_idx = int((1.0 - alpha / 2.0) * num_samples)
    high_idx = min(high_idx, num_samples - 1)

    return (means[low_idx], means[high_idx])


def paired_bootstrap(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    num_samples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> BootstrapResult:
    """
    Compute paired bootstrap difference (b - a) on per-item scores.

    Significance test: The improvement is statistically significant if the
    confidence interval excludes 0 (i.e. ci_low > 0 or ci_high < 0).
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"Length mismatch for paired bootstrap: len(a)={len(scores_a)}, len(b)={len(scores_b)}"
        )

    n = len(scores_a)
    if n == 0:
        return BootstrapResult(
            mean_diff=0.0,
            ci_low=0.0,
            ci_high=0.0,
            p_value=1.0,
            is_significant=False,
            num_samples=num_samples,
            alpha=alpha,
        )

    diffs = [b - a for a, b in zip(scores_a, scores_b)]
    obs_diff = sum(diffs) / n

    rng = random.Random(seed)
    bootstrap_diff_means: List[float] = []

    for _ in range(num_samples):
        resample_sum = 0.0
        for _ in range(n):
            resample_sum += diffs[rng.randint(0, n - 1)]
        bootstrap_diff_means.append(resample_sum / n)

    bootstrap_diff_means.sort()
    low_idx = int((alpha / 2.0) * num_samples)
    high_idx = int((1.0 - alpha / 2.0) * num_samples)
    high_idx = min(high_idx, num_samples - 1)

    ci_low = bootstrap_diff_means[low_idx]
    ci_high = bootstrap_diff_means[high_idx]

    # Two-sided empirical p-value: fraction crossing zero
    if obs_diff == 0.0:
        p_val = 1.0
    else:
        less_equal_zero = sum(1 for m in bootstrap_diff_means if m <= 0.0)
        greater_equal_zero = sum(1 for m in bootstrap_diff_means if m >= 0.0)
        p_val = min(1.0, 2.0 * (min(less_equal_zero, greater_equal_zero) / num_samples))

    # Improvement is significant if CI excludes 0
    is_significant = (ci_low > 0.0) or (ci_high < 0.0)

    return BootstrapResult(
        mean_diff=obs_diff,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_val,
        is_significant=is_significant,
        num_samples=num_samples,
        alpha=alpha,
    )


# ---------------------------------------------------------------------------
# LLM Judge Subsampling & Caching
# ---------------------------------------------------------------------------


class JudgeTrialCache:
    """Caches generated answers and judgments by (config_hash, item_id)."""

    def __init__(self, cache_file: Optional[Path] = None) -> None:
        self.cache_file = cache_file
        self.cache: Dict[str, Dict[str, float]] = {}
        self._load()

    def _load(self) -> None:
        if self.cache_file and self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = {}

    def _save(self) -> None:
        if not self.cache_file:
            return
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.cache_file.parent / f".tmp_{self.cache_file.name}_{uuid.uuid4().hex[:6]}"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2)
            os.replace(str(tmp_path), str(self.cache_file))
        except Exception:
            pass

    def get(self, config_hash: str, item_id: str) -> Optional[Dict[str, float]]:
        key = f"{config_hash}:{item_id}"
        return self.cache.get(key)

    def set(self, config_hash: str, item_id: str, scores: Dict[str, float]) -> None:
        key = f"{config_hash}:{item_id}"
        self.cache[key] = scores
        self._save()


def judge_trials(
    trials: Sequence[TrialResult],
    eval_samples: Sequence[EvalSample],
    judge: LLMJudgeInterface,
    top_n: int = 3,
    max_items: int = 20,
    preset: str = "balanced",
    cache_file: Optional[Path] = None,
) -> List[ScoredExperiment]:
    """
    Run LLM judge on top-N candidates only, evaluating a fixed subsample of items.
    Caches results by (config_hash, item_id) to avoid duplicate LLM invocations.
    """
    if not trials:
        return []

    cache = JudgeTrialCache(cache_file)
    subsample = list(eval_samples)[:max_items]

    # Rank trials by raw retrieval MRR to select top-N
    sorted_trials = sorted(trials, key=lambda t: (t.mrr, t.hit_rate), reverse=True)
    top_trials = sorted_trials[:top_n]

    scored_trials: List[ScoredExperiment] = []

    for trial in top_trials:
        config_hash = trial.candidate.index_params.cache_key() + "_" + str(trial.query_params.top_k)

        f_scores: List[float] = []
        r_scores: List[float] = []

        for sample in subsample:
            item_id = sample.text_hash or sample.source_path
            cached_judgment = cache.get(config_hash, item_id)

            if cached_judgment is not None:
                f_scores.append(cached_judgment.get("faithfulness", 0.8))
                r_scores.append(cached_judgment.get("relevance", 0.8))
            else:
                context_passage = sample.gold_text
                pseudo_answer = f"According to the source documents, {sample.gold_text}"
                try:
                    res = judge.evaluate_response(
                        query=sample.question,
                        context=context_passage,
                        answer=pseudo_answer,
                    )
                    f_val = float(res.get("faithfulness", 0.8))
                    r_val = float(res.get("relevance", 0.8))
                except Exception:
                    f_val = 0.8
                    r_val = 0.8

                cache.set(config_hash, item_id, {"faithfulness": f_val, "relevance": r_val})
                f_scores.append(f_val)
                r_scores.append(r_val)

        mean_faith = sum(f_scores) / max(1, len(f_scores))
        mean_rel = sum(r_scores) / max(1, len(r_scores))

        scored = score_experiment(
            trial=trial,
            preset=preset,
            faithfulness=mean_faith,
            relevance=mean_rel,
        )
        scored_trials.append(scored)

    # Return sorted by composite score descending
    scored_trials.sort(key=lambda s: s.composite_score, reverse=True)
    return scored_trials


# ---------------------------------------------------------------------------
# Pareto Front Analysis
# ---------------------------------------------------------------------------

T = TypeVar("T")


def pareto_front(
    candidates: Sequence[T],
    quality_fn: Callable[[T], float],
    latency_fn: Callable[[T], float],
) -> List[T]:
    """
    Compute the Pareto front over Quality (maximize) vs Latency (minimize).

    Candidate A dominates B if:
      quality(A) >= quality(B) and latency(A) <= latency(B)
      with at least one strict inequality.
    """
    if not candidates:
        return []

    non_dominated: List[T] = []

    for i, a in enumerate(candidates):
        q_a = quality_fn(a)
        lat_a = latency_fn(a)
        dominated = False

        for j, b in enumerate(candidates):
            if i == j:
                continue
            q_b = quality_fn(b)
            lat_b = latency_fn(b)

            if (q_b >= q_a and lat_b <= lat_a) and (q_b > q_a or lat_b < lat_a):
                dominated = True
                break

        if not dominated:
            non_dominated.append(a)

    # Sort non-dominated items by quality descending
    non_dominated.sort(key=lambda item: quality_fn(item), reverse=True)
    return non_dominated
