"""
Ablation Study Framework for AgenticRAG:
Isolates the empirical contribution of individual tools and architectural choices:
1. Full AgenticRAG
2. Ablation: Without FIND tool
3. Ablation: Without OPEN tool
4. Ablation: Without SUMMARIZE tool
5. Ablation: Single-query SEARCH only (no multi-query formulation)
6. Ablation: Step budget variation (1, 3, 5 steps)
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

from knowledge.agentic.controller import AgenticController
from knowledge.agentic.schemas import FindResult, OpenResult, SummaryResult


class AblationRunner:
    """
    Executes controlled ablation experiments by disabling specific agentic tools
    or constraining search behaviors, measuring the delta in retrieval quality.
    """

    def __init__(self, benchmark_path: str = "knowledge/evaluation/enterprise_benchmark.json") -> None:
        self.benchmark_path = Path(benchmark_path)
        with open(self.benchmark_path, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)

    def _create_controller(self, ablation_type: str, max_steps: int = 4) -> AgenticController:
        controller = AgenticController()

        if ablation_type == "no_find":
            # Mock FIND tool to always return zero matches
            mock_find = MagicMock()
            mock_find.execute.return_value = FindResult(
                document_id="", expression="", matches=[], total_matches=0, latency_ms=1.0
            )
            controller.find_tool = mock_find

        elif ablation_type == "no_open":
            # Mock OPEN tool to return empty views
            mock_open = MagicMock()
            mock_open.execute.return_value = OpenResult(
                document_id="", source_path="", views=[], total_chars=0, evidence_records=[], latency_ms=1.0
            )
            controller.open_tool = mock_open

        elif ablation_type == "no_summarize":
            # Mock SUMMARIZE tool to return empty summary
            mock_sum = MagicMock()
            mock_sum.execute.return_value = SummaryResult(
                summary_text="", source_references=[], evidence_ids_used=[], token_estimate=0, is_extractive=True, latency_ms=1.0
            )
            controller.summarize_tool = mock_sum

        elif ablation_type == "single_query_only":
            # Override plan_and_route to issue only 1 query
            orig_plan = controller.plan_and_route
            def single_q_plan(state):
                state = orig_plan(state)
                queries = state["current_action"]["args"].get("queries", [state["question"]])
                state["current_action"]["args"]["queries"] = [queries[0]]
                return state
            controller.plan_and_route = single_q_plan

        return controller

    def run_experiment(
        self,
        experiment_name: str,
        ablation_type: str,
        max_steps: int = 4,
        sample_size: int = 15,
    ) -> Dict[str, Any]:
        """Execute one ablation variant across a test slice."""
        controller = self._create_controller(ablation_type, max_steps=max_steps)
        test_items = self.dataset[:sample_size]

        hits = []
        latencies = []
        steps_list = []
        abstention_scores = []

        t_start = time.perf_counter()

        for item in test_items:
            q = item["question"]
            expected_docs = [d.lower() for d in item.get("expected_documents", [])]
            is_unanswerable = item.get("is_unanswerable", False)

            t0 = time.perf_counter()
            res = controller.run(q, max_steps=max_steps)
            lat = (time.perf_counter() - t0) * 1000

            retrieved = [d.lower() for d in res.get("discovered_docs", [])]
            abstained = res.get("abstained", False)

            if expected_docs:
                hit = 1.0 if any(any(ed in rd for rd in retrieved) for ed in expected_docs) else 0.0
            else:
                hit = 1.0 if is_unanswerable and abstained else 0.0

            abstention_score = 1.0 if (is_unanswerable == abstained) else 0.0

            hits.append(hit)
            latencies.append(lat)
            steps_list.append(res.get("steps", 1))
            abstention_scores.append(abstention_score)

        total_duration = time.perf_counter() - t_start
        n = len(test_items)

        return {
            "experiment": experiment_name,
            "ablation_type": ablation_type,
            "max_steps": max_steps,
            "samples_evaluated": n,
            "hit_rate": round(sum(hits) / n, 4) if n else 0.0,
            "abstention_accuracy": round(sum(abstention_scores) / n, 4) if n else 0.0,
            "avg_latency_ms": round(sum(latencies) / n, 2) if n else 0.0,
            "avg_steps": round(sum(steps_list) / n, 2) if n else 0.0,
            "total_experiment_sec": round(total_duration, 2),
        }

    def run_all_ablations(self, sample_size: int = 15) -> Dict[str, Any]:
        """Run the full suite of ablation experiments."""
        experiments = [
            ("Full AgenticRAG (Baseline C)", "none", 4),
            ("Ablation 1: Without FIND Tool", "no_find", 4),
            ("Ablation 2: Without OPEN Tool", "no_open", 4),
            ("Ablation 3: Without SUMMARIZE Tool", "no_summarize", 4),
            ("Ablation 4: Single-Query SEARCH Only", "single_query_only", 4),
            ("Ablation 5: Budget = 1 Step", "none", 1),
            ("Ablation 6: Budget = 2 Steps", "none", 2),
        ]

        results = []
        print(f"\n=== Executing AgenticRAG Ablation Studies ({sample_size} queries/variant) ===")
        for name, ab_type, steps in experiments:
            print(f"Running: {name}...")
            res = self.run_experiment(name, ab_type, max_steps=steps, sample_size=sample_size)
            results.append(res)
            print(f"  -> Hit Rate: {res['hit_rate']*100:.1f}%, Avg Latency: {res['avg_latency_ms']:.1f} ms, Avg Steps: {res['avg_steps']}")

        return {"ablations": results}


if __name__ == "__main__":
    runner = AblationRunner()
    results = runner.run_all_ablations(sample_size=10)
    out_file = Path("knowledge/evaluation/ablation_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nAblation results saved to {out_file}")
