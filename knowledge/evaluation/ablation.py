"""Exploratory tool-response ablations, not validated research measurements.

The current heuristic controller never schedules SUMMARIZE. A no-SUMMARIZE
experiment is therefore rejected rather than presented as an effective ablation.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from knowledge.agentic.controller import AgenticController
from knowledge.agentic.schemas import FindResult, OpenResult
from knowledge.evaluation.compare_systems import SystemComparator


class _DisabledFind:
    def execute(self, payload):
        return FindResult(
            document_id=payload.document_id,
            expression=payload.expression,
            matches=[],
            total_matches=0,
            latency_ms=0,
        )


class _DisabledOpen:
    def execute(self, payload):
        return OpenResult(
            document_id=payload.document_id,
            source_path="",
            views=[],
            total_chars=0,
            evidence_records=[],
            latency_ms=0,
        )


class _SingleQuerySearch:
    def __init__(self, tool):
        self.tool = tool

    def execute(self, payload):
        return self.tool.execute(payload.model_copy(update={"queries": payload.queries[:1]}))


class AblationRunner:
    def __init__(self, benchmark_path="knowledge/evaluation/enterprise_benchmark.json"):
        self.benchmark_path = Path(benchmark_path)
        self.dataset = json.loads(self.benchmark_path.read_text(encoding="utf-8"))

    def _create_controller(self, ablation_type, max_steps=4):
        if ablation_type not in {"none", "no_find", "no_open", "single_query_only"}:
            raise ValueError(f"Unsupported or inert ablation: {ablation_type}")
        controller = AgenticController()
        if ablation_type == "no_find":
            controller.find_tool = _DisabledFind()
        elif ablation_type == "no_open":
            controller.open_tool = _DisabledOpen()
        elif ablation_type == "single_query_only":
            controller.search_tool = _SingleQuerySearch(controller.search_tool)
        return controller

    def run_experiment(
        self, experiment_name, ablation_type, max_steps=4, sample_size=None, split=None
    ) -> Dict[str, Any]:
        controller = self._create_controller(ablation_type, max_steps)
        comparator = SystemComparator()
        comparator.agentic_controller = controller
        data = [item for item in self.dataset if split is None or item.get("split") == split]
        if sample_size is not None:
            if sample_size <= 0:
                raise ValueError("sample_size must be positive")
            data = data[:sample_size]
        if not data:
            raise ValueError("No benchmark questions selected")
        corpus_hash = comparator.validate_corpus(data)
        records = []
        for item in data:
            output = comparator.evaluate_system_c(item["question"], max_steps=max_steps)
            records.append(
                {
                    "id": item["id"],
                    "output": output,
                    "scores": comparator.score_single(item, output),
                }
            )
        summary = {}
        for key in records[0]["scores"]:
            values = [r["scores"][key] for r in records if r["scores"][key] is not None]
            summary[key] = sum(values) / len(values) if values else None
        return {
            "experiment": experiment_name,
            "ablation_type": ablation_type,
            "max_steps": max_steps,
            "split": split,
            "samples_evaluated": len(data),
            "chunks_sha256": corpus_hash,
            "summary": summary,
            "records": records,
        }

    def run_all_ablations(self, sample_size=None, split=None):
        variants = [
            ("Full heuristic controller", "none", 4),
            ("Empty FIND responses", "no_find", 4),
            ("Empty OPEN responses", "no_open", 4),
            ("Single-query SEARCH", "single_query_only", 4),
            ("One-step budget", "none", 1),
            ("Two-step budget", "none", 2),
        ]
        return {
            "ablations": [
                self.run_experiment(name, kind, steps, sample_size, split)
                for name, kind, steps in variants
            ],
            "limitations": "Empty responses change downstream routing. SUMMARIZE is never scheduled. Inspect per-query traces to establish whether each variant changed execution.",
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="knowledge/evaluation/enterprise_benchmark.json")
    parser.add_argument("--split", choices=["dev", "held_out"])
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    results = AblationRunner(args.dataset).run_all_ablations(args.max_samples, args.split)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved exploratory ablations to {output}")
