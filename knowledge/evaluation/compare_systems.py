"""
Three-System Comparative Evaluation Framework:
- Baseline A: Standard Single-Pass Hybrid RAG
- Baseline B: Existing Self-Correcting LangGraph Agent
- System C: Research-Inspired AgenticRAG (Microsoft AgenticRAG)
"""

import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from knowledge.agent.graph import run_agent_workflow
from knowledge.agentic.controller import AgenticController
from knowledge.api.service import RAGService


class SystemComparator:
    """
    Executes and scores queries across Baseline A, Baseline B, and System C.
    """

    def __init__(self, service: Optional[RAGService] = None) -> None:
        self.service = service or RAGService()
        self.agentic_controller = AgenticController()

    def evaluate_baseline_a(self, question: str) -> Dict[str, Any]:
        """Baseline A: Standard Single-Pass Hybrid Search + Generation."""
        t0 = time.perf_counter()
        try:
            res = self.service.ask(question, top_k=5)
            latency = (time.perf_counter() - t0) * 1000
            retrieved_docs = list(
                dict.fromkeys(Path(c["source_path"]).name for c in res.get("chunks", []))
            )
            return {
                "answer": res.get("answer", ""),
                "retrieved_docs": retrieved_docs,
                "latency_ms": latency,
                "steps": 1,
                "abstained": self.is_abstention(res.get("answer", "")),
                "citations": [f"[{d}]" for d in retrieved_docs],
            }
        except Exception as e:
            return {
                "error": str(e),
                "answer": "",
                "retrieved_docs": [],
                "latency_ms": (time.perf_counter() - t0) * 1000,
                "steps": 1,
                "abstained": True,
                "citations": [],
            }

    def evaluate_baseline_b(self, question: str) -> Dict[str, Any]:
        """Baseline B: Self-Correcting LangGraph Agent (rewrite loops)."""
        t0 = time.perf_counter()
        try:
            res = run_agent_workflow(question, max_retries=2)
            latency = (time.perf_counter() - t0) * 1000
            sources = res.get("sources", [])
            return {
                "error": res.get("error"),
                "answer": res.get("answer", ""),
                "retrieved_docs": sources,
                "latency_ms": latency,
                "steps": res.get("rewrites", 0) + 1,
                "abstained": self.is_abstention(res.get("answer", "")),
                "citations": [f"[{s}]" for s in sources],
            }
        except Exception as e:
            return {
                "error": str(e),
                "answer": "",
                "retrieved_docs": [],
                "latency_ms": (time.perf_counter() - t0) * 1000,
                "steps": 1,
                "abstained": True,
                "citations": [],
            }

    def evaluate_system_c(self, question: str, max_steps: int = 4) -> Dict[str, Any]:
        """System C: Research AgenticRAG with SEARCH, FIND, OPEN, SUMMARIZE."""
        t0 = time.perf_counter()
        try:
            res = self.agentic_controller.run(question, max_steps=max_steps)
            latency = (time.perf_counter() - t0) * 1000
            return {
                "error": res.get("error"),
                "answer": res.get("answer", ""),
                "retrieved_docs": res.get("discovered_docs", []),
                "latency_ms": latency,
                "steps": res.get("steps", 1),
                "abstained": res.get("abstained", False),
                "citations": [c.get("label", "") for c in res.get("citations", [])],
                "citation_integrity": res.get("citation_report", {}).get("citation_integrity", 1.0),
            }
        except Exception as e:
            return {
                "error": str(e),
                "answer": "",
                "retrieved_docs": [],
                "latency_ms": (time.perf_counter() - t0) * 1000,
                "steps": 1,
                "abstained": True,
                "citations": [],
                "citation_integrity": None,
            }

    @staticmethod
    def is_abstention(answer: str) -> bool:
        return any(
            phrase in answer.lower()
            for phrase in (
                "cannot answer",
                "could not find the answer",
                "insufficient evidence",
                "not enough information",
            )
        )

    def score_single(self, item, system_output):
        """Document-level proxies; keyword presence does not measure answer correctness."""
        expected = {Path(d).name.lower() for d in item.get("expected_documents", [])}
        retrieved = list(
            dict.fromkeys(Path(d).name.lower() for d in system_output.get("retrieved_docs", []))
        )
        keywords = [k.lower() for k in item.get("gold_keywords", [])]
        answer = system_output.get("answer", "").lower()
        unanswerable = item.get("is_unanswerable", False)
        # Infrastructure failures must never earn abstention or retrieval credit.
        failed = bool(system_output.get("error")) or answer.startswith("failed to generate answer:")
        abstained = system_output.get("abstained", False) or self.is_abstention(answer)
        hits = [int(doc in expected) for doc in retrieved] if not failed else [0] * len(retrieved)
        relevant = sum(hits)
        return {
            "retrieval_hit": float(bool(relevant)) if expected else None,
            "document_recall": relevant / len(expected) if expected else None,
            "document_precision": relevant / len(retrieved)
            if retrieved and expected
            else (0.0 if expected else None),
            "document_mrr": next((1 / (i + 1) for i, hit in enumerate(hits) if hit), 0.0)
            if expected
            else None,
            "keyword_recall": (
                sum(k in answer for k in keywords) / len(keywords)
                if keywords and not unanswerable and not failed
                else (0.0 if not unanswerable else None)
            ),
            "abstention_correctness": float(not failed and abstained == unanswerable),
            "error": float(failed),
            "latency_ms": system_output.get("latency_ms", 0.0),
            "steps": float(system_output.get("steps", 1)),
        }

    def validate_corpus(self, data):
        """Refuse a comparison when annotated source files are absent from the index."""
        with self.service.chunks_path.open(encoding="utf-8") as handle:
            index = json.load(handle)
        available = {Path(c["source_path"]).name.lower() for c in index["chunks"]}
        missing = sorted(
            {
                d
                for item in data
                for d in item.get("expected_documents", [])
                if Path(d).name.lower() not in available
            }
        )
        if missing:
            raise ValueError("Benchmark sources missing from index: " + ", ".join(missing))
        return hashlib.sha256(self.service.chunks_path.read_bytes()).hexdigest()

    def run_benchmark(
        self,
        dataset_path: str = "knowledge/evaluation/enterprise_benchmark.json",
        max_samples: Optional[int] = None,
        split: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the comparative evaluation across all three systems."""
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if split:
            data = [d for d in data if d.get("split") == split]

        if max_samples:
            data = data[:max_samples]

        if not data:
            raise ValueError("No benchmark questions selected")
        corpus_hash = self.validate_corpus(data)
        records = []
        systems = {
            "Baseline_A_StandardRAG": self.evaluate_baseline_a,
            "Baseline_B_SelfCorrecting": self.evaluate_baseline_b,
            "System_C_AgenticRAG": self.evaluate_system_c,
        }
        for item in data:
            for name, evaluate in systems.items():
                output = evaluate(item["question"])
                records.append(
                    {
                        "id": item["id"],
                        "category": item.get("category"),
                        "system": name,
                        "output": output,
                        "scores": self.score_single(item, output),
                    }
                )
        summary = {}
        for name in systems:
            rows = [r["scores"] for r in records if r["system"] == name]
            summary[name] = {"queries_evaluated": len(rows)}
            for key in rows[0]:
                values = [row[key] for row in rows if row[key] is not None]
                summary[name][key] = sum(values) / len(values) if values else None
        return {
            "metadata": {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "dataset_sha256": hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest(),
                "chunks_sha256": corpus_hash,
                "split": split,
                "max_samples": max_samples,
                "python": platform.python_version(),
                "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "limitations": "Document-level retrieval proxies; no correctness, faithfulness or citation support scoring.",
            },
            "summary": summary,
            "records": records,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="knowledge/evaluation/enterprise_benchmark.json")
    parser.add_argument("--split", choices=["dev", "held_out"])
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    results = SystemComparator().run_benchmark(args.dataset, args.max_samples, args.split)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["summary"], indent=2))
