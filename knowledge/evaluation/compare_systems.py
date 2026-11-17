"""
Three-System Comparative Evaluation Framework:
- Baseline A: Standard Single-Pass Hybrid RAG
- Baseline B: Existing Self-Correcting LangGraph Agent
- System C: Research-Inspired AgenticRAG (Microsoft AgenticRAG)
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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
            retrieved_docs = list(dict.fromkeys(Path(c["source_path"]).name for c in res.get("chunks", [])))
            return {
                "answer": res.get("answer", ""),
                "retrieved_docs": retrieved_docs,
                "latency_ms": latency,
                "steps": 1,
                "abstained": False,
                "citations": [f"[{d}]" for d in retrieved_docs],
            }
        except Exception as e:
            return {
                "answer": f"Error: {str(e)}",
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
                "answer": res.get("answer", ""),
                "retrieved_docs": sources,
                "latency_ms": latency,
                "steps": res.get("rewrites", 0) + 1,
                "abstained": False,
                "citations": [f"[{s}]" for s in sources],
            }
        except Exception as e:
            return {
                "answer": f"Error: {str(e)}",
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
                "answer": f"Error: {str(e)}",
                "retrieved_docs": [],
                "latency_ms": (time.perf_counter() - t0) * 1000,
                "steps": 1,
                "abstained": True,
                "citations": [],
                "citation_integrity": 1.0,
            }

    def score_single(
        self,
        item: Dict[str, Any],
        system_output: Dict[str, Any],
    ) -> Dict[str, float]:
        """Compute objective retrieval and generation scores for one question."""
        expected_docs = [d.lower() for d in item.get("expected_documents", [])]
        retrieved_docs = [d.lower() for d in system_output.get("retrieved_docs", [])]
        gold_keywords = [k.lower() for k in item.get("gold_keywords", [])]
        answer_lower = system_output.get("answer", "").lower()
        is_unanswerable = item.get("is_unanswerable", False)
        abstained = system_output.get("abstained", False)

        # 1. Retrieval Hit Rate: Did any expected document get retrieved?
        if expected_docs:
            doc_hit = 1.0 if any(any(ed in rd for rd in retrieved_docs) for ed in expected_docs) else 0.0
        else:
            doc_hit = 1.0 if is_unanswerable and abstained else 0.0

        # 2. Keyword Recall: What proportion of gold keywords are in the answer?
        if gold_keywords and not is_unanswerable:
            kw_hits = sum(1 for kw in gold_keywords if kw in answer_lower)
            kw_recall = kw_hits / len(gold_keywords)
        else:
            kw_recall = 1.0 if (is_unanswerable and abstained) else 0.0

        # 3. Abstention Correctness
        if is_unanswerable:
            abstention_score = 1.0 if (abstained or "cannot answer" in answer_lower or "insufficient" in answer_lower) else 0.0
        else:
            abstention_score = 1.0 if not abstained else 0.0

        return {
            "retrieval_hit": doc_hit,
            "keyword_recall": kw_recall,
            "abstention_correctness": abstention_score,
            "latency_ms": system_output.get("latency_ms", 0.0),
            "steps": float(system_output.get("steps", 1)),
        }

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

        metrics = {
            "Baseline_A_StandardRAG": {"hits": [], "keywords": [], "abstentions": [], "latencies": [], "steps": []},
            "Baseline_B_SelfCorrecting": {"hits": [], "keywords": [], "abstentions": [], "latencies": [], "steps": []},
            "System_C_AgenticRAG": {"hits": [], "keywords": [], "abstentions": [], "latencies": [], "steps": []},
        }

        print(f"\n--- Running 3-System Benchmark on {len(data)} questions ---")

        for idx, item in enumerate(data, 1):
            q = item["question"]
            print(f"[{idx}/{len(data)}] Evaluating: {q[:60]}...")

            # Run Baseline A
            out_a = self.evaluate_baseline_a(q)
            sc_a = self.score_single(item, out_a)
            metrics["Baseline_A_StandardRAG"]["hits"].append(sc_a["retrieval_hit"])
            metrics["Baseline_A_StandardRAG"]["keywords"].append(sc_a["keyword_recall"])
            metrics["Baseline_A_StandardRAG"]["abstentions"].append(sc_a["abstention_correctness"])
            metrics["Baseline_A_StandardRAG"]["latencies"].append(sc_a["latency_ms"])
            metrics["Baseline_A_StandardRAG"]["steps"].append(sc_a["steps"])

            # Run Baseline B
            out_b = self.evaluate_baseline_b(q)
            sc_b = self.score_single(item, out_b)
            metrics["Baseline_B_SelfCorrecting"]["hits"].append(sc_b["retrieval_hit"])
            metrics["Baseline_B_SelfCorrecting"]["keywords"].append(sc_b["keyword_recall"])
            metrics["Baseline_B_SelfCorrecting"]["abstentions"].append(sc_b["abstention_correctness"])
            metrics["Baseline_B_SelfCorrecting"]["latencies"].append(sc_b["latency_ms"])
            metrics["Baseline_B_SelfCorrecting"]["steps"].append(sc_b["steps"])

            # Run System C
            out_c = self.evaluate_system_c(q)
            sc_c = self.score_single(item, out_c)
            metrics["System_C_AgenticRAG"]["hits"].append(sc_c["retrieval_hit"])
            metrics["System_C_AgenticRAG"]["keywords"].append(sc_c["keyword_recall"])
            metrics["System_C_AgenticRAG"]["abstentions"].append(sc_c["abstention_correctness"])
            metrics["System_C_AgenticRAG"]["latencies"].append(sc_c["latency_ms"])
            metrics["System_C_AgenticRAG"]["steps"].append(sc_c["steps"])

        summary = {}
        for sys_name, m in metrics.items():
            n = len(m["hits"])
            summary[sys_name] = {
                "queries_evaluated": n,
                "retrieval_hit_rate": round(sum(m["hits"]) / n, 4) if n else 0.0,
                "keyword_recall": round(sum(m["keywords"]) / n, 4) if n else 0.0,
                "abstention_accuracy": round(sum(m["abstentions"]) / n, 4) if n else 0.0,
                "average_latency_ms": round(sum(m["latencies"]) / n, 2) if n else 0.0,
                "average_steps": round(sum(m["steps"]) / n, 2) if n else 0.0,
            }

        return summary


if __name__ == "__main__":
    comparator = SystemComparator()
    # Run a representative sample of 10 items for fast CLI verification
    results = comparator.run_benchmark(max_samples=10)
    print("\nBenchmark Summary Results:")
    print(json.dumps(results, indent=2))
