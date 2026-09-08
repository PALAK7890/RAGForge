"""
Evaluation pipeline for KnowledgeOS retrieval.
"""

import time
from pathlib import Path

from knowledge.embeddings.sentence_transformer import SentenceTransformerEmbedding
from knowledge.evaluation.dataset import EvaluationDataset
from knowledge.evaluation.judge import LLMJudgeEvaluator
from knowledge.evaluation.metrics import (
    accuracy_at_k,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from knowledge.llms.ollama_llm import OllamaLLM
from knowledge.rerankers.cross_encoder import CrossEncoderReranker
from knowledge.retrievers.bm25 import BM25Retriever
from knowledge.retrievers.hybrid import HybridRetriever
from knowledge.vectorstores.faiss_store import FAISSStore


class RetrievalEvaluator:

    def __init__(self) -> None:
        self.embedder = SentenceTransformerEmbedding()
        self.vector_store = FAISSStore()
        self.vector_store.load(".knowledge/index/faiss.index")
        self.reranker = CrossEncoderReranker()
        self.llm = OllamaLLM()
        self.judge = LLMJudgeEvaluator()

    def evaluate(
        self,
        dataset_path: str,
        judge: bool = False,
    ) -> dict:

        dataset = EvaluationDataset(dataset_path).load()

        with open(".knowledge/index/chunks.json", "r", encoding="utf-8") as f:
            import json

            data = json.load(f)

        chunks = data["chunks"]

        texts = [chunk["text"] for chunk in chunks]

        bm25 = BM25Retriever()
        bm25.build(texts)

        hybrid = HybridRetriever()

        accuracy1 = []
        accuracy3 = []
        precision3 = []
        recall3 = []
        rr_scores = []

        latencies = []
        faithfulness_scores = []
        relevance_scores = []

        results = []

        for sample in dataset:

            question = sample["question"]
            expected = sample["expected_document"]

            start = time.perf_counter()

            embedding = self.embedder.embed_query(question)

            scores, indices = self.vector_store.search(
                embedding,
                top_k=20,
            )

            faiss_results = []

            for score, idx in zip(scores[0], indices[0]):

                if idx == -1:
                    continue

                faiss_results.append((idx, float(score)))

            bm25_results = bm25.search(
                question,
                top_k=20,
            )

            fused = hybrid.fuse(
                faiss_results,
                bm25_results,
            )

            candidate_chunks = []

            for idx, score in fused:

                candidate_chunks.append(
                    {
                        "idx": idx,
                        "text": chunks[idx]["text"],
                        "score": score,
                    }
                )

            reranked = self.reranker.rerank(
                question,
                candidate_chunks,
                top_k=5,
            )

            retrieved = []

            for item in reranked:

                idx = item["idx"]
                retrieved.append(idx)

            relevant = []

            for i, chunk in enumerate(chunks):

                name = Path(chunk["source_path"]).name

                if name == expected:
                    relevant.append(i)

            accuracy1.append(
                accuracy_at_k(retrieved, relevant, 1)
            )

            accuracy3.append(
                accuracy_at_k(retrieved, relevant, 3)
            )

            precision3.append(
                precision_at_k(retrieved, relevant, 3)
            )

            recall3.append(
                recall_at_k(retrieved, relevant, 3)
            )

            rr_scores.append(
                reciprocal_rank(retrieved, relevant)
            )

            latency = (time.perf_counter() - start) * 1000

            latencies.append(latency)

            faith_score = None
            rel_score = None
            if judge:
                context_str = "\n\n".join(chunks[i]["text"] for i in retrieved)
                ans = self.llm.generate(question, context_str)
                j_res = self.judge.evaluate_response(question, context_str, ans)
                faith_score = j_res["faithfulness"]
                rel_score = j_res["relevance"]
                faithfulness_scores.append(faith_score)
                relevance_scores.append(rel_score)

            result_entry = {
                "question": question,
                "expected": expected,
                "retrieved": [
                    Path(chunks[i]["source_path"]).name
                    for i in retrieved
                ],
                "latency_ms": latency,
            }
            if judge:
                result_entry["faithfulness"] = faith_score
                result_entry["relevance"] = rel_score

            results.append(result_entry)

        report = {
            "queries": len(dataset),
            "accuracy@1": sum(accuracy1) / len(accuracy1) if accuracy1 else 0.0,
            "accuracy@3": sum(accuracy3) / len(accuracy3) if accuracy3 else 0.0,
            "precision@3": sum(precision3) / len(precision3) if precision3 else 0.0,
            "recall@3": sum(recall3) / len(recall3) if recall3 else 0.0,
            "mrr": mean_reciprocal_rank(rr_scores),
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
            "results": results,
        }

        if judge and faithfulness_scores:
            report["avg_faithfulness"] = sum(faithfulness_scores) / len(faithfulness_scores)
            report["avg_relevance"] = sum(relevance_scores) / len(relevance_scores)

        return report