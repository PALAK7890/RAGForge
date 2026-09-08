"""
Benchmark retrieval quality.
"""

import time
from typing import Dict

from knowledge.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class Benchmark:

    def evaluate(
        self,
        query: str,
        retrieved: list[int],
        relevant: list[int],
    ) -> Dict[str, float]:

        start = time.perf_counter()

        metrics = {
            "Recall@5": recall_at_k(
                retrieved,
                relevant,
                5,
            ),
            "Precision@5": precision_at_k(
                retrieved,
                relevant,
                5,
            ),
            "MRR": reciprocal_rank(
                retrieved,
                relevant,
            ),
        }

        metrics["Latency"] = (
            time.perf_counter() - start
        ) * 1000

        return metrics