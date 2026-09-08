"""
Feedback-driven reranker tuning pipeline.
Converts accumulated feedback judgments into training triplets and fine-tunes the CrossEncoder.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from knowledge.feedback.store import FeedbackStore
from knowledge.training.reranker_trainer import RerankerTrainer


class FeedbackRerankerTuner:
    """Uses logged feedback to fine-tune the CrossEncoder reranker."""

    def __init__(self, store: Optional[FeedbackStore] = None) -> None:
        self.store = store or FeedbackStore()
        self.trainer = RerankerTrainer()

    def tune_from_feedback(
        self,
        output_dir: str = ".knowledge/models/fine_tuned_reranker",
        epochs: int = 1,
        min_samples: int = 2,
    ) -> Dict[str, Any]:
        """Fine-tune the reranker model on accumulated user feedback."""
        samples = self.store.get_training_samples()
        if len(samples) < min_samples:
            raise ValueError(
                f"Need at least {min_samples} feedback samples to tune, but only have {len(samples)}. "
                "Record feedback first using 'knowledge feedback'."
            )

        # Save temporary pairs file
        temp_pairs = Path(".knowledge/temp_feedback_pairs.json")
        temp_pairs.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_pairs, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2)

        try:
            result = self.trainer.train(
                training_data_path=str(temp_pairs),
                output_dir=output_dir,
                epochs=epochs,
            )
            return result
        finally:
            if temp_pairs.exists():
                temp_pairs.unlink()
