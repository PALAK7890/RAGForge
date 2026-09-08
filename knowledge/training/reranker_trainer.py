"""
CrossEncoder reranker fine-tuning module.
"""

import json
from pathlib import Path
from typing import Any, Dict, List
from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder, InputExample


class RerankerTrainer:
    """Fine-tunes a CrossEncoder model on domain relevance judgments."""

    def __init__(self, base_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.base_model = base_model

    def train(
        self,
        training_data_path: str = ".knowledge/reranker_pairs.json",
        output_dir: str = ".knowledge/models/fine_tuned_reranker",
        epochs: int = 1,
        batch_size: int = 4,
        warmup_steps: int = 10,
    ) -> Dict[str, Any]:
        """Fine-tune the CrossEncoder model and save the weights."""
        data_file = Path(training_data_path)
        if not data_file.exists():
            raise FileNotFoundError(f"Training data not found at {training_data_path}")

        with open(data_file, "r", encoding="utf-8") as f:
            samples = json.load(f)

        if not samples:
            raise ValueError("Training dataset is empty.")

        train_examples = [
            InputExample(texts=[s["query"], s["text"]], label=float(s.get("label", 1.0)))
            for s in samples
            if "query" in s and "text" in s
        ]

        model = CrossEncoder(self.base_model, num_labels=1)
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        model.fit(
            train_dataloader=train_dataloader,
            epochs=epochs,
            warmup_steps=warmup_steps,
            output_path=str(out_path),
            show_progress_bar=False,
        )

        return {
            "status": "success",
            "model_path": str(out_path.resolve()),
            "epochs": epochs,
            "training_samples": len(train_examples),
            "base_model": self.base_model,
        }
