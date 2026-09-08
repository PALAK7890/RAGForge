"""
SentenceTransformer Bi-Encoder fine-tuning module using contrastive loss.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
from torch.utils.data import DataLoader
from sentence_transformers import InputExample, SentenceTransformer, losses


class BiEncoderTrainer:
    """Fine-tunes a SentenceTransformer model on domain query-document pairs."""

    def __init__(self, base_model: str = "all-MiniLM-L6-v2") -> None:
        self.base_model = base_model

    def train(
        self,
        training_pairs_path: str = ".knowledge/training_pairs.json",
        output_dir: str = ".knowledge/models/fine_tuned_bi_encoder",
        epochs: int = 1,
        batch_size: int = 4,
        warmup_steps: int = 10,
    ) -> Dict[str, Any]:
        """Train the model with MultipleNegativesRankingLoss and save weights."""
        pairs_file = Path(training_pairs_path)
        if not pairs_file.exists():
            raise FileNotFoundError(f"Training pairs not found at {training_pairs_path}. Generate synthetic data first.")

        with open(pairs_file, "r", encoding="utf-8") as f:
            pairs = json.load(f)

        if not pairs:
            raise ValueError("Training dataset is empty.")

        train_examples = [
            InputExample(texts=[p["query"], p["positive"]])
            for p in pairs
            if "query" in p and "positive" in p
        ]

        if not train_examples:
            raise ValueError("No valid query-positive pairs found in training file.")

        model = SentenceTransformer(self.base_model)
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        train_loss = losses.MultipleNegativesRankingLoss(model)

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
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
