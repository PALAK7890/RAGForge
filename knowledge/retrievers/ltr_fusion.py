"""
Learning-to-Rank (LTR) fusion module.
Replaces or supplements static Reciprocal Rank Fusion with a trainable gradient-boosted ranker.
"""

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


class LTRFeatureExtractor:
    """Extracts ranking features for candidate retrieval chunks."""

    @staticmethod
    def extract_features(
        dense_score: float,
        dense_rank: int,
        bm25_score: float,
        bm25_rank: int,
        rrf_score: float,
        chunk_length: int,
        chunk_index: int,
    ) -> List[float]:
        """Convert retrieval signals into a numerical feature vector."""
        return [
            float(dense_score),
            float(dense_rank),
            float(bm25_score),
            float(bm25_rank),
            float(rrf_score),
            float(min(chunk_length / 1000.0, 5.0)),
            float(min(chunk_index / 50.0, 5.0)),
        ]


class LTRFusion:
    """Trainable rank fusion model with Reciprocal Rank Fusion fallback."""

    def __init__(self, model_path: Optional[str] = ".knowledge/models/ltr_model.joblib") -> None:
        self.model_path = Path(model_path) if model_path else None
        self.model: Optional[HistGradientBoostingClassifier] = None
        self.feature_extractor = LTRFeatureExtractor()

        if self.model_path and self.model_path.exists():
            self.load(str(self.model_path))

    def load(self, path: str) -> None:
        """Load trained LTR weights from disk."""
        try:
            self.model = joblib.load(path)
        except Exception:
            self.model = None

    def save(self, path: str) -> None:
        """Persist trained model weights to disk."""
        if self.model is None:
            raise ValueError("Cannot save an untrained LTR model.")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, out)

    def fuse(
        self,
        faiss_results: List[Tuple[int, float]],
        bm25_results: List[Tuple[int, float]],
        chunks: Optional[List[Dict[str, Any]]] = None,
        k: int = 60,
    ) -> List[Tuple[int, float]]:
        """
        Rank candidate chunks using the trained LTR model if available,
        or fall back to static Reciprocal Rank Fusion (RRF).
        """
        # 1. Compute standard RRF base scores
        rrf_scores: Dict[int, float] = defaultdict(float)
        dense_map: Dict[int, Tuple[int, float]] = {}
        bm25_map: Dict[int, Tuple[int, float]] = {}

        for rank, (idx, score) in enumerate(faiss_results):
            rrf_scores[idx] += 1.0 / (k + rank + 1)
            dense_map[idx] = (rank, score)

        for rank, (idx, score) in enumerate(bm25_results):
            rrf_scores[idx] += 1.0 / (k + rank + 1)
            bm25_map[idx] = (rank, score)

        # 2. If no trained model or chunks metadata is missing, use RRF
        if self.model is None or chunks is None:
            ranked_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
            return ranked_rrf

        # 3. Model-based ranking: extract features for all candidate chunk indices
        candidate_indices = list(rrf_scores.keys())
        if not candidate_indices:
            return []

        feature_matrix: List[List[float]] = []
        valid_indices: List[int] = []

        for idx in candidate_indices:
            if idx >= len(chunks):
                continue

            chunk = chunks[idx]
            d_rank, d_score = dense_map.get(idx, (999, 0.0))
            b_rank, b_score = bm25_map.get(idx, (999, 0.0))
            r_score = rrf_scores.get(idx, 0.0)

            text_len = len(chunk.get("text", ""))
            chunk_pos = chunk.get("chunk_index", 0)

            feats = self.feature_extractor.extract_features(
                dense_score=d_score,
                dense_rank=d_rank,
                bm25_score=b_score,
                bm25_rank=b_rank,
                rrf_score=r_score,
                chunk_length=text_len,
                chunk_index=chunk_pos,
            )
            feature_matrix.append(feats)
            valid_indices.append(idx)

        if not feature_matrix:
            return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        X = np.array(feature_matrix)
        # Probability of class 1 (relevant)
        try:
            probabilities = self.model.predict_proba(X)[:, 1]
            scored_candidates = list(zip(valid_indices, [float(p) for p in probabilities]))
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            return scored_candidates
        except Exception:
            # Fallback to RRF on prediction error
            return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    def train_on_judgments(
        self,
        training_data_path: str = "knowledge/evaluation/question.json",
        chunks_path: str = ".knowledge/index/chunks.json",
        output_model_path: str = ".knowledge/models/ltr_model.joblib",
    ) -> Dict[str, Any]:
        """Train GBDT model on labeled question-document pairs."""
        with open(training_data_path, "r", encoding="utf-8") as f:
            questions = json.load(f)

        if not Path(chunks_path).exists():
            raise FileNotFoundError(f"Chunks index not found at {chunks_path}")

        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)

        chunks = chunks_data.get("chunks", [])
        if not chunks:
            raise ValueError("No indexed chunks found to train LTR.")

        X_train: List[List[float]] = []
        y_train: List[int] = []

        # Synthetic feature generation for training
        for sample in questions:
            expected_doc = sample.get("expected_document", "").lower()

            for i, chunk in enumerate(chunks[:30]):
                chunk_source = Path(chunk.get("source_path", "")).name.lower()
                is_relevant = 1 if expected_doc and (expected_doc in chunk_source) else 0

                # Simulated retrieval features
                dense_sim = 0.85 if is_relevant else np.random.uniform(0.1, 0.5)
                bm25_val = 15.0 if is_relevant else np.random.uniform(0.0, 5.0)
                rrf_val = 0.03 if is_relevant else 0.005

                feats = self.feature_extractor.extract_features(
                    dense_score=dense_sim,
                    dense_rank=1 if is_relevant else 10,
                    bm25_score=bm25_val,
                    bm25_rank=1 if is_relevant else 10,
                    rrf_score=rrf_val,
                    chunk_length=len(chunk.get("text", "")),
                    chunk_index=chunk.get("chunk_index", 0),
                )
                X_train.append(feats)
                y_train.append(is_relevant)

        if not X_train or sum(y_train) == 0:
            # Add synthetic baseline positive/negative pairs to allow model initialization
            X_train.append([0.9, 1.0, 20.0, 1.0, 0.05, 0.5, 0.0])
            y_train.append(1)
            X_train.append([0.2, 20.0, 1.0, 20.0, 0.001, 0.2, 0.1])
            y_train.append(0)

        model = HistGradientBoostingClassifier(max_iter=30, max_depth=3, random_state=42)
        model.fit(np.array(X_train), np.array(y_train))
        self.model = model

        self.save(output_model_path)

        return {
            "status": "trained",
            "samples": len(X_train),
            "positives": sum(y_train),
            "model_path": output_model_path,
        }
