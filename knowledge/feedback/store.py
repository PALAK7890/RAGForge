"""
Local feedback logging store for RAGForge retrieval judgments.
Records user ratings (thumbs up/down or relevance scores) to close the active learning loop.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class FeedbackStore:
    """Stores query-chunk feedback judgments locally."""

    def __init__(self, store_path: str = ".knowledge/feedback.json") -> None:
        self.store_path = Path(store_path)

    def _load(self) -> List[Dict[str, Any]]:
        if not self.store_path.exists():
            return []
        try:
            with open(self.store_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return list(data) if isinstance(data, list) else []
        except Exception:
            return []

    def _save(self, records: List[Dict[str, Any]]) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

    def record_feedback(
        self,
        query: str,
        chunk_text: str,
        rating: int,
        chunk_id: Optional[str] = None,
        source_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Record a relevance judgment.
        rating: +1 (relevant), -1 or 0 (irrelevant), or 1-5 scale.
        """
        records = self._load()
        entry = {
            "feedback_id": str(uuid.uuid4())[:8],
            "query": query,
            "chunk_id": chunk_id or f"chunk_{len(records)}",
            "chunk_text": chunk_text,
            "rating": int(rating),
            "source_path": source_path or "",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        records.append(entry)
        self._save(records)
        return entry

    def get_all(self) -> List[Dict[str, Any]]:
        """Retrieve all recorded feedback entries."""
        return self._load()

    def get_training_samples(self) -> List[Dict[str, Any]]:
        """Convert feedback records into (query, text, label) training samples."""
        records = self._load()
        samples: List[Dict[str, Any]] = []

        for r in records:
            # Normalize rating to binary or normalized float label (0.0 to 1.0)
            rating = r["rating"]
            if rating > 0:
                label = 1.0
            else:
                label = 0.0

            samples.append({
                "query": r["query"],
                "text": r["chunk_text"],
                "label": label,
            })

        return samples

    def clear(self) -> None:
        """Clear all stored feedback."""
        if self.store_path.exists():
            self.store_path.unlink()
