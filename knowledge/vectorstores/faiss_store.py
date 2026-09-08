"""
Simple FAISS vector store.
"""

from pathlib import Path
from typing import Optional

import faiss
import numpy as np


class FAISSStore:
    """Stores and searches dense embeddings using FAISS."""

    def __init__(self) -> None:
        self.index: Optional[faiss.Index] = None

    def add(self, embeddings: list[list[float]]) -> None:
        """Add embeddings to the index."""

        vectors = np.asarray(embeddings, dtype="float32")

        if self.index is None:
            dimension = vectors.shape[1]
            self.index = faiss.IndexFlatIP(dimension)

        self.index.add(vectors)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search the index."""
        if self.index is None:
            raise ValueError("Index is not loaded or initialized.")

        query = np.asarray([query_embedding], dtype="float32")
        scores, indices = self.index.search(query, top_k)
        return scores, indices

    def save(self, path: str) -> None:
        """Save index to disk."""
        if self.index is None:
            raise ValueError("Cannot save an empty or uninitialized index.")

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, path)

    def load(self, path: str) -> None:
        """Load index from disk."""
        self.index = faiss.read_index(path)