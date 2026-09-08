"""
SentenceTransformer embedding model implementation.
"""

from typing import cast

from sentence_transformers import SentenceTransformer

from knowledge.embeddings.base import BaseEmbeddingModel


class SentenceTransformerEmbedding(BaseEmbeddingModel):
    """Wrapper around SentenceTransformer."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return cast(list[list[float]], embeddings.tolist())

    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return cast(list[float], embedding.tolist())