"""
Unsupervised topic clustering module for RAGForge chunks.
Clusters dense vector embeddings with KMeans and extracts keyword tags via TF-IDF.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from knowledge.vectorstores.faiss_store import FAISSStore


class TopicClusterer:
    """Discovers thematic clusters across document chunks and attaches topic metadata."""

    def __init__(
        self,
        chunks_path: str = ".knowledge/index/chunks.json",
        faiss_path: str = ".knowledge/index/faiss.index",
        topics_path: str = ".knowledge/index/topics.json",
    ) -> None:
        self.chunks_path = Path(chunks_path)
        self.faiss_path = Path(faiss_path)
        self.topics_path = Path(topics_path)

    def cluster_chunks(self, n_clusters: int = 5) -> List[Dict[str, Any]]:
        """
        Cluster indexed chunks using their embeddings and assign topic tags.
        Saves updated chunks.json and topics.json.
        """
        if not self.chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found at {self.chunks_path}. Index documents first.")

        with open(self.chunks_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        chunks = data.get("chunks", [])
        if not chunks:
            raise ValueError("No indexed chunks found to cluster.")

        k = min(n_clusters, len(chunks))
        if k < 1:
            k = 1

        # Extract embeddings
        texts = [chunk["text"] for chunk in chunks]

        # Use FAISS index vectors or fallback to TF-IDF feature space
        if self.faiss_path.exists():
            try:
                vs = FAISSStore()
                vs.load(str(self.faiss_path))
                if vs.index is not None and vs.index.ntotal == len(chunks):
                    # Reconstruct FAISS vectors
                    embeddings = vs.index.reconstruct_n(0, vs.index.ntotal)
                else:
                    tfidf = TfidfVectorizer(max_features=256, stop_words="english")
                    embeddings = tfidf.fit_transform(texts).toarray()
            except Exception:
                tfidf = TfidfVectorizer(max_features=256, stop_words="english")
                embeddings = tfidf.fit_transform(texts).toarray()
        else:
            tfidf = TfidfVectorizer(max_features=256, stop_words="english")
            embeddings = tfidf.fit_transform(texts).toarray()

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        # Build topic summaries by cluster
        topics_summary: List[Dict[str, Any]] = []

        for cluster_id in range(k):
            cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
            cluster_texts = [texts[i] for i in cluster_indices]

            # Keyword extraction via TF-IDF
            keywords: List[str] = []
            if cluster_texts:
                try:
                    vec = TfidfVectorizer(stop_words="english", max_features=3)
                    vec.fit(cluster_texts)
                    keywords = list(vec.get_feature_names_out())
                except Exception:
                    keywords = [f"topic_{cluster_id}"]

            topic_label = "-".join(keywords) if keywords else f"topic-{cluster_id}"

            # Tag chunks
            for i in cluster_indices:
                chunks[i]["topic_id"] = int(cluster_id)
                chunks[i]["topic_name"] = topic_label

            sample_text = cluster_texts[0][:120] + "..." if cluster_texts else ""

            topics_summary.append({
                "topic_id": int(cluster_id),
                "name": topic_label,
                "keywords": keywords,
                "chunk_count": len(cluster_indices),
                "sample_text": sample_text,
            })

        # Save updated chunks.json
        with open(self.chunks_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Save topics.json
        self.topics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.topics_path, "w", encoding="utf-8") as f:
            json.dump(topics_summary, f, indent=2, ensure_ascii=False)

        return topics_summary

    def get_topics(self) -> List[Dict[str, Any]]:
        """Read saved topic clusters from topics.json."""
        if not self.topics_path.exists():
            return []

        with open(self.topics_path, "r", encoding="utf-8") as f:
            return json.load(f)
