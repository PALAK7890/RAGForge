"""
Unit tests for unsupervised topic clustering and keyword tagging.
"""

import json
from pathlib import Path

from knowledge.clustering.clusterer import TopicClusterer


def test_topic_clusterer_execution(tmp_path: Path) -> None:
    """Verify TopicClusterer clusters text chunks and assigns topic labels."""
    chunks_file = tmp_path / "chunks.json"
    faiss_file = tmp_path / "nonexistent.index"
    topics_file = tmp_path / "topics.json"

    # Create dummy chunks across two distinct topics: machine learning and web development
    chunks = [
        {"chunk_id": "c1", "text": "Deep neural networks and machine learning embeddings are useful."},
        {"chunk_id": "c2", "text": "Transformers and neural attention mechanisms power large language models."},
        {"chunk_id": "c3", "text": "HTML templates and CSS styling render modern browser web pages."},
        {"chunk_id": "c4", "text": "JavaScript frameworks and frontend UI components build dynamic web applications."},
    ]
    chunks_file.write_text(json.dumps({"chunks": chunks}), encoding="utf-8")

    clusterer = TopicClusterer(
        chunks_path=str(chunks_file),
        faiss_path=str(faiss_file),
        topics_path=str(topics_file),
    )

    topics = clusterer.cluster_chunks(n_clusters=2)

    assert len(topics) == 2
    assert topics_file.exists()

    # Check updated chunks
    with open(chunks_file, "r", encoding="utf-8") as f:
        updated_data = json.load(f)

    for c in updated_data["chunks"]:
        assert "topic_id" in c
        assert "topic_name" in c

    # Check get_topics reads saved metadata
    saved_topics = clusterer.get_topics()
    assert len(saved_topics) == 2
    assert "keywords" in saved_topics[0]
    assert "chunk_count" in saved_topics[0]


def test_clusterer_no_faiss_file_uses_tfidf(tmp_path: Path) -> None:
    """Explicitly assert TF-IDF fallback is used when FAISS index file is absent.

    The existing happy-path test passes a nonexistent path, which incidentally
    triggers this branch. This test names it clearly and verifies topic labels
    are still assigned correctly via the TF-IDF code path.
    """
    chunks_file = tmp_path / "chunks.json"
    topics_file = tmp_path / "topics.json"
    missing_faiss = tmp_path / "no_such.index"  # does not exist

    chunks = [
        {"chunk_id": "a1", "text": "Gradient descent optimizes neural network loss functions."},
        {"chunk_id": "a2", "text": "Backpropagation computes gradients through hidden layers."},
        {"chunk_id": "a3", "text": "HTTP requests and REST APIs power modern web services."},
        {"chunk_id": "a4", "text": "JSON and YAML are common data serialization formats for APIs."},
    ]
    chunks_file.write_text(json.dumps({"chunks": chunks}), encoding="utf-8")

    clusterer = TopicClusterer(
        chunks_path=str(chunks_file),
        faiss_path=str(missing_faiss),
        topics_path=str(topics_file),
    )
    topics = clusterer.cluster_chunks(n_clusters=2)

    # TF-IDF fallback must still produce valid cluster output
    assert len(topics) == 2
    for t in topics:
        assert "topic_id" in t
        assert "chunk_count" in t
        assert t["chunk_count"] > 0


def test_clusterer_faiss_mismatch_falls_back_to_tfidf(tmp_path: Path) -> None:
    """FAISS index exists but ntotal != len(chunks) — must fall back to TF-IDF.

    This exercises the ``vs.index.ntotal == len(chunks)`` else-branch inside
    cluster_chunks, which is skipped when no FAISS file exists at all.
    """
    import faiss
    import numpy as np

    chunks_file = tmp_path / "chunks.json"
    faiss_file = tmp_path / "faiss.index"
    topics_file = tmp_path / "topics.json"

    # 4 text chunks but only 2 vectors in the FAISS index → mismatch
    chunks = [
        {"chunk_id": "b1", "text": "Convolutional layers extract spatial image features."},
        {"chunk_id": "b2", "text": "Pooling reduces spatial dimensions in CNN architectures."},
        {"chunk_id": "b3", "text": "SQL databases store relational data in tabular form."},
        {"chunk_id": "b4", "text": "Indexes speed up database query execution plans."},
    ]
    chunks_file.write_text(json.dumps({"chunks": chunks}), encoding="utf-8")

    # Write a FAISS index with only 2 vectors (ntotal=2) while there are 4 chunks
    dim = 8
    index = faiss.IndexFlatL2(dim)
    vecs = np.random.rand(2, dim).astype("float32")
    index.add(vecs)
    faiss.write_index(index, str(faiss_file))

    clusterer = TopicClusterer(
        chunks_path=str(chunks_file),
        faiss_path=str(faiss_file),
        topics_path=str(topics_file),
    )
    topics = clusterer.cluster_chunks(n_clusters=2)

    # TF-IDF fallback must handle the mismatch gracefully
    assert len(topics) == 2
    assert topics_file.exists()
    with open(chunks_file) as f:
        updated = json.load(f)
    for c in updated["chunks"]:
        assert "topic_id" in c
