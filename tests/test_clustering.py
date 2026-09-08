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
