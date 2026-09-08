"""
Unit tests for feedback logging and active reranker tuning.
"""

from pathlib import Path

import pytest

from knowledge.feedback.store import FeedbackStore
from knowledge.feedback.tuner import FeedbackRerankerTuner


def test_feedback_store_record_and_retrieve(tmp_path: Path) -> None:
    """Verify recording judgments persists to disk and returns structured entries."""
    store_file = tmp_path / "feedback.json"
    store = FeedbackStore(store_path=str(store_file))

    entry1 = store.record_feedback(
        query="What is BM25?",
        chunk_text="BM25 is a probabilistic lexical ranking algorithm.",
        rating=1,
        chunk_id="c1",
    )
    assert entry1["rating"] == 1
    assert entry1["chunk_id"] == "c1"

    entry2 = store.record_feedback(
        query="What is BM25?",
        chunk_text="Deep neural nets use backpropagation.",
        rating=-1,
        chunk_id="c2",
    )
    assert entry2["rating"] == -1

    all_entries = store.get_all()
    assert len(all_entries) == 2

    # Check training samples formatting
    samples = store.get_training_samples()
    assert len(samples) == 2
    assert samples[0]["label"] == 1.0
    assert samples[1]["label"] == 0.0


def test_feedback_tuner_minimum_samples_validation(tmp_path: Path) -> None:
    """Verify tuner rejects datasets with fewer than min_samples."""
    store_file = tmp_path / "feedback.json"
    store = FeedbackStore(store_path=str(store_file))
    tuner = FeedbackRerankerTuner(store=store)

    # Empty store
    with pytest.raises(ValueError) as exc:
        tuner.tune_from_feedback(min_samples=2)
    assert "Need at least 2 feedback samples" in str(exc.value)

    # Only 1 sample
    store.record_feedback(query="Q", chunk_text="T", rating=1)
    with pytest.raises(ValueError):
        tuner.tune_from_feedback(min_samples=2)
