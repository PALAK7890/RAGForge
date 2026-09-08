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


def test_feedback_store_all_positive_labels(tmp_path: Path) -> None:
    """All-positive feedback should produce training samples with label == 1.0."""
    store = FeedbackStore(store_path=str(tmp_path / "fb.json"))
    for i in range(3):
        store.record_feedback(query=f"Q{i}", chunk_text=f"Text {i}", rating=1, chunk_id=f"c{i}")

    samples = store.get_training_samples()
    assert len(samples) == 3
    assert all(s["label"] == 1.0 for s in samples)


def test_feedback_tuner_succeeds_with_sufficient_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """tune_from_feedback completes and returns a success dict when enough samples exist.

    Monkeypatches RerankerTrainer.train to skip the actual model download so the
    test runs offline, while still exercising all FeedbackRerankerTuner logic.
    """
    store = FeedbackStore(store_path=str(tmp_path / "fb.json"))
    # 3 mixed-label samples — sufficient for min_samples=2
    store.record_feedback(query="What is BM25?", chunk_text="BM25 is a ranking function.", rating=1, chunk_id="c1")
    store.record_feedback(query="What is FAISS?", chunk_text="FAISS is a vector index.", rating=1, chunk_id="c2")
    store.record_feedback(query="What is BM25?", chunk_text="Unrelated text.", rating=-1, chunk_id="c3")

    tuner = FeedbackRerankerTuner(store=store)

    fake_result = {"status": "success", "model_path": str(tmp_path / "reranker"), "epochs": 1}

    from knowledge.training.reranker_trainer import RerankerTrainer
    monkeypatch.setattr(RerankerTrainer, "train", lambda self, **kwargs: fake_result)

    result = tuner.tune_from_feedback(
        output_dir=str(tmp_path / "reranker"),
        epochs=1,
        min_samples=2,
    )
    assert result["status"] == "success"
