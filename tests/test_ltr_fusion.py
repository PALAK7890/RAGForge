"""
Unit tests for Learning-to-Rank (LTR) fusion and feature extraction.
"""

import json
from pathlib import Path

from knowledge.retrievers.ltr_fusion import LTRFeatureExtractor, LTRFusion


def test_ltr_feature_extractor() -> None:
    """Verify LTR feature extractor produces 7 numerical features."""
    feats = LTRFeatureExtractor.extract_features(
        dense_score=0.85,
        dense_rank=1,
        bm25_score=12.4,
        bm25_rank=2,
        rrf_score=0.03,
        chunk_length=450,
        chunk_index=3,
    )
    assert len(feats) == 7
    assert feats[0] == 0.85
    assert feats[1] == 1.0
    assert feats[2] == 12.4
    assert feats[3] == 2.0
    assert feats[4] == 0.03
    assert feats[5] == 0.45
    assert feats[6] == 0.06


def test_ltr_fallback_to_rrf() -> None:
    """Verify LTRFusion falls back cleanly to RRF when no trained model is loaded."""
    ltr = LTRFusion(model_path=None)
    assert ltr.model is None

    faiss_results = [(0, 0.9), (1, 0.7)]
    bm25_results = [(1, 10.0), (0, 5.0)]

    ranked = ltr.fuse(faiss_results, bm25_results, chunks=None)
    assert len(ranked) == 2
    # Both items present in ranking
    indices = [idx for idx, _ in ranked]
    assert 0 in indices
    assert 1 in indices


def test_ltr_training_and_serialization(tmp_path: Path) -> None:
    """Verify LTR training creates model and saves to disk."""
    q_file = tmp_path / "questions.json"
    q_file.write_text(
        json.dumps([
            {"question": "What is Python?", "expected_document": "python.txt"}
        ]),
        encoding="utf-8",
    )

    chunks_file = tmp_path / "chunks.json"
    chunks_file.write_text(
        json.dumps({
            "chunks": [
                {"chunk_id": "c1", "text": "Python is a language.", "source_path": "python.txt", "chunk_index": 0},
                {"chunk_id": "c2", "text": "Java is another language.", "source_path": "java.txt", "chunk_index": 1},
            ]
        }),
        encoding="utf-8",
    )

    model_path = tmp_path / "ltr_test.joblib"

    ltr = LTRFusion(model_path=None)
    res = ltr.train_on_judgments(
        training_data_path=str(q_file),
        chunks_path=str(chunks_file),
        output_model_path=str(model_path),
    )

    assert res["status"] == "trained"
    assert model_path.exists()
    assert ltr.model is not None

    # Test reloading
    new_ltr = LTRFusion(model_path=str(model_path))
    assert new_ltr.model is not None


def test_ltr_predict_proba_exception_falls_back_to_rrf(tmp_path: Path) -> None:
    """Verify LTRFusion falls back to RRF when predict_proba raises during inference.

    This exercises the except branch at ltr_fusion.py:138 which is unreachable
    without forcing a runtime error on the trained model.
    """
    # Build a minimal trained model so self.model is not None
    q_file = tmp_path / "questions.json"
    q_file.write_text(
        json.dumps([{"question": "Q?", "expected_document": "doc.txt"}]),
        encoding="utf-8",
    )
    chunks_file = tmp_path / "chunks.json"
    chunks_file.write_text(
        json.dumps({"chunks": [
            {"chunk_id": "c1", "text": "Some text.", "source_path": "doc.txt", "chunk_index": 0},
            {"chunk_id": "c2", "text": "Other text.", "source_path": "other.txt", "chunk_index": 1},
        ]}),
        encoding="utf-8",
    )

    ltr = LTRFusion(model_path=None)
    ltr.train_on_judgments(
        training_data_path=str(q_file),
        chunks_path=str(chunks_file),
        output_model_path=str(tmp_path / "model.joblib"),
    )
    assert ltr.model is not None

    # Monkey-patch predict_proba to raise
    original_predict_proba = ltr.model.predict_proba

    def boom(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("Simulated predict_proba failure")

    ltr.model.predict_proba = boom  # type: ignore[method-assign]

    chunks = [
        {"chunk_id": "c1", "text": "Some text.", "source_path": "doc.txt", "chunk_index": 0},
        {"chunk_id": "c2", "text": "Other text.", "source_path": "other.txt", "chunk_index": 1},
    ]
    faiss_results = [(0, 0.9), (1, 0.5)]
    bm25_results = [(1, 8.0), (0, 3.0)]

    # Should not raise — must fall back to RRF silently
    ranked = ltr.fuse(faiss_results, bm25_results, chunks=chunks)
    assert len(ranked) == 2
    indices = [idx for idx, _ in ranked]
    assert 0 in indices and 1 in indices

    # Restore for hygiene
    ltr.model.predict_proba = original_predict_proba  # type: ignore[method-assign]
