"""
Unit tests for retrieval model fine-tuning and synthetic data generation.
"""

import json
from pathlib import Path

import pytest

from knowledge.training.bi_encoder_trainer import BiEncoderTrainer
from knowledge.training.reranker_trainer import RerankerTrainer
from knowledge.training.synthetic_generator import SyntheticDataGenerator


def test_synthetic_generator_with_mock_chunks(tmp_path: Path) -> None:
    """Verify synthetic query-document generation handles chunk records."""
    chunks_file = tmp_path / "chunks.json"
    out_file = tmp_path / "pairs.json"

    chunks_file.write_text(
        json.dumps({
            "chunks": [
                {
                    "chunk_id": "chunk_1",
                    "text": "RAGForge is a modular retrieval augmented generation framework for Python.",
                    "source_path": "readme.md",
                },
                {
                    "chunk_id": "chunk_2",
                    "text": "FAISS provides fast similarity search and clustering of dense vector coordinates.",
                    "source_path": "faiss.md",
                }
            ]
        }),
        encoding="utf-8",
    )

    generator = SyntheticDataGenerator(chunks_path=str(chunks_file))
    pairs = generator.generate_pairs(num_pairs=2, output_path=str(out_file))

    assert len(pairs) == 2
    assert "query" in pairs[0]
    assert "positive" in pairs[0]
    assert pairs[0]["chunk_id"] == "chunk_1"
    assert out_file.exists()


def test_bi_encoder_trainer_validation(tmp_path: Path) -> None:
    """Verify BiEncoderTrainer enforces file existence and non-empty checks."""
    trainer = BiEncoderTrainer()
    with pytest.raises(FileNotFoundError):
        trainer.train(training_pairs_path=str(tmp_path / "missing.json"))

    empty_file = tmp_path / "empty.json"
    empty_file.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        trainer.train(training_pairs_path=str(empty_file))


def test_reranker_trainer_validation(tmp_path: Path) -> None:
    """Verify RerankerTrainer enforces file existence and non-empty checks."""
    trainer = RerankerTrainer()
    with pytest.raises(FileNotFoundError):
        trainer.train(training_data_path=str(tmp_path / "missing.json"))

    empty_file = tmp_path / "empty.json"
    empty_file.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        trainer.train(training_data_path=str(empty_file))
