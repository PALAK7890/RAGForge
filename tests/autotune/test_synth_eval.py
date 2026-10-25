"""
Unit tests for SynthEvalGenerator and related eval data structures.
"""

from typing import List, Tuple

import pytest

from knowledge.autotune.synth_eval import (
    EvalSample,
    EvalSet,
    SynthEvalGenerator,
)


class MockLLM:
    """Deterministic mock LLM for testing question generation."""

    def __init__(self, should_fail: bool = False, return_short: bool = False) -> None:
        self.should_fail = should_fail
        self.return_short = return_short
        self.calls: List[Tuple[str, str]] = []

    def generate(self, question: str, context: str) -> str:
        self.calls.append((question, context))
        if self.should_fail:
            raise RuntimeError("Ollama service unavailable")
        if self.return_short:
            return "Too short"
        return "What is the core architecture of this system?"


def test_eval_sample_properties() -> None:
    sample = EvalSample(
        question="What is FAISS?",
        source_path="/docs/faiss.md",
        char_start=10,
        char_end=120,
        gold_text="FAISS is a library for efficient similarity search.",
    )
    assert sample.has_span() is True
    assert len(sample.text_hash) == 40
    d = sample.to_dict()
    assert d["question"] == "What is FAISS?"
    assert d["char_start"] == 10
    assert d["char_end"] == 120
    assert d["text_hash"] == sample.text_hash

    # Without span
    sample_no_span = EvalSample(
        question="What is FAISS?",
        source_path="/docs/faiss.md",
        char_start=None,
        char_end=None,
        gold_text="FAISS is a library...",
    )
    assert sample_no_span.has_span() is False
    assert len(sample_no_span.text_hash) == 40


def test_eval_set_lengths() -> None:
    s1 = EvalSample("q1", "p1", 0, 10, "t1")
    s2 = EvalSample("q2", "p2", 0, 10, "t2")
    s3 = EvalSample("q3", "p3", 0, 10, "t3")
    eval_set = EvalSet(train=[s1, s2], holdout=[s3])

    assert len(eval_set) == 3
    assert len(eval_set.train) == 2
    assert len(eval_set.holdout) == 1
    assert len(eval_set.all_samples) == 3


def test_synth_eval_generator_validation() -> None:
    mock_llm = MockLLM()
    with pytest.raises(ValueError, match="n must be >= 2"):
        SynthEvalGenerator(llm=mock_llm, n=1)

    with pytest.raises(ValueError, match="holdout_ratio must be in"):
        SynthEvalGenerator(llm=mock_llm, n=10, holdout_ratio=1.5)

    with pytest.raises(ValueError, match="holdout_ratio must be in"):
        SynthEvalGenerator(llm=mock_llm, n=10, holdout_ratio=0.0)


def test_synth_eval_generator_normal() -> None:
    mock_llm = MockLLM()
    generator = SynthEvalGenerator(llm=mock_llm, n=5, holdout_ratio=0.2, seed=42, min_chunk_len=50)

    chunks = [
        {
            "chunk_id": f"chunk_{i}",
            "text": f"This is chunk number {i} containing enough characters to pass the min_chunk_len threshold.",
            "source_path": f"/docs/doc_{i}.txt",
            "char_start": 0,
            "char_end": 85,
        }
        for i in range(10)
    ]

    eval_set = generator.generate(chunks)

    assert len(eval_set) == 5
    assert len(eval_set.holdout) == 1
    assert len(eval_set.train) == 4
    assert len(mock_llm.calls) == 5
    for s in eval_set.all_samples:
        assert s.question == "What is the core architecture of this system?"
        assert s.char_start == 0
        assert s.char_end == 85
        assert s.has_span() is True


def test_synth_eval_generator_deterministic() -> None:
    mock_llm = MockLLM()
    chunks = [
        {
            "chunk_id": f"chunk_{i}",
            "text": f"Document chunk payload {i} with sufficient text length to exceed minimum requirement.",
            "source_path": f"/doc_{i}.txt",
            "char_start": i * 10,
            "char_end": i * 10 + 50,
        }
        for i in range(20)
    ]

    g1 = SynthEvalGenerator(llm=mock_llm, n=6, seed=123)
    g2 = SynthEvalGenerator(llm=mock_llm, n=6, seed=123)

    set1 = g1.generate(chunks)
    set2 = g2.generate(chunks)

    assert [s.source_path for s in set1.train] == [s.source_path for s in set2.train]
    assert [s.source_path for s in set1.holdout] == [s.source_path for s in set2.holdout]


def test_synth_eval_generator_fallback_on_llm_error() -> None:
    failing_llm = MockLLM(should_fail=True)
    generator = SynthEvalGenerator(llm=failing_llm, n=3, seed=42, min_chunk_len=50)

    chunks = [
        {
            "chunk_id": f"c_{i}",
            "text": f"Sentence {i} of document. More words to reach length threshold.",
            "source_path": f"doc_{i}.txt",
            "char_start": 0,
            "char_end": 60,
        }
        for i in range(5)
    ]

    eval_set = generator.generate(chunks)
    assert len(eval_set) == 3
    # Heuristic fallback format: "What is <first sentence part>?"
    for s in eval_set.all_samples:
        assert s.question.startswith("What is ")


def test_synth_eval_generator_fallback_on_short_response() -> None:
    short_llm = MockLLM(return_short=True)
    generator = SynthEvalGenerator(llm=short_llm, n=2, seed=42, min_chunk_len=50)

    chunks = [
        {
            "chunk_id": f"c_{i}",
            "text": "Deep learning systems require optimization. Additional text to pad.",
            "source_path": f"doc_{i}.txt",
        }
        for i in range(3)
    ]

    eval_set = generator.generate(chunks)
    assert len(eval_set) == 2
    for s in eval_set.all_samples:
        assert s.question.startswith("What is ")


def test_synth_eval_generator_legacy_chunks_backward_compat() -> None:
    mock_llm = MockLLM()
    generator = SynthEvalGenerator(llm=mock_llm, n=2, seed=42, min_chunk_len=50)

    legacy_chunks = [
        {
            "chunk_id": "legacy_1",
            "text": "Legacy chunk without any char_start or char_end keys present in dictionary.",
            "source_path": "legacy.txt",
        },
        {
            "chunk_id": "legacy_2",
            "text": "Another legacy chunk without char offsets in serialization dictionary format.",
            "source_path": "legacy2.txt",
        },
    ]

    eval_set = generator.generate(legacy_chunks)
    assert len(eval_set) == 2
    for s in eval_set.all_samples:
        assert s.char_start is None
        assert s.char_end is None
        assert s.has_span() is False


def test_synth_eval_generator_no_eligible_chunks() -> None:
    mock_llm = MockLLM()
    generator = SynthEvalGenerator(llm=mock_llm, n=5, min_chunk_len=100)

    tiny_chunks = [{"chunk_id": "1", "text": "too short", "source_path": "t.txt"}]

    with pytest.raises(ValueError, match="No chunks with at least 100 characters found"):
        generator.generate(tiny_chunks)
