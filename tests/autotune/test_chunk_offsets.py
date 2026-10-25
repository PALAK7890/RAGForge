"""
Tests for DocumentChunk character offsets across chunkers.

Guarantees:
1. `doc.text.strip()[chunk.char_start:chunk.char_end] == chunk.text` across various chunk_sizes and overlaps.
2. Backward compatibility: old chunks.json serialization without char_start/char_end loads cleanly.
3. Edge cases: empty text, single sentence larger than chunk size, short doc.
"""

import pytest

from knowledge.chunkers.base import DocumentChunk
from knowledge.chunkers.fixed_size import FixedSizeChunker
from knowledge.chunkers.recursive import RecursiveChunker
from knowledge.chunkers.sentence import SentenceChunker
from knowledge.loaders.base import LoadedDocument


@pytest.fixture
def sample_document() -> LoadedDocument:
    text = (
        "Artificial intelligence is transforming industries across the globe. "
        "Machine learning models can learn patterns from vast amounts of data. "
        "Natural language processing allows computers to understand human languages. "
        "Retrieval-augmented generation combines search with large language models to reduce hallucinations. "
        "Dense vector retrieval uses embeddings to capture semantic similarity. "
        "BM25 provides lexical matching for exact keywords. "
        "Hybrid search merges the strengths of both dense and sparse retrieval paradigms."
    )
    return LoadedDocument(
        text=text,
        source_path="/docs/ai_overview.txt",
        document_id="doc_ai_01",
        metadata={"category": "ai"},
    )


def test_recursive_chunker_offsets(sample_document: LoadedDocument) -> None:
    chunker = RecursiveChunker()
    raw_text = sample_document.text.strip()

    for size, overlap in [(100, 20), (200, 40), (350, 50)]:
        chunks = chunker.chunk([sample_document], chunk_size=size, chunk_overlap=overlap)
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.char_start is not None
            assert chunk.char_end is not None
            assert chunk.char_end > chunk.char_start
            extracted = raw_text[chunk.char_start:chunk.char_end]
            assert extracted == chunk.text, f"Mismatch: expected {chunk.text!r}, got {extracted!r}"


def test_fixed_size_chunker_offsets(sample_document: LoadedDocument) -> None:
    chunker = FixedSizeChunker()
    raw_text = sample_document.text.strip()

    for size, overlap in [(80, 20), (150, 30), (300, 50)]:
        chunks = chunker.chunk([sample_document], chunk_size=size, chunk_overlap=overlap)
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.char_start is not None
            assert chunk.char_end is not None
            extracted = raw_text[chunk.char_start:chunk.char_end]
            assert extracted == chunk.text, f"Mismatch: expected {chunk.text!r}, got {extracted!r}"


def test_fixed_size_chunker_short_doc() -> None:
    doc = LoadedDocument(
        text="Short text here.",
        source_path="short.txt",
        document_id="doc_short",
    )
    chunker = FixedSizeChunker()
    chunks = chunker.chunk([doc], chunk_size=100, chunk_overlap=20)
    assert len(chunks) == 1
    assert chunks[0].char_start == 0
    assert chunks[0].char_end == len(doc.text.strip())
    assert doc.text.strip()[chunks[0].char_start:chunks[0].char_end] == chunks[0].text


def test_sentence_chunker_offsets(sample_document: LoadedDocument) -> None:
    chunker = SentenceChunker()
    raw_text = sample_document.text.strip()

    for size, overlap in [(150, 30), (250, 50), (400, 80)]:
        chunks = chunker.chunk([sample_document], chunk_size=size, chunk_overlap=overlap)
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.char_start is not None
            assert chunk.char_end is not None
            extracted = raw_text[chunk.char_start:chunk.char_end]
            assert extracted == chunk.text, f"Mismatch: expected {chunk.text!r}, got {extracted!r}"


def test_sentence_chunker_long_sentence_fallback() -> None:
    long_sentence = "A" * 250 + "."
    doc = LoadedDocument(
        text=long_sentence,
        source_path="long_sent.txt",
        document_id="doc_long_sent",
    )
    chunker = SentenceChunker()
    chunks = chunker.chunk([doc], chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.char_start is not None
        assert chunk.char_end is not None
        assert doc.text.strip()[chunk.char_start:chunk.char_end] == chunk.text


def test_backward_compatibility_old_chunks_json() -> None:
    # Simulate old chunks.json without char_start / char_end
    old_serialized_chunk = {
        "chunk_id": "doc_01_0",
        "text": "This is legacy chunk text.",
        "document_id": "doc_01",
        "source_path": "/old/path.txt",
        "page_number": 1,
        "chunk_index": 0,
        "metadata": {"source": "legacy"},
    }

    chunk = DocumentChunk(**old_serialized_chunk)  # type: ignore[arg-type]
    assert chunk.chunk_id == "doc_01_0"
    assert chunk.text == "This is legacy chunk text."
    assert chunk.char_start is None
    assert chunk.char_end is None

    # Dumping back to dict preserves optional None fields
    dumped = chunk.model_dump()
    assert dumped["char_start"] is None
    assert dumped["char_end"] is None
