"""
Unit tests for text chunking strategies (Fixed-size, Sentence-aware, Recursive).
"""

from knowledge.chunkers.fixed_size import FixedSizeChunker
from knowledge.chunkers.recursive import RecursiveChunker
from knowledge.chunkers.sentence import SentenceChunker
from knowledge.loaders.base import LoadedDocument


def test_fixed_size_chunker_basic() -> None:
    """Test fixed-size chunking with exact stride and overlap checks."""
    # Length: 43 characters
    text = "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFG"
    doc = LoadedDocument(
        text=text,
        source_path="dummy.txt",
        document_id="doc1",
        page_number=1,
        metadata={"format": "txt"}
    )

    # chunk_size = 15, overlap = 5, stride = 10
    # Expected chunks:
    # 0: "abcdefghijklmno" (length 15)
    # 1: "klmnopqrstuvwxy" (length 15, overlap "klmno" from chunk 0)
    # 2: "uvwxyz012345678" (length 15, overlap "uvwxy" from chunk 1)
    # 3: "456789ABCDEFG" (length 13, remainder tail)
    chunker = FixedSizeChunker()
    chunks = chunker.chunk([doc], chunk_size=15, chunk_overlap=5)

    assert len(chunks) == 4
    
    assert chunks[0].text == "abcdefghijklmno"
    assert chunks[0].chunk_index == 0
    assert chunks[0].chunk_id == "doc1_0"
    
    assert chunks[1].text == "klmnopqrstuvwxy"
    assert chunks[1].chunk_index == 1
    assert chunks[1].chunk_id == "doc1_1"
    
    assert chunks[2].text == "uvwxyz012345678"
    assert chunks[2].chunk_index == 2
    
    assert chunks[3].text == "456789ABCDEFG"
    assert chunks[3].chunk_index == 3

    # Check correct overlap between consecutive chunks
    assert chunks[0].text[-5:] == chunks[1].text[:5]  # "klmno"
    assert chunks[1].text[-5:] == chunks[2].text[:5]  # "uvwxy"
    assert chunks[2].text[-5:] == chunks[3].text[:5]  # "56789"


def test_multi_page_indexing_uniqueness() -> None:
    """Assert running chunk_index increases sequentially across pages without collision."""
    page1 = LoadedDocument(
        text="This is a long line on page one. It needs to produce multiple chunks.",
        source_path="multi.pdf",
        document_id="doc_multi",
        page_number=1,
        metadata={"format": "pdf"}
    )
    page2 = LoadedDocument(
        text="This is another long line on page two. It also needs multiple chunks.",
        source_path="multi.pdf",
        document_id="doc_multi",
        page_number=2,
        metadata={"format": "pdf"}
    )

    # chunk_size = 20, overlap = 5, stride = 15
    chunker = FixedSizeChunker()
    chunks = chunker.chunk([page1, page2], chunk_size=20, chunk_overlap=5)

    # Assert there are no duplicate chunk_ids
    chunk_ids = [c.chunk_id for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids))

    # Assert sequential indices starting from 0 and continuing without resetting on page 2
    assert chunks[0].chunk_index == 0
    assert chunks[0].chunk_id == "doc_multi_0"
    assert chunks[0].page_number == 1

    # Check page 2 boundary chunks have incremented index
    page2_chunks = [c for c in chunks if c.page_number == 2]
    assert len(page2_chunks) > 0
    for c in page2_chunks:
        assert c.chunk_index >= len(chunks) - len(page2_chunks)
        assert c.chunk_id == f"doc_multi_{c.chunk_index}"


def test_sentence_chunker_fallback_runon() -> None:
    """Test SentenceChunker fallback to char-level splits on runon sentences."""
    runon_text = "ThisIsAVeryLongSentenceWithoutAnySpacesOrPunctuationThatExceedsChunkSizeLimit."
    doc = LoadedDocument(
        text=runon_text,
        source_path="runon.txt",
        document_id="doc_runon",
        page_number=1,
        metadata={"format": "txt"}
    )

    chunker = SentenceChunker()
    # chunk_size = 20, overlap = 5. The sentence is 78 chars, so it should split character-wise.
    chunks = chunker.chunk([doc], chunk_size=20, chunk_overlap=5)

    assert len(chunks) > 1
    # Check that chunks are sequential and under chunk_size
    for c in chunks:
        assert len(c.text) <= 20
        assert c.document_id == "doc_runon"


def test_sentence_chunker_basic() -> None:
    """Test sentence boundaries are preserved and grouped correctly."""
    text = "Sentence one. Sentence two! Sentence three?"
    doc = LoadedDocument(
        text=text,
        source_path="sents.txt",
        document_id="doc_sents",
        page_number=1,
        metadata={"format": "txt"}
    )

    chunker = SentenceChunker()
    # chunk_size = 30, overlap = 5.
    # "Sentence one. Sentence two!" length is 27. Adding " Sentence three?" would make it 43 (exceeds 30).
    # Chunks should be:
    # 0: "Sentence one. Sentence two!"
    # 1: "Sentence three?" (no sentence-level overlap since len("Sentence two!") = 13 > 5 overlap)
    chunks = chunker.chunk([doc], chunk_size=30, chunk_overlap=5)

    assert len(chunks) == 2
    assert chunks[0].text == "Sentence one. Sentence two!"
    assert chunks[1].text == "Sentence three?"


def test_recursive_chunker() -> None:
    """Test recursive character splits along hierarchical delimiters."""
    text = "Paragraph one.\n\nParagraph two.\nParagraph three. Paragraph four."
    doc = LoadedDocument(
        text=text,
        source_path="recur.txt",
        document_id="doc_recur",
        page_number=1,
        metadata={"format": "txt"}
    )

    chunker = RecursiveChunker()
    # chunk_size = 35.
    # "Paragraph one." (len 14).
    # "Paragraph two.\nParagraph three. Paragraph four." (len 45).
    # Recursive chunker should split by \n\n first, yielding "Paragraph one."
    # Then split "Paragraph two.\nParagraph three. Paragraph four." by \n.
    # Yielding "Paragraph two." (len 14) and "Paragraph three. Paragraph four." (len 30).
    chunks = chunker.chunk([doc], chunk_size=35, chunk_overlap=5)

    assert len(chunks) >= 3
    assert chunks[0].text == "Paragraph one."
    assert chunks[1].text == "Paragraph two."
    assert "Paragraph three." in chunks[2].text


def test_recursive_chunker_empty_delimiter_fallback() -> None:
    """Verify recursive chunker terminates and splits by chars when reaching empty string."""
    text = "abcdefghij"
    doc = LoadedDocument(
        text=text,
        source_path="empty.txt",
        document_id="doc_empty",
        page_number=1,
        metadata={"format": "txt"}
    )

    # Use separators ending with empty string ""
    chunker = RecursiveChunker(separators=["\n\n", ""])
    chunks = chunker.chunk([doc], chunk_size=4, chunk_overlap=1)

    # Expected chunks: "abcd", "de", "efgh", "hi", "ij" (or similar character boundaries)
    assert len(chunks) > 0
    for c in chunks:
        assert len(c.text) <= 4
