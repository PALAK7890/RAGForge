"""
Unit tests for AgenticRAG tools: SEARCH, FIND, OPEN, SUMMARIZE and EvidenceStore.
"""

import json
from unittest.mock import MagicMock

import pytest

from knowledge.agentic.evidence_store import EvidenceStore
from knowledge.agentic.schemas import (
    EvidenceRecord,
    FindInput,
    OpenInput,
    SearchInput,
    SummarizeInput,
)
from knowledge.agentic.tools import FindTool, OpenTool, SearchTool, SummarizeTool


@pytest.fixture
def mock_chunks_file(tmp_path):
    """Fixture creating a mock chunks.json for in-document navigation testing."""
    chunks_path = tmp_path / "chunks.json"
    data = {
        "documents": [
            {"name": "annual_report.txt", "path": "/docs/annual_report.txt"},
            {"name": "policy_2024.txt", "path": "/docs/policy_2024.txt"},
        ],
        "chunks": [
            {
                "chunk_id": "chunk_001",
                "document_id": "annual_report.txt",
                "source_path": "/docs/annual_report.txt",
                "page_number": 1,
                "chunk_index": 0,
                "text": "In fiscal year 2024, total revenue grew by 18.5% reaching $4.2B. Cloud operations contributed 65%.",
                "metadata": {"char_start": 0, "char_end": 100},
            },
            {
                "chunk_id": "chunk_002",
                "document_id": "annual_report.txt",
                "source_path": "/docs/annual_report.txt",
                "page_number": 2,
                "chunk_index": 1,
                "text": "Operating margin improved to 28.4%. Research and development expenses were $620M.",
                "metadata": {"char_start": 101, "char_end": 185},
            },
            {
                "chunk_id": "chunk_003",
                "document_id": "policy_2024.txt",
                "source_path": "/docs/policy_2024.txt",
                "page_number": 1,
                "chunk_index": 0,
                "text": "Data retention period is strictly 7 years for financial records and 3 years for logs.",
                "metadata": {"char_start": 0, "char_end": 85},
            },
        ],
    }
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return str(chunks_path)


# =============================================================================
# SEARCH TOOL TESTS
# =============================================================================

def test_search_tool_multi_query_and_deduplication():
    mock_service = MagicMock()
    # Query 1 returns chunk 1 and 2
    # Query 2 returns chunk 2 (duplicate) and chunk 3
    mock_service.search.side_effect = [
        [
            {"chunk_id": "c1", "text": "Passage 1", "source_path": "/docs/a.txt", "hybrid_score": 0.8, "rerank_score": 0.9},
            {"chunk_id": "c2", "text": "Passage 2", "source_path": "/docs/a.txt", "hybrid_score": 0.7, "rerank_score": 0.85},
        ],
        [
            {"chunk_id": "c2", "text": "Passage 2", "source_path": "/docs/a.txt", "hybrid_score": 0.7, "rerank_score": 0.85},
            {"chunk_id": "c3", "text": "Passage 3", "source_path": "/docs/b.txt", "hybrid_score": 0.6, "rerank_score": 0.75},
        ],
    ]

    tool = SearchTool(service=mock_service)
    res = tool.execute(SearchInput(queries=["revenue growth", "cloud operations"], top_k=5))

    assert res.error is None
    assert len(res.queries_used) == 2
    # c2 must be deduplicated: total 3 unique chunks
    assert res.total_found == 3
    chunk_ids = [e.chunk_id for e in res.evidence]
    assert chunk_ids == ["c1", "c2", "c3"]
    assert res.latency_ms > 0


def test_search_tool_empty_queries():
    tool = SearchTool(service=MagicMock())
    res = tool.execute(SearchInput(queries=["   ", ""]))
    assert res.error is not None
    assert res.error.error_type == "EmptyQueryError"
    assert res.total_found == 0


def test_search_tool_metadata_filtering():
    mock_service = MagicMock()
    mock_service.search.return_value = [
        {"chunk_id": "c1", "text": "Passage 1", "source_path": "/docs/a.txt", "metadata": {"category": "finance"}},
        {"chunk_id": "c2", "text": "Passage 2", "source_path": "/docs/b.txt", "metadata": {"category": "legal"}},
    ]
    tool = SearchTool(service=mock_service)
    res = tool.execute(SearchInput(queries=["test"], metadata_filter={"category": "finance"}))
    assert res.total_found == 1
    assert res.evidence[0].chunk_id == "c1"


# =============================================================================
# FIND TOOL TESTS
# =============================================================================

def test_find_tool_keyword_matching(mock_chunks_file):
    tool = FindTool(chunks_path=mock_chunks_file)
    res = tool.execute(FindInput(document_id="annual_report.txt", expression="revenue"))

    assert res.error is None
    assert res.total_matches == 1
    match = res.matches[0]
    assert match.chunk_id == "chunk_001"
    assert "total revenue grew" in match.snippet
    assert match.page_number == 1
    assert match.char_start == 27  # 'revenue' start position: "In fiscal year 2024, total " is 27 chars


def test_find_tool_case_sensitivity(mock_chunks_file):
    tool = FindTool(chunks_path=mock_chunks_file)
    # Case-insensitive by default
    res_ci = tool.execute(FindInput(document_id="annual_report.txt", expression="REVENUE", case_sensitive=False))
    assert res_ci.total_matches == 1

    # Case-sensitive mismatch
    res_cs = tool.execute(FindInput(document_id="annual_report.txt", expression="REVENUE", case_sensitive=True))
    assert res_cs.total_matches == 0


def test_find_tool_no_matches(mock_chunks_file):
    tool = FindTool(chunks_path=mock_chunks_file)
    res = tool.execute(FindInput(document_id="annual_report.txt", expression="nonexistent_phrase_xyz"))
    assert res.total_matches == 0
    assert res.matches == []
    assert res.error is None


# =============================================================================
# OPEN TOOL TESTS
# =============================================================================

def test_open_tool_by_page(mock_chunks_file):
    tool = OpenTool(chunks_path=mock_chunks_file)
    res = tool.execute(OpenInput(document_id="annual_report.txt", page=2))

    assert res.error is None
    assert len(res.views) == 1
    assert res.views[0].page_number == 2
    assert "Operating margin" in res.views[0].text
    assert len(res.evidence_records) == 1
    assert res.evidence_records[0].retrieval_method == "open"


def test_open_tool_by_offsets(mock_chunks_file):
    tool = OpenTool(chunks_path=mock_chunks_file)
    # Range [50, 150] overlaps both chunk_001 (0-100) and chunk_002 (101-185)
    res = tool.execute(OpenInput(document_id="annual_report.txt", char_start=50, char_end=150))
    assert len(res.views) == 2
    assert res.views[0].chunk_id == "chunk_001"
    assert res.views[1].chunk_id == "chunk_002"


def test_open_tool_truncation(mock_chunks_file):
    tool = OpenTool(chunks_path=mock_chunks_file)
    res = tool.execute(OpenInput(document_id="annual_report.txt", max_output_chars=50))
    assert res.is_truncated is True
    assert "[TRUNCATED]" in res.views[0].text


def test_open_tool_missing_document(mock_chunks_file):
    tool = OpenTool(chunks_path=mock_chunks_file)
    res = tool.execute(OpenInput(document_id="missing_file.pdf"))
    assert res.error is not None
    assert res.error.error_type == "DocumentNotFoundError"
    assert res.views == []


# =============================================================================
# SUMMARIZE TOOL TESTS
# =============================================================================

def test_summarize_tool_extractive():
    tool = SummarizeTool(llm=None)
    records = {
        "e1": EvidenceRecord(
            evidence_id="e1",
            document_id="annual_report.txt",
            source_path="/docs/annual_report.txt",
            text="Revenue surged 18.5% reaching $4.2B in FY2024. Growth was driven by cloud.",
            rerank_score=0.9,
        ),
        "e2": EvidenceRecord(
            evidence_id="e2",
            document_id="policy_2024.txt",
            source_path="/docs/policy_2024.txt",
            text="Financial data retention is mandatory for 7 years.",
            rerank_score=0.8,
        ),
    }

    res = tool.execute(SummarizeInput(evidence_ids=["e1", "e2"], token_budget=500), evidence_store=records)
    assert res.error is None
    assert res.is_extractive is True
    assert len(res.source_references) == 2
    assert "18.5%" in res.summary_text or "Revenue" in res.summary_text
    assert res.token_estimate > 0


def test_summarize_tool_empty_evidence():
    tool = SummarizeTool()
    res = tool.execute(SummarizeInput(evidence_ids=["unknown_id"]), evidence_store={})
    assert res.error is not None
    assert res.error.error_type == "EvidenceNotFoundError"


# =============================================================================
# EVIDENCE STORE TESTS
# =============================================================================

def test_evidence_store_deduplication():
    store = EvidenceStore()
    store.add(EvidenceRecord(
        evidence_id="e1",
        chunk_id="c1",
        document_id="doc.txt",
        text="Sample text A",
        rerank_score=0.6,
    ))
    # Duplicate chunk with higher score
    store.add(EvidenceRecord(
        evidence_id="e2",
        chunk_id="c1",
        document_id="doc.txt",
        text="Sample text A",
        rerank_score=0.95,
    ))
    store.add(EvidenceRecord(
        evidence_id="e3",
        chunk_id="c2",
        document_id="doc.txt",
        text="Sample text B",
        rerank_score=0.5,
    ))

    assert len(store) == 3
    removed = store.deduplicate()
    assert removed == 1
    assert len(store) == 2
    # Verify e2 (higher score) was kept over e1
    assert store.get("e2") is not None
    assert store.get("e1") is None


def test_evidence_store_citation_verification():
    store = EvidenceStore()
    store.add(EvidenceRecord(
        evidence_id="e_rev_1",
        chunk_id="c1",
        document_id="annual_report.txt",
        source_path="/docs/annual_report.txt",
        page_number=1,
        text="Revenue was $4.2B.",
    ))

    # Test valid citation format
    answer_valid = "According to [annual_report.txt, p.1], revenue reached $4.2B."
    v_report = store.validate_citations(answer_valid)
    assert v_report["total_cited"] == 1
    assert v_report["verified_count"] == 1
    assert v_report["citation_integrity"] == 1.0

    # Test invalid citation
    answer_invalid = "Profits doubled according to [secret_ledger.pdf]."
    v_report_inv = store.validate_citations(answer_invalid)
    assert v_report_inv["total_cited"] == 1
    assert v_report_inv["unverified_count"] == 1
    assert v_report_inv["citation_integrity"] == 0.0


def test_evidence_store_serialization():
    store = EvidenceStore()
    store.add(EvidenceRecord(
        evidence_id="test1",
        document_id="doc.txt",
        text="Hello world",
        page_number=3,
    ))
    data = store.to_dict()
    restored = EvidenceStore.from_dict(data)
    assert len(restored) == 1
    assert restored.get("test1").text == "Hello world"
    assert restored.get("test1").page_number == 3
