"""
End-to-end integration and resilience tests for the AgenticRAG framework.
Tests multi-document reasoning, failure modes, budget enforcement, and CLI invocations.
"""

import json
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from knowledge.agentic.controller import AgenticController
from knowledge.agentic.schemas import EvidenceRecord, SearchResult
from knowledge.cli.main import app


@pytest.fixture
def multi_doc_fixture():
    """Two distinct documents for cross-document reasoning testing."""
    return [
        EvidenceRecord(
            evidence_id="ev_fin",
            chunk_id="chunk_fin_1",
            document_id="financials_2024.pdf",
            source_path="/docs/financials_2024.pdf",
            page_number=2,
            char_start=100,
            char_end=350,
            text="Cloud division revenue reached $4.2B in 2024. Year-over-year operational expansion was 22%.",
            retrieval_method="search",
            rerank_score=0.92,
        ),
        EvidenceRecord(
            evidence_id="ev_sec",
            chunk_id="chunk_sec_1",
            document_id="security_compliance.txt",
            source_path="/docs/security_compliance.txt",
            page_number=1,
            char_start=0,
            char_end=200,
            text="All cloud workloads require SOC2 Type II certification and continuous automated audits.",
            retrieval_method="open",
            rerank_score=0.88,
        ),
    ]


def test_multi_document_evidence_aggregation(multi_doc_fixture):
    """Verify agent combines facts across separate documents into a coherent answer."""
    mock_search = MagicMock()
    mock_search.execute.return_value = SearchResult(
        queries_used=["cloud requirements and revenue"],
        evidence=multi_doc_fixture,
        total_found=2,
        latency_ms=15.0,
    )

    mock_llm = MagicMock()
    mock_llm.generate.return_value = (
        "Cloud revenue reached $4.2B [financials_2024.pdf, p. 2], "
        "and operations comply with SOC2 Type II [security_compliance.txt, p. 1]."
    )

    controller = AgenticController(
        search_tool=mock_search,
        find_tool=MagicMock(),
        open_tool=MagicMock(),
        llm=mock_llm,
    )

    result = controller.run("Summarize cloud revenue and compliance status", max_steps=3)

    assert result["abstained"] is False
    assert len(result["discovered_docs"]) == 2
    assert "financials_2024.pdf" in result["discovered_docs"]
    assert "security_compliance.txt" in result["discovered_docs"]
    assert result["citation_report"]["citation_integrity"] == 1.0


def test_budget_enforcement_and_termination():
    """Verify agent strictly terminates when max_steps budget is reached."""
    mock_search = MagicMock()
    mock_search.execute.return_value = SearchResult(
        queries_used=["test"],
        evidence=[
            EvidenceRecord(
                evidence_id="e1",
                chunk_id="c1",
                document_id="doc.txt",
                text="Incomplete fragment...",
            )
        ],
        total_found=1,
        latency_ms=5.0,
    )

    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Answer text [doc.txt]."

    controller = AgenticController(
        search_tool=mock_search,
        find_tool=MagicMock(),
        open_tool=MagicMock(),
        llm=MagicMock(),
    )

    # Hard cap of 2 steps
    result = controller.run("Complex multi-hop question", max_steps=2)
    assert result["steps"] <= 2
    assert result["sufficiency"] in ("sufficient", "unanswerable")


def test_empty_index_graceful_handling():
    """Verify empty retrieval handles gracefully without raising unhandled exceptions."""
    mock_search = MagicMock()
    mock_search.execute.return_value = SearchResult(
        queries_used=["empty"],
        evidence=[],
        total_found=0,
        latency_ms=2.0,
    )

    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Answer text [doc.txt]."

    controller = AgenticController(
        search_tool=mock_search,
        find_tool=MagicMock(),
        open_tool=MagicMock(),
        llm=MagicMock(),
    )

    result = controller.run("Query against empty knowledge base", max_steps=2)
    assert result["abstained"] is True
    assert "cannot answer" in result["answer"].lower()


def test_llm_failure_resilience(multi_doc_fixture):
    """Verify that if the LLM generation throws an exception, controller provides extractive fallback."""
    mock_search = MagicMock()
    mock_search.execute.return_value = SearchResult(
        queries_used=["test"],
        evidence=multi_doc_fixture,
        total_found=2,
        latency_ms=5.0,
    )

    mock_llm = MagicMock()
    mock_llm.generate.side_effect = RuntimeError("Ollama connection timed out")

    controller = AgenticController(
        search_tool=mock_search,
        find_tool=MagicMock(),
        open_tool=MagicMock(),
        llm=mock_llm,
    )

    result = controller.run("What is the cloud revenue?", max_steps=2)
    assert result["abstained"] is False
    # Fallback answer generated from top evidence
    assert "Based on retrieved evidence" in result["answer"]
    assert len(result["citations"]) > 0


def test_cli_agentic_ask_json_output(monkeypatch):
    """Test CLI agentic-ask command with --output-json flag."""
    from knowledge.agentic.controller import AgenticController

    mock_run = MagicMock(return_value={
        "question": "What is RAGForge?",
        "answer": "RAGForge is a research-driven agentic framework [readme.md].",
        "abstained": False,
        "sufficiency": "sufficient",
        "sufficiency_reason": "Facts verified.",
        "steps": 2,
        "tool_history": [
            {"step": 1, "tool": "SEARCH", "results_count": 2, "duration_ms": 10.0},
        ],
        "discovered_docs": ["readme.md"],
        "citations": [],
        "citation_report": {"total_cited": 1, "verified_count": 1, "citation_integrity": 1.0},
        "latencies": {},
        "evidence_count": 2,
    })
    monkeypatch.setattr(AgenticController, "run", mock_run)

    runner = CliRunner()
    result = runner.invoke(app, ["agentic-ask", "What is RAGForge?", "--output-json", "--max-steps", "3"])
    assert result.exit_code == 0
    parsed = json.loads(result.stdout)
    assert parsed["question"] == "What is RAGForge?"
    assert parsed["abstained"] is False
    assert parsed["steps"] == 2
