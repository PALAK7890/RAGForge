"""
Unit tests for the AgenticRAG controller, state machine, and decision loop.
"""

from unittest.mock import MagicMock

import pytest

from knowledge.agentic.controller import AgenticController
from knowledge.agentic.schemas import (
    EvidenceRecord,
    FindMatch,
    FindResult,
    OpenResult,
    SearchResult,
)
from knowledge.agentic.state import AgenticGraphState
from knowledge.llms.ollama_llm import OllamaLLM


def test_plan_and_route_comparative_decomposition():
    controller = AgenticController(
        search_tool=MagicMock(),
        find_tool=MagicMock(),
        open_tool=MagicMock(),
        summarize_tool=MagicMock(),
        llm=MagicMock(),
    )

    state: AgenticGraphState = {
        "question": "Compare the revenue growth between 2023 and 2024",
        "step": 0,
        "max_steps": 5,
        "evidence_store_data": {},
        "tool_history": [],
        "discovered_docs": [],
        "sub_questions": [],
        "current_action": {},
        "sufficiency": "pending",
        "sufficiency_reason": "",
        "answer": "",
        "citations": [],
        "citation_report": {},
        "latencies": {},
        "error": None,
        "abstained": False,
    }

    routed = controller.plan_and_route(state)
    assert len(routed["sub_questions"]) > 0
    assert routed["current_action"]["tool"] == "search"
    assert "Compare the revenue growth" in routed["current_action"]["args"]["queries"][0]


def test_agentic_controller_full_loop_execution():
    mock_search = MagicMock()
    mock_find = MagicMock()
    mock_open = MagicMock()
    mock_llm = MagicMock()

    # Search returns 1 evidence record
    mock_search.execute.return_value = SearchResult(
        queries_used=["What is the cloud revenue?"],
        evidence=[
            EvidenceRecord(
                evidence_id="e_rev",
                chunk_id="c1",
                document_id="report.txt",
                source_path="/docs/report.txt",
                page_number=1,
                text="Cloud revenue reached $4.2B with 18.5% YoY growth.",
                hybrid_score=0.9,
                rerank_score=0.95,
            )
        ],
        total_found=1,
        latency_ms=10.0,
    )

    # Find returns a match in report.txt
    mock_find.execute.return_value = FindResult(
        document_id="report.txt",
        expression="revenue",
        matches=[
            FindMatch(
                chunk_id="c1",
                document_id="report.txt",
                source_path="/docs/report.txt",
                page_number=1,
                char_start=0,
                char_end=100,
                snippet="Cloud revenue reached $4.2B",
                match_offset_in_text=6,
            )
        ],
        total_matches=1,
        latency_ms=5.0,
    )

    # Open returns surrounding page
    mock_open.execute.return_value = OpenResult(
        document_id="report.txt",
        source_path="/docs/report.txt",
        views=[],
        total_chars=100,
        evidence_records=[],
        latency_ms=5.0,
    )

    mock_llm.generate.return_value = "Cloud revenue grew 18.5% to $4.2B [report.txt, p.1]."

    controller = AgenticController(
        search_tool=mock_search,
        find_tool=mock_find,
        open_tool=mock_open,
        llm=mock_llm,
    )

    res = controller.run("What is the cloud revenue?", max_steps=4)

    assert res["abstained"] is False
    assert res["steps"] > 1
    assert len(res["tool_history"]) >= 2
    assert "4.2B" in res["answer"]
    assert res["evidence_count"] >= 1
    # Check that citations were verified
    assert res["citation_report"]["verified_count"] >= 1


def test_agentic_controller_grounded_abstention():
    mock_search = MagicMock()
    # Search yields zero results
    mock_search.execute.return_value = SearchResult(
        queries_used=["nonexistent topic"],
        evidence=[],
        total_found=0,
        latency_ms=5.0,
    )

    controller = AgenticController(
        search_tool=mock_search,
        find_tool=MagicMock(),
        open_tool=MagicMock(),
        llm=MagicMock(),
    )

    res = controller.run("What is the secret patent formula?", max_steps=2)

    assert res["abstained"] is True
    assert "cannot answer" in res["answer"].lower()
    assert res["citations"] == []


@pytest.mark.parametrize("page,label", [(None, "[policy.md]"), (3, "[policy.md, p.3]")])
def test_generation_receives_copyable_source_labels_without_nested_prompt(monkeypatch, page, label):
    search = MagicMock()
    search.execute.return_value = SearchResult(
        queries_used=["How long?"],
        evidence=[EvidenceRecord(document_id="policy.md", page_number=page, text="Retain for 30 days.")],
        total_found=1,
        latency_ms=0,
    )
    chat = MagicMock(return_value={"message": {"content": f"Retain for 30 days {label}."}})
    monkeypatch.setattr("knowledge.llms.ollama_llm.ollama.chat", chat)
    result = AgenticController(search_tool=search, llm=OllamaLLM()).run("How long?", max_steps=1)
    prompt = chat.call_args.kwargs["messages"][0]["content"]
    assert f"Source: {label}\nRetain for 30 days." in prompt
    assert prompt.count("Question:") == 1
    assert "[DocumentName" not in prompt
    assert "Evidence:" not in prompt
    assert result["citation_report"]["verified_count"] == 1
    assert result["citation_report"]["unverified_count"] == 0


@pytest.mark.parametrize("draft", ["Retain for 30 days.", "Retain for 30 days [policy.md, p.1]."])
def test_missing_or_invented_citations_get_one_grounded_correction(draft):
    llm = MagicMock()
    llm.generate.side_effect = [draft, "Retain for 30 days [policy.md]."]
    controller = AgenticController(llm=llm)
    result = controller.synthesize_answer({
        "question": "How long?", "latencies": {},
        "evidence_store_data": {"e": EvidenceRecord(evidence_id="e", document_id="policy.md", text="Retain for 30 days.").model_dump()},
    })
    assert result["answer"] == "Retain for 30 days [policy.md]."
    assert llm.generate.call_count == 2
    assert result["citation_report"]["repair_attempted"] is True
    assert result["citation_report"]["missing_citations"] is False


@pytest.mark.parametrize("repair", ["Retain for 30 days.", "Retain for 30 days [fake.md].", RuntimeError("offline")])
def test_failed_correction_retains_answer_and_reports_failure_without_looping(repair):
    llm = MagicMock()
    llm.generate.side_effect = ["Retain for 30 days.", repair]
    result = AgenticController(llm=llm).synthesize_answer({
        "question": "How long?", "latencies": {},
        "evidence_store_data": {"e": EvidenceRecord(evidence_id="e", document_id="policy.md", text="Retain for 30 days.").model_dump()},
    })
    assert result["answer"] == "Retain for 30 days."
    assert llm.generate.call_count == 2
    assert result["citation_report"]["missing_citations"] is True
    if isinstance(repair, Exception):
        assert "Citation correction failed" in result["error"]


def test_plain_refusal_does_not_trigger_citation_correction():
    llm = MagicMock()
    llm.generate.return_value = "I could not find the answer in the indexed documents."
    result = AgenticController(llm=llm).synthesize_answer({
        "question": "How long?", "latencies": {},
        "evidence_store_data": {"e": EvidenceRecord(evidence_id="e", document_id="policy.md", text="Unrelated fact.").model_dump()},
    })
    assert llm.generate.call_count == 1
    assert result["citation_report"]["missing_citations"] is False
    assert result["abstained"] is True


def test_refusal_from_citation_correction_sets_abstention_flag():
    refusal = "I could not find the answer in the indexed documents."
    llm = MagicMock()
    llm.generate.side_effect = ["An unsupported answer.", refusal]
    result = AgenticController(llm=llm).synthesize_answer({
        "question": "How long?", "latencies": {},
        "evidence_store_data": {"e": EvidenceRecord(evidence_id="e", document_id="policy.md", text="Unrelated fact.").model_dump()},
    })
    assert result["answer"] == refusal
    assert result["abstained"] is True
    assert result["citation_report"]["repair_attempted"] is True
    assert result["citation_report"]["missing_citations"] is False
