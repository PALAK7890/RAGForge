"""Regressions for benchmark credibility and real boundary failures."""

import json
from unittest.mock import MagicMock

import pytest

from knowledge.agentic.citations import CitationValidator
from knowledge.agentic.controller import AgenticController
from knowledge.agentic.evidence_store import EvidenceStore
from knowledge.agentic.schemas import EvidenceRecord, FindInput, OpenInput, SearchResult, ToolError
from knowledge.agentic.tools import FindTool, OpenTool
from knowledge.evaluation.ablation import AblationRunner
from knowledge.evaluation.compare_systems import SystemComparator


@pytest.fixture
def comparator():
    return SystemComparator(service=MagicMock())


@pytest.mark.parametrize("unanswerable", [True, False])
def test_errors_never_receive_quality_credit(comparator, unanswerable):
    item = {
        "is_unanswerable": unanswerable,
        "expected_documents": [] if unanswerable else ["a.md"],
        "gold_keywords": ["gold"],
    }
    scores = comparator.score_single(
        item,
        {
            "error": "model unavailable",
            "abstained": True,
            "answer": "gold",
            "retrieved_docs": ["a.md"],
        },
    )
    assert scores["error"] == 1
    assert scores["abstention_correctness"] == 0
    assert scores["retrieval_hit"] == (None if unanswerable else 0)
    assert scores["keyword_recall"] == (None if unanswerable else 0)


def test_exact_ranked_document_metrics(comparator):
    item = {"expected_documents": ["report.md", "policy.md"], "gold_keywords": ["7"]}
    score = comparator.score_single(
        item, {"retrieved_docs": ["old-report.md", "report.md", "report.md"], "answer": "7"}
    )
    assert score["document_recall"] == 0.5
    assert score["document_precision"] == 0.5
    assert score["document_mrr"] == 0.5
    assert (
        comparator.score_single(item, {"retrieved_docs": ["old-report.md"]})["retrieval_hit"] == 0
    )


def test_unanswerable_excluded_from_retrieval(comparator):
    score = comparator.score_single(
        {"is_unanswerable": True},
        {"answer": "I could not find the answer in the indexed documents."},
    )
    assert score["retrieval_hit"] is None
    assert score["abstention_correctness"] == 1


def test_benchmark_fails_before_inference_for_missing_sources(tmp_path):
    chunks = tmp_path / "chunks.json"
    chunks.write_text(json.dumps({"chunks": [{"source_path": "/docs/a.md"}]}))
    service = MagicMock(chunks_path=chunks)
    comparator = SystemComparator(service)
    with pytest.raises(ValueError, match="missing.md"):
        comparator.validate_corpus([{"expected_documents": ["missing.md"]}])
    service.ask.assert_not_called()


def test_benchmark_saves_raw_records_and_hashes(tmp_path, monkeypatch):
    chunks = tmp_path / "chunks.json"
    chunks.write_text(json.dumps({"chunks": [{"source_path": "/docs/a.md"}]}))
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(
            [{"id": "one", "question": "q", "split": "held_out", "expected_documents": ["a.md"]}]
        )
    )
    comparator = SystemComparator(MagicMock(chunks_path=chunks))
    for name in ["evaluate_baseline_a", "evaluate_baseline_b", "evaluate_system_c"]:
        monkeypatch.setattr(
            comparator, name, lambda q: {"answer": "answer", "retrieved_docs": ["a.md"]}
        )
    result = comparator.run_benchmark(str(dataset), split="held_out")
    assert len(result["records"]) == 3
    assert len(result["metadata"]["dataset_sha256"]) == 64
    assert result["summary"]["System_C_AgenticRAG"]["retrieval_hit"] == 1
    with pytest.raises(ValueError, match="No benchmark"):
        comparator.run_benchmark(str(dataset), split="dev")


@pytest.fixture
def navigation(tmp_path):
    path = tmp_path / "chunks.json"
    path.write_text(
        json.dumps(
            {
                "chunks": [
                    {
                        "source_path": "/docs/policy.md",
                        "document_id": "doc1",
                        "chunk_id": "c",
                        "chunk_index": 0,
                        "page_number": 2,
                        "text": "Retention is seven years. " * 10,
                        "metadata": {"char_start": 0, "char_end": 260},
                    }
                ]
            }
        )
    )
    return FindTool(str(path)), OpenTool(str(path))


@pytest.mark.parametrize("doc", ["", "policy", "../policy.md", "old-policy.md"])
def test_navigation_rejects_partial_document_ids(navigation, doc):
    find, open_tool = navigation
    assert find.execute(FindInput(document_id=doc, expression="Retention")).total_matches == 0
    assert open_tool.execute(OpenInput(document_id=doc)).views == []


@pytest.mark.parametrize("selector", [{"page": 99}, {"chunk_index": 99}, {"char_start": 999}])
def test_open_out_of_range_never_returns_unrelated_content(navigation, selector):
    assert navigation[1].execute(OpenInput(document_id="policy.md", **selector)).views == []


def test_open_enforces_hard_character_cap(navigation):
    result = navigation[1].execute(OpenInput(document_id="policy.md", max_output_chars=50))
    assert result.total_chars <= 50
    assert sum(len(v.text) for v in result.views) <= 50


@pytest.mark.parametrize(
    "citation", ["[policy.md, p.99]", "[policy.md, chars 0-999]", "[old-policy.md]", "[policy]"]
)
def test_both_citation_validators_reject_invented_locations(citation):
    record = EvidenceRecord(
        evidence_id="e",
        document_id="policy.md",
        source_path="/docs/policy.md",
        page_number=2,
        char_start=10,
        char_end=100,
        text="source",
    )
    assert CitationValidator([record]).validate(citation)["valid_citations"] == 0
    store = EvidenceStore()
    store.add(record)
    assert store.validate_citations(citation)["verified_count"] == 0


def test_citation_can_resolve_later_page_and_evidence_prefix():
    records = [
        EvidenceRecord(evidence_id=f"e{p}", document_id="policy.md", page_number=p, text="source")
        for p in [1, 2]
    ]
    assert (
        CitationValidator(records).validate("[policy.md, p.2] [evidence:e1]")["valid_citations"]
        == 2
    )


def test_controller_surfaces_structured_search_failure():
    search = MagicMock()
    search.execute.return_value = SearchResult(
        queries_used=["q"],
        evidence=[],
        total_found=0,
        latency_ms=0,
        error=ToolError(tool_name="SEARCH", error_type="IOError", message="missing index"),
    )
    result = AgenticController(search_tool=search, llm=MagicMock()).run("question", max_steps=1)
    assert "missing index" in result["error"]


@pytest.mark.parametrize("steps", [0, -1, 16])
def test_invalid_step_budget(steps):
    with pytest.raises(ValueError, match="max_steps"):
        AgenticController().run("question", steps)


def test_inert_ablation_rejected():
    with pytest.raises(ValueError, match="inert"):
        AblationRunner()._create_controller("no_summarize")


def test_single_query_ablation_constrains_retry_searches():
    from knowledge.agentic.schemas import SearchInput
    from knowledge.evaluation.ablation import _SingleQuerySearch

    tool = MagicMock()
    wrapped = _SingleQuerySearch(tool)
    wrapped.execute(SearchInput(queries=["initial", "variant"]))
    wrapped.execute(SearchInput(queries=["retry", "retry overview"]))
    assert [call.args[0].queries for call in tool.execute.call_args_list] == [
        ["initial"],
        ["retry"],
    ]


def test_sse_reports_worker_failure(monkeypatch):
    from fastapi.testclient import TestClient

    from knowledge.api.app import app

    def fail(self, question, max_steps=5):
        raise RuntimeError("controlled failure")

    monkeypatch.setattr(AgenticController, "run", fail)
    response = TestClient(app).get("/v2/agentic-stream/question")
    assert response.status_code == 200
    assert "\n\n" in response.text
    events = [
        json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")
    ]
    assert [event["event"] for event in events] == ["start", "error"]
    assert events[-1]["message"] == "controlled failure"
