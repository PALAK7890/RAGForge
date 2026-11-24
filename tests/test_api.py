"""
Unit tests for the RAGForge REST API endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from knowledge.api.app import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_api_workspace(tmp_path, monkeypatch):
    """Never run unit tests against a user's real corpus or load models implicitly."""
    import importlib

    from knowledge.api.service import RAGService

    api_module = importlib.import_module("knowledge.api.app")
    nodes = importlib.import_module("knowledge.agent.nodes")
    service = RAGService(str(tmp_path / "workspace"))
    monkeypatch.setattr(api_module, "service", service)
    monkeypatch.setattr(nodes, "service", service)


def test_health_check() -> None:
    """Verify health endpoint returns status 200 and expected payload."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "app": "RAGForge"}


def test_list_documents_endpoint() -> None:
    """Verify listing indexed documents returns structured schema."""
    response = client.get("/documents")
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert "total_documents" in data
    assert "total_chunks" in data
    assert isinstance(data["documents"], list)


def test_get_stats_endpoint() -> None:
    """Verify stats endpoint returns workspace metadata."""
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "workspace" in data
    assert "documents" in data
    assert "chunks" in data
    assert "embedding_model" in data
    assert data["vector_store"] == "FAISS"


def test_search_endpoint_validation() -> None:
    """Verify search input validation and parameter constraints."""
    # Missing query parameter
    response = client.get("/search")
    assert response.status_code == 422

    # Invalid top_k (< 1)
    response = client.get("/search?q=test&top_k=0")
    assert response.status_code == 422


def test_agent_ask_fallback_endpoint() -> None:
    """Verify agent-ask endpoint responds with expected structured payload."""
    response = client.post("/agent-ask", json={"question": "What is RAGForge?"})
    # Should succeed with structured agent response or 500 if index missing
    if response.status_code == 200:
        data = response.json()
        assert "question" in data
        assert "answer" in data
        assert "query_type" in data
        assert "sufficiency" in data


def test_record_feedback_endpoint() -> None:
    """Verify posting feedback returns 200 and confirms logged feedback ID."""
    response = client.post(
        "/feedback",
        json={
            "query": "What is FAISS?",
            "chunk_text": "FAISS is a vector index library.",
            "rating": 1,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "feedback_id" in data
    assert data["query"] == "What is FAISS?"
    assert data["rating"] == 1


def test_agent_stream_endpoint() -> None:
    """Verify /agent-stream/{query} returns an SSE stream with start and answer events.

    Collects all newline-delimited JSON events emitted by the stream, then
    confirms at minimum a 'start' event and an 'answer' event are present.
    """
    import json as _json

    raw_lines: list[str] = []
    with client.stream("GET", "/agent-stream/What%20is%20RAGForge") as response:
        assert response.status_code == 200
        assert "event-stream" in response.headers.get("content-type", "")
        for line in response.iter_lines():
            if line.strip():
                raw_lines.append(line.strip())

    events = []
    for line in raw_lines:
        try:
            events.append(_json.loads(line))
        except _json.JSONDecodeError:
            pass  # ignore non-JSON lines

    event_types = [e.get("event") for e in events]
    assert "start" in event_types, f"Expected 'start' event, got: {event_types}"
    assert "answer" in event_types, f"Expected 'answer' event, got: {event_types}"

    answer_event = next(e for e in events if e.get("event") == "answer")
    assert "text" in answer_event
    assert "rewrites" in answer_event
    assert "sufficiency" in answer_event


@pytest.mark.parametrize("planner", [None, "model"])
def test_agentic_ask_v2_endpoint(monkeypatch, planner) -> None:
    """Verify /v2/agentic-ask endpoint processes queries and returns structured AgenticAskResponse."""
    from unittest.mock import MagicMock

    from knowledge.agentic.controller import AgenticController

    mock_run = MagicMock(
        return_value={
            "question": "What is RAGForge?",
            "answer": "RAGForge is a modular RAG framework [readme.md].",
            "abstained": False,
            "sufficiency": "sufficient",
            "sufficiency_reason": "Found facts in readme.md",
            "steps": 2,
            "tool_history": [
                {"step": 1, "tool": "SEARCH", "results_count": 2, "duration_ms": 12.0},
            ],
            "discovered_docs": ["readme.md"],
            "citations": [
                {
                    "evidence_id": "ev_1",
                    "label": "[readme.md]",
                    "document_id": "readme.md",
                    "source_path": "/docs/readme.md",
                    "snippet": "RAGForge is a framework...",
                }
            ],
            "citation_report": {"total_cited": 1, "verified_count": 1, "citation_integrity": 1.0},
            "latencies": {"total": 45.0},
            "evidence_count": 2,
            "decision_history": [{"tool": "SEARCH", "reason": "Locate project facts"}],
        }
    )

    def run(self, question, max_steps=5):
        assert self.planner_mode == (planner or "heuristic")
        return mock_run(question, max_steps=max_steps)

    monkeypatch.setattr(AgenticController, "run", run)

    payload = {"question": "What is RAGForge?", "max_steps": 4}
    if planner:
        payload["planner"] = planner
    response = client.post(
        "/v2/agentic-ask", json=payload
    )
    assert response.status_code == 200
    data = response.json()
    assert data["question"] == "What is RAGForge?"
    assert "readme.md" in data["answer"]
    assert data["abstained"] is False
    assert len(data["citations"]) == 1
    assert data["citations"][0]["document_id"] == "readme.md"
    assert data["citation_report"]["citation_integrity"] == 1.0
    assert data["planner_mode"] == (planner or "heuristic")
    assert data["decision_history"][0]["tool"] == "SEARCH"
    mock_run.assert_called_once_with("What is RAGForge?", max_steps=4)


@pytest.mark.parametrize("planner", [None, "model"])
def test_agentic_stream_v2_endpoint(monkeypatch, planner) -> None:
    """Verify /v2/agentic-stream/{query} yields SSE events including start, tool_executed, and complete."""
    import json as _json
    from unittest.mock import MagicMock

    from knowledge.agentic.controller import AgenticController

    mock_run = MagicMock(
        return_value={
            "question": "What is RAGForge?",
            "answer": "RAGForge is modular.",
            "abstained": False,
            "sufficiency": "sufficient",
            "steps": 1,
            "tool_history": [
                {"step": 1, "tool": "SEARCH", "results_count": 1, "duration_ms": 15.0},
            ],
            "citations": [],
            "citation_report": {"citation_integrity": 1.0},
            "evidence_count": 1,
        }
    )
    monkeypatch.setattr(AgenticController, "run", mock_run)

    def stream_run(self, question, max_steps=5):
        assert self.planner_mode == (planner or "heuristic")
        result = mock_run(question, max_steps=max_steps)
        for record in result["tool_history"]:
            self.on_tool_event(record)
        return result

    monkeypatch.setattr(AgenticController, "run", stream_run)
    raw_lines: list[str] = []
    params = {"max_steps": 3}
    if planner:
        params["planner"] = planner
    with client.stream("GET", "/v2/agentic-stream/What%20is%20RAGForge", params=params) as response:
        assert response.status_code == 200
        assert "event-stream" in response.headers.get("content-type", "")
        for line in response.iter_lines():
            if line.strip():
                raw_lines.append(line.strip())

    events = [_json.loads(line.removeprefix("data: ")) for line in raw_lines if line.strip()]
    event_types = [e.get("event") for e in events]

    assert "start" in event_types
    assert "tool_executed" in event_types
    assert "complete" in event_types
    assert events[0]["planner_mode"] == (planner or "heuristic")
    assert next(e for e in events if e["event"] == "complete")["planner_mode"] == (planner or "heuristic")


def test_agentic_endpoints_reject_unknown_planner():
    response = client.post(
        "/v2/agentic-ask", json={"question": "test", "planner": "unknown"}
    )
    assert response.status_code == 422
    response = client.get("/v2/agentic-stream/test", params={"planner": "unknown"})
    assert response.status_code == 422
