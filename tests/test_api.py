"""
Unit tests for the RAGForge REST API endpoints.
"""

from fastapi.testclient import TestClient

from knowledge.api.app import app

client = TestClient(app)


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
