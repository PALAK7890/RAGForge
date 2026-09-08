"""
Unit tests for the LangGraph agentic retrieval layer.
"""

from knowledge.agent.graph import build_agent_graph, should_continue
from knowledge.agent.nodes import grade_sufficiency, route_query
from knowledge.agent.state import AgentState


def test_route_query_classification() -> None:
    """Verify query router node classifies query types and assigns appropriate top_k."""
    # Factual query
    state_factual: AgentState = {
        "original_query": "What is FAISS?",
        "current_query": "What is FAISS?",
        "query_type": "",
        "top_k": 0,
        "retrieved_chunks": [],
        "sufficiency": "",
        "sufficiency_reason": "",
        "rewrite_count": 0,
        "max_retries": 2,
        "answer": "",
        "sources": [],
        "latencies": {},
    }
    res = route_query(state_factual)
    assert res["query_type"] == "factual"
    assert res["top_k"] == 5

    # Summarization query
    state_sum: AgentState = {
        "original_query": "Give me an overview and summarize the architecture.",
        "current_query": "",
        "query_type": "",
        "top_k": 0,
        "retrieved_chunks": [],
        "sufficiency": "",
        "sufficiency_reason": "",
        "rewrite_count": 0,
        "max_retries": 2,
        "answer": "",
        "sources": [],
        "latencies": {},
    }
    res_sum = route_query(state_sum)
    assert res_sum["query_type"] == "summarization"
    assert res_sum["top_k"] == 8

    # Multi-hop comparison query
    state_comp: AgentState = {
        "original_query": "Compare dense retrieval vs sparse BM25 search.",
        "current_query": "",
        "query_type": "",
        "top_k": 0,
        "retrieved_chunks": [],
        "sufficiency": "",
        "sufficiency_reason": "",
        "rewrite_count": 0,
        "max_retries": 2,
        "answer": "",
        "sources": [],
        "latencies": {},
    }
    res_comp = route_query(state_comp)
    assert res_comp["query_type"] == "multi-hop"
    assert res_comp["top_k"] == 6


def test_grade_sufficiency_empty_chunks() -> None:
    """Verify grade node marks sufficiency as insufficient when no chunks are found."""
    state: AgentState = {
        "original_query": "Nonexistent query",
        "current_query": "Nonexistent query",
        "query_type": "factual",
        "top_k": 5,
        "retrieved_chunks": [],
        "sufficiency": "pending",
        "sufficiency_reason": "",
        "rewrite_count": 0,
        "max_retries": 2,
        "answer": "",
        "sources": [],
        "latencies": {},
    }
    res = grade_sufficiency(state)
    assert res["sufficiency"] == "insufficient"
    assert "No chunks" in res["sufficiency_reason"]


def test_should_continue_logic() -> None:
    """Verify conditional edge router transitions between rewrite and generate."""
    # Sufficient -> generate
    state_suff: AgentState = {
        "original_query": "Q",
        "current_query": "Q",
        "query_type": "factual",
        "top_k": 5,
        "retrieved_chunks": [{"text": "Content"}],
        "sufficiency": "sufficient",
        "sufficiency_reason": "",
        "rewrite_count": 0,
        "max_retries": 2,
        "answer": "",
        "sources": [],
        "latencies": {},
    }
    assert should_continue(state_suff) == "generate"

    # Insufficient with remaining retries -> rewrite
    state_retry: AgentState = {
        "original_query": "Q",
        "current_query": "Q",
        "query_type": "factual",
        "top_k": 5,
        "retrieved_chunks": [],
        "sufficiency": "insufficient",
        "sufficiency_reason": "",
        "rewrite_count": 0,
        "max_retries": 2,
        "answer": "",
        "sources": [],
        "latencies": {},
    }
    assert should_continue(state_retry) == "rewrite"

    # Insufficient but exceeded retries -> generate
    state_exceeded: AgentState = {
        "original_query": "Q",
        "current_query": "Q",
        "query_type": "factual",
        "top_k": 5,
        "retrieved_chunks": [],
        "sufficiency": "insufficient",
        "sufficiency_reason": "",
        "rewrite_count": 2,
        "max_retries": 2,
        "answer": "",
        "sources": [],
        "latencies": {},
    }
    assert should_continue(state_exceeded) == "generate"


def test_build_agent_graph() -> None:
    """Verify StateGraph compiles successfully."""
    graph = build_agent_graph()
    assert graph is not None
