"""
State definition for the RAGForge LangGraph agent.
"""

from typing import Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict):
    """Execution state passed between nodes in the agentic graph."""
    original_query: str
    current_query: str
    query_type: str  # "factual", "summarization", "multi-hop"
    top_k: int
    retrieved_chunks: List[Dict[str, Any]]
    sufficiency: str  # "sufficient" or "insufficient"
    sufficiency_reason: str
    rewrite_count: int
    max_retries: int
    answer: str
    sources: List[str]
    latencies: Dict[str, float]
