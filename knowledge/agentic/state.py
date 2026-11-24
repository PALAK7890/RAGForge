"""
State definition for the AgenticRAG LangGraph controller.
"""

from typing import Any, Dict, List, Optional, TypedDict


class AgenticGraphState(TypedDict):
    """
    Execution state passed across nodes in the research-grade AgenticRAG state machine.
    """
    question: str
    step: int
    max_steps: int
    evidence_store_data: Dict[str, Any]
    tool_history: List[Dict[str, Any]]
    discovered_docs: List[str]
    sub_questions: List[str]
    current_action: Dict[str, Any]
    decision_history: List[Dict[str, Any]]
    clarification_question: str
    sufficiency: str  # "pending", "sufficient", "insufficient", "unanswerable"
    sufficiency_reason: str
    answer: str
    citations: List[Dict[str, Any]]
    citation_report: Dict[str, Any]
    latencies: Dict[str, float]
    error: Optional[str]
    abstained: bool
