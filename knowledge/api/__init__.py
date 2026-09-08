"""
REST API module for RAGForge.
"""

from knowledge.api.schemas import (
    AgentAskRequest,
    AgentAskResponse,
    AskRequest,
    AskResponse,
    BenchmarkRequest,
    BenchmarkResponse,
    DocumentIndexRequest,
    DocumentListResponse,
    EvaluateRequest,
    FeedbackRequest,
    FeedbackResponse,
    SearchRequest,
    SearchResponse,
    StatsResponse,
)

__all__ = [
    "DocumentIndexRequest",
    "DocumentListResponse",
    "SearchRequest",
    "SearchResponse",
    "AskRequest",
    "AskResponse",
    "AgentAskRequest",
    "AgentAskResponse",
    "StatsResponse",
    "BenchmarkRequest",
    "BenchmarkResponse",
    "EvaluateRequest",
    "FeedbackRequest",
    "FeedbackResponse",
]
