"""
REST API module for RAGForge.
"""

from knowledge.api.schemas import (
    DocumentIndexRequest,
    DocumentListResponse,
    SearchRequest,
    SearchResponse,
    AskRequest,
    AskResponse,
    AgentAskRequest,
    AgentAskResponse,
    StatsResponse,
    BenchmarkRequest,
    BenchmarkResponse,
    EvaluateRequest,
    FeedbackRequest,
    FeedbackResponse,
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
