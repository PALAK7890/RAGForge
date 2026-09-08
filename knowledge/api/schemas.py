"""
Pydantic schemas for the RAGForge REST API.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class DocumentIndexRequest(BaseModel):
    """Payload for indexing a file or directory."""
    path: str = Field(description="File or folder path to index")


class DocumentRecord(BaseModel):
    """Metadata for an indexed document."""
    name: str
    path: str
    chunks: int = 0


class DocumentListResponse(BaseModel):
    """Response containing all indexed documents."""
    documents: List[DocumentRecord]
    total_documents: int
    total_chunks: int


class SearchRequest(BaseModel):
    """Search query parameters."""
    query: str = Field(description="Search text query")
    top_k: int = Field(default=5, description="Number of results to return")


class SearchResultChunk(BaseModel):
    """Individual retrieved chunk with ranking scores."""
    chunk_id: Optional[str] = None
    text: str
    source_path: str
    page_number: Optional[int] = None
    cross_score: Optional[float] = None
    hybrid_score: Optional[float] = None


class SearchResponse(BaseModel):
    """Search response payload."""
    query: str
    results: List[SearchResultChunk]


class AskRequest(BaseModel):
    """Question answering query payload."""
    question: str = Field(description="User question")
    top_k: int = Field(default=5, description="Number of chunks to retrieve for context")
    stream: bool = Field(default=False, description="Whether to stream the generated answer")


class AskResponse(BaseModel):
    """Standard RAG answer response."""
    question: str
    answer: str
    sources: List[str]
    chunks: List[SearchResultChunk]


class AgentAskRequest(BaseModel):
    """Agentic RAG question request."""
    question: str = Field(description="User question")
    max_retries: int = Field(default=2, description="Maximum retrieval rewrite attempts")


class AgentAskResponse(BaseModel):
    """Agentic RAG answer response with execution metadata."""
    question: str
    answer: str
    query_type: str
    sufficiency: str
    rewrites: int
    sources: List[str]
    latencies: Dict[str, float]


class StatsResponse(BaseModel):
    """Workspace index statistics."""
    workspace: str
    documents: int
    chunks: int
    embedding_model: str
    embedding_dim: str
    vector_store: str
    index_size: str
    chunk_file_size: str


class BenchmarkRequest(BaseModel):
    """Payload to trigger retrieval benchmark."""
    query: str = Field(description="Query to benchmark")


class BenchmarkResponse(BaseModel):
    """Latency benchmark breakdown."""
    query: str
    embedding_ms: float
    faiss_ms: float
    bm25_ms: float
    hybrid_ms: float
    rerank_ms: float
    llm_ms: float
    total_ms: float
    answer: str


class EvaluateRequest(BaseModel):
    """Payload to evaluate against a benchmark dataset."""
    dataset_path: str = Field(default="knowledge/evaluation/question.json", description="Path to evaluation questions dataset")


class FeedbackRequest(BaseModel):
    """Payload to record relevance judgment."""
    query: str
    chunk_text: str
    rating: int = Field(default=1, description="+1 for positive, 0 or -1 for negative")
    chunk_id: Optional[str] = None
    source_path: Optional[str] = None


class FeedbackResponse(BaseModel):
    """Response returning confirmation of logged feedback."""
    feedback_id: str
    query: str
    rating: int
    timestamp: str
