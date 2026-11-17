"""
Pydantic data schemas for AgenticRAG tools, evidence records, and results.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class EvidenceRecord(BaseModel):
    """
    Standardized, atomic evidence record maintaining end-to-end source provenance.
    """
    evidence_id: str = Field(default_factory=lambda: uuid4().hex[:12], description="Unique evidence identifier")
    chunk_id: Optional[str] = Field(default=None, description="Stable chunk identifier")
    document_id: str = Field(default="", description="Identifier or filename of source document")
    source_path: str = Field(default="", description="Path to source document file")
    page_number: Optional[int] = Field(default=None, description="Page number if document is paginated")
    char_start: Optional[int] = Field(default=None, description="Character start offset in source text")
    char_end: Optional[int] = Field(default=None, description="Character end offset in source text")
    text: str = Field(description="Raw text content of the retrieved evidence passage")
    retrieval_method: str = Field(default="hybrid", description="Method used to retrieve evidence: search, find, open")
    hybrid_score: Optional[float] = Field(default=None, description="Reciprocal Rank Fusion score")
    rerank_score: Optional[float] = Field(default=None, description="Cross-encoder neural reranking score")
    search_query: Optional[str] = Field(default=None, description="Query formulation that retrieved this record")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(), description="UTC timestamp")


class ToolError(BaseModel):
    """Structured error payload emitted by agentic tools."""
    tool_name: str
    error_type: str
    message: str
    details: Optional[Dict[str, Any]] = None


# --- SEARCH TOOL SCHEMAS ---

class SearchInput(BaseModel):
    """Input payload for the SEARCH tool."""
    queries: List[str] = Field(description="One or more query formulations to search across the corpus")
    top_k: int = Field(default=5, ge=1, le=50, description="Maximum candidate chunks to return per search")
    rerank: bool = Field(default=True, description="Whether to apply neural CrossEncoder reranking")
    metadata_filter: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata filter key-values")


class SearchResult(BaseModel):
    """Output payload from the SEARCH tool."""
    query_id: str = Field(default_factory=lambda: uuid4().hex[:8])
    queries_used: List[str]
    evidence: List[EvidenceRecord]
    total_found: int
    latency_ms: float
    error: Optional[ToolError] = None


# --- FIND TOOL SCHEMAS ---

class FindInput(BaseModel):
    """Input payload for the FIND in-document navigation tool."""
    document_id: str = Field(description="Target document name or ID to search within")
    expression: str = Field(description="Exact phrase, keyword, or regex pattern to search for")
    case_sensitive: bool = Field(default=False, description="Whether matching is case-sensitive")
    max_results: int = Field(default=10, ge=1, le=50, description="Maximum matches to return")


class FindMatch(BaseModel):
    """A single in-document match with surrounding context window."""
    chunk_id: Optional[str]
    document_id: str
    source_path: str
    page_number: Optional[int]
    char_start: Optional[int]
    char_end: Optional[int]
    snippet: str = Field(description="Surrounding text window around match location")
    match_offset_in_text: int


class FindResult(BaseModel):
    """Output payload from the FIND tool."""
    document_id: str
    expression: str
    matches: List[FindMatch]
    total_matches: int
    latency_ms: float
    error: Optional[ToolError] = None


# --- OPEN TOOL SCHEMAS ---

class OpenInput(BaseModel):
    """Input payload for the OPEN document reading tool."""
    document_id: str = Field(description="Target document name or ID to open")
    page: Optional[int] = Field(default=None, description="Specific page number to inspect")
    char_start: Optional[int] = Field(default=None, description="Start character offset to read from")
    char_end: Optional[int] = Field(default=None, description="End character offset to read to")
    chunk_index: Optional[int] = Field(default=None, description="Specific chunk index to inspect")
    max_output_chars: int = Field(default=4000, ge=50, le=16000, description="Hard cap on returned characters")


class ChunkView(BaseModel):
    """A rendered slice or chunk of a document."""
    chunk_id: Optional[str]
    chunk_index: Optional[int]
    page_number: Optional[int]
    char_start: Optional[int]
    char_end: Optional[int]
    text: str


class OpenResult(BaseModel):
    """Output payload from the OPEN tool."""
    document_id: str
    source_path: str
    views: List[ChunkView]
    total_chars: int
    is_truncated: bool = False
    evidence_records: List[EvidenceRecord] = Field(default_factory=list)
    latency_ms: float
    error: Optional[ToolError] = None


# --- SUMMARIZE TOOL SCHEMAS ---

class SummarizeInput(BaseModel):
    """Input payload for the SUMMARIZE evidence compression tool."""
    evidence_ids: List[str] = Field(description="List of evidence IDs in the evidence store to summarize")
    token_budget: int = Field(default=1500, ge=100, le=8000, description="Approximate token budget for summary")
    focus_question: Optional[str] = Field(default=None, description="Optional question to guide extractive focus")


class SourceReference(BaseModel):
    """Citation reference mapping back to underlying evidence."""
    evidence_id: str
    source_path: str
    page_number: Optional[int]
    chunk_id: Optional[str]


class SummaryResult(BaseModel):
    """Output payload from the SUMMARIZE tool."""
    summary_text: str
    source_references: List[SourceReference]
    evidence_ids_used: List[str]
    token_estimate: int
    is_extractive: bool
    latency_ms: float
    error: Optional[ToolError] = None
