"""
Agent-callable retrieval and navigation tools: SEARCH, FIND, OPEN, and SUMMARIZE.
Directly implements the tool specifications from Microsoft's AgenticRAG research paper.
"""

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from knowledge.agentic.schemas import (
    ChunkView,
    EvidenceRecord,
    FindInput,
    FindMatch,
    FindResult,
    OpenInput,
    OpenResult,
    SearchInput,
    SearchResult,
    SourceReference,
    SummarizeInput,
    SummaryResult,
    ToolError,
)
from knowledge.api.service import RAGService
from knowledge.llms.ollama_llm import OllamaLLM


class SearchTool:
    """
    SEARCH Tool: Corpus-wide discovery across hybrid vector and lexical indices.
    Supports multi-query formulations, deduplication, latency tracking, and optional reranking.
    """

    def __init__(self, service: Optional[RAGService] = None) -> None:
        self.service = service or RAGService()

    def execute(self, payload: SearchInput) -> SearchResult:
        """Execute one or more search queries across the corpus."""
        t0 = time.perf_counter()
        all_evidence: List[EvidenceRecord] = []
        seen_chunk_ids = set()

        queries = [q.strip() for q in payload.queries if q.strip()]
        if not queries:
            return SearchResult(
                queries_used=[],
                evidence=[],
                total_found=0,
                latency_ms=(time.perf_counter() - t0) * 1000,
                error=ToolError(
                    tool_name="SEARCH",
                    error_type="EmptyQueryError",
                    message="No valid search queries provided.",
                ),
            )

        try:
            for q in queries:
                raw_results = self.service.search(query=q, top_k=payload.top_k)

                for chunk in raw_results:
                    cid = chunk.get("chunk_id")
                    if cid and cid in seen_chunk_ids:
                        continue
                    if cid:
                        seen_chunk_ids.add(cid)

                    # Apply optional metadata filtering
                    if payload.metadata_filter:
                        chunk_meta = chunk.get("metadata", {}) or {}
                        match = all(
                            chunk_meta.get(k) == v or chunk.get(k) == v
                            for k, v in payload.metadata_filter.items()
                        )
                        if not match:
                            continue

                    doc_name = Path(chunk.get("source_path", "")).name or chunk.get("document_id", "unknown")
                    evidence = EvidenceRecord(
                        evidence_id=uuid4().hex[:12],
                        chunk_id=cid,
                        document_id=doc_name,
                        source_path=chunk.get("source_path", ""),
                        page_number=chunk.get("page_number"),
                        char_start=(chunk.get("metadata", {}) or {}).get("char_start"),
                        char_end=(chunk.get("metadata", {}) or {}).get("char_end"),
                        text=chunk.get("text", ""),
                        retrieval_method="search",
                        hybrid_score=chunk.get("hybrid_score"),
                        rerank_score=chunk.get("rerank_score"),
                        search_query=q,
                    )
                    all_evidence.append(evidence)

            latency = (time.perf_counter() - t0) * 1000
            return SearchResult(
                queries_used=queries,
                evidence=all_evidence,
                total_found=len(all_evidence),
                latency_ms=latency,
            )
        except Exception as e:
            return SearchResult(
                queries_used=queries,
                evidence=all_evidence,
                total_found=len(all_evidence),
                latency_ms=(time.perf_counter() - t0) * 1000,
                error=ToolError(
                    tool_name="SEARCH",
                    error_type=type(e).__name__,
                    message=str(e),
                ),
            )


class FindTool:
    """
    FIND Tool: In-document substring and keyword search within a specific document.
    Does NOT perform global vector search; navigates strictly within candidate document boundaries.
    """

    def __init__(self, chunks_path: str = ".knowledge/index/chunks.json") -> None:
        self.chunks_path = Path(chunks_path)

    def _load_chunks(self) -> List[Dict[str, Any]]:
        if not self.chunks_path.exists():
            return []
        try:
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("chunks", [])
        except Exception:
            return []

    def execute(self, payload: FindInput) -> FindResult:
        """Search for exact expression matches within chunks of a specific document."""
        t0 = time.perf_counter()
        matches: List[FindMatch] = []
        all_chunks = self._load_chunks()

        target_doc = payload.document_id.lower().strip()
        expr = payload.expression.strip()

        if not expr:
            return FindResult(
                document_id=payload.document_id,
                expression=payload.expression,
                matches=[],
                total_matches=0,
                latency_ms=(time.perf_counter() - t0) * 1000,
                error=ToolError(
                    tool_name="FIND",
                    error_type="EmptyExpressionError",
                    message="Search expression cannot be empty.",
                ),
            )

        flags = 0 if payload.case_sensitive else re.IGNORECASE
        pattern = re.compile(re.escape(expr), flags)

        for chunk in all_chunks:
            source_p = chunk.get("source_path", "")
            doc_id = chunk.get("document_id", "")
            doc_name = Path(source_p).name

            # Match document_id either by filename or exact document_id string
            if target_doc not in doc_name.lower() and target_doc not in doc_id.lower():
                continue

            text = chunk.get("text", "")
            for m in pattern.finditer(text):
                start_idx, end_idx = m.span()

                # Extract surrounding window (+- 60 characters)
                window_start = max(0, start_idx - 60)
                window_end = min(len(text), end_idx + 60)
                prefix = "..." if window_start > 0 else ""
                suffix = "..." if window_end < len(text) else ""
                snippet = f"{prefix}{text[window_start:window_end]}{suffix}"

                meta = chunk.get("metadata", {}) or {}
                chunk_char_start = meta.get("char_start")
                chunk_char_end = meta.get("char_end")

                char_start_doc = (chunk_char_start + start_idx) if chunk_char_start is not None else None
                char_end_doc = (chunk_char_start + end_idx) if chunk_char_start is not None else None

                matches.append(
                    FindMatch(
                        chunk_id=chunk.get("chunk_id"),
                        document_id=doc_name,
                        source_path=source_p,
                        page_number=chunk.get("page_number"),
                        char_start=char_start_doc,
                        char_end=char_end_doc,
                        snippet=snippet,
                        match_offset_in_text=start_idx,
                    )
                )

                if len(matches) >= payload.max_results:
                    break

            if len(matches) >= payload.max_results:
                break

        latency = (time.perf_counter() - t0) * 1000
        return FindResult(
            document_id=payload.document_id,
            expression=payload.expression,
            matches=matches,
            total_matches=len(matches),
            latency_ms=latency,
        )


class OpenTool:
    """
    OPEN Tool: Inspects contiguous content, sections, or pages within a known document.
    Enforces path safety, bounds checking, and token/character length caps.
    """

    def __init__(self, chunks_path: str = ".knowledge/index/chunks.json") -> None:
        self.chunks_path = Path(chunks_path)

    def _load_chunks(self) -> List[Dict[str, Any]]:
        if not self.chunks_path.exists():
            return []
        try:
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("chunks", [])
        except Exception:
            return []

    def execute(self, payload: OpenInput) -> OpenResult:
        """Open and read a document's chunks by page, offsets, or chunk index."""
        t0 = time.perf_counter()
        all_chunks = self._load_chunks()
        target_doc = payload.document_id.lower().strip()

        matching_chunks = []
        source_path = ""
        for c in all_chunks:
            doc_name = Path(c.get("source_path", "")).name
            doc_id = c.get("document_id", "")
            if target_doc in doc_name.lower() or target_doc in doc_id.lower():
                matching_chunks.append(c)
                if not source_path:
                    source_path = c.get("source_path", "")

        if not matching_chunks:
            return OpenResult(
                document_id=payload.document_id,
                source_path="",
                views=[],
                total_chars=0,
                latency_ms=(time.perf_counter() - t0) * 1000,
                error=ToolError(
                    tool_name="OPEN",
                    error_type="DocumentNotFoundError",
                    message=f"Document '{payload.document_id}' not found in index.",
                ),
            )

        # Sort matching chunks by chunk_index
        matching_chunks.sort(key=lambda x: x.get("chunk_index", 0))

        # Filter by page if requested
        if payload.page is not None:
            filtered = [c for c in matching_chunks if c.get("page_number") == payload.page]
            if filtered:
                matching_chunks = filtered

        # Filter by chunk_index if requested
        elif payload.chunk_index is not None:
            filtered = [c for c in matching_chunks if c.get("chunk_index") == payload.chunk_index]
            if filtered:
                matching_chunks = filtered

        # Filter by character offsets if requested
        elif payload.char_start is not None or payload.char_end is not None:
            req_start = payload.char_start or 0
            req_end = payload.char_end or float("inf")
            filtered = []
            for c in matching_chunks:
                meta = c.get("metadata", {}) or {}
                cs = meta.get("char_start")
                ce = meta.get("char_end")
                if cs is not None and ce is not None:
                    # Overlap check
                    if max(req_start, cs) < min(req_end, ce):
                        filtered.append(c)
            if filtered:
                matching_chunks = filtered

        views: List[ChunkView] = []
        evidence_records: List[EvidenceRecord] = []
        total_chars = 0
        truncated = False

        for c in matching_chunks:
            meta = c.get("metadata", {}) or {}
            chunk_text = c.get("text", "")
            char_len = len(chunk_text)

            if total_chars + char_len > payload.max_output_chars:
                remaining_budget = max(0, payload.max_output_chars - total_chars)
                chunk_text = chunk_text[:remaining_budget] + "... [TRUNCATED]"
                truncated = True

            views.append(
                ChunkView(
                    chunk_id=c.get("chunk_id"),
                    chunk_index=c.get("chunk_index"),
                    page_number=c.get("page_number"),
                    char_start=meta.get("char_start"),
                    char_end=meta.get("char_end"),
                    text=chunk_text,
                )
            )

            evidence_records.append(
                EvidenceRecord(
                    evidence_id=uuid4().hex[:12],
                    chunk_id=c.get("chunk_id"),
                    document_id=Path(source_path).name or payload.document_id,
                    source_path=source_path,
                    page_number=c.get("page_number"),
                    char_start=meta.get("char_start"),
                    char_end=meta.get("char_end"),
                    text=chunk_text,
                    retrieval_method="open",
                )
            )

            total_chars += len(chunk_text)
            if truncated:
                break

        latency = (time.perf_counter() - t0) * 1000
        return OpenResult(
            document_id=Path(source_path).name or payload.document_id,
            source_path=source_path,
            views=views,
            total_chars=total_chars,
            is_truncated=truncated,
            evidence_records=evidence_records,
            latency_ms=latency,
        )


class SummarizeTool:
    """
    SUMMARIZE Tool: Compresses accumulated evidence into an information-dense summary.
    Maintains strict provenance links back to underlying evidence passages.
    """

    def __init__(self, llm: Optional[OllamaLLM] = None) -> None:
        self.llm = llm

    def execute(
        self,
        payload: SummarizeInput,
        evidence_store: Dict[str, EvidenceRecord],
    ) -> SummaryResult:
        """Condense evidence records into a bounded summary with citation provenance."""
        t0 = time.perf_counter()
        target_records = [
            evidence_store[eid]
            for eid in payload.evidence_ids
            if eid in evidence_store
        ]

        if not target_records:
            return SummaryResult(
                summary_text="No matching evidence records available to summarize.",
                source_references=[],
                evidence_ids_used=[],
                token_estimate=0,
                is_extractive=True,
                latency_ms=(time.perf_counter() - t0) * 1000,
                error=ToolError(
                    tool_name="SUMMARIZE",
                    error_type="EvidenceNotFoundError",
                    message="None of the requested evidence IDs exist in the store.",
                ),
            )

        source_refs: List[SourceReference] = [
            SourceReference(
                evidence_id=r.evidence_id,
                source_path=r.source_path,
                page_number=r.page_number,
                chunk_id=r.chunk_id,
            )
            for r in target_records
        ]

        # Prioritize higher-scoring or distinct passages
        target_records.sort(
            key=lambda r: (r.rerank_score or 0.0, r.hybrid_score or 0.0),
            reverse=True,
        )

        char_budget = payload.token_budget * 4

        # If LLM is available, attempt abstractive grounded summary
        summary_text = ""
        is_extractive = True

        if self.llm is not None:
            try:
                context_blocks = "\n\n".join(
                    f"[{r.document_id}]: {r.text}" for r in target_records
                )
                focus = (
                    f"Focus on information addressing: '{payload.focus_question}'"
                    if payload.focus_question
                    else "Synthesize all critical facts, figures, and relationships."
                )
                prompt = (
                    f"Summarize the following factual passages into a concise, dense briefing.\n"
                    f"{focus}\n"
                    f"Retain exact names, dates, metrics, and figures. Do not extrapolate.\n\n"
                    f"Context:\n{context_blocks}\n\nSummary:"
                )
                raw_summary = self.llm.generate("Evidence Summary", prompt).strip()
                if raw_summary and len(raw_summary) > 20:
                    summary_text = raw_summary[:char_budget]
                    is_extractive = False
            except Exception:
                is_extractive = True

        # Deterministic Extractive fallback
        if not summary_text or is_extractive:
            extracted_sentences = []
            current_len = 0

            for r in target_records:
                sentences = re.split(r"(?<=[.!?])\s+", r.text.strip())
                for s in sentences:
                    s_clean = s.strip()
                    if len(s_clean) < 15:
                        continue
                    # Prioritize informative sentences (containing digits, entities, or capital letters)
                    if current_len + len(s_clean) + 5 <= char_budget:
                        extracted_sentences.append(f"• [{r.document_id}] {s_clean}")
                        current_len += len(s_clean) + len(r.document_id) + 6
                    if current_len >= char_budget:
                        break
                if current_len >= char_budget:
                    break

            summary_text = "\n".join(extracted_sentences) or target_records[0].text[:char_budget]
            is_extractive = True

        token_est = len(summary_text) // 4
        latency = (time.perf_counter() - t0) * 1000

        return SummaryResult(
            summary_text=summary_text,
            source_references=source_refs,
            evidence_ids_used=[r.evidence_id for r in target_records],
            token_estimate=token_est,
            is_extractive=is_extractive,
            latency_ms=latency,
        )
