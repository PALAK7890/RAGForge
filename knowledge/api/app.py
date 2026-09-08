"""
FastAPI application for RAGForge.
"""

from typing import Any, Dict, Iterator

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from knowledge.api.schemas import (
    AgentAskRequest,
    AgentAskResponse,
    AskRequest,
    AskResponse,
    BenchmarkRequest,
    BenchmarkResponse,
    DocumentIndexRequest,
    DocumentListResponse,
    DocumentRecord,
    EvaluateRequest,
    FeedbackRequest,
    FeedbackResponse,
    SearchResponse,
    SearchResultChunk,
    StatsResponse,
)
from knowledge.api.service import RAGService

app = FastAPI(
    title="RAGForge REST API",
    description="Production-inspired modular RAG framework API for document indexing, hybrid search, and local LLM generation.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = RAGService()


@app.get("/health")
def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "app": "RAGForge"}


@app.post("/documents", response_model=Dict[str, Any])
def index_documents(payload: DocumentIndexRequest) -> Dict[str, Any]:
    """Index new documents from a file or folder path."""
    try:
        result = service.index_documents(payload.path)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}") from e


@app.get("/documents", response_model=DocumentListResponse)
def list_documents() -> DocumentListResponse:
    """List all indexed documents and total chunk counts."""
    data = service.list_documents()
    records = [DocumentRecord(**d) for d in data["documents"]]
    return DocumentListResponse(
        documents=records,
        total_documents=data["total_documents"],
        total_chunks=data["total_chunks"],
    )


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., description="Query string"),
    top_k: int = Query(5, ge=1, le=50, description="Top-k results"),
) -> SearchResponse:
    """Perform hybrid (FAISS + BM25) search with Cross-Encoder reranking."""
    try:
        ranked_chunks = service.search(query=q, top_k=top_k)
        results = [
            SearchResultChunk(
                chunk_id=c.get("chunk_id"),
                text=c["text"],
                source_path=c["source_path"],
                page_number=c.get("page_number"),
                cross_score=c.get("rerank_score"),
                hybrid_score=c.get("hybrid_score"),
            )
            for c in ranked_chunks
        ]
        return SearchResponse(query=q, results=results)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/ask")
def ask(payload: AskRequest) -> Any:
    """Answer questions from indexed documents, with optional streaming token support."""
    try:
        if payload.stream:
            sources, chunks, token_stream = service.ask_stream(payload.question, top_k=payload.top_k)

            def event_generator() -> Iterator[str]:
                for token in token_stream:
                    yield token

            return StreamingResponse(event_generator(), media_type="text/plain")
        else:
            data = service.ask(payload.question, top_k=payload.top_k)
            chunk_models = [SearchResultChunk(**c) for c in data["chunks"]]
            return AskResponse(
                question=data["question"],
                answer=data["answer"],
                sources=data["sources"],
                chunks=chunk_models,
            )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/agent-ask", response_model=AgentAskResponse)
def agent_ask(payload: AgentAskRequest) -> AgentAskResponse:
    """Agentic RAG with self-correcting retrieval loop (wired via LangGraph)."""
    try:
        # Import dynamically or fallback if agent module not yet initialized
        try:
            from knowledge.agent.graph import run_agent_workflow
            result = run_agent_workflow(payload.question, max_retries=payload.max_retries)
            return AgentAskResponse(**result)
        except ImportError:
            # Fallback to standard QA until agent module is compiled
            qa = service.ask(payload.question)
            return AgentAskResponse(
                question=payload.question,
                answer=qa["answer"],
                query_type="factual (direct)",
                sufficiency="sufficient",
                rewrites=0,
                sources=qa["sources"],
                latencies={"fallback": 0.0},
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/stats", response_model=StatsResponse)
def get_stats() -> StatsResponse:
    """Return workspace and vector index statistics."""
    data = service.get_stats()
    return StatsResponse(**data)


@app.post("/benchmark", response_model=BenchmarkResponse)
def run_benchmark(payload: BenchmarkRequest) -> BenchmarkResponse:
    """Benchmark end-to-end retrieval and generation latency."""
    try:
        result = service.benchmark(payload.query)
        return BenchmarkResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/evaluate", response_model=Dict[str, Any])
def run_evaluation(payload: EvaluateRequest) -> Dict[str, Any]:
    """Run evaluation metrics on a golden Q&A dataset."""
    try:
        report = service.evaluate(payload.dataset_path)
        return report
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/feedback", response_model=FeedbackResponse)
def record_feedback(payload: FeedbackRequest) -> FeedbackResponse:
    """Record user relevance judgment for active learning."""
    from knowledge.feedback.store import FeedbackStore
    store = FeedbackStore()
    res = store.record_feedback(
        query=payload.query,
        chunk_text=payload.chunk_text,
        rating=payload.rating,
        chunk_id=payload.chunk_id,
        source_path=payload.source_path,
    )
    return FeedbackResponse(**res)


@app.get("/agent-stream/{query}")
def agent_stream(
    query: str,
    max_retries: int = Query(default=3, ge=1, le=10, description="Max retrieve-rewrite cycles"),
) -> StreamingResponse:
    """Stream agentic RAG progress events via Server-Sent Events.

    Yields newline-delimited JSON strings, one per logical step:
    - ``{"event": "start", "attempt": 1, "query": "..."}``
    - ``{"event": "retrieved", "attempt": 1, "chunks": <count>}``
    - ``{"event": "graded", "attempt": 1, "sufficiency": "insufficient"}``
    - ``{"event": "rewriting", "attempt": 2, "query": "..."}``
    - ``{"event": "answer", "sufficiency": "...", "rewrites": N, "text": "..."}``

    This is a manual loop mirroring run_agent_workflow so it works across
    LangGraph versions without depending on internal streaming APIs.
    """
    import json as _json

    from knowledge.agent.nodes import (
        generate_answer as _generate,
    )
    from knowledge.agent.nodes import (
        grade_sufficiency as _grade,
    )
    from knowledge.agent.nodes import (
        retrieve_context as _retrieve,
    )
    from knowledge.agent.nodes import (
        rewrite_query as _rewrite,
    )
    from knowledge.agent.nodes import (
        route_query as _route,
    )
    from knowledge.agent.state import AgentState

    def event_stream() -> Iterator[str]:
        state: AgentState = {
            "original_query": query,
            "current_query": query,
            "query_type": "factual",
            "top_k": 5,
            "retrieved_chunks": [],
            "sufficiency": "pending",
            "sufficiency_reason": "",
            "rewrite_count": 0,
            "max_retries": max_retries,
            "answer": "",
            "sources": [],
            "latencies": {},
        }

        # Route once to classify intent
        state = _route(state)

        attempt = 1
        while True:
            yield _json.dumps({"event": "start", "attempt": attempt, "query": state["current_query"]}) + "\n"

            state = _retrieve(state)
            yield _json.dumps({"event": "retrieved", "attempt": attempt, "chunks": len(state["retrieved_chunks"])}) + "\n"

            state = _grade(state)
            yield _json.dumps({"event": "graded", "attempt": attempt, "sufficiency": state["sufficiency"]}) + "\n"

            if state["sufficiency"] == "sufficient" or state["rewrite_count"] >= max_retries:
                break

            state = _rewrite(state)
            attempt += 1
            yield _json.dumps({"event": "rewriting", "attempt": attempt, "query": state["current_query"]}) + "\n"

        state = _generate(state)
        yield _json.dumps({
            "event": "answer",
            "sufficiency": state["sufficiency"],
            "rewrites": state["rewrite_count"],
            "sources": state["sources"],
            "text": state["answer"],
        }) + "\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
