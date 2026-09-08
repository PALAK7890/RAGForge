"""
Graph nodes for the RAGForge self-correcting agent.
"""

from pathlib import Path
import time
from typing import Any, Dict, List

from knowledge.agent.state import AgentState
from knowledge.api.service import RAGService
from knowledge.llms.ollama_llm import OllamaLLM

service = RAGService()
llm = OllamaLLM()


def route_query(state: AgentState) -> AgentState:
    """Classify the query intent and adjust retrieval parameters."""
    t0 = time.perf_counter()
    query_lower = state["original_query"].lower()

    if any(k in query_lower for k in ["summarize", "summary", "overview", "all", "explain everything"]):
        q_type = "summarization"
        top_k = 8
    elif any(k in query_lower for k in ["compare", "difference", "vs", "versus", "both", "relationship"]):
        q_type = "multi-hop"
        top_k = 6
    else:
        q_type = "factual"
        top_k = 5

    state["query_type"] = q_type
    state["top_k"] = top_k
    state["current_query"] = state["original_query"]
    state["latencies"]["route"] = (time.perf_counter() - t0) * 1000
    return state


def retrieve_context(state: AgentState) -> AgentState:
    """Execute hybrid retrieval + cross-encoder reranking."""
    t0 = time.perf_counter()
    try:
        chunks = service.search(query=state["current_query"], top_k=state["top_k"])
    except Exception:
        chunks = []

    state["retrieved_chunks"] = chunks
    state["latencies"]["retrieve"] = (time.perf_counter() - t0) * 1000
    return state


def grade_sufficiency(state: AgentState) -> AgentState:
    """Use the local LLM to grade if the retrieved context is sufficient."""
    t0 = time.perf_counter()
    chunks = state.get("retrieved_chunks", [])

    if not chunks:
        state["sufficiency"] = "insufficient"
        state["sufficiency_reason"] = "No chunks were retrieved for the query."
        state["latencies"]["grade"] = (time.perf_counter() - t0) * 1000
        return state

    context = "\n\n".join(c["text"] for c in chunks[:3])
    prompt = f"""Assess whether the context contains sufficient facts to answer the question.
Question: {state['current_query']}
Context:
{context}

Respond with only one word: SUFFICIENT or INSUFFICIENT."""

    try:
        verdict = llm.generate(state["current_query"], prompt).strip().upper()
        if "INSUFFICIENT" in verdict:
            state["sufficiency"] = "insufficient"
            state["sufficiency_reason"] = "Retrieved chunks do not contain enough facts to answer."
        else:
            state["sufficiency"] = "sufficient"
            state["sufficiency_reason"] = "Context contains relevant facts."
    except Exception:
        # Graceful fallback: assume sufficient if LLM grading fails
        state["sufficiency"] = "sufficient"
        state["sufficiency_reason"] = "LLM grader bypassed."

    state["latencies"]["grade"] = (time.perf_counter() - t0) * 1000
    return state


def rewrite_query(state: AgentState) -> AgentState:
    """Rewrite query to improve semantic retrieval on the next iteration."""
    t0 = time.perf_counter()
    state["rewrite_count"] += 1

    prompt = f"""The previous query "{state['current_query']}" did not retrieve sufficient context to answer: "{state['original_query']}".
Formulate a better, more specific keyword search query to find the relevant document passages.
Output ONLY the rewritten query with no explanation."""

    try:
        new_q = llm.generate(state["original_query"], prompt).strip().strip('"')
        if new_q and len(new_q) > 3:
            state["current_query"] = new_q
    except Exception:
        # Fallback: append domain terms
        state["current_query"] = f"{state['original_query']} details overview"

    state["latencies"][f"rewrite_{state['rewrite_count']}"] = (time.perf_counter() - t0) * 1000
    return state


def generate_answer(state: AgentState) -> AgentState:
    """Synthesize the final answer using retrieved context."""
    t0 = time.perf_counter()
    chunks = state.get("retrieved_chunks", [])

    if not chunks:
        state["answer"] = "I could not find the answer in the indexed documents."
        state["sources"] = []
        state["latencies"]["generate"] = (time.perf_counter() - t0) * 1000
        return state

    context = "\n\n".join(c["text"] for c in chunks)
    prefix = ""
    if state["sufficiency"] == "insufficient":
        prefix = "[Note: Context may be incomplete despite query reformulation.]\n\n"

    try:
        raw_answer = llm.generate(state["original_query"], context)
        state["answer"] = f"{prefix}{raw_answer}"
    except Exception as e:
        state["answer"] = f"Failed to generate answer: {str(e)}"

    state["sources"] = list(dict.fromkeys(Path(c["source_path"]).name for c in chunks if "source_path" in c))
    state["latencies"]["generate"] = (time.perf_counter() - t0) * 1000
    return state
