"""
LangGraph StateGraph definition and execution wrapper for RAGForge agent.
"""

from functools import partial
from typing import Any, Dict, Optional

from langgraph.graph import END, StateGraph

from knowledge.agent.nodes import (
    generate_answer,
    grade_sufficiency,
    retrieve_context,
    rewrite_query,
    route_query,
)
from knowledge.agent.state import AgentState
from knowledge.config.manager import AppConfig


def should_continue(state: AgentState) -> str:
    """Determine whether to generate an answer or rewrite the query and retry."""
    if state.get("sufficiency") == "sufficient":
        return "generate"
    if state.get("rewrite_count", 0) >= state.get("max_retries", 2):
        return "generate"
    return "rewrite"


def build_agent_graph(service=None, llm=None, top_k=None) -> Any:
    """Build and compile the LangGraph state machine."""
    workflow = StateGraph(AgentState)

    def route(state):
        state = route_query(state)
        if top_k is not None:
            state["top_k"] = top_k
        return state

    workflow.add_node("route", route)
    workflow.add_node(
        "retrieve",
        partial(retrieve_context, rag_service=service) if service is not None else retrieve_context,
    )
    workflow.add_node(
        "grade",
        partial(grade_sufficiency, llm_client=llm) if llm is not None else grade_sufficiency,
    )
    workflow.add_node(
        "rewrite", partial(rewrite_query, llm_client=llm) if llm is not None else rewrite_query
    )
    workflow.add_node(
        "generate", partial(generate_answer, llm_client=llm) if llm is not None else generate_answer
    )

    workflow.set_entry_point("route")
    workflow.add_edge("route", "retrieve")
    workflow.add_edge("retrieve", "grade")

    workflow.add_conditional_edges(
        "grade",
        should_continue,
        {
            "generate": "generate",
            "rewrite": "rewrite",
        },
    )

    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("generate", END)

    return workflow.compile()


def run_agent_workflow(
    question: str, max_retries: Optional[int] = None, *, service=None, llm=None, top_k=None
) -> Dict[str, Any]:
    """Execute the full agentic loop for a given question.

    Args:
        question: The user's question.
        max_retries: Hard cap on retrieve-grade-rewrite cycles. Defaults to
            ``AppConfig.max_retrieval_attempts`` (config.yaml, default 3).
    """
    if max_retries is None:
        max_retries = AppConfig().max_retrieval_attempts
    initial_state: AgentState = {
        "error": None,
        "original_query": question,
        "current_query": question,
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

    graph = (
        build_agent_graph(service=service, llm=llm, top_k=top_k)
        if any(x is not None for x in (service, llm, top_k))
        else build_agent_graph()
    )
    final_state = graph.invoke(initial_state, config={"recursion_limit": 4 * max_retries + 10})

    return {
        "error": final_state.get("error"),
        "question": question,
        "answer": final_state["answer"],
        "query_type": final_state["query_type"],
        "sufficiency": final_state["sufficiency"],
        "rewrites": final_state["rewrite_count"],
        "sources": final_state["sources"],
        "retrieved_chunks": final_state["retrieved_chunks"],
        "latencies": final_state["latencies"],
    }
