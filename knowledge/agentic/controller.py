"""
Agentic retrieval with heuristic or bounded model-directed tool selection.
"""

import json
import re
import time
from typing import Any, Callable, Dict, List, Optional

from langgraph.graph import END, StateGraph

from knowledge.agentic.evidence_store import EvidenceStore
from knowledge.agentic.schemas import (
    FindInput,
    OpenInput,
    PlannerDecision,
    SearchInput,
    SummarizeInput,
)
from knowledge.agentic.state import AgenticGraphState
from knowledge.agentic.tools import FindTool, OpenTool, SearchTool, SummarizeTool
from knowledge.llms.ollama_llm import OllamaLLM


class AgenticController:
    """
    Controller managing the AgenticRAG workflow graph and coordinating SEARCH, FIND, OPEN, and SUMMARIZE tools.
    """

    def __init__(
        self,
        search_tool: Optional[SearchTool] = None,
        find_tool: Optional[FindTool] = None,
        open_tool: Optional[OpenTool] = None,
        summarize_tool: Optional[SummarizeTool] = None,
        llm: Optional[OllamaLLM] = None,
        on_tool_event: Optional[Callable[[Dict[str, Any]], None]] = None,
        planner_mode: str = "heuristic",
    ) -> None:
        if planner_mode not in {"heuristic", "model"}:
            raise ValueError("planner_mode must be 'heuristic' or 'model'")
        self.planner_mode = planner_mode
        self.top_k = 5
        self.on_tool_event = on_tool_event
        self.search_tool = search_tool or SearchTool()
        self.find_tool = find_tool or FindTool()
        self.open_tool = open_tool or OpenTool()
        self.llm = llm or OllamaLLM()
        self.summarize_tool = summarize_tool or SummarizeTool(llm=self.llm)

    def _planner_context(self, state: AgenticGraphState) -> str:
        """Bound the decision prompt while showing new evidence and navigation results."""
        records = list(state["evidence_store_data"].values())
        evidence, remaining = [], 10000
        # Recent discoveries must remain visible even when earlier evidence filled the context.
        for record in reversed(records):
            text = record["text"][:min(2000, remaining)]
            if not text:
                break
            evidence.append({
                key: record.get(key)
                for key in ("document_id", "source_path", "page_number", "char_start", "char_end")
            } | {"text": text, "truncated": len(text) < len(record["text"])})
            remaining -= len(text)
        history = [
            {key: item[key] for key in ("tool", "inputs", "results_count", "matches", "error") if key in item}
            for item in state["tool_history"][-4:]
        ]
        return json.dumps({
            "remaining_tool_calls": state["max_steps"] - state["step"],
            "recent_tool_results": history,
            "omitted_evidence_records": len(records) - len(evidence),
            "evidence": evidence,
        }, ensure_ascii=False)

    def _model_arguments(self, decision: PlannerDecision, state: AgenticGraphState) -> dict:
        """Validate model input before it reaches any tool; keep retrieval limits fixed."""
        action, args = decision.action, decision.arguments
        allowed = {
            "search": {"queries"},
            "find": {"document_id", "expression"},
            "open": {"document_id", "page", "char_start", "char_end", "chunk_index"},
            "answer": set(), "abstain": set(), "clarify": {"question"},
        }
        if set(args) - allowed[action]:
            raise ValueError(f"Unexpected arguments for {action}")
        if action == "search":
            queries = args.get("queries")
            if not isinstance(queries, list) or not 1 <= len(queries) <= 3:
                raise ValueError("SEARCH requires one to three queries")
            if any(not isinstance(q, str) or not q.strip() or len(q) > 500 for q in queries):
                raise ValueError("SEARCH queries must be nonempty strings of at most 500 characters")
            return {"queries": list(dict.fromkeys(q.strip() for q in queries)), "top_k": self.top_k}
        if action in {"find", "open"}:
            doc = args.get("document_id")
            records = list(state["evidence_store_data"].values())
            matches = [r for r in records if doc and doc in {r.get("document_id"), r.get("source_path")}]
            if not matches:
                raise ValueError("Navigation requires an already discovered document ID")
            if len({r.get("source_path") for r in matches}) > 1:
                raise ValueError("Ambiguous document ID; use the full source_path")
            if action == "find":
                payload = FindInput.model_validate(args, strict=True)
                if not payload.expression.strip() or len(payload.expression) > 500:
                    raise ValueError("FIND requires a nonempty literal expression of at most 500 characters")
                return {"document_id": doc, "expression": payload.expression.strip()}
            payload = OpenInput.model_validate(args, strict=True)
            selectors = (payload.page, payload.char_start, payload.char_end, payload.chunk_index)
            if any(value is not None and value < 0 for value in selectors):
                raise ValueError("OPEN locations cannot be negative")
            if payload.char_end is not None and payload.char_end <= (payload.char_start or 0):
                raise ValueError("OPEN char_end must follow char_start")
            selector_count = sum((
                payload.page is not None,
                payload.chunk_index is not None,
                payload.char_start is not None or payload.char_end is not None,
            ))
            if selector_count > 1:
                raise ValueError("OPEN must use one location selector")
            return {key: value for key, value in args.items() if value is not None} | {"max_output_chars": 4000}
        if action == "clarify":
            question = args.get("question")
            if not isinstance(question, str) or not question.strip() or len(question) > 1000:
                raise ValueError("CLARIFY requires a concise question")
            return {"question": question.strip()}
        return args

    def _choose_model_action(self, state: AgenticGraphState) -> AgenticGraphState:
        t0 = time.perf_counter()
        history = state.setdefault("decision_history", [])
        entry = {"decision": len(history) + 1, "after_step": state["step"]}
        try:
            decision = PlannerDecision.model_validate(
                self.llm.plan_action(state["question"], self._planner_context(state))
            )
            entry.update(decision.model_dump())
            args = self._model_arguments(decision, state)
            state["sufficiency_reason"] = decision.reason
            state["current_action"] = {}
            if decision.action in {"search", "find", "open"}:
                if state["step"] >= state["max_steps"]:
                    state["sufficiency"] = "insufficient"
                    state["sufficiency_reason"] = "Tool budget exhausted before the planner established sufficient evidence."
                else:
                    signature = json.dumps({"tool": decision.action, "args": args}, sort_keys=True).casefold()
                    previous = [
                        json.dumps({"tool": item["tool"].lower(), "args": item["inputs"]}, sort_keys=True).casefold()
                        for item in state["tool_history"]
                    ]
                    if signature in previous:
                        raise ValueError("Planner repeated an identical tool action")
                    state["current_action"] = {"tool": decision.action, "args": args}
                    state["sufficiency"] = "pending"
            elif decision.action == "answer":
                if not state["evidence_store_data"]:
                    raise ValueError("Planner cannot answer without retrieved evidence")
                state["sufficiency"] = "sufficient"
            elif decision.action == "abstain":
                state["sufficiency"] = "unanswerable"
            else:
                state["sufficiency"] = "needs_clarification"
                state["clarification_question"] = args["question"]
        except Exception as exc:
            state["error"] = f"Planning failed: {exc}"
            state["sufficiency"] = "insufficient"
            state["sufficiency_reason"] = state["error"]
            state["current_action"] = {}
            entry["error"] = state["error"]
        entry["duration_ms"] = (time.perf_counter() - t0) * 1000
        history.append(entry)
        state["latencies"][f"planning_{len(history)}"] = entry["duration_ms"]
        return state

    # -------------------------------------------------------------------------
    # Graph Nodes
    # -------------------------------------------------------------------------

    def plan_and_route(self, state: AgenticGraphState) -> AgenticGraphState:
        """
        Analyze question intent, decompose multi-hop/comparative needs, and initialize retrieval trajectory.
        """
        if self.planner_mode == "model":
            return self._choose_model_action(state)
        t0 = time.perf_counter()
        question = state["question"].strip()
        q_lower = question.lower()

        # Check for multi-hop or comparative query indicators
        sub_questions: List[str] = []
        is_comparative = any(
            term in q_lower
            for term in ["compare", "difference between", "versus", "vs", "changes between"]
        )

        if is_comparative and (" and " in q_lower or " vs " in q_lower):
            # Decompose into sub-queries for comparative analysis
            sub_questions = [
                f"{question} aspect 1",
                f"{question} aspect 2",
            ]

        # Detect if a specific document is mentioned in the query (e.g., "in policy.txt", "according to readme.md")
        doc_match = re.search(r"\b([\w-]+\.(?:txt|pdf|docx|md|html))\b", q_lower)
        mentioned_doc = doc_match.group(1) if doc_match else None

        initial_queries = [question]
        if mentioned_doc:
            clean_q = re.sub(
                r"\b(?:in|according to|from)\s+[\w-]+\.\w+\b", "", question, flags=re.IGNORECASE
            ).strip()
            if clean_q:
                initial_queries.append(clean_q)

        state["sub_questions"] = sub_questions
        state["current_action"] = {
            "tool": "search",
            "args": {"queries": initial_queries, "top_k": self.top_k},
        }
        state["latencies"]["plan"] = (time.perf_counter() - t0) * 1000
        return state

    def execute_tool(self, state: AgenticGraphState) -> AgenticGraphState:
        """
        Dispatch the current planned tool action, ingest results into EvidenceStore, and record execution trace.
        """
        t0 = time.perf_counter()
        action = state.get("current_action", {})
        tool_name = action.get("tool", "search").lower()
        args = action.get("args", {})

        store = EvidenceStore.from_dict(state["evidence_store_data"])
        step_num = state["step"] + 1

        history_entry: Dict[str, Any] = {
            "step": step_num,
            "tool": tool_name.upper(),
            "inputs": args,
            "timestamp": time.time(),
        }

        try:
            if tool_name == "search":
                queries = args.get("queries", [state["question"]])
                top_k = args.get("top_k", 5)
                res = self.search_tool.execute(SearchInput(queries=queries, top_k=top_k))
                if res.error:
                    raise RuntimeError(f"SEARCH: {res.error.message}")
                store.add_many(res.evidence)
                history_entry["results_count"] = res.total_found
                history_entry["queries_used"] = res.queries_used
                history_entry["documents"] = list(dict.fromkeys(r.document_id for r in res.evidence))

            elif tool_name == "find":
                doc_id = args.get("document_id", "")
                expr = args.get("expression", "")
                res_find = self.find_tool.execute(FindInput(document_id=doc_id, expression=expr))
                if res_find.error:
                    raise RuntimeError(f"FIND: {res_find.error.message}")
                history_entry["results_count"] = res_find.total_matches
                history_entry["matches"] = [m.model_dump() for m in res_find.matches[:3]]

            elif tool_name == "open":
                doc_id = args.get("document_id", "")
                page = args.get("page")
                char_start = args.get("char_start")
                char_end = args.get("char_end")
                res_open = self.open_tool.execute(
                    OpenInput(
                        document_id=doc_id,
                        page=page,
                        char_start=char_start,
                        char_end=char_end,
                        chunk_index=args.get("chunk_index"),
                        max_output_chars=args.get("max_output_chars", 4000),
                    )
                )
                if res_open.error:
                    raise RuntimeError(f"OPEN: {res_open.error.message}")
                store.add_many(res_open.evidence_records)
                history_entry["results_count"] = len(res_open.views)
                history_entry["total_chars"] = res_open.total_chars

            elif tool_name == "summarize":
                eids = args.get("evidence_ids", list(store.to_dict().keys()))
                res_sum = self.summarize_tool.execute(
                    SummarizeInput(evidence_ids=eids, focus_question=state["question"]),
                    evidence_store=store.to_dict(),
                )
                history_entry["results_count"] = len(res_sum.evidence_ids_used)
                history_entry["is_extractive"] = res_sum.is_extractive
                history_entry["summary_preview"] = res_sum.summary_text[:150]

        except Exception as e:
            history_entry["error"] = str(e)
            state["error"] = str(e)

        # Deduplicate evidence
        dedup_count = store.deduplicate()
        history_entry["dedup_removed"] = dedup_count

        # Update discovered documents list
        discovered = list(dict.fromkeys(r.document_id for r in store.get_all() if r.document_id))
        state["discovered_docs"] = discovered
        state["evidence_store_data"] = store.to_dict()

        history_entry["duration_ms"] = (time.perf_counter() - t0) * 1000
        state["tool_history"].append(history_entry)
        if self.on_tool_event is not None:
            self.on_tool_event(dict(history_entry))
        state["step"] = step_num
        state["latencies"][f"step_{step_num}_{tool_name}"] = history_entry["duration_ms"]

        return state

    def evaluate_evidence(self, state: AgenticGraphState) -> AgenticGraphState:
        """
        Evaluate whether collected evidence is sufficient to answer the question, or choose the next navigation step.
        """
        if self.planner_mode == "model":
            if state.get("error"):
                state["sufficiency"] = "insufficient"
                state["sufficiency_reason"] = state["error"]
                return state
            return self._choose_model_action(state)
        t0 = time.perf_counter()
        store = EvidenceStore.from_dict(state["evidence_store_data"])
        evidence_list = store.get_all()
        q_lower = state["question"].lower()

        # Check for hard budget termination
        if state["step"] >= state["max_steps"]:
            if len(evidence_list) > 0:
                state["sufficiency"] = "sufficient"
                state["sufficiency_reason"] = (
                    f"Budget reached ({state['max_steps']} steps); proceeding with best evidence."
                )
            else:
                state["sufficiency"] = "unanswerable"
                state["sufficiency_reason"] = (
                    "Budget exhausted without finding any relevant documents or passages."
                )
            state["latencies"][f"eval_step_{state['step']}"] = (time.perf_counter() - t0) * 1000
            return state

        # If zero evidence was found on first search
        if not evidence_list:
            if state["step"] == 1:
                # Try reformulated search with relaxed keywords
                relaxed_q = re.sub(r"[^\w\s]", "", state["question"]).strip()
                state["current_action"] = {
                    "tool": "search",
                    "args": {"queries": [relaxed_q, f"{relaxed_q} overview"], "top_k": self.top_k},
                }
                state["sufficiency"] = "pending"
                state["sufficiency_reason"] = (
                    "Initial search yielded no results; retrying with relaxed query."
                )
            else:
                state["sufficiency"] = "unanswerable"
                state["sufficiency_reason"] = (
                    "Repeated searches found no matching documents in knowledge base."
                )
            state["latencies"][f"eval_step_{state['step']}"] = (time.perf_counter() - t0) * 1000
            return state

        # Check if we should perform in-document navigation (FIND or OPEN)
        last_action = state["tool_history"][-1] if state["tool_history"] else {}
        last_tool = last_action.get("tool", "")

        # Look for specific target keywords or numbers in question (e.g., "retention", "revenue", "2024")
        key_terms = [
            w
            for w in re.findall(r"\b\w{4,}\b", q_lower)
            if w
            not in [
                "what",
                "when",
                "where",
                "which",
                "explain",
                "describe",
                "about",
                "could",
                "would",
            ]
        ]

        # Decision: If we just searched and found anchor documents, inspect specific candidate with FIND
        if last_tool == "SEARCH" and state["discovered_docs"] and key_terms:
            target_doc = state["discovered_docs"][0]
            term_to_find = key_terms[0]
            state["current_action"] = {
                "tool": "find",
                "args": {"document_id": target_doc, "expression": term_to_find},
            }
            state["sufficiency"] = "pending"
            state["sufficiency_reason"] = (
                f"Anchored in document '{target_doc}'; locating keyword '{term_to_find}'."
            )
            state["latencies"][f"eval_step_{state['step']}"] = (time.perf_counter() - t0) * 1000
            return state

        # Decision: If we just did FIND and located matches, inspect surrounding page or offsets with OPEN
        if last_tool == "FIND" and last_action.get("matches"):
            first_match = last_action["matches"][0]
            doc_id = first_match.get("document_id")
            page = first_match.get("page_number")
            state["current_action"] = {
                "tool": "open",
                "args": {
                    "document_id": doc_id,
                    "page": page,
                    "char_start": first_match.get("char_start") if page is None else None,
                    "char_end": first_match.get("char_end") if page is None else None,
                    "max_output_chars": 2000,
                },
            }
            state["sufficiency"] = "pending"
            state["sufficiency_reason"] = (
                f"Found match in '{doc_id}'; opening page {page} for complete context."
            )
            state["latencies"][f"eval_step_{state['step']}"] = (time.perf_counter() - t0) * 1000
            return state

        # Default: Evidence is sufficient
        state["sufficiency"] = "sufficient"
        state["sufficiency_reason"] = (
            f"Collected {len(evidence_list)} verified evidence passages across {len(state['discovered_docs'])} documents."
        )
        state["latencies"][f"eval_step_{state['step']}"] = (time.perf_counter() - t0) * 1000
        return state

    def should_continue(self, state: AgenticGraphState) -> str:
        """Route to synthesis when evidence is sufficient or budget exhausted, else continue loop."""
        if state.get("sufficiency") in ("sufficient", "unanswerable", "insufficient", "needs_clarification"):
            return "synthesize"
        if state.get("step", 0) >= state.get("max_steps", 5):
            return "synthesize"
        return "continue"

    def synthesize_answer(self, state: AgenticGraphState) -> AgenticGraphState:
        """
        Synthesize the final answer grounded in verified evidence, or abstain when evidence is missing.
        Validates citation references against actual evidence records.
        """
        t0 = time.perf_counter()
        store = EvidenceStore.from_dict(state["evidence_store_data"])

        if state.get("sufficiency") == "needs_clarification":
            state["answer"] = state["clarification_question"]
            state["abstained"] = False
            state["citations"] = []
            state["citation_report"] = {
                **store.validate_citations(""), "missing_citations": False,
            }
            state["latencies"]["synthesize"] = (time.perf_counter() - t0) * 1000
            return state

        # Grounded Abstention Case
        if state.get("sufficiency") == "unanswerable" or len(store) == 0:
            state["abstained"] = True
            state["answer"] = (
                f"I cannot answer '{state['question']}' based on the available indexed enterprise documents. "
                f"Reason: {state.get('sufficiency_reason', 'No matching evidence found in corpus.')}"
            )
            state["citations"] = []
            state["citation_report"] = {
                "total_cited": 0,
                "verified_count": 0,
                "unverified_count": 0,
                "citation_integrity": 1.0,
            }
            state["latencies"]["synthesize"] = (time.perf_counter() - t0) * 1000
            return state

        context = store.format_context_for_prompt(max_tokens=3000)

        try:
            question = state["question"]
            if state.get("sufficiency") == "insufficient":
                question += (
                    "\nInvestigation ended before evidence completeness was established. "
                    "Answer only supported parts, state what remains unknown, and do not guess."
                )
            raw_gen = self.llm.generate(question, context)
            raw_answer = str(raw_gen).strip() if raw_gen is not None else ""
            if not raw_answer:
                raise RuntimeError("LLM returned an empty answer")
        except Exception as e:
            state["error"] = f"Generation failed: {e}"
            # Deterministic fallback when LLM fails or is unavailable
            top_evidence = sorted(
                store.get_all(),
                key=lambda r: (r.rerank_score or 0.0, r.hybrid_score or 0.0),
                reverse=True,
            )
            snippets = [f"[{r.document_id}] {r.text.strip()}" for r in top_evidence[:3]]
            raw_answer = "Based on retrieved evidence:\n\n" + "\n\n".join(snippets)

        # Validate citation integrity
        citation_report = store.validate_citations(raw_answer)
        refusal = "I could not find the answer in the indexed documents."
        needs_citations = bool(raw_answer) and raw_answer != refusal
        # One bounded repair; never manufacture references by appending sources
        # to claims the model did not attribute.
        if not state.get("error") and needs_citations and (
            not citation_report["total_cited"] or citation_report["unverified_count"]
        ):
            try:
                repaired = self.llm.generate(
                    f"{state['question']}\n\n"
                    f"Draft answer:\n{raw_answer}\n\n"
                    "Rewrite the draft using only the supplied evidence. Cite each factual "
                    "claim with an exact Source label from the context. Remove unsupported "
                    "claims and invented locations. Return only the corrected answer.",
                    context,
                )
                repaired = str(repaired or "").strip()
                checked = store.validate_citations(repaired)
                if repaired == refusal or (checked["total_cited"] and not checked["unverified_count"]):
                    raw_answer, citation_report = repaired, checked
            except Exception as exc:
                state["error"] = f"Citation correction failed: {exc}"
            citation_report["repair_attempted"] = True
        citation_report["missing_citations"] = bool(raw_answer) and raw_answer != refusal and not citation_report["total_cited"]
        citations = store.generate_citations()

        state["abstained"] = raw_answer == refusal
        state["answer"] = raw_answer
        state["citations"] = citations
        state["citation_report"] = citation_report
        state["latencies"]["synthesize"] = (time.perf_counter() - t0) * 1000
        return state

    # -------------------------------------------------------------------------
    # Graph Construction
    # -------------------------------------------------------------------------

    def build_graph(self) -> Any:
        """Compile the LangGraph state machine."""
        workflow = StateGraph(AgenticGraphState)

        workflow.add_node("plan", self.plan_and_route)
        workflow.add_node("execute_tool", self.execute_tool)
        workflow.add_node("evaluate_evidence", self.evaluate_evidence)
        workflow.add_node("synthesize", self.synthesize_answer)

        workflow.set_entry_point("plan")
        workflow.add_conditional_edges(
            "plan", self.should_continue,
            {"continue": "execute_tool", "synthesize": "synthesize"},
        )
        workflow.add_edge("execute_tool", "evaluate_evidence")

        workflow.add_conditional_edges(
            "evaluate_evidence",
            self.should_continue,
            {
                "continue": "execute_tool",
                "synthesize": "synthesize",
            },
        )

        workflow.add_edge("synthesize", END)
        return workflow.compile()

    def run(self, question: str, max_steps: int = 5) -> Dict[str, Any]:
        """
        Execute the full AgenticRAG workflow for a user question.
        """
        if not question.strip():
            raise ValueError("Question cannot be empty")
        if not 1 <= max_steps <= 15:
            raise ValueError("max_steps must be between 1 and 15")
        initial_state: AgenticGraphState = {
            "question": question,
            "step": 0,
            "max_steps": max_steps,
            "evidence_store_data": {},
            "tool_history": [],
            "discovered_docs": [],
            "sub_questions": [],
            "current_action": {},
            "decision_history": [],
            "clarification_question": "",
            "sufficiency": "pending",
            "sufficiency_reason": "",
            "answer": "",
            "citations": [],
            "citation_report": {},
            "latencies": {},
            "error": None,
            "abstained": False,
        }

        graph = self.build_graph()
        final_state = graph.invoke(initial_state, config={"recursion_limit": max_steps * 3 + 5})

        return {
            "planner_mode": self.planner_mode,
            "decision_history": final_state.get("decision_history", []),
            "error": final_state.get("error"),
            "question": question,
            "answer": final_state["answer"],
            "abstained": final_state["abstained"],
            "sufficiency": final_state["sufficiency"],
            "sufficiency_reason": final_state["sufficiency_reason"],
            "steps": final_state["step"],
            "tool_history": final_state["tool_history"],
            "discovered_docs": final_state["discovered_docs"],
            "citations": final_state["citations"],
            "citation_report": final_state["citation_report"],
            "latencies": final_state["latencies"],
            "evidence_count": len(final_state["evidence_store_data"]),
            "evidence_records": list(final_state["evidence_store_data"].values()),
        }
