"""Model-directed investigation, with scripted models and no external inference."""

import json
from unittest.mock import MagicMock

import pytest

from knowledge.agentic.controller import AgenticController
from knowledge.agentic.schemas import (
    EvidenceRecord,
    FindMatch,
    FindResult,
    OpenResult,
    SearchResult,
    ToolError,
)
from knowledge.llms.ollama_llm import OllamaLLM


def decision(action, reason="Choose the next step from available evidence.", **arguments):
    return {"action": action, "reason": reason, "arguments": arguments}


def evidence(document="finance.md", text="Cloud revenue reached $4.2B."):
    return EvidenceRecord(
        evidence_id=document,
        chunk_id=document,
        document_id=document,
        source_path=f"/docs/{document}",
        text=text,
    )


def search_result(*records):
    return SearchResult(
        queries_used=["requested query"],
        evidence=list(records),
        total_found=len(records),
        latency_ms=0,
    )


class ScriptedLLM:
    def __init__(self, decisions, answer="Cloud revenue reached $4.2B [finance.md]."):
        self.decisions = iter(decisions)
        self.contexts = []
        self.generation_contexts = []
        self.answer = answer

    def plan_action(self, question, context):
        self.contexts.append(context)
        value = next(self.decisions)
        if isinstance(value, Exception):
            raise value
        return value

    def generate(self, question, context):
        self.generation_contexts.append(context)
        return self.answer


def make_controller(llm, search=None, find=None, open_tool=None):
    search = search if search is not None else MagicMock()
    if not search.execute.side_effect and not isinstance(search.execute.return_value, SearchResult):
        search.execute.return_value = search_result(evidence())
    return AgenticController(
        search_tool=search,
        find_tool=find if find is not None else MagicMock(),
        open_tool=open_tool if open_tool is not None else MagicMock(),
        summarize_tool=MagicMock(),
        llm=llm,
        planner_mode="model",
    )


def test_investigation_searches_again_for_missing_fact_and_combines_two_sources():
    """A second query must be chosen after inspecting the first search's evidence."""
    finance = evidence()
    security = evidence("security.md", "Cloud workloads require SOC2 Type II certification.")
    search = MagicMock()

    def retrieve(payload):
        if payload.queries == ["cloud revenue"]:
            return search_result(finance)
        assert payload.queries == ["cloud certification requirements"]
        return search_result(security)

    search.execute.side_effect = retrieve

    class AdaptiveLLM(ScriptedLLM):
        def plan_action(self, question, context):
            self.contexts.append(context)
            if "Cloud revenue reached $4.2B." not in context:
                return decision("search", queries=["cloud revenue"])
            if "Cloud workloads require SOC2 Type II certification." not in context:
                return decision(
                    "search",
                    reason="Revenue is known, but the certification requirement is missing.",
                    queries=["cloud certification requirements"],
                )
            return decision("answer", reason="Both requested facts have source evidence.")

    llm = AdaptiveLLM(
        [],
        "Revenue reached $4.2B [finance.md]; workloads require SOC2 Type II [security.md].",
    )
    controller = make_controller(llm, search=search)
    result = controller.run("What are cloud revenue and certification requirements?", max_steps=5)

    assert result["error"] is None
    assert result["steps"] == 2
    assert len(llm.contexts) == 3
    assert result["discovered_docs"] == ["finance.md", "security.md"]
    assert result["evidence_count"] == 2
    assert [step["tool"] for step in result["tool_history"]] == ["SEARCH", "SEARCH"]
    assert len(result["decision_history"]) == 3
    assert "Cloud revenue reached $4.2B." in llm.generation_contexts[0]
    assert "Cloud workloads require SOC2 Type II certification." in llm.generation_contexts[0]
    assert result["citation_report"]["verified_count"] == 2
    controller.find_tool.execute.assert_not_called()
    controller.open_tool.execute.assert_not_called()


def test_model_can_finish_after_one_search_without_forced_navigation():
    llm = ScriptedLLM([decision("search", queries=["cloud revenue"]), decision("answer")])
    controller = make_controller(llm)

    result = controller.run("What is cloud revenue?", max_steps=5)

    assert result["steps"] == 1
    assert result["sufficiency"] == "sufficient"
    assert not result["abstained"]
    assert len(llm.contexts) == 2
    controller.find_tool.execute.assert_not_called()
    controller.open_tool.execute.assert_not_called()


def test_find_results_are_visible_before_model_selects_open_location():
    search = MagicMock()
    search.execute.return_value = search_result(evidence("policy.md", "The retention policy index."))
    find = MagicMock()
    find.execute.return_value = FindResult(
        document_id="policy.md",
        expression="retention",
        matches=[FindMatch(
            chunk_id="policy-section",
            document_id="policy.md",
            source_path="/docs/policy.md",
            page_number=None,
            char_start=240,
            char_end=280,
            snippet="The retention exception applies for 90 days.",
            match_offset_in_text=0,
        )],
        total_matches=1,
        latency_ms=0,
    )
    opened = evidence("policy.md", "The retention exception applies for 90 days.")
    open_tool = MagicMock()
    open_tool.execute.return_value = OpenResult(
        document_id="policy.md",
        source_path="/docs/policy.md",
        views=[],
        total_chars=len(opened.text),
        evidence_records=[opened],
        latency_ms=0,
    )
    llm = ScriptedLLM([
        decision("search", queries=["retention policy"]),
        decision("find", document_id="policy.md", expression="retention"),
        decision("open", document_id="policy.md", char_start=240, char_end=280),
        decision("answer"),
    ], answer="The exception applies for 90 days [policy.md].")

    result = make_controller(llm, search, find, open_tool).run("How long is the exception?", max_steps=5)

    assert result["error"] is None
    assert "The retention exception applies for 90 days." in llm.contexts[2]
    assert "240" in llm.contexts[2]
    assert open_tool.execute.call_args.args[0].char_start == 240
    assert open_tool.execute.call_args.args[0].char_end == 280
    assert [step["tool"] for step in result["tool_history"]] == ["SEARCH", "FIND", "OPEN"]


def test_ambiguous_request_can_ask_for_clarification_without_retrieval():
    clarification = "Do you mean the 2025 policy or the 2026 policy?"
    llm = ScriptedLLM([decision("clarify", question=clarification)])
    controller = make_controller(llm)

    result = controller.run("What is the policy?", max_steps=5)

    assert result["error"] is None
    assert result["answer"] == clarification
    assert result["steps"] == 0
    assert not result["abstained"]
    assert not llm.generation_contexts
    controller.search_tool.execute.assert_not_called()


def test_model_can_abstain_despite_nonempty_but_irrelevant_evidence():
    llm = ScriptedLLM([
        decision("search", queries=["cloud revenue forecast"]),
        decision("abstain", reason="The source reports past revenue but no forecast."),
    ])

    result = make_controller(llm).run("What is next year's cloud revenue forecast?", max_steps=5)

    assert result["error"] is None
    assert result["abstained"]
    assert result["evidence_count"] == 1
    assert result["sufficiency"] != "sufficient"
    assert not llm.generation_contexts


@pytest.mark.parametrize("invalid", [
    None,
    "not a decision",
    decision("delete", path="/docs/finance.md"),
    decision("search", queries=[]),
    decision("search", queries="cloud revenue"),
    decision("find", expression="revenue"),
])
def test_invalid_decision_stops_with_error_without_executing_tools(invalid):
    llm = ScriptedLLM([invalid])
    controller = make_controller(llm)

    result = controller.run("What is cloud revenue?", max_steps=5)

    assert result["error"]
    assert result["steps"] == 0
    assert result["abstained"]
    assert len(llm.contexts) == 1
    assert result["decision_history"]
    controller.search_tool.execute.assert_not_called()
    controller.find_tool.execute.assert_not_called()
    controller.open_tool.execute.assert_not_called()


@pytest.mark.parametrize("action,arguments", [
    ("find", {"document_id": "invented.md", "expression": "revenue"}),
    ("open", {"document_id": "invented.md", "page": 1}),
])
def test_model_cannot_navigate_to_an_unknown_document(action, arguments):
    llm = ScriptedLLM([
        decision("search", queries=["cloud revenue"]),
        decision(action, **arguments),
    ])
    controller = make_controller(llm)

    result = controller.run("What is cloud revenue?", max_steps=5)

    assert result["error"]
    assert result["steps"] == 1
    assert result["evidence_count"] == 1
    controller.find_tool.execute.assert_not_called()
    controller.open_tool.execute.assert_not_called()


def test_repeated_tool_action_is_bounded_and_reported():
    search_decision = decision("search", queries=["cloud revenue"])
    llm = ScriptedLLM([search_decision, search_decision])
    controller = make_controller(llm)

    result = controller.run("What is cloud revenue?", max_steps=5)

    assert result["error"]
    assert result["steps"] == 1
    assert controller.search_tool.execute.call_count == 1
    assert len(llm.contexts) == 2


def test_budget_exhaustion_does_not_claim_incomplete_evidence_is_sufficient():
    llm = ScriptedLLM([
        decision("search", queries=["cloud revenue"]),
        decision("search", reason="Certification evidence is still missing.", queries=["cloud certification"]),
    ])
    controller = make_controller(llm)

    result = controller.run("What are cloud revenue and certification requirements?", max_steps=1)

    assert result["steps"] == 1
    assert controller.search_tool.execute.call_count == 1
    assert len(llm.contexts) <= 2
    assert result["sufficiency"] != "sufficient"
    assert result["evidence_count"] == 1


def test_planner_failure_retains_evidence_and_does_not_revert_to_heuristics():
    llm = ScriptedLLM([
        decision("search", queries=["cloud revenue"]),
        RuntimeError("planner unavailable"),
    ])
    controller = make_controller(llm)

    result = controller.run("What is cloud revenue?", max_steps=5)

    assert "planner unavailable" in result["error"]
    assert result["steps"] == 1
    assert result["evidence_count"] == 1
    assert len(llm.contexts) == 2
    controller.find_tool.execute.assert_not_called()
    controller.open_tool.execute.assert_not_called()


def test_tool_error_remains_visible_and_is_not_counted_as_evidence():
    search = MagicMock()
    search.execute.return_value = SearchResult(
        queries_used=["cloud revenue"],
        evidence=[],
        total_found=0,
        latency_ms=0,
        error=ToolError(tool_name="SEARCH", error_type="IndexError", message="index unavailable"),
    )
    llm = ScriptedLLM([
        decision("search", queries=["cloud revenue"]),
        decision("abstain", reason="Search failed."),
    ])

    result = make_controller(llm, search=search).run("What is cloud revenue?", max_steps=5)

    assert "index unavailable" in result["error"]
    assert result["evidence_count"] == 0
    assert result["tool_history"][0]["error"]
    assert search.execute.call_count == 1


def test_answer_without_any_evidence_does_not_generate_an_unsupported_answer():
    llm = ScriptedLLM([decision("answer", reason="I already know this from memory.")])
    controller = make_controller(llm)

    result = controller.run("What is cloud revenue?", max_steps=5)

    assert result["abstained"]
    assert not llm.generation_contexts
    assert result["sufficiency"] != "sufficient"
    controller.search_tool.execute.assert_not_called()


def test_ollama_planning_uses_schema_and_bounded_request(monkeypatch):
    expected = decision("search", queries=["missing fact"])
    client = MagicMock()
    client.chat.return_value = {"message": {"content": json.dumps(expected)}}
    factory = MagicMock(return_value=client)
    monkeypatch.setattr("ollama.Client", factory)
    assert OllamaLLM("local-model").plan_action("question", "observations") == expected
    factory.assert_called_once_with(timeout=120)
    request = client.chat.call_args.kwargs
    assert request["model"] == "local-model"
    search_schema = request["format"]["oneOf"][0]
    assert search_schema["additionalProperties"] is False
    assert search_schema["properties"]["action"] == {"const": "search"}
    assert search_schema["properties"]["arguments"]["required"] == ["queries"]
    assert request["options"] == {"temperature": 0, "seed": 17, "num_ctx": 8192, "num_predict": 1024}
    assert request["messages"][0]["role"] == "system"
    assert "observations" in request["messages"][-1]["content"]


@pytest.mark.parametrize("content", [
    "Not JSON", '{"action":"delete","reason":"bad","arguments":{}}',
    'prefix {"action":"answer","reason":"done","arguments":{}}',
])
def test_ollama_rejects_invalid_planning_output_without_guessing(monkeypatch, content):
    from pydantic import ValidationError

    llm = OllamaLLM()
    monkeypatch.setattr(llm, "_chat", lambda **kwargs: {"message": {"content": content}})
    with pytest.raises(ValidationError):
        llm.plan_action("question", "evidence")


@pytest.mark.parametrize("selectors", [
    {"page": 1, "char_end": 20},
    {"chunk_index": 0, "char_start": 0},
    {"page": 1, "chunk_index": 0},
])
def test_conflicting_open_selectors_fail_before_tool_execution(selectors):
    llm = ScriptedLLM([
        decision("search", queries=["cloud revenue"]),
        decision("open", document_id="finance.md", **selectors),
    ])
    controller = make_controller(llm)
    result = controller.run("What is revenue?", max_steps=5)
    assert result["error"]
    controller.open_tool.execute.assert_not_called()


def test_null_selectors_do_not_bypass_repeated_open_detection():
    llm = ScriptedLLM([
        decision("search", queries=["cloud revenue"]),
        decision("open", document_id="finance.md"),
        decision("open", document_id="finance.md", page=None, char_start=None, char_end=None),
    ])
    open_tool = MagicMock()
    open_tool.execute.return_value = OpenResult(
        document_id="finance.md", source_path="/docs/finance.md", views=[], total_chars=0, latency_ms=0,
    )
    controller = make_controller(llm, open_tool=open_tool)
    result = controller.run("What is revenue?", max_steps=5)
    assert result["error"]
    assert result["steps"] == 2
    assert open_tool.execute.call_count == 1
