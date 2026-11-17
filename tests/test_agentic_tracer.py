"""
Unit tests for ExecutionTracer, trace redaction, JSON export, and AgentBudget.
"""

import json

from knowledge.agentic.tracer import AgentBudget, ExecutionTracer


def test_agent_budget_exhaustion():
    budget = AgentBudget(max_steps=3, timeout_sec=10.0)
    assert budget.is_step_exhausted(2) is False
    assert budget.is_step_exhausted(3) is True
    assert budget.is_step_exhausted(4) is True


def test_execution_tracer_lifecycle_and_redaction(tmp_path):
    tracer = ExecutionTracer(trace_dir=str(tmp_path))
    trace = tracer.start_trace("What is the quarterly revenue?", budget=AgentBudget(max_steps=4))

    assert trace.trace_id is not None
    assert trace.question == "What is the quarterly revenue?"

    # Record tool step with sensitive input
    tracer.record_step(
        step_number=1,
        tool_name="SEARCH",
        inputs={"queries": ["revenue"], "api_key": "secret_xyz_12345", "auth_token": "token_abc"},
        duration_ms=18.5,
        evidence_ids=["ev_1", "ev_2"],
        results_count=2,
    )

    # Check sensitive fields were redacted
    step_rec = trace.steps[0]
    assert step_rec.redacted_inputs["api_key"] == "[REDACTED]"
    assert step_rec.redacted_inputs["auth_token"] == "[REDACTED]"
    assert step_rec.redacted_inputs["queries"] == ["revenue"]
    assert step_rec.duration_ms == 18.5

    # End trace
    final_trace = tracer.end_trace(
        sufficiency="sufficient",
        termination_reason="Verified facts found in financials.",
        abstained=False,
        evidence_count=2,
    )

    assert final_trace.total_duration_ms > 0
    assert final_trace.final_sufficiency == "sufficient"
    assert final_trace.abstained is False

    # Export to JSON
    json_path = tmp_path / "trace.json"
    json_str = tracer.export_json(str(json_path))
    assert json_path.exists()
    parsed = json.loads(json_str)
    assert parsed["trace_id"] == trace.trace_id
    assert len(parsed["steps"]) == 1
