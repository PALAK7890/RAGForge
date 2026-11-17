"""
Structured execution tracing and execution budget guards for AgenticRAG.
Captures granular step timings, redacted tool calls, evidence counts, and exports JSON traces.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4


@dataclass
class AgentBudget:
    """Configurable execution budgets preventing unbounded resource consumption."""
    max_steps: int = 5
    max_tool_calls: int = 10
    max_context_tokens: int = 4000
    timeout_sec: float = 60.0
    max_output_chars: int = 4000

    def is_step_exhausted(self, current_step: int) -> bool:
        return current_step >= self.max_steps

    def is_timed_out(self, start_time: float) -> bool:
        return (time.perf_counter() - start_time) > self.timeout_sec


@dataclass
class StepTraceRecord:
    """Trace record for a single tool execution step."""
    step_number: int
    tool_name: str
    redacted_inputs: Dict[str, Any]
    duration_ms: float
    evidence_ids: List[str]
    results_count: int
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class ExecutionTrace:
    """Full end-to-end execution trace for an AgenticRAG query."""
    trace_id: str
    question: str
    start_time: float
    end_time: Optional[float] = None
    total_duration_ms: Optional[float] = None
    steps: List[StepTraceRecord] = field(default_factory=list)
    final_sufficiency: Optional[str] = None
    termination_reason: Optional[str] = None
    abstained: bool = False
    evidence_count: int = 0
    budget: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


class ExecutionTracer:
    """
    Manages trace recording, sensitive data redaction, and persistence.
    """

    def __init__(self, trace_dir: str = ".knowledge/traces") -> None:
        self.trace_dir = Path(trace_dir)
        self.active_trace: Optional[ExecutionTrace] = None
        self._start_perf: float = 0.0

    def start_trace(self, question: str, budget: Optional[AgentBudget] = None) -> ExecutionTrace:
        """Initialize a new query execution trace."""
        self._start_perf = time.perf_counter()
        b_dict = asdict(budget) if budget else asdict(AgentBudget())

        self.active_trace = ExecutionTrace(
            trace_id=uuid4().hex[:12],
            question=question,
            start_time=time.time(),
            budget=b_dict,
        )
        return self.active_trace

    def record_step(
        self,
        step_number: int,
        tool_name: str,
        inputs: Dict[str, Any],
        duration_ms: float,
        evidence_ids: List[str],
        results_count: int,
        error: Optional[str] = None,
    ) -> None:
        """Record a single tool execution event with sensitive parameter redaction."""
        if not self.active_trace:
            return

        # Redact sensitive parameters (passwords, tokens, keys)
        redacted = {}
        for k, v in inputs.items():
            if any(s in k.lower() for s in ["key", "token", "password", "secret", "auth"]):
                redacted[k] = "[REDACTED]"
            else:
                redacted[k] = v

        step_record = StepTraceRecord(
            step_number=step_number,
            tool_name=tool_name.upper(),
            redacted_inputs=redacted,
            duration_ms=round(duration_ms, 2),
            evidence_ids=evidence_ids,
            results_count=results_count,
            error=error,
        )
        self.active_trace.steps.append(step_record)

    def end_trace(
        self,
        sufficiency: str,
        termination_reason: str,
        abstained: bool,
        evidence_count: int,
    ) -> ExecutionTrace:
        """Finalize the active trace."""
        if not self.active_trace:
            raise RuntimeError("No active trace to end.")

        elapsed_ms = (time.perf_counter() - self._start_perf) * 1000
        self.active_trace.end_time = time.time()
        self.active_trace.total_duration_ms = round(elapsed_ms, 2)
        self.active_trace.final_sufficiency = sufficiency
        self.active_trace.termination_reason = termination_reason
        self.active_trace.abstained = abstained
        self.active_trace.evidence_count = evidence_count

        return self.active_trace

    def export_json(self, file_path: Optional[str] = None) -> str:
        """Export the current trace to a JSON string and optionally save to disk."""
        if not self.active_trace:
            return "{}"

        json_str = self.active_trace.to_json()
        if file_path:
            p = Path(file_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(json_str)
        return json_str
