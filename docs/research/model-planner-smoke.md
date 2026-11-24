# Model-directed investigation: implementation and smoke checks

Review date: 24 November 2026.

`agentic-ask --planner model` enables an experimental LLM decision loop. After
each tool result, the model can search again, inspect a discovered document,
answer, abstain, or ask for clarification. This replaces fixed routing in the
opt-in mode. The default `heuristic` mode and its saved benchmark remain available.

## Implementation

- Ollama receives an action-specific JSON schema, the question, bounded evidence,
  recent tool observations, and the remaining tool budget. Generic examples show
  a missing fact followed by sufficient evidence; they are not benchmark answers.
- The controller validates arguments and discovered document identities before
  executing SEARCH, FIND, or OPEN. It rejects unsupported actions, malformed
  arguments, ambiguous document names, and identical repeated tool calls.
- At most `max_steps` tools and `max_steps + 1` planner requests execute. Each
  planner request has a 120-second timeout. This is not an overall wall-clock
  deadline, token-cost budget, or cancellation guarantee.
- Nonempty evidence and budget exhaustion do not automatically imply sufficiency
  in model mode. Planning errors remain explicit; the system does not silently
  fall back to heuristic routing. Existing answer generation and bounded citation
  correction are reused. A degraded answer carrying an error is not a success.
- `decision_history` records actions, short rationales, durations, and errors.
  CLI JSON, v2 API responses, the v2 stream's final event, and evaluation outputs
  expose the selected mode. The benchmark names it `agentic_model` separately
  from the existing `agentic` system.

SUMMARIZE is still not scheduled, and FIND/OPEN still navigate indexed chunks.
Model decisions can be incorrect even when their syntax and arguments are valid.

## Controlled live investigation

The reproducible [smoke script](../../scripts/check_model_planner.py) creates three
fictional documents in an isolated temporary workspace. One maps the KESTREL audit
packet to the Meridian Steward role. Another maps that role to Nisha Rao. The
third is unrelated. With per-query `top_k=1`, the first tool call retrieved only
the routing document. The model then searched for the newly discovered role and
retrieved the ownership register.

The final development attempt used real SentenceTransformer embeddings, FAISS,
BM25, cross-encoder reranking, and local `llama3.2` (3.2B, Q4_K_M). Planning and
generation used temperature 0, seed 17, context 8192, and maximum output 1024 tokens.
Model fingerprints, source text/hashes, implementation hashes, full decisions,
tool histories, answers, and timings are in the [raw artifacts](model-planner-smoke.json).

Observed sequence: **SEARCH -> SEARCH -> FIND -> SEARCH -> ANSWER**. Four tool
calls and five planner calls completed without an execution error. The model
returned the correct person's name and selected ANSWER before exhausting the
five-tool budget. This demonstrates evidence-dependent retrieval, not reliable
planning in general.

There were two redundant tool calls after the ownership register had been found.
One FIND rationale even said no further call was needed while selecting FIND.
The answer cited `owners.md` but omitted `routing.md`, leaving the full two-source
attribution incomplete despite valid reference identity. The approximately
60-second investigation time is descriptive, not a controlled latency result.

Five development attempts are retained, including the earlier repeated-action
errors, a missing-argument failure, and budget exhaustion. Prompts and the output
schema were refined using these failures. This fixture is therefore development
evidence, not a holdout or an independent quality test.

## Small comparison on unchanged benchmark v2 questions

The first three development questions (`v2_001` through `v2_003`) were run through
both heuristic and model-directed workflows. The final six-case run preserved
all outputs and reported `partial_failure`.

| Question | Heuristic workflow | Model-directed workflow |
|---|---|---|
| ORBIT-17 owner and incident role | Correct requested facts with source references; no execution error. | Correct requested facts, uncited; five-tool budget exhausted, `sufficiency: insufficient`. |
| SC-COPPER preventive visit frequency | Correct six-week interval with a reference; no execution error. | Repeated-action error after one search; false refusal despite retrieved evidence. |
| Unopened BP-42 warranty | Correct 12 months from dispatch with a reference; no execution error. | Correct quantity in degraded output, uncited; repeated-action error after two searches. |

Both workflows retrieved all annotated passage requirements for these three
questions. The heuristic workflow had zero execution errors; the model-directed
workflow had two. An earlier comparison had three model-planning errors and is
also retained. The final run does not demonstrate improved answer quality.

These descriptions are assistant reviews of saved answers against the unchanged
references, not independent human scores. Full-run correctness, faithfulness,
and citation-support fields remain unscored. No new 120-question comparison or
statistical superiority claim is made. The `held_out` split was not used here.

The raw metadata identifies the starting revision and dirty working tree because
these runs occurred during development. The final controlled run's controller,
LLM wrapper, and schema hashes match the implementation saved with this report.
Temporary source paths no longer exist; the script and stored source text recreate
the corpus. Outputs need not be identical across model/runtime versions.

## Validation and reproduction

The full regression suite passes **295 tests**, with six existing dependency
deprecation warnings. Coverage includes adaptive second-source retrieval,
early completion, clarification, abstention with irrelevant evidence, invalid
decisions, unknown documents, repeated actions, budget termination, planning/tool
errors, API/CLI selection, and consistent benchmark generation settings.
Targeted lint, whitespace, README link, and command-syntax checks passed.

```bash
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 python scripts/check_model_planner.py \
  --output reports/planner-investigation.json

HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 python -m knowledge.evaluation.benchmark_v2 \
  --index --workspace .knowledge/planner-v2 --split dev --max-samples 3 \
  --systems agentic agentic_model --output reports/planner-comparison.json
```

Offline mode requires cached retrieval models; Ollama must be running with the
selected model installed. The smoke script returns nonzero if the specified
adaptive retrieval is not observed and saves the result, including failures.

The model-directed mode remains opt-in. Reliable stopping, complete attribution,
and broader answer-quality validation remain unresolved with the tested model.
