# Project documentation: live answer review

All 12 cases produced nonempty answers with zero recorded execution errors using
the actual AgenticRAG pipeline at revision `dce1361`. This application check used
local `llama3.2` with Ollama runtime defaults, not the fixed generation options of
the 360-case benchmark. Its performance numbers must not be compared directly
with that experiment.

This is an assistant review of the saved answers, reference annotations, collected
evidence and reconstructed generation context. It is not independent human
validation. No semantic accuracy percentage is asserted.

| Case | Review finding |
|---|---|
| `project_001` benchmark size | Incomplete response. The model refused although the context supplied 120 questions and 24 documents. The category table was not retrieved, so the eight-category fact had a retrieval gap. A partial answer with that limitation would be more useful. |
| `project_002` thread setting | False refusal. The rendered context contained `OMP_NUM_THREADS=1` and the native-library issue, but the model said it could not find the answer. |
| `project_003` model names | Correct answer with supporting references: `all-MiniLM-L6-v2` and `cross-encoder/ms-marco-MiniLM-L-6-v2`. |
| `project_004` unsupported research claims | Correctly rejects automatic SUMMARIZE scheduling and unreviewed semantic-correctness claims, citing the architecture and generation report. |
| `project_005` historical availability | False refusal despite both the old Ollama blocker and the later completed run being in the rendered context. The answer should distinguish the two reports in time. |
| `project_006` citation subtraction | Wrong quantity. It reports 81 uncited answers instead of calculating 39 cited answers minus 37 with invalid references = 2. The reference exists and supports the number 81, but that fact does not answer the question. |
| `project_007` latency subtraction | Correctly calculates 1.093 seconds and rejects a controlled speed claim. The quoted “not a randomized performance study” wording comes from the retrieved benchmark guide; citing that document alongside the numeric report would improve quotation attribution. |
| `project_008` review rubric | Incorrect and uncited. It confuses null with a score of zero. The retrieved rubric says factual answers missing required citations receive 0; null is appropriate for a refusal without a factual claim requiring support. |
| `project_009` ambiguous accuracy | Does not clarify the requested benchmark, system or metric. Neither annotated gold passage was retrieved. The plain refusal avoids inventing a score but does not satisfy the expected clarification behavior. |
| `project_010` ambiguous attempt budget | Selects the architecture's documented default of 3 without clarifying which agent or action was meant. The context also contained the benchmark's two rewrites and five tool steps. The quoted default is source-backed; the unresolved ambiguity makes the answer incomplete. |
| `project_011` human-reviewed correctness | Appropriate refusal: no independently reviewed correctness percentage is supplied by the corpus. |
| `project_012` production bill and SLA | Appropriate refusal: the corpus supplies neither a hosting bill nor a guaranteed production uptime SLA. |

Five answers contain bracket citations; all five pass reference identity/location
validation. That does not make all five answers correct, as `project_006` shows.
Six answers are canonical refusals and correctly expose `abstained: true` after
the flag fix; four of those concern questions intended to be answered or clarified.
The remaining uncited factual answer (`project_008`) exposes `missing_citations: true`.

All ten evidence-bearing cases found their required source files. An offline
check of the retrieval-only run found **85% mean exact-passage coverage**, with
all annotated passages retrieved for **8/10 questions**. The two unanswerables
are excluded. These coverage results are not answer-correctness measurements.

The source snapshots and reference labels remain frozen. An independent reviewer
should check these annotations before they support a formal quality claim. Future
work should address using available evidence, numerical question interpretation,
and clarification behavior; adding repeated citation retries alone does not solve
those failures.

- [Raw live answers and evidence](project-docs-agentic.json).
- [Generation environment](project-docs-generation-environment.json).
- [Retrieval-only results](project-docs-retrieval.json) and [exact-passage check](project-docs-passage-coverage.json).
- [Questions, source snapshots and reproduction commands](../../examples/project_docs/README.md).
