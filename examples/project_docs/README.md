# RAGForge documentation practice questions

Twelve questions cover all eight benchmark categories using four snapshots of
RAGForge's existing documentation. Ten have supporting passages; two request
information the selected corpus does not provide. Each question includes an
expected answer, behavior, filenames, exact quotes and character spans.

`sources.json` records original repository paths, snapshot revision and SHA-256
hashes. The historical audit and later generation report intentionally disagree
about Ollama availability: answers must distinguish the dates, not treat the old
blocker as current. Numerical questions specify the requested quantity explicitly.

These are assistant-authored practice cases with assistant source review, not an
independent human-reviewed benchmark. All cases are development examples. They
do not establish generalization or semantic answer accuracy. Source claims are
evaluated as documented at the snapshot revision, not as current runtime facts.
An independent reviewer should check labels and record disagreements before this
set is used for any formal quality claim. The existing 120-question corpus remains
unchanged.

Reuse the existing runner with a separate workspace:

```bash
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 python -m knowledge.evaluation.examples \
  --examples examples/project_docs --workspace .knowledge/project-docs \
  --index --output reports/project-docs-retrieval.json

HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 python -m knowledge.evaluation.examples \
  --examples examples/project_docs --workspace .knowledge/project-docs \
  --mode agentic --output reports/project-docs-agentic.json
```

Index only `corpus/`; never index `questions.json`, reference answers or reviews.
Document recall measures source-file coverage, not passage coverage or answer
quality. Check raw answers for missing or unsupported claims and citations.

## Recorded checks

All 12 questions completed real retrieval and AgenticRAG generation without an
execution error. All ten evidence-bearing questions found their required files,
but exact-passage coverage averaged 85%; only 8/10 retrieved every annotated
passage. Live answers still showed false refusals, a wrong numerical interpretation,
rubric confusion and unresolved ambiguity. See the
[answer review and raw outputs](../../docs/research/project-docs-answer-review.md).
These are assistant-reviewed findings, not independently validated accuracy scores.
