# RAG practice examples

This set adds **40 authored questions over eight fictional source documents**.
All company names, policies, prices, incidents and financial figures are invented
for testing. They are not real company guidance or empirical research results.

There are 32 answerable questions and eight deliberately unanswerable questions.
The set covers factual lookups, exact identifiers, navigation, calculations,
comparisons, multiple-source reasoning, superseded policies and false premises.
Examples include:

- What is the default value of `ATLAS_REQUEST_TIMEOUT_MS`?
- For a Premium customer affected by INC-042, what is the first-response target?
- What percentage of FY2025 revenue came from cloud?
- How did the leave carryover cap change between 2025 and 2026?
- Does Atlas Pro automatically include 24/7 Premium support?
- What database engine does Atlas API use? (Not specified.)

Each answerable item includes an expected answer, source filenames, exact quoted
passages and character offsets. Numerical answers show their calculations. The
loader checks quoted spans against the corpus before running. Unanswerable items
have empty evidence labels; some ask for values explicitly undisclosed in the
sources, while others concern facts the corpus never mentions.

The 20 dev and 20 held_out labels are **practice partitions only**. Both were
authored together and are visible in the repository; neither establishes an
independent held-out research evaluation. Near-related questions share sources.
Keyword labels are convenience hints, not correctness judgments.

## Run actual retrieval without Ollama

From the repository root, with dependencies and embedding/reranker models ready:

```bash
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 venv/bin/python -m knowledge.evaluation.examples \
  --index --output reports/examples-retrieval.json
```

`HF_HUB_OFFLINE=1` requires cached models. Omit it when intentionally downloading
missing models. `OMP_NUM_THREADS=1` avoids the installed native-library threading
issue observed in the implementation audit.

The default workspace is `.knowledge/examples`, separate from the usual index.
The runner rejects indexes containing other source files. `--index` builds or
updates only the example corpus; it does not clear unrelated documents. To use
another isolated workspace, pass `--workspace .knowledge/practice-v2` on every run.

## Generate answers when Ollama is running

```bash
venv/bin/python -m knowledge.evaluation.examples --mode standard \
  --split dev --max-samples 5 --output reports/examples-standard.json

venv/bin/python -m knowledge.evaluation.examples --mode agentic \
  --split dev --max-samples 5 --output reports/examples-agentic.json
```

Use the same thread/offline settings as appropriate for your machine. Agentic
mode uses the heuristic controller and its current default step budget; `--top-k`
applies to retrieval/standard mode only. Remove `--max-samples` for the full
selected partition, or omit `--split` for all 40 cases. `--index` is necessary on
the first run for a fresh workspace.

Outputs retain each expected answer, gold evidence, retrieved content, generated
answer or error, document recall, and corpus/question fingerprints. Retrieval
mode performs no LLM inference. Unanswerable cases have no retrieval recall score.
There is no automatic correctness or faithfulness score: manually inspect whether
the answer states the required facts, uses the right policy version, performs
arithmetic correctly, cites supporting sources and declines unsupported details.
An extractive fallback carrying an error is not a successful generated answer.

This is a runnable practice set, separate from the original unverified
120-question repository benchmark. It is intended to reveal failures, not prove
that one architecture is superior.
