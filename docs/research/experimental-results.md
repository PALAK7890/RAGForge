# AgenticRAG evaluation status

## Audit finding (2026-11-17)

The previously published comparison (68.3%, 76.7%, 91.7% retrieval hit rates),
citation scores, latency measurements, token multipliers, and ablation deltas
are withdrawn. No per-question outputs or machine-readable experiment artifacts
supporting those tables were found in the repository or supplied execution log.
The log shows file creation and tests, but no completed comparative experiment.
These figures must not be used as measured RAGForge results.

The old comparison entry point evaluated only ten questions, and the old ablation
entry point used ten per variant. Neither reproduced the claimed 120-question
experiment. Baseline citation scoring was not implemented. Unanswerable cases
and infrastructure errors could earn retrieval credit, and filename substring
matching could mark the wrong source as relevant.

## Dataset facts and limits

`enterprise_benchmark.json` contains 120 authored questions: 62 dev and 58
held_out. Category counts are factual 20, exact_keyword 15, multihop 20,
comparative 15, numerical 15, document_navigation 15, ambiguous 10, unanswerable 10.
It is a repository-documentation question set, not an independently validated
enterprise benchmark. The split labels alone do not establish that the held-out
questions were never used during development.

Gold labels are filenames and keyword lists, not verified passage spans or full
reference answers. Many questions reuse the same documents. Some refer to source
files (`main.py`, `hybrid.py`, etc.) and `config.yaml`; a docs-only index lacks
those files, and a clean checkout has no tracked config.yaml. The loader does not
accept .py or .yaml. Some alternatives listed in expected_documents may not each
contain the requested answer. Labels need human review against a fixed corpus
before scientific comparisons. Do not index the entire repository: it can include
private runtime state and generated artifacts.

The generator is portable and deterministically recreates the current dataset.
It does not create the corpus or verify gold answers.

## Current scoring and reproduction

The comparison CLI now evaluates all selected questions by default and writes
per-question outputs, explicit errors, scores, dataset/chunk hashes, split,
Python version, and commit identifier. Retrieval uses exact normalized filenames
and reports any-source hit rate, document recall, document precision, and document
MRR. These are document-level proxies over each system's returned evidence,
not passage Recall@K or a fixed-K comparison. Unanswerable questions are excluded
from retrieval and keyword denominators. Execution errors receive zero quality
credit and a separate error rate. Keyword recall measures string presence;
it does not establish correctness or faithfulness. Abstention detection for
baselines is a limited phrase heuristic. Citation validity is not citation support.
No confidence intervals, inference costs, answer correctness, or faithfulness are
claimed by this runner.

First curate a public corpus with every annotated source, review gold labels,
and build a fresh `.knowledge` index. Preflight rejects missing source filenames.
Use identical default models and workspace for all three systems; changing only
an injected Baseline A service does not reconfigure the legacy agent. Ensure
Ollama and `llama3.2` are available before attempting generation.

```bash
python -m knowledge.evaluation.enterprise_benchmark_gen
python -m knowledge.evaluation.compare_systems --split dev --output reports/comparison-dev.json
python -m knowledge.evaluation.compare_systems --split held_out --output reports/comparison-held-out.json
python -m knowledge.evaluation.ablation --split held_out --output reports/ablations-held-out.json
```

Use `--max-samples 2` for an explicitly limited smoke run. Save environment/model
versions and the exact corpus alongside outputs before publishing results. Chunk
hashes identify indexed text but do not replace an archived corpus/configuration.

Ablations use empty FIND/OPEN responses and restrict every SEARCH action for the
single-query variant. They change downstream routing and are exploratory, not
clean estimates of each component's contribution. Per-query traces are retained
so ineffective variants are visible. SUMMARIZE is never selected by the current
controller; no-SUMMARIZE is rejected as an inert experiment.

## Verified local checks

The pre-change suite reproduced 198 passed with six warnings. Real ingestion
and hybrid FAISS/BM25/RRF/CrossEncoder search succeeded using cached models on
public documentation (six files, 137 chunks at the time of that check).
A docs-only held_out comparison is correctly rejected because annotated sources
are missing. Ollama was unreachable both inside and outside the restricted
execution environment. No full model-generated comparative or ablation results
were produced or substituted with mocks.

See `audit-report.md` for final regression results and remaining implementation
limits. The original two-question and historical latency sections in README are
historical claims, not new measurements from this audit.
