# RAGForge benchmark v2 — fictional evidence-grounded evaluation

Version **2.0.0** preserves the original benchmark's 120-question category counts
while replacing its missing-corpus assumptions with 24 supplied Markdown files.
The original `enterprise_benchmark.json` and the 40 practice examples are unchanged.

All organisations, assets, policies, accounts and prices are fictional. This is
an authored engineering benchmark, not a paper reproduction or an independently
reviewed enterprise dataset. Its scores cannot be compared directly with the
withdrawn v1 numbers or with the smaller practice corpus's document hit rate.

## Data design

| Category | Development | Evaluation | Total |
|---|---:|---:|---:|
| Factual retrieval | 10 | 10 | 20 |
| Exact keyword | 8 | 7 | 15 |
| Multi-hop | 10 | 10 | 20 |
| Comparative | 8 | 7 | 15 |
| Numerical | 8 | 7 | 15 |
| Document navigation | 8 | 7 | 15 |
| Ambiguous | 5 | 5 | 10 |
| Unanswerable | 5 | 5 | 10 |
| **Total** | **62** | **58** | **120** |

Development uses Aster Robotics: support routing, parts, transport, safety,
accounts, records and workshop competence. Evaluation uses Bracken Observatory:
instruments, grants, suppliers, archive access, visitor rules and observing budgets.
Each split has 12 distinct documents, including rejected proposals and confusingly
named equipment. The corpus includes obsolete/current policy pairs and Markdown
tables. Every multi-hop question requires evidence from at least two documents.
Numerical questions include worked calculations in their reference annotations.

Both families share broad category concepts but not source documents or
organisation-specific investigation chains. Draft retrieval and integrity checks
occurred during construction; the versioned snapshot was committed before the
recorded run. All files and labels are public; `held_out` is a reserved evaluation label,
not a blind external test. This initial audit evaluates both splits and therefore
exposes their failures. Any subsequent tuning on evaluation failures invalidates
that reservation; create a new version for a fresh final evaluation.

`benchmark.json` records source fingerprints and exact quoted passages with
character spans. Each question has `expected_answer`, `expected_behavior`, and
`evidence_groups`: every group is required, while IDs within a group are acceptable
alternatives. An answer that needs two facts has two groups, not two alternative
filenames. Unanswerable questions have no positive evidence groups. Ambiguous
questions require clarification or explicit conditional answers rather than an
arbitrary choice. Unsupported numerical requests must not be answered by guessing.

`build.py` is the deterministic fixture-authoring source. It is not imported by the
RAG pipeline and its questions/reference answers are never indexed. Regenerating
it must reproduce the snapshot byte-for-byte. The loader checks counts, unique
IDs/questions, source hashes, exact passage spans, cross-split evidence isolation
and multi-document requirements. These checks establish structural integrity;
human review is still needed to certify semantic labels and coverage.

## Reproduce retrieval without Ollama

Run from the repository root with the existing environment and cached models:

```bash
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 venv/bin/python -m knowledge.evaluation.benchmark_v2 \
  --index --split dev --output reports/v2-dev-retrieval.json

HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 venv/bin/python -m knowledge.evaluation.benchmark_v2 \
  --split held_out --output reports/v2-evaluation-retrieval.json
```

Use `--split all` for 120 questions or `--max-samples 3` for an explicitly limited
smoke run. Default K is 5. The isolated workspace is `.knowledge/benchmark-v2`;
choose a fresh directory with `--workspace` if a previous fixture version is indexed.
The runner rejects unrelated sources, stale chunks, mismatched vector/chunk counts,
and any required passage not represented in the index. It does not clear user data.
Do not index this entire examples directory: only `corpus/` belongs in the index.
Omit offline mode only when intentionally downloading missing models. The thread
setting addresses the native-library instability recorded in the prior audit.

## Compare the three systems

First start Ollama and install the selected model (`llama3.2` by default). Then:

```bash
OMP_NUM_THREADS=1 venv/bin/python -m knowledge.evaluation.benchmark_v2 \
  --index --split dev --systems standard self_correcting agentic \
  --output reports/v2-dev-comparison.json --review-template reports/v2-dev-review.json

OMP_NUM_THREADS=1 venv/bin/python -m knowledge.evaluation.benchmark_v2 \
  --split held_out --systems standard self_correcting agentic \
  --output reports/v2-evaluation-comparison.json --review-template reports/v2-evaluation-review.json
```

All systems share one isolated index, embedding/reranker instances, Ollama model
and per-search K. Embedding is `all-MiniLM-L6-v2`; reranker is
`cross-encoder/ms-marco-MiniLM-L-6-v2`. Final generation requests source citations.
Generation uses temperature 0, seed 17, context 8192 tokens and maximum output 1024
tokens, with a 120-second request timeout. These settings improve reproducibility
but do not guarantee bit-identical output across runtimes/hardware.

Standard RAG retrieves once; self-correcting RAG permits two rewrites; AgenticRAG
has five tool steps. Their retrieval-call counts and context-selection policies
are intentionally different. This is equal per-search depth, not equal total
compute. Search models are warmed once before timing. Systems run in a fixed
order with shared warm models; latency is descriptive, not a randomized performance
study. The AgenticRAG controller remains heuristic; benchmark v2 does not add an
autonomous planner or activate SUMMARIZE.

If Ollama/model availability fails, the runner writes `status: blocked`, the
reason, no query records, and no fabricated scores, then exits with code 2. Mid-run
errors are retained and produce `partial_failure`. Generation-error fallbacks are
not successful answers. Retrieval coverage may still be reported if real chunks
were collected before generation failed; it is independent of generation success.

## Metrics and their denominators

- **Passage recall@K:** macro average of the fraction of required evidence groups
  covered by the first K unique chunks. A matching filename is insufficient: a
  chunk must contain an exact gold passage from that source. Alternative passages
  satisfy the same group once. This measures annotated evidence, not every possible
  semantically relevant passage in a document.
- **All requirements@K:** fraction of evidence-bearing questions with every group
  covered. This is stricter than finding any relevant document.
- **MRR@K:** reciprocal rank of the first chunk containing an annotated passage;
  zero when absent within K.
- **Precision@K:** supporting chunks divided by K, including unfilled slots as
  misses. Overlapping unique chunks can each be relevant.
- **nDCG@K:** binary chunk relevance against all indexed chunks containing a gold
  passage, discounted by rank. Alternative supporting chunks are all relevant.
- **All-evidence group recall:** coverage over all collected chunks, shown separately
  from fixed-K metrics. Agentic fixed-K order is evidence collection order; the
  self-correcting result uses its final retrieval. This ordering is explicit and
  must not be mistaken for a common cross-system ranking algorithm.
- **Latency:** wall-clock time per query/system, including generation when enabled;
  initial indexing/model warmup is excluded. Errors remain in latency totals.
- **Exact answer match proxy:** normalized text equality with the authored answer.
  It penalizes valid paraphrases and is not semantic answer correctness.
- **Abstention heuristic:** simple phrase/flag detection compared with the
  unanswerable label. Read the actual answers, especially ambiguous cases.
- **Citation reference validity:** syntactic identity/page/span validation against
  retrieved evidence. No citation produces null, not a perfect score. This does
  not verify claim support or citation completeness.

Every aggregate includes its contributing `n` and each system's error count.
The ten unanswerable questions are excluded from retrieval denominators; the ten
clarification questions do have positive evidence requirements. Missing human
scores remain null with n=0. No confidence intervals, statistical superiority,
token-cost claims or semantic generation scores are invented.

Outputs preserve raw answers, errors, retrieved chunks, per-query metrics,
category breakdowns, corpus/question/index hashes, package versions, model options,
code revision, dirty-state flag and a run identifier. Store outputs securely when
using this infrastructure with a different, private corpus.

## Score answers and citations with human review

The review template is tied to a run ID. Inspect each output alongside the
reference answer and gold evidence in `benchmark.json`. Enter scores in [0,1]:

- `answer_correctness`: 1 for a complete correct answer, 0 for wrong/unsupported;
  use a fractional score only with an explanatory note. Clarification cases need
  ambiguity acknowledged and supported alternatives; unanswerables need refusal
  to invent the missing fact.
- `faithfulness`: whether factual claims follow from the supplied source evidence.
- `citation_support`: whether cited passages actually support the associated claims.
  Use 0 for factual answers requiring citations that omit them; null is appropriate
  for a refusal with no factual claim requiring support. Record a note explaining
  partial scores or disputed labels.

Prefer an independent reviewer and keep any disagreement notes. These are human
judgments, not unquestionable ground truth. Failed outputs are excluded from
review templates and remain visible in the system error count; always report both
review coverage and failures when interpreting mean review scores.

```bash
venv/bin/python -m knowledge.evaluation.benchmark_v2 \
  --review-input reports/v2-dev-comparison.json --reviews reports/v2-dev-review.json \
  --output reports/v2-dev-reviewed.json
```

Unknown/duplicate result keys, mismatched runs and nonfinite/out-of-range review
scores are rejected. Original raw results remain unchanged when a different
output file is used.

See [the measured retrieval report](../../docs/research/benchmark-v2-results.md)
for verified results and blockers. A high score on this finite synthetic corpus
does not demonstrate production robustness.
