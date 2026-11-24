# RAGForge

RAGForge is a Python framework for indexing local documents, retrieving evidence,
and generating answers with Ollama. It combines dense retrieval, BM25, reciprocal
rank fusion, and cross-encoder reranking. A CLI and FastAPI service expose the
retrieval pipeline and three question-answering workflows.

The repository includes versioned evaluation corpora, raw model outputs, and
reproduction commands. AgenticRAG supports the existing heuristic controller and
an experimental model-directed planner. Recorded retrieval and citation scores do not establish semantic
answer accuracy or production readiness.

## Architecture

```text
Documents -> Loaders -> Chunks with source metadata -> Embeddings -> FAISS index

Question -> Dense search + BM25 -> Rank fusion -> Cross-encoder reranking
         -> Retrieved evidence -> Ollama -> Answer
```

Supported inputs are PDF, DOCX, TXT, Markdown, and HTML. PDF loading extracts
existing text; scanned documents require an external OCR step. Recursive,
sentence-boundary, and fixed-size chunkers preserve source location metadata
where available.

Default retrieval components:

| Component | Implementation |
|---|---|
| Embeddings | SentenceTransformers `all-MiniLM-L6-v2` |
| Dense index | FAISS exact inner-product search |
| Lexical retrieval | `rank_bm25` BM25Okapi |
| Fusion | Weighted reciprocal rank fusion |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Generation | Ollama `llama3.2` |

The three answer workflows share these retrieval components:

| Workflow | CLI command | Behavior |
|---|---|---|
| Standard RAG | `ask` | Retrieves context once and generates an answer. |
| Self-correcting RAG | `agent-ask` | Uses LLM context grading and bounded query rewriting through LangGraph. |
| AgenticRAG | `agentic-ask` | Accumulates evidence with SEARCH, FIND, and OPEN; choose heuristic or model-directed tool selection. |

AgenticRAG exposes execution traces, evidence provenance, and citation-reference
checks. Missing or invalid citations trigger at most one generation correction.
The default tool budget is five steps, configurable from one to fifteen.
SUMMARIZE is implemented as a callable tool but is not scheduled by the controller.
The [research mapping](docs/research/agenticrag-paper-analysis.md) describes the
local adaptation and its differences from the referenced AgenticRAG paper.

## Local setup

Package metadata declares Python 3.9 or newer. Recorded local checks used
Python 3.13.7; the full declared version range has not been tested. Git is needed
to clone the repository. Ollama is required for answer generation, but retrieval
can run without it.

```bash
git clone https://github.com/PALAK7890/RAGForge.git
cd RAGForge
python -m venv venv
source venv/bin/activate
python -m pip install -e .
```

On Windows PowerShell, activate the environment with `venv\Scripts\Activate.ps1`.
The installed command is `knowledge`; `python -m knowledge.cli.main` is equivalent.
Embedding and reranking models download on first use. The CLI currently constructs
these models at import time, including for `init` and `--help`.

### Index and search documents

Initialize the workspace and run the API from the repository root:

```bash
knowledge init
knowledge serve --port 8000
```

In another terminal, index the supplied project-documentation corpus:

```bash
curl -X POST http://127.0.0.1:8000/documents \
  -H 'Content-Type: application/json' \
  -d '{"path":"examples/project_docs/corpus"}'

curl --get http://127.0.0.1:8000/search \
  --data-urlencode 'q=How many questions were used in the full generation run?'
```

`path` identifies a file or directory on the server's filesystem, relative to the
server's working directory when not absolute. This endpoint does not upload files.
The service stores its index under `.knowledge/` and preserves other indexed
documents when adding or updating sources. Keep the server and CLI in the same
working directory to use the same default index.

The legacy CLI `add` command remains available, but adding new or changed files
can replace the index with only those files. Use `POST /documents` for incremental
indexing. For evaluation, use the dedicated runners below; they isolate each corpus
and avoid indexing reference answers.

### Generate answers locally

Start Ollama locally, using `ollama serve` in a separate terminal if it is not
already running. Download the model, then run queries from the activated project
environment:

```bash
ollama pull llama3.2

knowledge ask "How many questions were used in the full generation run?"
knowledge agent-ask "Why were the original benchmark claims withdrawn?"
knowledge agentic-ask "What does citation validity measure?" --show-trace
knowledge agentic-ask "What does citation validity measure?" --max-steps 4 --output-json
```

The sample corpus contains documentation snapshots. Answers should reflect those
snapshots; they do not describe the current runtime environment automatically.

### Model-directed investigation

Enable the experimental planner explicitly:

```bash
knowledge agentic-ask "Why were the original benchmark claims withdrawn?" \
  --planner model --max-steps 5 --output-json
```

The model reviews the question, collected evidence, recent tool results, and
remaining tool budget before choosing SEARCH, FIND, OPEN, ANSWER, ABSTAIN, or
CLARIFY. It can search a second document for a missing fact, finish without forced
navigation, or ask a clarification question. JSON output includes `decision_history`
with each action, its short rationale, timing, and any planning error.

Tool arguments and document identities are validated before execution. Repeated
actions stop with an explicit error. A run permits at most `max_steps` tool calls
and `max_steps + 1` planner calls; each planner request has a 120-second timeout.
Budget exhaustion alone does not establish sufficient evidence. Planning failures
remain visible and do not silently switch to heuristics.

The default remains `--planner heuristic`, preserving the workflow used by the
saved benchmark. Model planning adds inference calls and can choose poor actions;
it does not guarantee correct evidence assessment or answers. It does not add
full-source navigation or automatic summarization. The API accepts `"planner":
"model"` on `/v2/agentic-ask`, or `planner=model` on the v2 stream endpoint.

The [live investigation checks](docs/research/model-planner-smoke.md) demonstrate
a real second-document lookup and record the remaining failures. In the final
three-question comparison, model planning hit two repeated-action errors while
the heuristic baseline had none. This mode is not yet a demonstrated quality upgrade.

## CLI and API reference

| CLI command | Purpose |
|---|---|
| `knowledge init` | Create the default workspace directories and initial configuration file. |
| `knowledge search "question"` | Retrieve and rerank passages without generation. |
| `knowledge list` / `knowledge stats` | Inspect indexed documents and index statistics. |
| `knowledge ask "question"` | Generate an answer from retrieved context. |
| `knowledge agent-ask "question" --max-retries 2` | Run the self-correcting workflow. |
| `knowledge agentic-ask "question" --show-trace` | Run the heuristic tool controller with a trace. |
| `knowledge benchmark "question"` | Profile retrieval and generation stages; requires Ollama. |
| `knowledge evaluate knowledge/evaluation/questions.json` | Run the original two-question retrieval smoke test against a matching index. |
| `knowledge serve --port 8000` | Start the HTTP service on `127.0.0.1`. |

Synthetic training data, bi-encoder training, reranker tuning, relevance feedback,
and autotuning are also implemented. See [training](knowledge/training/),
[feedback](knowledge/feedback/), and [autotuning](knowledge/autotune/).
Use `knowledge COMMAND --help` for supported CLI options.

The running service provides request schemas and interactive documentation at
`http://127.0.0.1:8000/docs`. Its main endpoints are:

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Service health. |
| POST / GET | `/documents` | Index a server-local path or list documents. |
| GET | `/search` | Retrieve passages using the `q` query parameter and optional `top_k`. |
| POST | `/ask` | Standard RAG; `stream: true` returns a plain-text token stream. |
| POST | `/agent-ask` | Self-correcting RAG. |
| GET | `/agent-stream/{query}` | Self-correcting progress as newline-delimited JSON. |
| POST | `/v2/agentic-ask` | AgenticRAG answer, evidence, citation report, and execution metadata. |
| GET | `/v2/agentic-stream/{query}` | AgenticRAG progress using Server-Sent Events. |
| GET | `/stats` | Workspace and index statistics. |
| POST | `/benchmark` | Stage latency profiling. |
| POST | `/evaluate` | Retrieval evaluation against a server-local dataset. |
| POST | `/feedback` | Record a relevance judgment. |

## Evaluation datasets

| Dataset | Questions | Sources | Intended use |
|---|---:|---:|---|
| [Original smoke test](knowledge/evaluation/questions.json) | 2 | 2 documents | Basic retrieval checks; insufficient for quality comparisons. |
| [Fictional practice examples](examples/evaluation/README.md) | 40 | 8 documents | Policy versions, identifiers, calculations, multiple-source questions, and unsupported requests. |
| [Project-documentation examples](examples/project_docs/README.md) | 12 | 4 snapshots | RAGForge-specific questions, conflicting historical reports, ambiguity, and missing information. |
| [Benchmark v2](examples/benchmark_v2/README.md) | 120 | 24 fictional documents | Three-system evaluation with exact supporting passages and versioned source hashes. |

Benchmark v2 contains 62 development and 58 evaluation questions across eight
categories: factual, exact keyword, multi-hop, comparative, numerical, document
navigation, ambiguous, and unanswerable. Ten questions are unanswerable; the
remaining 110, including clarification cases, have annotated evidence requirements.
The public, authored `held_out` split is not a blind external evaluation.

The original 120-question repository dataset is separate from v2. Its previously
published comparative and ablation scores were withdrawn because saved runs did
not support them and its matching corpus was incomplete. V2 results do not validate
those claims. See the [evaluation audit](docs/research/experimental-results.md).

### Recorded benchmark v2 results

The [citation-correction run](docs/research/benchmark-v2-citation-fix.md) preserves
360 real outputs: 120 questions through each of three systems, with zero execution
errors. The saved experiment used code revision `7d4523c` and Ollama `llama3.2`
with temperature 0, seed 17, context length 8192, and maximum output 1024 tokens.
The report records the model fingerprint and interrupted-run recovery details.

| Metric | Standard RAG | Self-correcting RAG | AgenticRAG |
|---|---:|---:|---:|
| Passage recall@5, 110 evidence-bearing questions | 98.18% | 96.89% | 98.18% |
| All evidence requirements@5, 110 questions | 95.45% | 92.73% | 95.45% |
| Answers containing citations, out of 120 | 41 | 42 | 83 |
| Answers containing invalid references | 1 | 1 | 0 |
| Evidence-bearing answers without citations, out of 110 | 70 | 68 | 27 |
| Mean reference validity among cited answers | 97.56% (n=41) | 97.62% (n=42) | 100.00% (n=83) |
| Execution errors, out of 120 | 0 | 0 | 0 |

Passage recall measures the fraction of annotated evidence requirements covered by
the first five unique chunks, averaged across questions. All requirements@5 counts
questions for which every required evidence group is covered. A filename match
alone does not satisfy either metric. Each workflow has its own evidence ordering
and retrieval budget, so this is not an equal-compute comparison.

Reference validity checks source identity and available page/span metadata. It
does not establish claim support, complete attribution, or answer correctness.
Full-run semantic correctness, faithfulness, and citation-support scores remain
unmeasured; independent human review is pending. Exact timing data is unavailable
for 169 recovered cases, so the combined run does not support a full-run latency
comparison. See the [sample answer review](docs/research/citation-sample-review.md)
and [remaining failures](docs/research/citation-remaining-failures.md).

The 12 project-documentation examples also completed retrieval and AgenticRAG
generation without execution errors. Exact-passage coverage averaged 85%, with
all required passages found for 8 of 10 evidence-bearing questions. The
[answer review](docs/research/project-docs-answer-review.md) records false refusals,
numerical mistakes, and unresolved ambiguity despite successful execution.

### Reproduce the checks

Run from the repository root with dependencies installed. The following commands
index only each dataset's source corpus into a separate workspace and save raw
results. Retrieval-only runs do not need Ollama.

```bash
OMP_NUM_THREADS=1 python -m knowledge.evaluation.examples \
  --index --output reports/examples-retrieval.json

OMP_NUM_THREADS=1 python -m knowledge.evaluation.examples \
  --examples examples/project_docs --workspace .knowledge/project-docs \
  --index --output reports/project-docs-retrieval.json

OMP_NUM_THREADS=1 python -m knowledge.evaluation.benchmark_v2 \
  --index --split all --output reports/v2-retrieval.json
```

For the full three-system comparison, start Ollama and install `llama3.2` first:

```bash
OMP_NUM_THREADS=1 python -m knowledge.evaluation.benchmark_v2 \
  --index --split all --systems standard self_correcting agentic \
  --workspace .knowledge/benchmark-v2 \
  --output reports/v2-comparison.json --review-template reports/v2-review.json
```

Use `--split dev --max-samples 3` for a limited v2 smoke run. Practice runners
support `--mode standard` and `--mode agentic` for generation. Add `--planner model`
to agentic practice runs. In benchmark v2, select the separate `agentic_model`
system to evaluate model planning alongside the existing `agentic` baseline:

```bash
OMP_NUM_THREADS=1 python -m knowledge.evaluation.benchmark_v2 \
  --index --split dev --max-samples 3 --systems agentic agentic_model \
  --output reports/v2-planner-smoke.json
```

The historical 360-case results above describe the heuristic system only.
Never index question
files, reference answers, or review artifacts as source evidence.

`OMP_NUM_THREADS=1` addresses the native-library threading failure observed during
the local audit. Add `HF_HUB_OFFLINE=1` only when the embedding and reranking models
are already cached. Identical settings do not guarantee identical model outputs
across hardware and runtime versions. Detailed metric definitions, provenance,
and the human-review workflow are in the [v2 guide](examples/benchmark_v2/README.md).

## Development and limitations

Install test dependencies and run the suite from the repository root:

```bash
python -m pip install -r requirements.txt
OMP_NUM_THREADS=1 python -m pytest tests/ -q
```

Tests cover loaders, chunking, service indexing, retrieval evaluation, agent tools,
citation validation, API behavior, and benchmark integrity. Unit and regression
tests complement the recorded model runs; they do not measure generated-answer
quality.

Current implementation limits:

- Heuristic routing remains the default. The opt-in model planner requires its
  own quality evaluation. Citation correction is bounded and can still leave answers uncited.
- False refusals, incomplete attribution, and incorrect reasoning remain in saved
  outputs even when the required evidence was retrieved.
- The legacy CLI indexing path differs from the service path. Service updates
  re-embed retained chunks and are not transactional under concurrent indexing.
- Configuration is not uniformly applied across entry points. Editing
  `.knowledge/config.yaml` does not control every runtime component.
- Evaluation corpora and labels are authored fixtures. Public split labels and
  successful execution do not establish generalization or independent validation.

Implementation details and historical findings are documented in the
[architecture](docs/architecture.md), [design decisions](docs/design_decisions.md),
and [audit report](docs/research/audit-report.md). Historical reports retain the
code revisions and limitations of their recorded runs.

## License and author

Package metadata declares the MIT license. A standalone license file is not
currently included in the repository.

Maintained by Palak. [GitHub](https://github.com/PALAK7890) |
[LinkedIn](https://www.linkedin.com/in/palak-dahiya-430b39259) |
[Email](mailto:palakdahiya78@gmail.com).
