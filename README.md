# RAGForge

<p align="center">

**A modular Retrieval-Augmented Generation (RAG) framework for document indexing, hybrid retrieval, reranking, evaluation, and local LLM inference.**

</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![FAISS](https://img.shields.io/badge/Vector%20Store-FAISS-orange)
![SentenceTransformers](https://img.shields.io/badge/Embeddings-SentenceTransformers-red)
![BM25](https://img.shields.io/badge/Retrieval-BM25-yellow)
![Ollama](https://img.shields.io/badge/LLM-Ollama-lightgrey)

</p>

---

## Overview

RAGForge is a modular Retrieval-Augmented Generation (RAG) framework that enables indexing, searching, and querying local document collections using a multi-stage retrieval pipeline.

The framework combines dense vector retrieval, sparse lexical search, hybrid rank fusion, Cross-Encoder reranking, and local LLM inference into an extensible, inspectable architecture.

Rather than treating RAG as an opaque black box, RAGForge emphasizes retrieval quality, evaluation, benchmarking, modularity, and reproducible experimentation. Every stage of the pipeline is implemented as an independent component, making it straightforward to inspect intermediate rankings or swap embedding models, chunking strategies, fusion weights, or language models without modifying the rest of the system.

---

# Why RAGForge?

Standard naive RAG implementations often rely solely on single-pass vector similarity search, which frequently struggles with exact keyword queries, domain terminology, and noisy context.

RAGForge demonstrates a complete, transparent retrieval and evaluation workflow:

- Multi-format document ingestion (PDF, DOCX, TXT, HTML, Markdown)
- Character-offset chunking (Recursive, Sentence-boundary, and Fixed-size)
- Dense semantic retrieval using SentenceTransformers and FAISS FlatIP
- Sparse lexical retrieval using BM25
- Parameterized Reciprocal Rank Fusion (RRF) with configurable dense/sparse weighting
- Second-stage neural reranking with Cross-Encoder models
- Local LLM inference and streaming via Ollama
- Retrieval benchmarking with stage-by-stage latency profiling
- Rigorous retrieval evaluation (Accuracy@K, Precision@K, Recall@K, MRR)
- Incremental document indexing with SHA-1 hash tracking

---

# Core Retrieval Pipeline

The primary, fully-supported feature set in RAGForge is the multi-stage indexing, retrieval, and evaluation engine:

```
[ Documents ]
      │
      ▼
[ Document Loaders ] (PDF, DOCX, TXT, HTML, MD)
      │
      ▼
[ Text Chunkers ] (Recursive / Sentence / Fixed + Char Offsets)
      │
      ▼
[ Dense Embeddings ] (SentenceTransformers)
      │
      ▼
[ Vector Store ] (FAISS FlatIP Index) + [ BM25 Inverted Index ]

─── Query Phase ───

[ User Query ]
      │
      ├───────────────────────────────┐
      ▼                               ▼
[ Dense Retrieval ] (FAISS)       [ Sparse Retrieval ] (BM25)
      │                               │
      └───────────────┬───────────────┘
                      ▼
        [ Reciprocal Rank Fusion ] (Configurable dense_weight)
                      │
                      ▼
         [ Cross-Encoder Reranking ] (ms-marco-MiniLM-L-6-v2)
                      │
                      ▼
         [ Context Construction ]
                      │
                      ▼
         [ Local LLM Inference ] (Ollama / Llama 3.2)
                      │
                      ▼
              [ Generated Answer ]
```

### Pipeline Components

| Stage | Implementation | Purpose |
| :--- | :--- | :--- |
| **Document Loaders** | `PDFLoader`, `DocxLoader`, `TextLoader`, `MarkdownLoader`, `HTMLLoader` | Normalizes heterogeneous local documents into standard `LoadedDocument` objects. |
| **Chunking** | `RecursiveChunker`, `SentenceChunker`, `FixedSizeChunker` | Splits text into manageable segments while preserving source page `(char_start, char_end)` offsets for exact span evaluation. |
| **Dense Retrieval** | `SentenceTransformerEmbedding` + `FAISSStore` | Computes semantic dense embeddings and searches them with inner-product similarity. |
| **Sparse Retrieval** | `BM25Retriever` (`rank_bm25`) | Indexes word frequencies to handle exact keywords, identifiers, and entity matches. |
| **Hybrid Fusion** | `reciprocal_rank_fusion` (`HybridRetriever`) | Blends dense and sparse rankings: $\text{score} = w_d \cdot \frac{1}{k + r_d} + (1 - w_d) \cdot \frac{1}{k + r_s}$ with deterministic tie-breaking. |
| **Reranking** | `CrossEncoderReranker` | Jointly encodes query and passage tokens through a cross-attention transformer to rescore the top candidate pool. |
| **Inference** | `OllamaLLM` | Generates final answers grounded in retrieved passages via local Ollama instances. |

---

# Retrieval Evaluation & Real Benchmarks

RAGForge includes a dedicated evaluation framework to measure retrieval quality independently of language model generation.

### Supported Metrics

- **Accuracy@1**: Fraction of queries where the expected document is the top-ranked result.
- **Accuracy@K**: Fraction of queries where the expected document appears in the top $K$ results.
- **Precision@K**: Proportion of retrieved chunks in the top $K$ that are relevant.
- **Recall@K**: Proportion of relevant chunks retrieved in the top $K$.
- **Mean Reciprocal Rank (MRR)**: Average reciprocal rank $\frac{1}{\text{rank}}$ of the first relevant document.
- **Average Latency**: Mean wall-clock retrieval time per query.

### Real Evaluation Output

Running `knowledge evaluate` against the default evaluation set (`knowledge/evaluation/questions.json`):

```bash
python -m knowledge.cli.main evaluate knowledge/evaluation/questions.json
```

The default evaluation set is a **smoke test**: 2 questions hand-written by the author against a 2-document, 2-chunk corpus (`docs/readme.md` and `docs/test.txt`, one chunk each). It verifies that the pipeline returns any result at all; it has no statistical power for comparing retrieval configurations. Run `knowledge generate-synthetic-data` to build a larger set from your own documents.

Output (reproduced 2026-11-05, `HF_HUB_OFFLINE=1`, warm model, 2-query eval set):

```text
RAGForge Evaluation Report

Queries              : 2
Accuracy@1           : 100.00%
Accuracy@3           : 100.00%
Precision@3          : 0.500
Recall@3             : 1.000
Mean Reciprocal Rank : 1.000
Average Latency      : 262.16 ms  # mean of 473 ms + 51 ms; each query runs in a fresh subprocess (cold)

Per Query Results

Question: What is KnowledgeOS?
Expected: test.txt
Retrieved: test.txt, readme.md
Latency: 473.37 ms
----------------------------------------------------------------------
Question: What does this project demonstrate?
Expected: readme.md
Retrieved: readme.md, test.txt
Latency: 50.95 ms
----------------------------------------------------------------------
```

Custom evaluation sets can be provided as JSON files specifying `question` and `expected_document` (or gold span offsets).

### Single-Process Retrieval Latency

Measured 2026-11-09 with `HF_HUB_OFFLINE=1 python scripts/measure_latency.py`, single Python process, 2-chunk corpus:

| Run | Latency | Notes |
| :--- | :--- | :--- |
| **Cold** (includes model load) | **1563.5 ms** | SentenceTransformer + CrossEncoder loaded from disk |
| **Warm** (models already in memory) | **43.6 ms** | Second query in the same process |

These numbers were produced by [`scripts/measure_latency.py`](scripts/measure_latency.py), which loads all components once and runs two sequential queries in the same process. `knowledge evaluate` spawns a fresh subprocess per query so both runs appear cold at the OS level and the per-query latencies shown above (473 ms / 51 ms) are both cold.

### Stage-by-Stage Latency Profiling

The `benchmark` command profiles execution time across each pipeline stage independently to locate bottlenecks:

```bash
python -m knowledge.cli.main benchmark "What is RAGForge?"
```

Example breakdown (requires Ollama running locally; timings vary with hardware and model size and cannot be reproduced without a running Ollama instance):

```text
KnowledgeOS Benchmark

Query              : What is RAGForge?

Embedding          : depends on cold/warm state (see Single-Process Latency above)
FAISS Search       : sub-millisecond (in-memory flat index)
BM25 Search        : sub-millisecond
Hybrid Fusion      : sub-millisecond
Cross Encoder      : scales linearly with candidate pool size; dominates retrieval latency
LLM Generation     : seconds (Ollama local model; varies with model size and hardware)
```

> **Note**: Specific millisecond values for each stage require Ollama running locally and vary significantly with hardware. Run `knowledge benchmark "..."` on your own machine to get reproducible numbers for your environment.

---

# Advanced Extensions

Beyond the base retrieval pipeline, RAGForge includes modular extensions designed to address specific real-world retrieval challenges:

### 1. FastAPI Service Layer (`knowledge/api/`)
- **What it does**: Exposes the indexing, retrieval, evaluation, and agent pipelines over asynchronous HTTP REST endpoints (`/query`, `/stream/{query}`, `/agent-ask`, `/agent-stream/{query}`, `/ingest`, `/health`, `/stats`).
- **Why it was added**: To decouple the engine from local CLI invocations so it can power web frontends or external services. Server-Sent Events (SSE) provide live token-by-token streaming and per-step agent progress updates without the connection state overhead of WebSockets.

### 2. Self-Correcting Agentic Retrieval (`knowledge/agent/`)
- **What it does**: Executes a LangGraph cyclic state graph: `route_query` $\rightarrow$ `retrieve_context` $\rightarrow$ `grade_sufficiency` $\rightarrow$ (`rewrite_query` | `generate_answer`).
- **Why it was added**: Single-pass retrieval fails when queries are poorly phrased or missing critical keywords. The agent inspects retrieved chunks, verifies whether they contain enough information to answer the question, and rewrites the query if context is insufficient. An enforced iteration cap (`max_retrieval_attempts: 3`) prevents infinite retry loops.

### 3. LLM-as-a-Judge Evaluation (`knowledge/evaluation/judge.py`)
- **What it does**: Prompts an LLM to evaluate generated responses along two structured axes: **Faithfulness** (is the answer strictly grounded in the retrieved text?) and **Relevance** (does the answer directly address the prompt?).
- **Why it was added**: Retrieval metrics (MRR, Precision) only evaluate whether the correct passage was retrieved, not whether the LLM faithfully used that context. The judge module parses structured JSON scores and includes regex fallbacks to handle conversational output from local models.

### 4. Autotune Optimization Engine (`knowledge/autotune/`)
- **What it does**: Performs a two-tier hyperparameter search across index configurations (chunk size, overlap, chunking strategy, embedding model) and query configurations (top-k, dense weight, RRF $k$, reranking enabled, query expansion). It scores candidates using composite objectives (`quality`, `balanced`, `fast`) and validates improvements using paired bootstrap significance testing on a held-out split.
- **Why it was added**: Eliminates manual guesswork when tuning RAG parameters. It uses offline candidate pooling so query configurations are swept without rerunning index builds, caches cross-encoder scores on disk, and never modifies production indexes without explicit confirmation.

### 5. Domain Adaptation Fine-Tuning (`knowledge/training/`)
- **What it does**: Provides contrastive training pipelines for fine-tuning SentenceTransformer bi-encoders (via `MultipleNegativesRankingLoss`) and CrossEncoder rerankers using synthetic question-passage pairs generated from local documents.
- **Why it was added**: Pre-trained public embedding models often underperform on proprietary jargon, internal abbreviations, or specialized formats. Fine-tuning aligns the vector space with domain-specific vocabulary.

### 6. Relevance Feedback Loop (`knowledge/feedback/`)
- **What it does**: Records user relevance judgments (thumbs-up / thumbs-down) in structured logs (`.knowledge/feedback.json`) and exposes a sample-size-gated trigger to fine-tune the cross-encoder reranker.
- **Why it was added**: Allows retrieval systems to learn from actual user interactions over time while requiring a minimum sample threshold (`--min-samples`) to prevent overfitting on sparse feedback.

---

# CLI Reference

Initialize workspace:

```bash
python -m knowledge.cli.main init
```

Index documents:

```bash
python -m knowledge.cli.main add docs/
```

Search documents:

```bash
python -m knowledge.cli.main search "hybrid retrieval"
```

List indexed documents:

```bash
python -m knowledge.cli.main list
```

Show index statistics:

```bash
python -m knowledge.cli.main stats
```

Ask questions via local RAG:

```bash
python -m knowledge.cli.main ask "Explain hybrid retrieval."
```

Run agentic retrieval with query grading:

```bash
python -m knowledge.cli.main agent-ask "Compare dense and sparse retrieval"
```

Launch the FastAPI REST service:

```bash
python -m knowledge.cli.main serve --port 8000
```

Evaluate retrieval quality:

```bash
python -m knowledge.cli.main evaluate knowledge/evaluation/questions.json
# With LLM-as-a-judge faithfulness and relevance:
python -m knowledge.cli.main evaluate knowledge/evaluation/questions.json --judge
```

Run latency benchmark:

```bash
python -m knowledge.cli.main benchmark "What is RAGForge?"
```

Generate synthetic training data & fine-tune bi-encoder:

```bash
python -m knowledge.cli.main generate-synthetic-data --num-pairs 20
python -m knowledge.cli.main train-bi-encoder --epochs 1
```

Record feedback and tune reranker:

```bash
python -m knowledge.cli.main feedback "What is BM25?" --text "BM25 is a keyword ranking..." --rating 1
python -m knowledge.cli.main list-feedback
python -m knowledge.cli.main tune-reranker --min-samples 5
```

---

# Installation & Local Setup

### Prerequisites

- Python 3.11 or later
- [Ollama](https://ollama.com) installed and running locally
- Git

Clone the repository and install dependencies:

```bash
git clone https://github.com/PALAK7890/RAGForge.git
cd RAGForge

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -e .
```

Pull the local language model:

```bash
ollama pull llama3.2
ollama serve
```

---

# Technology Stack

| Category | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.11+ | Core implementation |
| **CLI Framework** | Typer & Rich | Command-line interface and formatting |
| **Dense Embeddings** | SentenceTransformers | Vector representations (`all-MiniLM-L6-v2`) |
| **Vector Index** | FAISS | Exact inner product dense search |
| **Sparse Retrieval** | rank-bm25 | BM25Okapi lexical retrieval |
| **Reranking** | CrossEncoder | Second-stage neural attention rescoring |
| **Local LLM** | Ollama | Local model inference and answer generation |
| **API Layer** | FastAPI & Uvicorn | Asynchronous REST endpoints and SSE streaming |
| **Agent Orchestration** | LangGraph | State graph for cyclic retrieval and grading |
| **Evaluation** | NumPy & Scikit-Learn | Retrieval metrics, bootstrap CIs, and LLM judge |

---

# Experimental / In Progress

The following modules were explored during development and are preserved on the **`experimental/extensions`** branch rather than active in `main`:

- **Learning-to-Rank (LTR) Fusion**: A gradient-boosted decision tree (`HistGradientBoostingClassifier`) that combines dense similarity scores, BM25 ranks, and document position features, archived as an experimental alternative to CrossEncoder reranking.
- **Topic Clustering**: Unsupervised K-Means clustering over FAISS index vectors with TF-IDF keyword extraction to discover thematic topics across document chunks, archived as an exploratory visualization feature outside the core query loop.

Both modules remain accessible on the `experimental/extensions` branch for ongoing experimentation.

---

# Future Improvements

Concrete items not yet implemented:

- **`knowledge remove` and `knowledge rebuild` commands**: There is currently no CLI command to remove a specific document from the index or to do a full clean rebuild. Removing a document requires manually editing `chunks.json` and rebuilding the FAISS index.
- **Lazy-load the embedding model**: `SentenceTransformerEmbedding` is instantiated at module import time in `knowledge/cli/main.py`, which means every `knowledge --help` invocation loads the full model (~100 MB). Moving model construction into each command's body would make help and lightweight commands instant.
- **Harder hand-written eval set with unanswerable questions**: The current 2-question eval set is too easy (100% scores on every configuration). A larger set that includes ambiguous, multi-hop, and deliberately unanswerable questions would make retrieval metric comparisons meaningful.
- **`feedback tune --watch` mode**: The feedback reranker tuner is currently triggered manually. An auto-trigger once the feedback store passes a configurable sample threshold would close the active-learning loop without requiring a manual CLI call.
- **`knowledge autotune` progress persistence (`--resume`)**: The autotune runner caches individual index builds but does not checkpoint trial results mid-sweep. If a long run is interrupted, completed trials are lost and the sweep restarts from scratch.
- **Metadata-aware filtering**: Retrieval currently searches all indexed chunks regardless of source document, date, or file type. A filter parameter on `search` and `ask` would allow scoping queries to a document subset.

---

# License

This project is licensed under the MIT License.

---

# Author

**Palak**

B.Tech Data Science Student

- GitHub: [github.com/PALAK7890](https://github.com/PALAK7890)
- LinkedIn: [linkedin.com/in/palak-dahiya-430b39259](https://www.linkedin.com/in/palak-dahiya-430b39259)
- Email: palakdahiya78@gmail.com
