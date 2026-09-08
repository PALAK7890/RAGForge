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

RAGForge is a modular Retrieval-Augmented Generation (RAG) framework that enables indexing, searching, and querying local document collections using a production-inspired retrieval pipeline.

The framework combines dense vector retrieval, sparse lexical search, hybrid rank fusion, Cross-Encoder reranking, and local LLM inference into a single extensible architecture.

Unlike many educational RAG implementations that focus only on question answering, RAGForge emphasizes retrieval quality, benchmarking, evaluation, modularity, and experimentation. Each stage of the retrieval pipeline is implemented as an independent component, making it straightforward to replace embedding models, retrieval algorithms, rerankers, or language models without affecting the rest of the system.

---

# Why RAGForge?

Modern Retrieval-Augmented Generation systems are more than vector search. High-quality retrieval requires multiple ranking stages, evaluation, benchmarking, and modular system design.

RAGForge demonstrates an end-to-end retrieval pipeline including:

- Multi-format document ingestion
- Recursive document chunking
- Dense semantic retrieval
- Sparse lexical retrieval
- Hybrid rank fusion
- Cross-Encoder reranking
- Local LLM inference with Ollama
- Retrieval benchmarking
- Retrieval evaluation
- Incremental document indexing

The project is designed for developers, students, and researchers interested in understanding how production-inspired retrieval systems are engineered and evaluated.

---

# Key Features

| Component | Description |
|-----------|-------------|
| Multi-format Loaders | PDF, TXT, Markdown, HTML and DOCX support |
| Recursive Chunking | Configurable chunk size and overlap |
| Dense Retrieval | SentenceTransformers embeddings with FAISS |
| Sparse Retrieval | BM25 lexical ranking |
| Hybrid Retrieval | Reciprocal Rank Fusion (RRF) & Trainable LTR GBDT |
| Query Expansion | Improves retrieval recall |
| Cross-Encoder Reranking | Second-stage neural reranking |
| Local LLM | Ollama integration |
| Incremental Indexing | Detects new, modified and unchanged documents |
| Self-Correcting Agent | LangGraph query routing, sufficiency grading & query rewrite |
| REST API | FastAPI async endpoints with streaming answer responses |
| Domain Adaptation | SentenceTransformer bi-encoder & CrossEncoder fine-tuning |
| Synthetic Data Gen | Automatic query-chunk pair generation using local LLM |
| LLM-as-a-Judge | Faithfulness and Answer Relevance evaluation scoring |
| Topic Clustering | Unsupervised KMeans & TF-IDF keyword extraction |
| Active Learning | Feedback collection and continuous reranker tuning |
| Benchmarking | End-to-end latency analysis per pipeline stage |

---

# System Architecture

The complete architecture diagrams are available in:

```

docs/architecture.md

```

The repository contains dedicated diagrams for:

- Indexing pipeline
- Retrieval pipeline
- End-to-end RAG workflow

These diagrams illustrate how documents move through indexing, retrieval, reranking, and answer generation.

---

# Project Goals

RAGForge is designed around five engineering principles.

- Modular architecture
- Local-first deployment
- Reproducible experimentation
- Retrieval quality evaluation
- Component-level benchmarking

Rather than treating RAG as a single black-box model, the framework exposes every stage of the retrieval pipeline for inspection, replacement, and experimentation.

---

# Core Pipeline

```
Documents
      │
      ▼
Document Loaders
      │
      ▼
Chunking
      │
      ▼
Embeddings
      │
      ▼
FAISS Index

User Query
      │
      ▼
Query Expansion
      │
      ▼
Semantic Retrieval
      │
Sparse Retrieval
      │
      ▼
Hybrid Fusion
      │
      ▼
Cross Encoder
      │
      ▼
Context Construction
      │
      ▼
Ollama
      │
      ▼
Generated Answer
```

---

# Installation

## Prerequisites

- Python 3.11 or later
- Ollama
- Git

Clone the repository.

```bash
git clone https://github.com/PALAK7890/RAGForge.git
cd RAGForge
```

Create and activate a virtual environment.

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

Install the project.

```bash
pip install -e .
```

---

# Installing Ollama

Install Ollama from the official website.

https://ollama.com

Pull a supported model.

```bash
ollama pull llama3.2
```

Start the Ollama server.

```bash
ollama serve
```

---

# Repository Structure

```
RAGForge/
│
├── knowledge/
│   ├── agent/          # LangGraph agentic retrieval graph
│   ├── api/            # FastAPI REST endpoints + RAGService layer
│   ├── chunkers/
│   ├── cli/
│   ├── clustering/     # KMeans + TF-IDF topic clustering
│   ├── config/
│   ├── embeddings/
│   ├── evaluation/     # Metrics + LLM-as-a-Judge
│   ├── feedback/       # Feedback store + reranker tuner
│   ├── indexing/
│   ├── llms/
│   ├── loaders/
│   ├── rerankers/
│   ├── retrievers/     # Hybrid retrieval + LTR fusion
│   ├── training/       # Bi-encoder + reranker fine-tuning
│   └── vectorstores/
│
├── docs/
│   └── architecture.md
│
├── tests/
├── README.md
├── pyproject.toml
└── requirements.txt
```

---

# Advanced Usage

RAGForge ships with 7 extensions beyond the core retrieval pipeline. Full architecture diagrams for each are in [`docs/architecture.md`](docs/architecture.md).

### REST API

Launch the FastAPI server with `knowledge serve`. Endpoints cover indexing, hybrid search, streaming Q&A, agentic retrieval, evaluation, feedback, and statistics. The `/agent-stream/{query}` endpoint streams per-step SSE progress events during multi-attempt retrieval cycles.

### Self-Correcting Agent

`knowledge agent-ask` runs a LangGraph state machine that classifies query intent, retrieves context, grades sufficiency via an LLM, and rewrites the query if context is lacking. A hard cap (`max_retrieval_attempts` in `config.yaml`, default 3) prevents infinite loops. The `/agent-stream/{query}` API endpoint exposes the same loop with SSE progress events.

### Model Fine-Tuning

`knowledge finetune` commands cover synthetic training data generation (LLM-prompted query–document pairs), bi-encoder fine-tuning with `MultipleNegativesRankingLoss`, and CrossEncoder reranker fine-tuning on labeled triplets.

### Learning-to-Rank Fusion

`knowledge retrieve-ltr` replaces static RRF with a trainable GBDT ranker (`HistGradientBoostingClassifier`) that combines dense + sparse retrieval signals. Falls back to RRF automatically when no trained model is present or inference fails.

### LLM-as-a-Judge Evaluation

`knowledge evaluate --judge` adds faithfulness and answer-relevance scoring to the standard evaluation pass. The judge LLM returns structured JSON scores that are aggregated across the question set.

### Topic Clustering

`knowledge topics` runs unsupervised KMeans over the FAISS index vectors (or TF-IDF features as fallback) to group chunks by theme and extract keyword tags. Results are saved to `.knowledge/index/topics.json` and tagged into `chunks.json`.

### Feedback Loop

`knowledge feedback up/down` logs relevance judgments to `.knowledge/feedback.json`. `knowledge feedback tune` converts accumulated judgments into training triplets and fine-tunes the CrossEncoder. This is a **manual trigger** — it does not run automatically after each judgment.

---

# CLI Commands

Initialize a new workspace.

```bash
python -m knowledge.cli.main init
```

Index documents.

```bash
python -m knowledge.cli.main add docs/
```

Search indexed documents.

```bash
python -m knowledge.cli.main search "hybrid retrieval"
```

Ask questions using local RAG.

```bash
python -m knowledge.cli.main ask "What is Retrieval-Augmented Generation?"
```

List indexed documents.

```bash
python -m knowledge.cli.main list
```

Show workspace statistics.

```bash
python -m knowledge.cli.main stats
```

Benchmark the retrieval pipeline.

```bash
python -m knowledge.cli.main benchmark "What is RAGForge?"
```

Evaluate retrieval quality (with optional LLM-as-a-judge faithfulness and relevance).

```bash
python -m knowledge.cli.main evaluate knowledge/evaluation/question.json --judge
```

Run self-correcting agentic Q&A (LangGraph loop).

```bash
python -m knowledge.cli.main agent-ask "Compare dense and sparse retrieval"
```

Launch the FastAPI REST server.

```bash
python -m knowledge.cli.main serve --port 8000
```

Generate synthetic training data & fine-tune the bi-encoder.

```bash
python -m knowledge.cli.main generate-synthetic-data --num-pairs 20
python -m knowledge.cli.main train-bi-encoder --epochs 1
```

Train the Learning-to-Rank (LTR) fusion model.

```bash
python -m knowledge.cli.main train-ltr
```

Discover unsupervised topic clusters across chunks.

```bash
python -m knowledge.cli.main cluster --n-clusters 5
python -m knowledge.cli.main topics
```

Record feedback judgments and fine-tune the reranker.

```bash
python -m knowledge.cli.main feedback "What is BM25?" --text "BM25 is..." --rating 1
python -m knowledge.cli.main list-feedback
python -m knowledge.cli.main tune-reranker
```

---

# Example Workflow

### 1. Index documents

```bash
python -m knowledge.cli.main add docs/
```

Output

```
Found 12 document(s)

✓ handbook.pdf
✓ report.docx
✓ notes.txt

Index saved successfully.

Total Chunks : 148
Embedding Dimension : 384
```

---

### 2. Search

```bash
python -m knowledge.cli.main search "hybrid retrieval"
```

Output

```
Expanded Query:
hybrid retrieval semantic search dense retrieval BM25

1. handbook.pdf

Cross Score : 8.52

Hybrid Score : 0.93

Hybrid retrieval combines dense vector search with sparse lexical retrieval.

------------------------------------------------------------
```

---

### 3. Ask Questions

```bash
python -m knowledge.cli.main ask "Explain hybrid retrieval."
```

Output

```
Answer

Hybrid retrieval combines semantic vector search and lexical keyword
search to improve retrieval accuracy. The retrieved passages are
reranked using a Cross-Encoder before being provided to the language
model.

Sources

- handbook.pdf
- retrieval_notes.md
```

---

# Retrieval Pipeline

RAGForge follows a multi-stage retrieval architecture.

```
User Query
      │
      ▼
Query Expansion
      │
      ▼
Sentence Embedding
      │
      ▼
FAISS Retrieval
      │
BM25 Retrieval
      │
      ▼
Hybrid Rank Fusion
      │
      ▼
Cross-Encoder Reranking
      │
      ▼
Context Construction
      │
      ▼
Local LLM
      │
      ▼
Answer Generation
```

Each component can be replaced independently, making experimentation with retrieval techniques straightforward.

---

# Retrieval Components

| Stage | Implementation |
|--------|----------------|
| Document Loading | PDF, DOCX, TXT, HTML, Markdown |
| Chunking | Recursive Character Splitter |
| Embeddings | SentenceTransformers |
| Vector Store | FAISS |
| Sparse Retrieval | BM25 |
| Fusion | Reciprocal Rank Fusion (RRF) |
| Reranking | CrossEncoder |
| LLM | Ollama |
| Evaluation | Accuracy@K, Precision@K, Recall@K, MRR |
| Benchmarking | Latency Profiling |

---

# Benchmarking

RAGForge includes a built-in benchmarking framework that measures the latency of each stage in the retrieval pipeline independently.

Run the benchmark using:

```bash
python -m knowledge.cli.main benchmark "What is RAGForge?"
```

Example output:

```text
KnowledgeOS Benchmark

Query              : What is RAGForge?

Embedding          : 541.23 ms
FAISS Search       : 59.54 ms
BM25 Search        : 1.32 ms
Hybrid Fusion      : 0.06 ms
Cross Encoder      : 491.40 ms
LLM Generation     : 8755.71 ms

---------------------------------------------

Total Pipeline     : 9855.17 ms

Retrieved Chunks   : 2
Embedding Size     : 384
```

The benchmark is intended to identify latency bottlenecks and compare retrieval strategies independently from language model inference.

---

# Evaluation Framework

RAGForge includes a modular retrieval evaluation framework for measuring retrieval quality.

Supported metrics include:

- Accuracy@1
- Accuracy@3
- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)
- Average Retrieval Latency

Run the evaluation.

```bash
python -m knowledge.cli.main evaluate evaluation/questions.json
```

Example output:

```text
KnowledgeOS Evaluation

Queries            : 50

Accuracy@1         : 92.00%
Accuracy@3         : 98.00%

Precision@3        : 0.91
Recall@3           : 0.98

Mean Reciprocal Rank : 0.95

Average Latency    : 81.34 ms
```

The evaluation framework is designed to benchmark retrieval performance independently of the language model.

---

# Technology Stack

| Category | Technology |
|------------|------------|
| Language | Python |
| CLI | Typer |
| Console | Rich |
| Embeddings | SentenceTransformers |
| Vector Search | FAISS |
| Sparse Retrieval | BM25 |
| Rank Fusion | Reciprocal Rank Fusion & Trainable GBDT |
| Reranker | CrossEncoder |
| Local LLM | Ollama |
| REST API | FastAPI + Uvicorn |
| Agent Orchestration | LangGraph |
| LTR Ranker | scikit-learn HistGradientBoostingClassifier |
| Topic Clustering | KMeans + TF-IDF |
| Active Learning | Feedback logging + CrossEncoder fine-tuning |
| Configuration | YAML |
| Evaluation | Custom Metrics + LLM-as-a-Judge |

---

# Design Principles

The framework is built around several core engineering principles.

- Modular architecture
- Production-inspired retrieval pipeline
- Local-first inference
- Reproducible experimentation
- Independent benchmarking
- Retrieval quality evaluation
- Easily replaceable components

Each subsystem is isolated behind a simple interface, allowing retrieval algorithms, embedding models, rerankers, and language models to be swapped without affecting the rest of the pipeline.

---

# Roadmap

## Completed

- Multi-format document loaders (PDF, DOCX, TXT, HTML)
- Recursive chunking with configurable overlap
- SentenceTransformer embeddings
- FAISS vector indexing
- BM25 lexical retrieval
- Hybrid Reciprocal Rank Fusion (RRF)
- Query expansion
- Cross-Encoder reranking
- Ollama LLM integration
- Incremental indexing
- Benchmarking framework
- Retrieval evaluation metrics (Accuracy@K, Precision, Recall, MRR)
- FastAPI REST API with streaming `/ask` and SSE `/agent-stream`
- LangGraph self-correcting agentic retrieval with configurable retry cap
- SentenceTransformer bi-encoder fine-tuning
- CrossEncoder reranker fine-tuning
- Synthetic training pair generation
- Learning-to-Rank fusion (GBDT + RRF fallback)
- LLM-as-a-Judge evaluation (faithfulness + relevance)
- Unsupervised topic clustering (KMeans + TF-IDF fallback)
- Feedback logging and manual reranker retraining

## Planned

- `feedback tune --watch` — automatic retraining when feedback store reaches threshold
- Scheduled fine-tuning via cron or background thread
- Docker support
- Metadata-aware filtering (date, source, topic)
- Additional embedding backends (OpenAI, Cohere)
- Multi-vector retrieval (ColBERT-style)
- Automated benchmark report generation

---

# Contributing

Contributions are welcome.

If you would like to improve RAGForge, please:

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Open a Pull Request.

For significant changes, please open an issue first to discuss the proposed modification.

---

# License

This project is licensed under the MIT License.

---

# Acknowledgements

RAGForge builds upon several excellent open-source projects.

- FAISS
- SentenceTransformers
- rank-bm25
- Ollama
- Typer
- Rich

Their contributions make modern Retrieval-Augmented Generation systems accessible to the community.

---

# Author

**Palak**

B.Tech Data Science Student

Interested in:

- Machine Learning
- Retrieval-Augmented Generation
- Information Retrieval
- Large Language Models
- AI Systems Engineering

GitHub:
https://github.com/PALAK7890

---

If you found this project useful, consider giving the repository a ⭐.