# RAGForge System Architecture

RAGForge is a modular, production-inspired Retrieval-Augmented Generation (RAG) framework with hybrid retrieval, neural reranking, agentic self-correction, domain fine-tuning, and active learning.

---

## 1. End-to-End Hybrid RAG Pipeline

```mermaid
flowchart TD
    subgraph Ingestion["Document Ingestion & Indexing"]
        A[Raw Documents: PDF, DOCX, TXT, HTML] --> R[Router & Loaders]
        R --> C[Recursive Chunker]
        C --> EMB[SentenceTransformer Embeddings]
        EMB --> FAISS[(FAISS Dense Index)]
        C --> BM25[(BM25 Sparse Index)]
        C --> CLUST[Topic Clustering: KMeans & TF-IDF]
    end

    subgraph QueryExecution["Hybrid Search & Ranking"]
        Q[User Query] --> QE[Query Expansion]
        QE --> D_SEARCH[Dense Semantic Search]
        QE --> S_SEARCH[BM25 Lexical Search]
        FAISS -.-> D_SEARCH
        BM25 -.-> S_SEARCH
        D_SEARCH --> FUSION{Fusion Layer: LTR GBDT or RRF}
        S_SEARCH --> FUSION
        FUSION --> CE[Cross-Encoder Neural Reranker]
        CE --> TOPK[Top-K Grounded Context]
    end

    subgraph Generation["Inference & Attribution"]
        TOPK --> LLM[Local Ollama LLM]
        LLM --> ANS[Generated Answer + Source Citations]
    end
```

---

## 2. Agentic Self-Correcting Graph Layer (LangGraph)

The agent layer introduces query intent classification, retrieval sufficiency grading, and automated query rewriting. A configurable hard cap (`max_retrieval_attempts` in `config.yaml`, default 3) ensures the rewrite loop always terminates.

```mermaid
flowchart TD
    START([User Query]) --> ROUTE["Route Query Node: Classify Intent and Top-K"]
    ROUTE --> RETRIEVE[Retrieve Node: Hybrid Search and Rerank]
    RETRIEVE --> GRADE[Grade Node: LLM Evaluates Context Sufficiency]

    GRADE --> CHECK{Context Sufficient?}
    CHECK -- Yes --> GEN[Generate Answer Node]
    CHECK -- "No and retries remaining" --> REWRITE[Rewrite Query Node: Semantic Reformulation]
    REWRITE --> RETRIEVE
    CHECK -- "No and max_retrieval_attempts reached" --> GEN_WARN[Generate with Sufficiency Warning]

    GEN --> END([Final Answer + Attribution])
    GEN_WARN --> END
```

> **Retry hard cap:** `should_continue()` checks `rewrite_count >= max_retries` before returning `"rewrite"`. The `max_retries` value is seeded from `AppConfig.max_retrieval_attempts` (default 3) and passed through `AgentState`, so it can never be exceeded regardless of LLM grading behavior.

---

## 3. FastAPI REST Service Architecture

The REST API exposes pipeline operations as asynchronous endpoints sharing the unified `RAGService` layer with the Typer CLI. The `/agent-stream` endpoint provides SSE progress events during agentic retrieval cycles.

```mermaid
flowchart LR
    CLIENT[HTTP Client / Frontend] --> APP[FastAPI REST API]

    APP --> ENDP1["GET /search"]
    APP --> ENDP2["POST /ask — Streaming Supported"]
    APP --> ENDP3["POST /agent-ask — Blocking"]
    APP --> ENDP4["GET /agent-stream/query — SSE Progress Events"]
    APP --> ENDP5["POST /documents"]
    APP --> ENDP6["POST /feedback"]
    APP --> ENDP7["GET /stats"]
    APP --> ENDP8["POST /evaluate"]

    ENDP1 & ENDP2 & ENDP3 & ENDP5 & ENDP7 & ENDP8 --> SVC[Unified RAGService]
    ENDP4 --> AGENT[Manual Loop: route to retrieve to grade to rewrite to generate]
    ENDP6 --> FSTORE[FeedbackStore]
```

**SSE event sequence for `/agent-stream/{query}`:** `start` → `retrieved` → `graded` → (optional) `rewriting` → `start` → … → `answer`. Each event is a newline-delimited JSON object.

---

## 4. Active Learning & Feedback Loop

> **Manual trigger note:** `knowledge feedback tune` is **intentionally manual**. Feedback accumulates in `.knowledge/feedback.json` across sessions. Run `tune` once you have collected enough samples (default threshold: 2). It does not trigger automatically after each `feedback up/down` call. A future `--watch` flag is planned but not yet implemented.

```mermaid
flowchart LR
    ANS[Retrieved Context and Answer] --> USER[User Judgment: Thumbs Up or Down]
    USER --> STORE["FeedbackStore (.knowledge/feedback.json)"]
    STORE --> MANUAL["Manual trigger: knowledge feedback tune"]
    MANUAL --> TUNER[FeedbackRerankerTuner]
    TUNER --> MODEL[Fine-Tuned Cross-Encoder Reranker]
    MODEL --> RETRIEVE[Improved Production Retrieval]
```

---

## 5. Learning-to-Rank Fusion

The LTR layer combines dense and sparse retrieval signals into a feature vector and re-ranks candidates using a trained GBDT classifier, with automatic fallback to static RRF when no model is loaded or inference fails.

```mermaid
flowchart TD
    FAISS[FAISS Dense Results: rank and score] --> FEAT[LTRFeatureExtractor: 7 features]
    BM25_R[BM25 Sparse Results: rank and score] --> FEAT
    META[Chunk Metadata: length and position] --> FEAT

    FEAT --> MODEL_CHECK{Trained GBDT available?}

    MODEL_CHECK -- Yes --> GBDT[HistGradientBoostingClassifier predict_proba]
    MODEL_CHECK -- No --> RRF[Static Reciprocal Rank Fusion]
    GBDT -- predict_proba raises --> RRF

    GBDT --> RANKED[Re-ranked Candidate List]
    RRF --> RANKED
    RANKED --> CE_R[Cross-Encoder Reranker]
```

---

## 6. Topic Clustering

Unsupervised KMeans groups indexed chunks by theme using FAISS embedding vectors when available, with TF-IDF fallback when the index is absent or mismatched.

```mermaid
flowchart TD
    CHUNKS[chunks.json] --> VCHECK{FAISS exists and ntotal equals len chunks?}
    VCHECK -- Yes --> FAISS_V[Reconstruct FAISS Embedding Vectors]
    VCHECK -- No --> TFIDF_F[TF-IDF Feature Matrix: max 256 features]

    FAISS_V --> KMEANS[KMeans: k = min n_clusters len chunks]
    TFIDF_F --> KMEANS

    KMEANS --> LABELS[Cluster Label per Chunk]
    LABELS --> KW[Per-cluster TF-IDF Keywords: top 3]
    KW --> TOPICS[topics.json saved]
    LABELS --> TAGGED[chunks.json updated with topic_id and topic_name]
```