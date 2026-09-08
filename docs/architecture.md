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

The agent layer introduces query intent classification, retrieval sufficiency grading, and automated query rewriting.

```mermaid
flowchart TD
    START([User Query]) --> ROUTE[Route Query Node: Classify Intent & Top-K]
    ROUTE --> RETRIEVE[Retrieve Node: Hybrid Search & Rerank]
    RETRIEVE --> GRADE[Grade Node: LLM Evaluates Context Sufficiency]
    
    GRADE --> CHECK{Context Sufficient?}
    CHECK -- Yes --> GEN[Generate Answer Node]
    CHECK -- No & Retries Left --> REWRITE[Rewrite Query Node: Semantic Reformulation]
    REWRITE --> RETRIEVE
    CHECK -- No & Max Retries Reached --> GEN_WARN[Generate with Sufficiency Warning]
    
    GEN --> END([Final Answer + Attribution])
    GEN_WARN --> END
```

---

## 3. FastAPI REST Service Architecture

The REST API exposes pipeline operations as asynchronous endpoints sharing the unified `RAGService` layer with the Typer CLI:

```mermaid
flowchart LR
    CLIENT[HTTP Client / Frontend] --> APP[FastAPI REST API]
    
    APP --> ENDP1["GET /search"]
    APP --> ENDP2["POST /ask (Streaming Supported)"]
    APP --> ENDP3["POST /agent-ask"]
    APP --> ENDP4["POST /documents"]
    APP --> ENDP5["POST /feedback"]
    APP --> ENDP6["GET /stats"]
    APP --> ENDP7["POST /evaluate"]
    
    ENDP1 & ENDP2 & ENDP3 & ENDP4 & ENDP5 & ENDP6 & ENDP7 --> SVC[Unified RAGService]
```

---

## 4. Active Learning & Feedback Loop

User feedback continually enhances retrieval accuracy through fine-tuning:

```mermaid
flowchart LR
    ANS[Retrieved Context & Answer] --> USER[User Judgment: Thumbs Up/Down]
    USER --> STORE[Feedback Store: .knowledge/feedback.json]
    STORE --> TUNER[Feedback Reranker Tuner]
    TUNER --> MODEL[Fine-Tuned Cross-Encoder Reranker]
    MODEL --> RETRIEVE[Improved Production Retrieval]
```