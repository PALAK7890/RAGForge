# Research Analysis: AgenticRAG — Agentic Retrieval for Enterprise Knowledge Bases

**Authors:** Susheel Suresh, Hazel Mak, Shangpo Chou, Fred Kroon, Sahil Bhatnagar  
**Organization:** Microsoft Corporation  
**Preprint:** arXiv:2605.05538  
**Analysis Date:** 2026-11-17  
**Implementing Framework:** RAGForge (`feature/agenticrag-research`)

---

## 1. Executive Summary & Research Motivation

Traditional Retrieval-Augmented Generation (RAG) systems adhere to a rigid, synchronous **single-shot pipeline**:

$$\text{Query } q \longrightarrow \text{Embed / Tokenize } \longrightarrow \text{Retrieve top-}K \text{ Chunks } \longrightarrow \text{LLM Context Window } \longrightarrow \text{Answer } a$$

While straightforward and low-latency, this paradigm suffers from severe structural vulnerabilities when applied to complex enterprise knowledge bases:

1. **Premature Candidate Commit:** The retrieval engine selects a static top-$K$ candidate set before any reasoning model inspects the content. If critical evidence is split across disparate documents, buried deeper in rank lists, or phrased with non-overlapping vocabulary, the generator has no recourse.
2. **Context Fragmentation & Lost Document Structure:** Standard chunking slices documents into 300–800 token passages, discarding structural context (sections, headers, tables, cross-references). Single-shot RAG cannot "read forward" or "turn the page" within a promising document.
3. **Inability to Reformulate Based on Partial Evidence:** Multi-hop queries require discovering fact $A$ to know what query is needed for fact $B$. A static single-shot retriever cannot form iterative retrieval trajectories.
4. **Noise Vulnerability:** High top-$K$ retrieval injects irrelevant distractor chunks into the context window, causing LLM attention dilution and hallucinations.

### The AgenticRAG Core Hypothesis

Microsoft's **AgenticRAG** reformulates retrieval from a **passive filtering stage** into an **active, iterative agentic investigation**. Rather than replacing enterprise search infrastructure with complex custom vector databases or graph engines, AgenticRAG places a **lightweight agentic software harness** directly over the existing search layer. 

The reasoning LLM acts as an autonomous research agent equipped with four primitive operational tools—**SEARCH**, **FIND**, **OPEN**, and **SUMMARIZE**—enabling it to iteratively query, navigate inside candidate documents, synthesize partial findings, and determine when evidence is sufficient to answer or abstain.

---

## 2. Agentic Retrieval Architecture

```
                  ┌──────────────────────────────────────────────┐
                  │              User Query (q)                  │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │          Agentic Retrieval Planner           │
                  │   (Maintains Graph State & Evidence Store)   │
                  └──────────────────────┬───────────────────────┘
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   ▼                                           ▼
         [Action Selection]                          [Stopping Evaluation]
                   │                                           │
    ┌──────────────┼──────────────┬──────────────┐             │
    ▼              ▼              ▼              ▼             │
┌────────┐    ┌────────┐    ┌────────┐    ┌───────────┐        │
│ SEARCH │    │  FIND  │    │  OPEN  │    │ SUMMARIZE │        │
└────────┘    └────────┘    └────────┘    └───────────┘        │
    │              │              │              │             │
    └──────────────┼──────────────┴──────────────┘             │
                   ▼                                           │
         [Evidence Record]                                     │
                 (Dedup, Provenance, Spans)                    │
                   │                                           │
                   ▼                                           ▼
         [Evidence Evaluation] ───────► [Sufficient?] ──No────►┘
                                              │
                                             Yes
                                              ▼
                                 ┌─────────────────────────┐
                                 │ Grounded Answer with    │
                                 │ Validated Citations     │
                                 │ (or Grounded Abstention)│
                                 └─────────────────────────┘
```

The system operates as a closed-loop controller:
- **Global Discovery (Corpus-Level):** Uses `SEARCH` to retrieve top-level candidate documents and passages across hybrid indices.
- **Local Navigation (Document-Level):** Uses `FIND` (intra-document keyword/expression matching) and `OPEN` (inspecting surrounding spans, pages, or sections) to navigate inside promising documents.
- **Context Synthesis (Memory-Level):** Uses `SUMMARIZE` to compress accumulated evidence into a cohesive factual summary while preserving explicit document provenance.
- **Termination Guard:** Terminates when the evidence evaluation node marks information sufficiency, when the step budget is exhausted, or by gracefully abstaining if evidence is missing or contradictory.

---

## 3. Tool Specifications

The paper establishes four discrete agent tools with strict functional boundaries:

### Tool 1: SEARCH
- **Objective:** Discover relevant documents and candidate passages across the entire corpus.
- **Behavior:** Wraps the underlying enterprise search stack (dense vector search + BM25 sparse search + Reciprocal Rank Fusion + optional cross-encoder reranking).
- **Key Capability:** Multi-query search. The agent can issue multiple distinct query formulations in a single step or over successive iterations based on intermediate findings.
- **Output:** Structured list of passages with document identifiers, chunk IDs, hybrid scores, and source metadata.

### Tool 2: FIND
- **Objective:** Navigate *within* a known candidate document to pinpoint exact terms, identifiers, figures, or clause headers.
- **Behavior:** Targeted lexical/substring search restricted to the specified `document_id`. Case-insensitive matching with surrounding context windows (e.g., ±50 characters).
- **Distinction from SEARCH:** Does *not* execute global vector similarity. Operates deterministically on the full text or indexed chunk sequence of a single document.
- **Output:** Matching offsets, page numbers, and snippet windows. Returns an empty match set without error if terms are absent.

### Tool 3: OPEN
- **Objective:** Read detailed, contiguous content from a document around a specific location (page, section, or character offset).
- **Behavior:** Retrieves raw or chunked content surrounding an offset or page. Provides the agent the ability to "zoom in" and inspect tables, multi-paragraph arguments, or adjacent definitions that were split by chunking boundaries.
- **Safety Guards:** Enforces strict maximum output length (character/token limit) and validates document IDs to prevent unbounded context consumption.
- **Output:** Contiguous text blocks with verified source provenance and character span coordinates.

### Tool 4: SUMMARIZE
- **Objective:** Condense accumulated evidence across multiple retrieved passages into a concise, factual digest under an explicit token budget.
- **Behavior:** Synthesizes extracted text segments without hallucinating external facts. Preserves crucial numerical values, dates, and entity relations.
- **Provenance Invariant:** Explicitly maps every synthesized point back to its underlying `evidence_id`. A summary is never treated as a primary evidence record; the original source passages remain the ground truth.
- **Output:** Condensed text digest, list of contributing evidence IDs, and token count.

---

## 4. Agent Decision-Making & Control Loop

The agent follows an explicit StateGraph execution flow:

1. **Information Need Decomposition:**
   - Evaluates whether the user's prompt is a simple single-fact query, a multi-hop reasoning problem, or a cross-document comparative analysis.
   - Initial search strategy is determined (e.g., broad query vs. multiple sub-queries).

2. **Iterative Evidence Gathering:**
   - **Step 1:** Issue `SEARCH` to obtain initial anchor documents.
   - **Step 2:** If anchor documents contain references to relevant sections, dates, or specific identifiers, invoke `FIND` within those specific documents.
   - **Step 3:** Use `OPEN` on the identified pages or offsets to read the complete context (e.g., complete table or section).
   - **Step 4:** If context grows large, invoke `SUMMARIZE` on intermediate findings to free context budget while retaining provenance.

3. **Sufficiency & Stopping Conditions:**
   - The agent continuously grades its evidence against the user question:
     - **Sufficient:** All necessary sub-claims are supported by verified evidence records. Transition to answer generation.
     - **Insufficient & Budget Remaining:** Identify the specific missing variable or fact; formulate a targeted `SEARCH` or `FIND` action.
     - **Budget Exhausted:** Hard iteration cap reached (e.g., max 5 iterations or 10 tool calls). Generate a partial answer with explicit limitation caveats.
     - **Unanswerable / Conflict Detected:** If documents directly conflict or if evidence does not exist in the corpus, trigger grounded abstention rather than hallucinating.

---

## 5. Evaluation Methodology & Published Benchmark Results

The Microsoft paper evaluates AgenticRAG on three challenging benchmarks representing distinct failure modes of standard RAG:

| Benchmark | Domain & Characteristics | Baseline Performance | AgenticRAG Performance | Relative / Absolute Delta |
|:---|:---|:---|:---|:---|
| **BRIGHT** | Long-context retrieval benchmark testing reasoning across complex documents | 27.8% Recall@1 (Best embedding baseline) | **49.6% Recall@1** | **+21.8 pp** absolute gain |
| **WixQA** | Enterprise customer support & technical troubleshooting | 0.85 Factuality | **0.96 Factuality** | **+13.0%** relative improvement |
| **FinanceBench** | Complex financial documents (10-K, 10-Q), tables, ratios, cross-year filings | ~70% Answer Correctness | **92.0% Answer Correctness** | Within **2.0 pp of human Oracle** |

### Key Ablation Insights (From Paper)
1. **Agentic Tool Harness:** Shifting from static single-pass retrieval to iterative agentic tool use yielded a **5.9× improvement** in end-to-end task success.
2. **Document Navigation (`FIND` + `OPEN`):** Responsible for the majority of the gain on tabular and long-document benchmarks (FinanceBench), where chunks frequently truncate financial notes.
3. **Multi-Query Reformulation:** Prevented early dead-ends when initial enterprise search terminology mismatched the document vernacular.

---

## 6. Implementation Architecture in RAGForge

To implement the research ideas in RAGForge while preserving existing production capabilities, the following architectural mapping is established:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   RAGForge AgenticRAG Architecture                     │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Core Tool Layer: knowledge/agentic/tools.py                         │
│    - SearchTool      --> Wraps RAGService.search (FAISS + BM25 + RRF)  │
│    - FindTool        --> In-document regex/substring chunk search      │
│    - OpenTool        --> Chunks/spans reconstruction & offset reader   │
│    - SummarizeTool   --> Extractive/LLM evidence compression           │
├────────────────────────────────────────────────────────────────────────┤
│ 2. Evidence Management Layer: knowledge/agentic/evidence_store.py      │
│    - EvidenceRecord  --> UUID, chunk_id, doc_id, spans, scores, text  │
│    - EvidenceStore   --> Dedup, provenance tracking, citation engine   │
├────────────────────────────────────────────────────────────────────────┤
│ 3. Agentic Controller Layer: knowledge/agentic/controller.py           │
│    - AgenticState    --> LangGraph TypedDict with full tool trajectory │
│    - Planner / Loop  --> Multi-step tool dispatch with hard budgets    │
├────────────────────────────────────────────────────────────────────────┤
│ 4. Execution Observability: knowledge/agentic/tracer.py                │
│    - ExecutionTracer --> Step latencies, tool calls, JSON export       │
└────────────────────────────────────────────────────────────────────────┘
```

### Component Classification

#### A. Directly Reproduced from Paper
- The four-tool ontology: `SEARCH`, `FIND`, `OPEN`, `SUMMARIZE`.
- Iterative document navigation (global discovery via `SEARCH` followed by local inspection via `FIND` and `OPEN`).
- Structured evidence accumulation with strict provenance tracking.
- Dynamic stopping based on evidence sufficiency evaluation.
- Multi-query reformulation across retrieval steps.

#### B. Engineering Adaptations for RAGForge
- **Local/Offline Operation:** Built to operate with local Ollama models (`llama3.2`) and SentenceTransformers (`all-MiniLM-L6-v2`) in addition to cloud models.
- **Local Hybrid Search:** Uses in-memory FAISS FlatIP and BM25Okapi with deterministic Reciprocal Rank Fusion instead of Azure AI Search APIs.
- **Character-Offset Invariance:** Leverages RAGForge's existing `(char_start, char_end)` chunker offset metadata to map `OPEN` and citations back to exact source text spans.
- **Extractive Fallback for SUMMARIZE:** When LLM inference is constrained or offline, provides deterministic extractive synthesis to prevent pipeline stalls.

#### C. Original RAGForge Extensions Beyond Paper
- **Autotune Integration:** Hyperparameter tuning of tool budgets (`max_iterations`, `top_k`, `dense_weight`) using RAGForge's Pareto-optimal autotuning engine.
- **Active Relevance Feedback Loop:** Integration with `.knowledge/feedback.json` to allow user ratings to guide reranking within agentic search steps.
- **Three-Way Benchmark Harness:** Automated evaluation framework comparing Baseline A (Standard RAG), Baseline B (Self-Correcting RAG), and System C (AgenticRAG) across accuracy, citation integrity, and latency.
- **FastAPI SSE Streaming:** Event-by-event Server-Sent Events emitting granular tool actions (`tool_start`, `tool_result`, `evidence_collected`, `answer`) to client applications.
