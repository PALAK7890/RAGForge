"""
Generates the 120-question research evaluation benchmark for AgenticRAG.
Divided into 8 categories across dev (62 questions) and held_out (58 questions) splits.
"""

import json
from pathlib import Path

def generate_benchmark(output_path=None) -> None:
    questions = []

    # 1. FACTUAL RETRIEVAL (20 questions: 10 dev, 10 held_out)
    factual_specs = [
        ("What vector index algorithm is utilized by RAGForge for dense inner-product similarity search?", ["architecture.md", "design_decisions.md"], ["FAISS", "FlatIP"], "dev"),
        ("What neural reranker model is employed for second-stage cross-attention scoring?", ["architecture.md", "design_decisions.md"], ["ms-marco-MiniLM-L-6-v2", "CrossEncoder"], "dev"),
        ("What default loss function is utilized by BiEncoderTrainer for contrastive domain adaptation?", ["design_decisions.md"], ["MultipleNegativesRankingLoss"], "dev"),
        ("What is the primary sparse lexical retrieval algorithm integrated in RAGForge?", ["architecture.md", "readme.md"], ["BM25", "BM25Okapi"], "dev"),
        ("What default port does the RAGForge FastAPI server listen on?", ["architecture.md", "main.py"], ["8000"], "dev"),
        ("What embedding model family is used for semantic dense embeddings in RAGForge?", ["architecture.md", "design_decisions.md"], ["SentenceTransformers", "all-MiniLM-L6-v2"], "dev"),
        ("What parameter governs the relative contribution of dense versus sparse search in RRF?", ["design_decisions.md"], ["dense_weight"], "dev"),
        ("What format are intermediate chunks and metadata serialized to in the index directory?", ["design_decisions.md"], ["chunks.json"], "dev"),
        ("What default smoothing constant k is used in the reciprocal rank fusion formula?", ["hybrid.py", "design_decisions.md"], ["60"], "dev"),
        ("What local inference engine powers language model generation in RAGForge?", ["readme.md", "architecture.md"], ["Ollama"], "dev"),
        # Held-out
        ("What is the purpose of the DocumentRegistry class during document indexing?", ["architecture.md"], ["SHA-1", "hash", "incremental"], "held_out"),
        ("What chunking strategies are supported by RAGForge text chunkers?", ["design_decisions.md"], ["Recursive", "Sentence", "Fixed-size"], "held_out"),
        ("What metric evaluates the reciprocal rank of the first relevant document retrieved?", ["design_decisions.md"], ["MRR", "Mean Reciprocal Rank"], "held_out"),
        ("How does RAGForge ensure deterministic tie-breaking during hybrid rank fusion?", ["design_decisions.md", "hybrid.py"], ["chunk_id"], "held_out"),
        ("What structured log file stores user relevance judgments in RAGForge?", ["design_decisions.md"], ["feedback.json"], "held_out"),
        ("What HTTP streaming standard is used by RAGForge for agent progress updates?", ["design_decisions.md"], ["Server-Sent Events", "SSE"], "held_out"),
        ("What library is used by RAGForge to build state graph agentic workflows?", ["architecture.md", "design_decisions.md"], ["LangGraph"], "held_out"),
        ("What is the maximum token sequence limit for the ms-marco-MiniLM-L-6-v2 reranker?", ["design_decisions.md"], ["512"], "held_out"),
        ("What loss function is used for CrossEncoder binary classification tuning?", ["design_decisions.md"], ["binary cross-entropy", "BCE"], "held_out"),
        ("What statistical method does the autotune engine use to validate candidate improvements?", ["design_decisions.md"], ["paired bootstrap"], "held_out"),
    ]

    for q, docs, keywords, split in factual_specs:
        questions.append({
            "id": f"fact_{len(questions)+1:03d}",
            "question": q,
            "category": "factual",
            "split": split,
            "expected_documents": docs,
            "gold_keywords": keywords,
            "is_unanswerable": False,
        })

    # 2. EXACT KEYWORD RETRIEVAL (15 questions)
    keyword_specs = [
        ("What exact Python class name implements recursive character chunking?", ["architecture.md"], ["RecursiveChunker"], "dev"),
        ("Which specific CLI command initiates hyperparameter sweep tuning?", ["main.py"], ["autotune"], "dev"),
        ("What exact module houses the reciprocal_rank_fusion algorithm?", ["architecture.md"], ["knowledge.retrievers.hybrid"], "dev"),
        ("What specific Pydantic schema represents an indexed document record?", ["schemas.py"], ["DocumentRecord"], "dev"),
        ("What exact function name builds and compiles the LangGraph state machine?", ["graph.py"], ["build_agent_graph"], "dev"),
        ("What specific environment variable enables offline operation without HuggingFace calls?", ["design_decisions.md"], ["HF_HUB_OFFLINE"], "dev"),
        ("What CLI option sets the maximum retrieval rewrite attempts in agent-ask?", ["main.py"], ["--max-retries"], "dev"),
        ("What exact class represents vector index storage in FAISS?", ["architecture.md"], ["FAISSStore"], "dev"),
        # Held-out
        ("What exact method name embeds multiple documents simultaneously in SentenceTransformerEmbedding?", ["design_decisions.md"], ["embed_documents"], "held_out"),
        ("What exact endpoint returns workspace statistics in the REST API?", ["app.py"], ["/stats"], "held_out"),
        ("What specific CLI command benchmarks stage-by-stage latency?", ["main.py"], ["benchmark"], "held_out"),
        ("What exact class routes incoming files to appropriate loaders?", ["architecture.md"], ["LoaderRouter"], "held_out"),
        ("What exact method loads the FAISS index from disk in FAISSStore?", ["design_decisions.md"], ["load"], "held_out"),
        ("What exact method in BM25Retriever constructs the inverted index from text corpus?", ["design_decisions.md"], ["build"], "held_out"),
        ("What CLI flag specifies the minimum feedback sample size for reranker tuning?", ["main.py"], ["--min-samples"], "held_out"),
    ]

    for q, docs, keywords, split in keyword_specs:
        questions.append({
            "id": f"kw_{len(questions)+1:03d}",
            "question": q,
            "category": "exact_keyword",
            "split": split,
            "expected_documents": docs,
            "gold_keywords": keywords,
            "is_unanswerable": False,
        })

    # 3. MULTI-HOP REASONING (20 questions)
    multihop_specs = [
        ("How does the document ingestion process in RAGForge enable the autotune evaluator to compare chunks across different chunk sizes?", ["design_decisions.md"], ["character offsets", "char_start", "char_end", "immutable source text"], "dev"),
        ("Why does the cross-encoder stage only process a top candidate pool rather than the full corpus indexed by FAISS?", ["design_decisions.md"], ["O(N) forward passes", "bi-encoder narrows", "latency"], "dev"),
        ("How does the self-correcting agent determine whether to terminate retrieval when an LLM fails to output a numeric iteration count?", ["design_decisions.md"], ["AgentState", "hard integer cap", "conditional routing edge"], "dev"),
        ("Why does reciprocal rank fusion avoid the score distribution mismatch between FAISS cosine similarities and BM25 scores?", ["design_decisions.md"], ["incompatible distributions", "ordinal positions", "unbounded BM25"], "dev"),
        ("How does the offline candidate pooling in the autotune runner prevent repeated index rebuilds during query parameter sweeps?", ["design_decisions.md"], ["retrieves top-50 pool once", "in-memory array sorting", "two-tier"], "dev"),
        ("What happens to span-level evaluation when a document loader normalizes whitespace or strips line breaks before chunking?", ["design_decisions.md"], ["str.find fails", "char_start=None", "document-level hit detection"], "dev"),
        ("How does the LLM-as-a-judge module evaluate answer quality differently from traditional retrieval metrics like MRR?", ["design_decisions.md"], ["faithfulness", "relevance", "hallucination", "retrieval vs generation"], "dev"),
        ("Why does the feedback tuning mechanism require an explicit manual trigger rather than retraining online after every user rating?", ["design_decisions.md"], ["catastrophic forgetting", "accidental clicks", "minimum sample threshold"], "dev"),
        ("How does the agent controller handle situations where the retrieved context is graded as insufficient by the sufficiency grader?", ["graph.py", "design_decisions.md"], ["rewrite_query", "retrieve again", "max_retries cap"], "dev"),
        ("Why are in-batch random negatives insufficient for teaching the cross-encoder to distinguish fine-grained passage differences?", ["design_decisions.md"], ["easy to distinguish", "coarse domain vocabulary", "hard negative mining"], "dev"),
        # Held-out
        ("How does RAGForge's incremental indexing determine whether an existing file needs re-indexing?", ["architecture.md"], ["DocumentRegistry", "SHA-1 hash", "UNCHANGED"], "held_out"),
        ("Why does paired bootstrap significance testing outperform Student's t-test for evaluating information retrieval improvements?", ["design_decisions.md"], ["bimodal distribution", "bounded [0, 1]", "no normality assumption"], "held_out"),
        ("How does the Server-Sent Events architecture in FastAPI avoid the stateful overhead associated with WebSockets for RAG queries?", ["design_decisions.md"], ["unidirectional push", "plain HTTP", "auto-reconnection"], "held_out"),
        ("Why does RRF fusion fail to distinguish high-confidence outlier documents with 0.99 similarity from marginal matches?", ["design_decisions.md"], ["discards score magnitudes", "ordinal rank only", "position 1"], "held_out"),
        ("How does the judge fallback score of 1.0 affect aggregate benchmark evaluations when local model timeouts occur?", ["design_decisions.md"], ["inflates faithfulness", "silent error", "appears to agree"], "held_out"),
        ("Why does the autotune neighborhood priority heuristic risk getting trapped in local parameter minima?", ["design_decisions.md"], ["assumes metric smoothness", "baseline local minimum", "exhausts budget"], "held_out"),
        ("How does the synthetic evaluation generator verify whether source passage text changed after dataset generation?", ["design_decisions.md"], ["SHA-1 hash of extracted text", "char_start", "detects stale labels"], "held_out"),
        ("What prevents the cross-encoder reranker from offloading inference to an available GPU in the current implementation?", ["design_decisions.md"], ["no device argument", "CrossEncoder(model_name)", "defaults to CPU"], "held_out"),
        ("Why does binary cross-entropy fine-tuning fail when users submit only positive ratings in the relevance feedback loop?", ["design_decisions.md"], ["only one class", "no negative examples", "trivial model"], "held_out"),
        ("How does the autotune runner ensure atomicity when saving rebuilt FAISS index caches to disk?", ["design_decisions.md"], ["temp-dir rename", "avoids partial writes"], "held_out"),
    ]

    for q, docs, keywords, split in multihop_specs:
        questions.append({
            "id": f"mh_{len(questions)+1:03d}",
            "question": q,
            "category": "multihop",
            "split": split,
            "expected_documents": docs,
            "gold_keywords": keywords,
            "is_unanswerable": False,
        })

    # 4. COMPARATIVE ANALYSIS (15 questions)
    comparative_specs = [
        ("Compare the strengths and weaknesses of dense vector search versus sparse lexical search in hybrid retrieval.", ["design_decisions.md", "architecture.md"], ["semantic meaning vs exact keywords", "out-of-vocabulary", "FAISS vs BM25"], "dev"),
        ("Compare the latency and computational complexity of bi-encoder retrieval versus cross-encoder reranking.", ["design_decisions.md"], ["pre-computed embeddings", "O(N) cross-attention", ">80% latency"], "dev"),
        ("Compare the architectural role of standard single-shot RAG with self-correcting agentic RAG.", ["architecture.md", "design_decisions.md"], ["fixed top-k vs iterative rewriting", "sufficiency grading", "LangGraph"], "dev"),
        ("Compare Server-Sent Events (SSE) and WebSockets for streaming agentic RAG query progress to clients.", ["design_decisions.md"], ["unidirectional vs bidirectional", "HTTP compatibility", "stateful overhead"], "dev"),
        ("Compare character offset tracking with chunk ID matching for tracking ground truth evaluation spans.", ["design_decisions.md"], ["immutable source text vs unstable chunk IDs", "survives re-chunking"], "dev"),
        ("Compare the trade-offs of in-batch random negatives versus hard negative mining for embedding model fine-tuning.", ["design_decisions.md"], ["faster data prep vs discriminative signal", "second retrieval pass"], "dev"),
        ("Compare the design of flat parameter grid search versus two-tier indexed parameter budgeting in autotune.", ["design_decisions.md"], ["54 index builds vs 8 budget builds", "offline candidate pooling"], "dev"),
        ("Compare Student's t-test with paired bootstrap resampling for evaluating retrieval metric significance.", ["design_decisions.md"], ["parametric normality assumption vs non-parametric empirical CI"], "dev"),
        # Held-out
        ("Compare the role of SEARCH tool versus FIND tool in Microsoft AgenticRAG architecture.", ["agenticrag-paper-analysis.md"], ["corpus-wide discovery vs intra-document keyword location"], "held_out"),
        ("Compare the role of FIND tool versus OPEN tool in document navigation.", ["agenticrag-paper-analysis.md"], ["identifying match locations vs inspecting contiguous sections/pages"], "held_out"),
        ("Compare extractive evidence summarization versus abstractive LLM summarization in context compression.", ["agenticrag-paper-analysis.md", "tools.py"], ["deterministic speed vs synthesis capability", "token budget"], "held_out"),
        ("Compare RAGForge's default 2-question smoke test with enterprise benchmark evaluation sets.", ["design_decisions.md", "agenticrag-paper-analysis.md"], ["pipeline verification vs statistical power", "BRIGHT", "FinanceBench"], "held_out"),
        ("Compare recursive character chunking with sentence-boundary chunking on structured technical text.", ["design_decisions.md"], ["paragraph-sentence hierarchy vs linguistic sentence models"], "held_out"),
        ("Compare the fallback behavior of LLMJudge versus deterministic rule-based evaluation.", ["design_decisions.md"], ["1.0 fallback inflates scores vs strict error propagation"], "held_out"),
        ("Compare the index caching strategy in autotune with the trial score persistence across sweep interruptions.", ["design_decisions.md"], ["index caches survive vs trial results lost mid-sweep"], "held_out"),
    ]

    for q, docs, keywords, split in comparative_specs:
        questions.append({
            "id": f"comp_{len(questions)+1:03d}",
            "question": q,
            "category": "comparative",
            "split": split,
            "expected_documents": docs,
            "gold_keywords": keywords,
            "is_unanswerable": False,
        })

    # 5. NUMERICAL & QUANTITATIVE REASONING (15 questions)
    numerical_specs = [
        ("What is the default chunk size and chunk overlap configured in RAGForge's default config.yaml?", ["config.yaml", "main.py"], ["500", "100"], "dev"),
        ("What is the default top_k number of chunks retrieved by RAGForge during standard search?", ["config.yaml", "service.py"], ["5"], "dev"),
        ("What is the minimum sample count required before tune-reranker will execute fine-tuning?", ["main.py", "design_decisions.md"], ["5"], "dev"),
        ("What is the default maximum number of retrieval rewrite attempts in the self-correcting agent?", ["graph.py", "design_decisions.md"], ["2", "3"], "dev"),
        ("What percentage of total retrieval latency does the cross-encoder reranker typically account for on large candidate pools?", ["design_decisions.md"], [">80%"], "dev"),
        ("How many bootstrap resampling iterations does the autotune composite scorer run to establish confidence intervals?", ["design_decisions.md"], ["1000", "1,000"], "dev"),
        ("What is the maximum token context limit for the ms-marco-MiniLM-L-6-v2 cross-encoder model?", ["design_decisions.md"], ["512"], "dev"),
        ("What was the warm-model single-process retrieval latency measured in RAGForge on 2026-11-09?", ["README.md", "measure_latency.py"], ["43.6 ms"], "dev"),
        # Held-out
        ("What was the cold single-process retrieval latency including model load measured on 2026-11-09?", ["README.md"], ["1563.5 ms"], "held_out"),
        ("What Recall@1 did Microsoft's AgenticRAG achieve on the BRIGHT long-context benchmark?", ["agenticrag-paper-analysis.md"], ["49.6%"], "held_out"),
        ("What was the absolute percentage point improvement of AgenticRAG over baseline models on the BRIGHT benchmark?", ["agenticrag-paper-analysis.md"], ["+21.8"], "held_out"),
        ("What factuality score did AgenticRAG achieve on the WixQA customer support benchmark?", ["agenticrag-paper-analysis.md"], ["0.96"], "held_out"),
        ("What answer correctness percentage did AgenticRAG reach on FinanceBench?", ["agenticrag-paper-analysis.md"], ["92%"], "held_out"),
        ("How many percentage points was AgenticRAG's FinanceBench score from human oracle performance?", ["agenticrag-paper-analysis.md"], ["2", "2.0"], "held_out"),
        ("What performance multiplier did Microsoft researchers isolate from switching to agentic tool use in their ablation study?", ["agenticrag-paper-analysis.md"], ["5.9x", "5.9"], "held_out"),
    ]

    for q, docs, keywords, split in numerical_specs:
        questions.append({
            "id": f"num_{len(questions)+1:03d}",
            "question": q,
            "category": "numerical",
            "split": split,
            "expected_documents": docs,
            "gold_keywords": keywords,
            "is_unanswerable": False,
        })

    # 6. DOCUMENT NAVIGATION (15 questions)
    nav_specs = [
        ("In docs/design_decisions.md, what is described as the key known limitation of the document chunker character offset locator?", ["design_decisions.md"], ["str.find", "char_start=None", "whitespace normalization"], "dev"),
        ("In docs/architecture.md, what are the primary stages shown in the retrieval pipeline diagram?", ["architecture.md"], ["Dense Retrieval", "Sparse Retrieval", "RRF", "Cross-Encoder"], "dev"),
        ("In docs/design_decisions.md under Section 3, what code reference is given for the CrossEncoder model initialization?", ["design_decisions.md"], ["CrossEncoderReranker.__init__", "CrossEncoder(model_name)"], "dev"),
        ("In docs/design_decisions.md Section 6, what static fallback score is returned when JSON parsing fails in LLMJudge?", ["design_decisions.md"], ["1.0", "LLMJudge.score_faithfulness"], "dev"),
        ("In docs/design_decisions.md Section 4, what architectural remedy is proposed for blocking CPU calls inside async FastAPI routes?", ["design_decisions.md"], ["run_in_executor", "asyncio"], "dev"),
        ("In docs/design_decisions.md Section 5, how does the sufficiency grader handle polite preambles like 'partially yes' from small local LLMs?", ["design_decisions.md"], ["interprets as no", "redundant rewrite cycle"], "dev"),
        ("In docs/design_decisions.md Section 9, what happens to trial results if an autotune run is interrupted after 40 completed trials?", ["design_decisions.md"], ["lost", "not checkpointed mid-sweep", "restarts from scratch"], "dev"),
        ("In docs/design_decisions.md Section 10, why are evaluation queries not guaranteed to be strictly independent and identically distributed?", ["design_decisions.md"], ["adjacent chunks", "correlated performance", "narrows CI"], "dev"),
        # Held-out
        ("In docs/research/agenticrag-paper-analysis.md, what are the four core tools specified by the Microsoft research team?", ["agenticrag-paper-analysis.md"], ["SEARCH", "FIND", "OPEN", "SUMMARIZE"], "held_out"),
        ("In docs/research/agenticrag-paper-analysis.md, which benchmark measures performance on financial reports like 10-K and 10-Q filings?", ["agenticrag-paper-analysis.md"], ["FinanceBench"], "held_out"),
        ("In docs/research/agenticrag-paper-analysis.md Section 3, how does the FIND tool differ functionally from SEARCH?", ["agenticrag-paper-analysis.md"], ["intra-document substring matching", "no global vector search"], "held_out"),
        ("In docs/research/agenticrag-paper-analysis.md, what is the reported limitation regarding multi-step agent reasoning latency?", ["agenticrag-paper-analysis.md"], ["multiple LLM reasoning steps", "token overhead"], "held_out"),
        ("In docs/design_decisions.md Section 8, why does the synthetic evaluation generator store character offsets and text hashes rather than full text?", ["design_decisions.md"], ["double file size", "noisy diffs", "re-extract on demand"], "held_out"),
        ("In docs/design_decisions.md Section 11, what is the documented failure mode of models trained only with in-batch random negatives?", ["design_decisions.md"], ["cannot discriminate closely related chunks", "within same section"], "held_out"),
        ("In docs/design_decisions.md Section 12, what risk is cited if user relevance judgments are applied for online model retraining on every click?", ["design_decisions.md"], ["catastrophic forgetting", "accidental clicks", "transient errors"], "held_out"),
    ]

    for q, docs, keywords, split in nav_specs:
        questions.append({
            "id": f"nav_{len(questions)+1:03d}",
            "question": q,
            "category": "document_navigation",
            "split": split,
            "expected_documents": docs,
            "gold_keywords": keywords,
            "is_unanswerable": False,
        })

    # 7. AMBIGUOUS QUERIES (10 questions: 5 dev, 5 held_out)
    ambiguous_specs = [
        ("How does RAGForge optimize retrieval latency?", ["design_decisions.md", "architecture.md"], ["two-stage retrieval", "candidate pooling", "caching"], "dev"),
        ("Explain the role of weights in RAGForge ranking.", ["design_decisions.md", "hybrid.py"], ["dense_weight", "RRF", "reciprocal rank"], "dev"),
        ("What parameters can be tuned in RAGForge?", ["design_decisions.md"], ["chunk size", "overlap", "top-k", "dense_weight", "RRF k"], "dev"),
        ("How does RAGForge handle document updates?", ["architecture.md"], ["DocumentRegistry", "incremental indexing", "SHA-1"], "dev"),
        ("What makes RAGForge inspectable?", ["README.md", "architecture.md"], ["modular stages", "char offsets", "benchmarks", "traces"], "dev"),
        # Held-out
        ("How is quality measured in RAGForge pipelines?", ["design_decisions.md"], ["MRR", "Accuracy@K", "Precision", "Recall", "LLM judge"], "held_out"),
        ("What happens when context is incomplete during generation?", ["graph.py", "design_decisions.md"], ["rewrite query", "retry cap", "abstention"], "held_out"),
        ("How does RAGForge support external frontends?", ["app.py", "design_decisions.md"], ["FastAPI", "REST", "SSE streaming"], "held_out"),
        ("What are the primary failure points of RAGForge reranking?", ["design_decisions.md"], ["CPU only", "512 token limit", "high latency"], "held_out"),
        ("How does active learning work in the system?", ["design_decisions.md"], ["feedback records", "min-samples", "tune-reranker"], "held_out"),
    ]

    for q, docs, keywords, split in ambiguous_specs:
        questions.append({
            "id": f"amb_{len(questions)+1:03d}",
            "question": q,
            "category": "ambiguous",
            "split": split,
            "expected_documents": docs,
            "gold_keywords": keywords,
            "is_unanswerable": False,
        })

    # 8. UNANSWERABLE QUESTIONS (10 questions: 5 dev, 5 held_out)
    unanswerable_specs = [
        ("What is the AWS Bedrock pricing model configured in RAGForge for Claude 3.5 Sonnet?", [], ["abstain", "not configured", "unanswerable"], "dev"),
        ("Which PostgreSQL connection pool size is specified in database_config.yaml?", [], ["abstain", "no postgresql", "unanswerable"], "dev"),
        ("What was Microsoft's quarterly revenue from Xbox Game Pass in Q3 2021 according to the index?", [], ["abstain", "not found", "unanswerable"], "dev"),
        ("What is the secret AES-256 encryption key used for securing SQLite customer sessions?", [], ["abstain", "no secret key", "unanswerable"], "dev"),
        ("How many Kubernetes pods are deployed across the RAGForge production cluster?", [], ["abstain", "no kubernetes", "unanswerable"], "dev"),
        # Held-out
        ("What is the Snowflake warehouse cluster size used for daily vector exports in RAGForge?", [], ["abstain", "no snowflake", "unanswerable"], "held_out"),
        ("Which Redis caching layer port is designated for session management in production?", [], ["abstain", "no redis", "unanswerable"], "held_out"),
        ("What is the Kafka topic partition count configured for document ingestion streaming?", [], ["abstain", "no kafka", "unanswerable"], "held_out"),
        ("What was Apple's total iPhone sales volume in mainland China during fiscal year 2018?", [], ["abstain", "not indexed", "unanswerable"], "held_out"),
        ("What is the OAuth2 client secret for Google Workspace SSO integration in RAGForge?", [], ["abstain", "no oauth secret", "unanswerable"], "held_out"),
    ]

    for q, docs, keywords, split in unanswerable_specs:
        questions.append({
            "id": f"unans_{len(questions)+1:03d}",
            "question": q,
            "category": "unanswerable",
            "split": split,
            "expected_documents": docs,
            "gold_keywords": keywords,
            "is_unanswerable": True,
        })

    out_path = Path(output_path) if output_path else Path(__file__).with_name("enterprise_benchmark.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2)

    dev_count = sum(1 for q in questions if q["split"] == "dev")
    held_count = sum(1 for q in questions if q["split"] == "held_out")
    print(f"Generated {len(questions)} evaluation questions: {dev_count} dev, {held_count} held-out.")

if __name__ == "__main__":
    generate_benchmark()
