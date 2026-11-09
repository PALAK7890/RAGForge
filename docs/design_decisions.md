# RAGForge — Design Decisions & Known Limitations

For each major component this document records: what problem it solves, the
key design decision made, the alternative that was not chosen and why, and
a known limitation or edge case the current implementation does not handle
well.

---

## 1. Document Chunkers & Character Offsets (`knowledge/chunkers/`)

**What it does**
Splits loaded document text into bounded segments using one of three
strategies — Recursive (split on paragraph → sentence → word boundaries),
Sentence-boundary, and Fixed-size — while recording the exact
`(char_start, char_end)` byte offsets of each chunk inside the raw page text.

**Key design decision — store offsets, not just chunk IDs**
Chunk IDs were considered as the primary identity key for evaluation hits.
They were rejected because IDs are unstable: any change to chunk size,
overlap, or strategy shifts every boundary and invalidates all stored IDs.
Character offsets point directly into the immutable source text and survive
across completely different indexing configurations, which is essential for
the autotune evaluator to compare results across a parameter sweep.

**Known limitation**
The offset locator uses `str.find(chunk_text, search_from)`. If a document
loader normalises whitespace, strips internal line breaks, or alters
punctuation, `find()` fails to locate the transformed substring in the raw
source and silently returns `char_start=None, char_end=None`. That chunk
loses span-level evaluation and falls back to document-level hit detection.

Code references:
- `RecursiveChunker.chunk()` — `pos = text.find(chunk_text, search_from)`; sets `char_start = None, char_end = None` when `find()` returns `-1`
- `SentenceChunker._flush_buffer()` — same `str.find` pattern; `cs = None, ce = None` on failure

---

## 2. Hybrid Retrieval & Parameterised RRF (`knowledge/retrievers/`)

**What it does**
Combines dense semantic vector search (SentenceTransformers + FAISS FlatIP)
with sparse BM25 keyword search using a shared `reciprocal_rank_fusion`
function. A single `dense_weight` parameter in `[0.0, 1.0]` controls the
relative contribution of each signal. Ties are broken deterministically by
`chunk_id` so results are reproducible across runs.

**Key design decision — rank fusion over score normalisation**
An alternative is to L2-normalise cosine similarities and BM25 scores to
`[0, 1]` and linearly combine them. This was not chosen because the two
score distributions are incompatible: cosine similarities cluster in a
narrow upper range while BM25 scores are unbounded and scale with document
length and term frequency. Any fixed normalisation constant would need
re-tuning per corpus. Rank fusion operates purely on ordinal positions
(`1 / (k + rank)`) and is corpus-agnostic. The configurable `dense_weight`
was added after the autotune runner showed that the optimal weight varies
across document collections.

**Known limitation**
RRF discards score magnitudes entirely. A document with 0.99 cosine
similarity and a document with 0.52 cosine similarity both rank first if they
land in position 1; the high-confidence outlier cannot distinguish itself.
This matters when one retrieval signal is very confident and the other is
noisy.

---

## 3. Cross-Encoder Neural Reranking (`knowledge/rerankers/cross_encoder.py`)

**What it does**
Performs full query-passage cross-attention scoring using
`ms-marco-MiniLM-L-6-v2` on the top candidates returned by hybrid fusion.
Unlike bi-encoder embeddings, the cross-encoder sees query and passage
jointly through all transformer layers before producing a relevance score.

**Key design decision — two-stage retrieval (bi-encoder first, cross-encoder second)**
Running the cross-encoder over the full corpus would be accurate but
prohibitively slow: each query requires `O(N)` transformer forward passes.
The bi-encoder + FAISS retrieval stage narrows candidates from thousands to
20–50 in sub-millisecond time (pre-computed document vectors). The
cross-encoder is then restricted to that small pool, giving near-full
accuracy at a fraction of the cost.

**Known limitation**
`self.model.predict(pairs)` passes all query-passage pairs to the
SentenceTransformers `CrossEncoder` in a single call, so the library handles
internal batching. However, there is no `device=` argument passed to the
constructor, which means inference always runs on CPU regardless of available
GPU hardware. There is also no configurable `batch_size` or `show_progress_bar`
parameter. On long chunks (>800 characters) and large candidate pools the
cross-encoder accounts for >80 % of total retrieval latency.

Code references:
- `CrossEncoderReranker.__init__()` — `self.model = CrossEncoder(model_name)` with no `device=` argument; always defaults to CPU
- `CrossEncoderReranker.rerank()` — `scores = self.model.predict(pairs)` passes all pairs in one call; there is no `batch_size` or GPU configuration

---

## 4. FastAPI Service Layer (`knowledge/api/`)

**What it does**
Exposes the retrieval, evaluation, and agent pipelines over async HTTP REST
endpoints. The `/agent-stream/{query}` endpoint emits Server-Sent Events
(SSE) so callers receive live progress updates at each agentic step (retrieved,
graded, rewriting, answer) without polling.

**Key design decision — SSE over WebSockets for streaming**
WebSockets support bidirectional communication, but RAG queries follow a
strictly unidirectional server-push pattern once submitted. SSE runs over
plain HTTP, traverses proxies and load balancers without protocol upgrades,
and handles auto-reconnection at the client layer. WebSockets would add
stateful connection management and keepalive overhead without providing
any benefit for this use case.

**Known limitation**
The underlying `RAGService` calls FAISS and runs cross-encoder inference
synchronously. Under concurrent load, these blocking CPU operations inside
async route handlers can stall the event loop and reduce throughput. The
fix is `asyncio.get_event_loop().run_in_executor()` around blocking calls,
which is not currently implemented.

---

## 5. LangGraph Self-Correcting Agent (`knowledge/agent/`)

**What it does**
Implements a cyclic state graph: `route_query` → `retrieve_context` →
`grade_sufficiency` → (`rewrite_query` | `generate_answer`). If the grader
decides retrieved context is insufficient, the query is rewritten and
retrieval runs again, up to `max_retrieval_attempts` times (default 3).

**Key design decision — hard integer cap in graph state, not in LLM prompt**
The iteration limit could be expressed as a prompt instruction ("stop after
3 rewrites"). This was rejected because local LLMs do not reliably follow
numeric counting instructions; they may miscount or ignore the constraint
entirely. The cap is instead stored as an integer in `AgentState` and
enforced by the conditional routing edge in Python code, making termination
deterministic regardless of LLM behaviour.

**Known limitation**
The sufficiency grader prompts the LLM to output a binary "yes" or "no".
Small local models (e.g., Llama-3.2-3B) frequently produce preambles like
"Based on the provided context, partially yes", which the string parser
interprets as "no", triggering a redundant rewrite cycle even when the
context was adequate.

---

## 6. LLM-as-a-Judge Evaluation (`knowledge/evaluation/judge.py`)

**What it does**
Automates response quality scoring by prompting an LLM to evaluate generated
answers on two axes: **Faithfulness** (is every claim grounded in the
retrieved text?) and **Relevance** (does the answer address the user's
question?). Scores are parsed from structured JSON with a regex fallback for
conversational output.

**Key design decision — two orthogonal axes over a single composite score**
A single "quality" score was considered. It was rejected because faithfulness
and relevance capture independent failure modes. A model can faithfully
paraphrase an irrelevant passage (high faithfulness, low relevance) or
produce a relevant-sounding answer not supported by the retrieved text (low
faithfulness, high relevance). Separating them makes it possible to diagnose
whether a problem is in retrieval (wrong passage retrieved) or generation
(LLM hallucinated beyond the context).

**Known limitation**
When parsing fails completely (JSON decode error or unexpected output format),
`LLMJudge.score_faithfulness()` and `LLMJudge.score_relevance()` each return
a hardcoded fallback score of `1.0` — the maximum possible value. In an
environment with frequent model timeouts or degraded local inference this
silently inflates aggregate faithfulness and relevance averages, making the
judge appear to agree with every answer even when it never ran. The failure
mode is caught by a bare `except Exception` block; it is not surfaced as a
distinct error state in evaluation reports.

---

## 7. Autotune: Search Space & Budgeting (`knowledge/autotune/search_space.py`)

**What it does**
Defines a two-tier parameter grid: `IndexParams` (chunk size, overlap,
chunking strategy, embedding model — require a full FAISS rebuild each) and
`QueryParams` (top-k, dense weight, RRF k, reranking, query expansion —
evaluated in-memory from a pre-retrieved candidate pool). A configurable
`budget` caps the number of index builds; for each built index every
`QueryParams` combination is swept exhaustively.

**Key design decision — separate index budget from query sweep**
Treating all parameters as a flat grid would make every combination require
a fresh FAISS rebuild. With 3 chunk sizes × 2 overlaps × 3 strategies ×
3 embedding models = 54 potential index builds, and 72 query configurations,
a flat sweep would be 54 × 72 = 3,888 retrieval passes. The two-tier design
reduces this to `budget` index builds (default 8) × 72 query sweeps = 576
passes, with the query sweep being almost free (in-memory array sorting).

**Known limitation**
The neighbourhood priority heuristic assumes metric smoothness around the
baseline. If the baseline is in a local minimum (e.g., chunk size 100 on
documents requiring 800-character context), the budget may be exhausted
exploring nearby ineffective configurations before reaching high-performing
distant regions of the space.

---

## 8. Autotune: Synthetic Evaluation Generator (`knowledge/autotune/synth_eval.py`)

**What it does**
Generates an evaluation set from unannotated local documents by prompting an
LLM to write natural search queries paired with their source passage. Each
item stores gold character offsets (`char_start`, `char_end`) and a SHA-1
hash of the extracted text so the evaluator can detect if the source passage
changed between runs. The set is split 80/20 into train and held-out splits
used by the scorer.

**Key design decision — store offsets and text hash, not extracted text**
Storing the full extracted passage would double the size of evaluation files
and make diffs noisy. Storing only the offset and a hash is sufficient:
the evaluator can re-extract the text from the source document on demand and
the hash detects stale labels if the source file is edited after generation.

**Known limitation**
Questions are generated one chunk at a time, so the synthetic set only
covers single-hop factual retrieval. Multi-hop questions (requiring synthesis
across multiple passages) and comparative questions cannot be generated by
this approach.

---

## 9. Autotune: Experiment Runner (`knowledge/autotune/runner.py`)

**What it does**
Orchestrates the two-tier sweep: builds and caches FAISS indexes on disk
(with atomic temp-dir → rename to avoid partial writes), retrieves a
top-50 candidate pool per query once per `(index, query_expansion)` pair,
pre-scores the union of candidates through the cross-encoder in a single
batch, and applies offline RRF fusion across all `QueryParams` configurations
from the cached pool.

**Key design decision — offline candidate pooling over per-config retrieval**
An earlier design re-ran FAISS + BM25 retrieval for every `QueryParams`
combination. With 72 configurations × eval set size, this was 72× more
retrieval calls. The pooling design retrieves once and recomputes fusion
weights in memory. The tradeoff is a fixed pool ceiling: if `dense_weight`
would have promoted a chunk outside the top-50 pool, that chunk is
permanently unavailable to any downstream configuration.

**Known limitation**
Trial results are not checkpointed mid-sweep. If a long autotune run is
interrupted after completing 40 of 576 trials, all 40 results are lost and
the sweep restarts from scratch on the next run. Index caches survive
interruption; trial scores do not.

---

## 10. Autotune: Composite Scorer & Bootstrap Statistics (`knowledge/autotune/scorer.py`)

**What it does**
Computes per-configuration composite scores (weighted MRR + Hit Rate +
latency penalty) under three presets (`quality`, `balanced`, `fast`).
Statistical significance between baseline and candidate is evaluated using
paired bootstrap resampling on per-query metric vectors; a configuration is
accepted only when its bootstrap confidence interval excludes zero. Estimated
vs measured latencies are strictly separated and never compared.

**Key design decision — paired bootstrap over Student's t-test**
Information retrieval metrics like MRR and Hit Rate are bounded in `[0, 1]`
and heavily bimodal (many queries score 0 or 1). Student's t-test assumes
normality, which these distributions violate. Paired bootstrap makes no
parametric assumptions: it resamples paired per-query differences
`(candidate_i − baseline_i)` with replacement across 1,000 iterations.
Significance is declared when the empirical CI strictly excludes zero.

**Known limitation**
Bootstrap resampling assumes i.i.d. evaluation queries. If multiple synthetic
questions come from adjacent chunks in the same document, inter-query
performance is correlated, which can narrow the CI and inflate significance
rates slightly.

---

## 11. Domain Adaptation Fine-Tuning (`knowledge/training/`)

**What it does**
Provides two fine-tuning pipelines: `BiEncoderTrainer` adapts a
SentenceTransformer using `MultipleNegativesRankingLoss` on
`(query, positive_passage)` pairs; `CrossEncoderTrainer` fine-tunes a
reranker using binary cross-entropy on `(query, passage, label)` triples.
Both pipelines consume synthetic pairs generated by `generate-synthetic-data`.

**Key design decision — in-batch negatives over hard negative mining**
Hard negative mining (selecting high-scoring BM25 false positives as
negatives) would produce more discriminative training signal. It was not
implemented because it requires a second retrieval pass over the corpus per
training query, which doubles data preparation time and complexity. In-batch
random negatives are simpler and effective for coarse domain vocabulary
adaptation. The limitation is documented below.

**Known limitation**
In-batch random negatives are unlikely to be semantically similar to the
query, making them easy to distinguish. The model therefore improves broad
domain vocabulary alignment but does not learn to discriminate between
closely related chunks within the same document section, which is exactly the
failure mode cross-encoder reranking is meant to address.

---

## 12. Relevance Feedback Loop (`knowledge/feedback/`)

**What it does**
Persists user relevance judgments (thumbs-up / thumbs-down) as structured
JSON records in `.knowledge/feedback.json`. The `tune-reranker` command reads
these records, filters by a minimum sample threshold (`--min-samples`,
default 5), and fine-tunes the cross-encoder reranker on the accumulated
judgments.

**Key design decision — explicit manual trigger with minimum sample gate**
Automatic retraining on every click was considered. It was rejected because
single-click online updates risk catastrophic forgetting and make the model
vulnerable to accidental clicks or transient user errors. Requiring explicit
invocation of `tune-reranker` and a minimum sample threshold ensures updates
happen on aggregated, representative batches. It also gives developers the
opportunity to inspect the feedback distribution before committing to a
fine-tuning run.

**Known limitation**
If users only submit positive ratings on helpful answers and do not rate
unhelpful ones, the feedback store accumulates no negative examples. Binary
cross-entropy fine-tuning on a dataset with only one class either fails
to converge or produces a trivial model that scores every passage positively.
