# AgenticRAG attribution and implementation scope

Reference: [AgenticRAG: Agentic Retrieval for Enterprise Knowledge Bases](https://arxiv.org/abs/2605.05538),
Susheel Suresh, Hazel Mak, Shangpo Chou, Fred Kroon, Sahil Bhatnagar, 2026.
The paper describes a reasoning LLM using SEARCH, FIND, OPEN and SUMMARIZE over
enterprise search. Its abstract reports BRIGHT Recall@1 of 49.6%, WixQA factuality
of 0.96 and FinanceBench answer correctness of 92%, plus a 5.9x tool-use ablation
gain. These are paper results, not measurements of this repository. This note
is an implementation mapping, not an independently complete paper reproduction.

## What the code actually implements

- SEARCH wraps local FAISS, BM25, reciprocal rank fusion and optional CrossEncoder
  reranking through RAGService. It accepts multiple queries and metadata filters.
- FIND performs literal, escaped substring matching over indexed chunks of a
  named document. It is not a general regex interface or full original-document search.
- OPEN selects indexed chunks by page, chunk index or overlapping offsets with
  a strict character cap. It does not reconstruct missing original sections or
  return an exact clipped character span. Unavailable offsets cannot be invented.
- SUMMARIZE is separately callable with provenance references and an approximate
  character-based budget. It is not scheduled or consumed by the default controller.
- The default LangGraph controller follows deterministic heuristics: SEARCH, optionally
  FIND in the first discovered document, optionally OPEN, then synthesis. Empty
  search can trigger a relaxed retry. Comparative subquestions are placeholders
  and are not independently investigated in this mode. The opt-in `--planner model`
  mode asks the LLM for a validated next action after each tool result. It can
  reformulate searches, investigate another discovered document, answer, abstain,
  or ask for clarification. Decisions and errors are retained in `decision_history`.
  This adds model-directed investigation; it does not reproduce the paper's
  complete navigation, context management, or experimental protocol.
- In heuristic mode, nonempty retrieved evidence is treated as sufficient. This does
  not prove relevance, detect contradictory sources, or guarantee abstention on
  unanswerable queries. A step cap is enforced; wall-clock cancellation, per-query
  token accounting are not implemented. Model mode bounds both tool calls and
  planner calls, gives each planner request a timeout, and stops invalid or repeated
  actions explicitly. Model judgments about evidence sufficiency can still be wrong.
- Citation validation checks reference identity and available page/span bounds.
  It does not verify semantic support for each claim. Missing or invalid citations
  trigger one bounded generation correction. A citation-free response has vacuous integrity of 1.0 and
  must not be interpreted as fully grounded.
- Failed generation retains an extractive fallback and an explicit error. That
  degraded output cannot count as a successful benchmark answer.
- The v2 API runs blocking work in a worker thread. SSE uses framed data events
  with progress emitted after each tool finishes. Disconnect cancellation stops
  event delivery but cannot forcibly interrupt running synchronous model work.

## Adaptations and limitations

Local hybrid retrieval and Ollama are engineering adaptations. EvidenceStore,
citation metadata, CLI/API integration and the repository question set are local
extensions. The standalone tracer is not wired into every controller action;
controller history still includes inputs and snippets and should be treated as
potentially sensitive. No measured superiority or exact paper reproduction is
established. Autotuning and relevance feedback remain existing features rather
than a verified integrated agentic research experiment.

Filenames remain the public document identifiers; duplicate basenames across
folders can be ambiguous. Original-source navigation, stable corpus-wide IDs,
reliable multi-hop planning and independently reviewed evidence labels remain
future work. The historical benchmark describes heuristic mode; its scores do not
validate the opt-in planner. See [evaluation status](experimental-results.md).
