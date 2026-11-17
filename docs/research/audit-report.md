# RAGForge implementation audit — 2026-11-17

## Repository and evidence

Located `/Users/PalakNst/Desktop/cli_ai` by searching Desktop rather than assuming
its path. Its working tree was clean on `feature/agenticrag-research`, commit
`5688464`. GitHub's research branch was independently checked and matched that
commit. Work was prepared in an isolated local clone on
`feature/agenticrag-audit`; no history was rewritten or private runtime data committed.
Read the referenced conversation and supplied execution-log attachment. The
historical README was assessed through the tracked README and the
conversation's review, rather than treating prior assistant claims as evidence.

The dataset genuinely contains 120 questions and regenerates byte-for-byte.
The old 198-test result also reproduced. The claimed 91.7% result, baseline
comparisons, citation scores and ablation deltas lack saved experimental evidence
and cannot be reproduced by the old ten-question entry points. Their tables and
unsupported interpretations have been withdrawn. The reference paper exists;
its abstract's reported metrics are distinct from this project's measurements.

## Changes

1. Preserve unrelated/unchanged documents during service indexing, replace
   modified documents without retaining their old chunks, keep vector/chunk order
   aligned, use the selected workspace's manifest, and expose indexing errors.
   Retained chunks are re-embedded during updates; this favors correctness over
   update speed and is not a transactional or concurrent indexing redesign.
   Declare missing runtime dependencies `ollama` and `rank-bm25`.
2. Require exact document identifiers for FIND/OPEN, return no unrelated chunks
   for out-of-range selectors, enforce OPEN's hard character cap, pass the
   controller's OPEN budget and FIND offsets, and honor SEARCH's rerank option.
3. Resolve citations against exact evidence/document/path identities and actual
   page/span metadata through a shared validator. Preserve distinct passages in
   evidence deduplication. Citation-free responses still have vacuous integrity;
   reference validation is not semantic claim verification.
4. Surface structured tool failures and failed generation instead of silently
   treating extractive fallbacks as successful model output. Validate controller
   step budgets and size LangGraph's recursion allowance for the chosen budget.
   Preserve existing fallback answers with an explicit error field.
5. Emit properly framed v2 SSE events after each actual tool completes, with a
   terminal error event if the worker fails. Keep v1 behavior compatible. Emit
   CLI JSON without Rich markup/wrapping, validate CLI budgets, and show errors.
6. Compare exact filenames, exclude unanswerables from retrieval denominators,
   deny quality credit for failed runs, expose error rates, and save per-question
   outputs plus dataset/chunk fingerprints and commit/split metadata. Reject
   missing corpus sources before inference. Run the entire selected split by
   default instead of silently slicing ten questions. Remove production use of
   unittest mocks from ablations, apply single-query constraints on retries too,
   and reject the inactive no-SUMMARIZE experiment.
7. Replace research claims with an accurate account of the heuristic controller,
   experimental dataset, reproduction steps and remaining limits.

## Verification

- Baseline: `HF_HUB_OFFLINE=1 venv/bin/python -m pytest tests/ -q`:
  **198 passed, 6 warnings**.
- Final regression suite using the existing desktop environment from the audit
  checkout: **225 passed, 6 warnings**, 5.14 seconds. Added 27 regression cases.
  API unit tests use isolated workspaces, avoiding accidental model loading or
  changes to the user's real index.
- Targeted import/unused-name lint and whitespace checks were run for changed
  backend/evaluation/test files. These do not imply a clean repository-wide lint
  or type-check baseline.
- Real public-document ingestion: **6 files / 137 chunks**, followed by actual
  SentenceTransformer → FAISS + BM25 → RRF → CrossEncoder retrieval using cached
  models. No mocked search results in this check.
- Real CLI: JSON parsed successfully, 3 tool steps, 11 evidence records, explicit
  generation error because Ollama was unreachable.
- Real FastAPI v2 POST: HTTP 200 with 11 evidence records and explicit generation
  error. Actual v2 stream: start → 3 tool_executed events → complete, with framed
  `data:` events and the same generation error. These verify retrieval and
  degraded integration behavior, not successful LLM answers.
- Held-out benchmark preflight correctly rejected missing indexed sources:
  app.py, graph.py, hybrid.py, main.py, tools.py. The full dataset also references
  config.yaml absent from a clean tracked checkout.
- Real-model checks used `OMP_NUM_THREADS=1`: a native-library segmentation fault
  occurred with the default thread setup when earlier API tests implicitly used
  the generated index. Isolation fixed the unit-suite behavior; the installed
  FAISS/PyTorch/macOS threading combination still needs investigation before
  claiming unrestricted concurrent model-serving reliability.

## Blockers and remaining scope

Ollama was unreachable both inside and outside the restricted environment.
Therefore full answer-generation comparisons and ablations were not run. Corpus
coverage and gold-label review are also required. No replacement benchmark
numbers are asserted.

The controller still uses heuristics, not an LLM choosing tools. It does not
schedule SUMMARIZE or genuinely investigate its placeholder subquestions. It
cannot guarantee relevance, contradictions detection or correct abstention.
OPEN returns overlapping indexed chunks rather than exact original-source spans.
Duplicate basenames may be ambiguous. Full wall-clock enforcement, prompt/output
budgets, token/cost accounting, semantic citation support, statistical inference,
concurrent index transactions and complete tracer integration remain outside
these verified fixes. Existing autotuning, feedback and training functionality
was retained and covered by the regression suite.

## Commits

- `30c849c`: preserve incremental indexing and declare runtime dependencies.
- `0e2c599`: navigation/citation boundaries, error visibility and live SSE framing.
- `988065c`: evaluation credibility, portable generation, reproducibility records
  and regression tests.
- Documentation commit: withdraw unsupported results and document this audit.
