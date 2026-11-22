# Benchmark v2: citation correction rerun

All **360 cases (120 questions × three systems)** have saved outputs, with status `complete`. Code revision: `7d4523c43994fa9d33ab95ca1041930de53d6b1b`. Combined result ID: `resumed-ba64ac46da24943ec497`. The working tree was clean at startup.

Execution stopped after 169 cases. Their real answers, chunks and generation scores were recovered from the incremental journal; the remaining 191 cases ran in separately saved question segments. The resumed segments verified the same model fingerprint and used the same code, corpus and settings. Exact per-case latencies and the original in-memory run ID were not saved for the interrupted prefix. Those timings are null, not reconstructed from cumulative elapsed time. The combined ID and segment provenance are recorded in the JSON.

The corpus, questions, index fingerprints, model and generation options match the original run. The local model fingerprint checked for this run is `llama3.2:latest`, SHA-256 `a80c4f17acd55265feec403c7aef86be0c25983ab279d83f3bcd3abbcb5b8b72` (3.2B, Q4_K_M).

## Changes and validation

Agentic evidence now supplies copyable source labels, includes page numbers only when present, and avoids a nested answer prompt. Missing or invalid citations trigger at most one correction using the same evidence. Invalid corrections do not silently replace the answer; missing citations and correction failures remain visible. Malformed location suffixes are rejected. These checks validate reference identity/location, not semantic claim support.

Baseline and grading prompts were restored byte-for-byte after the development sample exposed a regression. Agentic prompt bytes match the successful 27-case smoke run. An earlier full attempt was stopped after eight cases to make that correction; it is not included in this completed experiment.

All 120 Standard and all 120 self-correcting answers match the original saved answers exactly. The final regression suite passes **255 tests**, with six existing dependency deprecation warnings. After preserving the experiment, revision `dce1361` corrected the `abstained` flag for canonical model refusals, including refusals returned by citation correction. That flag-only change leaves prompts and generated text unchanged; the benchmark already recognized this refusal through its text heuristic. The raw experiment remains the output of `7d4523c`.

## Measured results

| Metric | Standard RAG | Self-correcting RAG | AgenticRAG |
|---|---:|---:|---:|
| Passage recall@5 | 98.18% (n=110) | 96.89% (n=110) | 98.18% (n=110) |
| All requirements@5 | 95.45% (n=110) | 92.73% (n=110) | 95.45% (n=110) |
| Mean latency, resumed cases only (seconds) | 3.40 (n=63) | 9.50 (n=64) | 4.71 (n=64) |
| Answers with citations | 41 | 42 | 83 |
| Answers with invalid references | 1 | 1 | 0 |
| Answers without citations (all 120) | 79 | 78 | 37 |
| Evidence-bearing answers without citations (110) | 70 | 68 | 27 |
| Citation correction attempts | 0 | 0 | 70 |
| Citation-reference validity, cited answers only | 97.56% (n=41) | 97.62% (n=42) | 100.00% (n=83) |
| Execution errors | 0 | 0 | 0 |

## Before/after AgenticRAG citations

Using the **same current validator** on both sets of saved answers, citation-reference validity changed from **3.85% (n=39)** to **100.00% (n=83)**. Answers containing invalid references changed from 38 to 0; evidence-bearing answers without citations changed from 71 to 27.

The historical report retains its original 6.41% score. Rescoring rejects one previously credited malformed reference in `v2_022`: `[dev_shipping.md, p.c43d4c3c2fc5]`. Original artifacts remain unchanged. The new report does not use that permissive result as its comparison baseline.

## Interpretation and limits

The 110 evidence-bearing questions include clarification cases; ten unanswerables are excluded from retrieval denominators. Citation validity averages only answers containing citations, so missing-citation counts are shown alongside it. A valid filename does not prove complete attribution, correct reasoning or source support.

The [assistant review of 54 saved sample answers](citation-sample-review.md) checks actual answers and source evidence. It found repaired false refusals, incomplete attribution for a multi-document answer, and an ambiguous numerical gold label. It is not independent human validation and does not supply semantic scores for the full run. Full-run correctness, faithfulness and citation-support fields remain null.

[Illustrative remaining failures](citation-remaining-failures.md) include a correct but uncited paging route, an incompletely attributed multi-document answer, and a false refusal despite all required passages being retrieved. The bounded correction does not guarantee citation coverage or answer correctness.

The correction may add one generation call for AgenticRAG. Total compute is not equalized. Latency means cover only 63 Standard and 64 cases for each other system from the resumed portion; this subset contains different questions from the missing prefix. Do not compare these subset means with the original whole-benchmark latency averages. Each resumed question warmed the shared retrieval models before measured calls. No complete-rerun speed claim is available. The public synthetic benchmark has been inspected during development and is not a fresh blind holdout.

## Artifacts and reproduction

- [Raw completed run](benchmark-v2-citation-fix.json) and [blank run-bound human-review template](benchmark-v2-citation-fix-review.json).
- [Original completed run](benchmark-v2-full-generation.json), preserved unchanged.
- [Unmodified interrupted prefix journal](benchmark-v2-citation-fix-prefix.jsonl), with the recovery hash and missing-timing disclosure in the combined result.
- [27-case correction smoke run](citation-repair-smoke.json).
- [New project-documentation examples](../../examples/project_docs/README.md) and [their 12-case retrieval results](project-docs-retrieval.json). All ten evidence-bearing examples retrieved every required source file; two unanswerables receive no retrieval score. The [offline exact-passage check](project-docs-passage-coverage.json) finds 85% macro passage coverage and all required passages for 8/10 questions. These are retrieval measures, not answer correctness.
- [Live generation review for all 12 documentation examples](project-docs-answer-review.md), with raw answers and evidence. There were zero execution errors, but the review exposes false refusals and reasoning failures. This application check uses Ollama runtime defaults and is separate from the fixed-settings benchmark.

```bash
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 python -m knowledge.evaluation.benchmark_v2 \
  --index --split all --systems standard self_correcting agentic \
  --workspace .knowledge/citation-rerun \
  --output reports/citation-rerun.json --review-template reports/citation-review.json
```
