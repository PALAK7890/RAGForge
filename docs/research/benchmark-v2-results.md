# Benchmark v2: verified retrieval results

Recorded 18 November 2026 against code revision `e05fa93e9e2762a7f213c746631d1c2dbc16e62a` with a clean working tree.

This is a real retrieval-only run over the supplied fictional corpus, not an LLM-generated-answer comparison. The source snapshot, gold labels and runner are versioned; the original benchmark is unchanged.

The index contains **24 documents and 54 chunks**. All 120 questions completed with **zero execution errors**. Retrieval denominators contain 110 evidence-bearing questions (including 10 clarification cases); the 10 unanswerable questions have no positive retrieval labels.

| Metric | Development | Evaluation (`held_out`) | All |
|---|---:|---:|---:|
| Passage requirement recall@5 | 0.9942 (n=57) | 0.9686 (n=53) | 0.9818 (n=110) |
| All requirements@5 | 0.9825 (n=57) | 0.9245 (n=53) | 0.9545 (n=110) |
| MRR@5 | 0.9596 (n=57) | 0.9717 (n=53) | 0.9655 (n=110) |
| Precision@5 | 0.2807 (n=57) | 0.2830 (n=53) | 0.2818 (n=110) |
| nDCG@5 | 0.9584 (n=57) | 0.9419 (n=53) | 0.9504 (n=110) |
| Mean latency (ms) | 189.40 (n=62) | 190.53 (n=58) | 189.95 (n=120) |

Recall 0.9818 means **98.18% macro-average evidence requirement coverage**, not 98.18% answer accuracy. All requirements@5 is **105/110 (95.45%)**. MRR and nDCG are bounded ranking scores. Precision counts supporting chunks against five available slots; it is not document-level precision. Read the [metric definitions](../../examples/benchmark_v2/README.md) before comparing scores.

## Actual incomplete-evidence cases

| ID | Split | Required-evidence coverage | Question |
|---|---|---:|---|
| v2_026 | dev | 66.67% | Which production asset battery requires a 9-month handling certificate, and what is that certificate called? |
| v2_081 | held_out | 50.00% | Does one replacement detector for LUMEN-8 cross its grant quotation threshold? |
| v2_082 | held_out | 66.67% | Where will the approved supplier ship a replacement detector for LUMEN-8 from? |
| v2_089 | held_out | 50.00% | Can a visitor host keep the LUMEN-8 external platform operating when sustained wind is 40 km/h in August 2026? |
| v2_095 | held_out | 66.67% | How does the NIGHTGLASS expansion bid differ from approved terms for quarterly hours and detector price? |

These failures remain in the aggregate. A filename hit would hide missing passages; v2 uses exact source quotes. Inspect the raw chunks before changing retrieval or labels. If evaluation failures guide tuning, this snapshot can no longer serve as a fresh final holdout.

## Artifacts and reproduction

- [Raw outputs, per-query metrics, category summaries and fingerprints](benchmark-v2-retrieval.json).
- [Blocked three-system generation run](benchmark-v2-generation-status.json).
- [Dataset, category counts, review rubric and reproduction guide](../../examples/benchmark_v2/README.md).

```bash
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 venv/bin/python -m knowledge.evaluation.benchmark_v2 \
  --index --split all --output reports/v2-retrieval.json
```

Real execution used cached SentenceTransformer and CrossEncoder models, FAISS, BM25 and RRF. Model-loading and a warmup search are excluded from measured per-query latency. The native-library thread limit was set to one. Timing is hardware-dependent and descriptive; no confidence intervals or speed superiority claims are made.

## Generation and statistical limitations

Ollama was unreachable, including an availability check outside the restricted environment. The comparison run records a blocker with zero scored queries, not zeros masquerading as model quality. Answer correctness, faithfulness, citation support and abstention accuracy remain **unmeasured**. The runner provides a run-bound human review workflow for semantic metrics once generation is available.

The development and evaluation document families are disjoint, but their public synthetic questions were authored together. Draft retrieval/integrity checks occurred during construction; the committed snapshot was frozen before the recorded run. This is not an independently reviewed or blind benchmark, and its high scores do not establish production robustness or reproduce the paper.

Full regression verification: **239 passed, 6 warnings**. New checks cover source drift, split isolation, genuine multi-document labels, exact passage ranking, alternative evidence, failed-generation credit, run-bound review scores, adapter configuration, and deterministic fixture regeneration. Targeted lint and whitespace checks passed.

## Subsequent full generation run

Ollama later became available. All 120 questions were run through the three systems.
See [the complete generation report](benchmark-v2-full-generation.md) for actual outputs, automatic scores,
execution failures and remaining semantic-review gaps. Earlier blocked-run artifacts
remain historical records.
