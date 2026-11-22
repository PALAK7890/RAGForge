# Benchmark v2: full local generation run

This run executed all **120 questions through three systems (360 cases)** using local Ollama `llama3.2`. It supersedes the earlier availability blocker for this later run; the old blocked-run artifact is retained as historical evidence.

Code revision: `91db7782279b2c55f08e79dbef756ea91538e2de`. Run ID: `89db52cc6c5cf91be22e`. Status: `complete`. The working tree was clean when execution began. The external monitor retained incremental outputs without changing scoring or model behavior.

| Metric | Standard RAG | Self-correcting RAG | AgenticRAG |
|---|---:|---:|---:|
| Passage requirement recall@5 | 98.18% (n=110) | 96.89% (n=110) | 98.18% (n=110) |
| All requirements@5 | 95.45% (n=110) | 92.73% (n=110) | 95.45% (n=110) |
| All collected evidence coverage | 98.18% (n=110) | 96.89% (n=110) | 98.18% (n=110) |
| MRR@5 | 0.965 (n=110) | 0.971 (n=110) | 0.954 (n=110) |
| Precision@5 | 28.18% (n=110) | 27.64% (n=110) | 28.18% (n=110) |
| nDCG@5 | 0.950 (n=110) | 0.946 (n=110) | 0.946 (n=110) |
| Mean end-to-end latency (seconds) | 14.626 (n=120) | 24.875 (n=120) | 13.533 (n=120) |
| Citation-reference validity (cited answers only) | 97.56% (n=41) | 97.62% (n=42) | 6.41% (n=39) |
| Abstention accuracy — phrase/flag heuristic | 89.17% (n=120) | 90.00% (n=120) | 55.83% (n=120) |
| Execution errors | 0 | 0 | 0 |
| Answers without bracket citations | 79 | 78 | 81 |
| Answers with invalid citation references | 1 | 1 | 37 |
| Correct abstention detections on 10 unanswerable questions (heuristic) | 9/10 | 10/10 | 10/10 |

## Interpretation

Execution success means the pipeline returned without a recorded exception; it does not establish that the answer is correct. Passage metrics use 110 evidence-bearing questions (including clarification cases). Ten unanswerable questions are excluded from retrieval denominators. Citation-reference validity averages only answers that cite something; missing-citation counts are therefore reported alongside it. Reference validity checks existence/location, not semantic support.

Abstention detection is a phrase/flag heuristic, not a reviewed judgment. Exact-answer-match proxies and category breakdowns are retained in the raw JSON. None of these automatic scores substitutes for semantic answer correctness, faithfulness or citation-support review. Those fields remain null. The supplied review template is run-bound and does not contain fabricated judgments.

All systems share the frozen synthetic corpus, local model and per-search depth. They use different context-selection and iteration policies. Agentic fixed-K ranks use evidence collection order; the self-correcting baseline uses its final retrieval. Total compute is not equalized. Latency is descriptive for this local sequential run, not a statistically controlled performance claim.

## Observed failures

| Question | System | Issue | Answer excerpt |
|---|---|---|---|
| v2_002 | agentic | At least one bracket reference fails identity/location validation | According to the provided evidence, SC-COPPER includes a scheduled preventive visit every 6 weeks. [dev_contracts.md, p.1] |
| v2_003 | agentic | At least one bracket reference fails identity/location validation | The stated warranty period for unopened BP-42 packs is 12 months from dispatch. [dev_parts.md, p.1] |
| v2_005 | agentic | At least one bracket reference fails identity/location validation | According to the evidence, a rover that has failed an insulation check must go to Bay Q2 for quarantine. [dev_safety_2026.md, p.1] |
| v2_006 | agentic | At least one bracket reference fails identity/location validation | According to [dev_records.md, p.1], Records Operations owns the retention schedule. A legal hold suspends deletion until Legal explicitly releases the hold. |
| v2_008 | agentic | At least one bracket reference fails identity/location validation | The approved ledger listed 36 BP-42 packs available on 4 February 2026. [Document: dev_parts.md, p.1] |
| v2_009 | agentic | At least one bracket reference fails identity/location validation | The fiscal 2025 year-end order backlog was EUR 1.1 million. [Document: dev_finance.md, p.335f2c5a2f76] |
| v2_012 | agentic | At least one bracket reference fails identity/location validation | The gateway readiness path is GET /ready/rover [dev_operations.md, p.1].   Liveness is reported by GET /health/rover [dev_operations.md, p.1]. |
| v2_013 | agentic | At least one bracket reference fails identity/location validation | According to the evidence, the paging route that reaches the Field Autonomy duty coordinator is "field-auto-primary". [dev_routing.md, p.1] |

The table is illustrative, not the complete failure list. All original answers, errors, supporting chunks and metrics are retained below. Review any claimed page numbers against the Markdown corpus; those files have no page-number metadata.

## Reproduction and artifacts

- [Raw full-run results](benchmark-v2-full-generation.json).
- [Blank semantic-review template](benchmark-v2-full-generation-review.json).
- [Dataset, metric definitions and review rubric](../../examples/benchmark_v2/README.md).

```bash
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 venv/bin/python -m knowledge.evaluation.benchmark_v2 \
  --index --split all --systems standard self_correcting agentic \
  --output reports/v2-full-generation.json --review-template reports/v2-full-review.json
```

The full corpus and questions are authored fictional fixtures, not independent real-world enterprise data. Draft checks exposed the evaluation split during construction, so this is not a blind held-out study. No claim of research-paper reproduction or general superiority follows from these results.

## Development and evaluation breakdown

| Split | System | Cases | Passage recall@5 | All requirements@5 |
|---|---|---:|---:|---:|
| dev | standard | 62 | 99.42% (n=57) | 98.25% (n=57) |
| dev | self_correcting | 62 | 96.49% (n=57) | 92.98% (n=57) |
| dev | agentic | 62 | 99.42% (n=57) | 98.25% (n=57) |
| held_out | standard | 58 | 96.86% (n=53) | 92.45% (n=53) |
| held_out | self_correcting | 58 | 97.33% (n=53) | 92.45% (n=53) |
| held_out | agentic | 58 | 96.86% (n=53) | 92.45% (n=53) |

The low AgenticRAG citation-reference score is largely explained by invented page markers such as `[dev_contracts.md, p.1]` for unpaginated Markdown sources. Of 39 AgenticRAG answers containing bracket citations, 37 contain at least one invalid reference; another 81 answers contain no bracket citations. These are observed reference-format/location failures, not a semantic correctness assessment.

The full run produced nonempty answers for all 360 cases, and all semantic review fields remain null. This verification checks completeness and artifact consistency; it does not replace human review or establish answer quality.
