# Citation-fix sample: assistant answer review

This is an **assistant review of 54 saved answers**, not independent human
validation and not a semantic score for the full benchmark. It covers nine
development questions across all eight categories, three systems, and two runs.
The sample includes a known citation failure and is not a random or blind sample.

Before: run `89db52cc6c5cf91be22e`, revision `91db778`, saved in
`docs/research/benchmark-v2-full-generation.json`. After: run
`256fd6fed936abd05a30`, revision `327fc74`, saved in
[citation-repair-smoke.json](citation-repair-smoke.json).
Both recorded clean code at startup, the same dataset/index hashes, `llama3.2`,
temperature 0, seed 17, context 8192 and output limit 1024. The correction can add
one generation call, so total compute is not equal. All 27 selected cases in each
run have no recorded execution error.

I checked each answer against its reference answer, exact gold passages, and the
actual retrieved chunks. Every positive gold-evidence group was present in the
retrieved chunks for all selected cases in both runs. This makes the observed
false refusals generation failures rather than missing annotated retrieval.

| Question | Before | After |
|---|---|---|
| `v2_001` ownership | All systems correctly identify Field Autonomy and the duty coordinator but omit citations. Agentic also supplies a supported paging route. | All preserve the correct ownership answer. Agentic cites `dev_routing.md`; both baselines still omit citations. |
| `v2_002` preventive visit | All correctly say every six weeks. Agentic invents page 1 for unpaginated Markdown. | All answers have a valid `dev_contracts.md` reference supporting the six-week interval. Baselines unnecessarily copy a document heading. |
| `v2_011` feature flag | Baselines correctly say enabled; Agentic refuses despite that fact being in its context. | All correctly answer enabled with a supporting `dev_operations.md` citation. Self-correcting retains an unnecessary incomplete-context warning. |
| `v2_019` contract chain | Baselines correctly give 25 minutes and explain ORBIT-17 → SC-COPPER, but cite only the contract document. Agentic incorrectly refuses despite both required passages being present. | All give the correct SC-COPPER target. Agentic cites the contract; both baselines omit citations. All omit the explicit ORBIT-17 → SC-COPPER link and its `dev_routing.md` citation. |
| `v2_029` changed interval | All correctly compare 120 with 80 operating hours; Agentic omits citations. | All correctly compare the two versions and reference both supporting policy documents. |
| `v2_037` revenue growth | Baselines correctly calculate the absolute increase, EUR 0.6 million, with a supporting source. Agentic falsely says fiscal-2024 revenue is absent. | All correctly derive EUR 0.6 million and 20%. Agentic cites the underlying finance figures; both baselines omit citations. |
| `v2_045` ledger fields | All give the four required fields. Agentic invents page 1. | All give the correct fields with a supporting filename-only citation. |
| `v2_053` ambiguous telemetry | All correctly give the two conditional durations—45 days after collection and 18 months after incident closure—but omit citations. | All retain both alternatives. Agentic cites the supporting retention schedule; both baselines remain uncited. This satisfies the rubric's conditional-answer option without a follow-up question. |
| `v2_058` unavailable total | All appropriately refuse to invent a tax-inclusive international total. The catalog explicitly lacks an applicable tax rate. | All preserve the appropriate refusal. No citation is required for the plain refusal. Explaining which inputs are missing would be more helpful. |

For the eight evidence-bearing questions, Agentic changed from **two cited
answers, both with invalid page references, to eight cited answers with valid
references**. The baselines each changed from six to four cited answers. That
regression in citation coverage remains visible; the shared prompt change did
not solve citation omission for the baseline systems.

Reference validity still does not imply complete claim support. In `v2_019`,
`dev_contracts.md` supports SC-COPPER's 25-minute target, while `dev_routing.md`
establishes the asset's assignment. A valid contract citation alone does not show
the full two-document reasoning chain. The bounded repair detects missing or
invalid references, not missing citations for individual claims in an otherwise
cited answer.

The numerical gold label also deserves review: `v2_037` asks for “revenue growth”
without specifying a percentage, but its reference answer accepts only 20%.
EUR 0.6 million is a correct absolute increase. Do not treat the older baseline
answer as mathematically wrong solely because it differs from the reference.
Keep this frozen question unchanged for the current comparison; explicitly ask
for a growth rate or accept both interpretations in a future dataset version.

Next priorities are baseline citation omissions and complete source attribution
for multi-document answers. An independent reviewer should adjudicate these
annotations before publishing semantic correctness, faithfulness or citation-
support percentages. No full-run semantic fields were filled by this review.

## Subsequent isolation correction

The sample exposed reduced baseline citation coverage. Revision `7d4523c` restored
the original baseline/grading prompt bytes and preserved the successful AgenticRAG
prompt bytes. The subsequent full rerun uses that isolated change; the table above
remains an accurate record of the earlier sample and was not rewritten.
