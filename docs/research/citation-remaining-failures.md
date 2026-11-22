# Illustrative remaining failures after citation fixes

This assistant review examines only the completed AgenticRAG records for
`v2_013`, `v2_021` and `v2_023` in
[the saved complete result](benchmark-v2-citation-fix.json) from the final
rerun at revision `7d4523c`. It compares actual answers and retrieved chunks with
the frozen gold passages. It is not independent human review or a full-run
semantic score.

| Case | Observed answer | Remaining issue and supporting evidence |
|---|---|---|
| `v2_013`: Field Autonomy paging route | Correctly answers `field-auto-primary`. | Omits a citation despite one correction attempt. The retrieved `dev_routing.md` explicitly states that this route reaches the duty coordinator. The report correctly retains `missing_citations: true`. |
| `v2_021`: signed document for ORBIT-17's replacement battery | Correctly states that DG-9 shipments require a signed dangerous-goods declaration. | Omits citations despite one correction attempt and leaves the applicability chain implicit. Retrieved `dev_parts.md` identifies BP-42 as ORBIT-17's approved pack; `dev_shipping.md` assigns BP-42 to DG-9 and requires the declaration. A complete attributed answer should connect and cite both documents. |
| `v2_023`: support-contract payment for the replacement battery | Refuses: “I could not find the answer in the indexed documents.” | False refusal despite all three gold evidence groups being retrieved. `dev_routing.md` assigns ORBIT-17 to SC-COPPER, `dev_parts.md` identifies BP-42, and `dev_contracts.md` excludes consumable batteries and requires a separately approved parts order. The supported answer is no. The plain-refusal exemption skips citation repair; this is an answer-quality failure, not a missing-reference-format failure. |

None of these cases records an execution error. The first two retain useful
answers and expose their unresolved citation omissions; the bounded correction
does not guarantee compliance. The third shows why absence of a citation need
not mean the same thing as a citation failure: the refusal is syntactically
permitted but semantically wrong here.

The raw citation report's `citation_integrity: 1.0` with zero references is
vacuous. Use the zero citation count, `missing_citations` flag and benchmark's
null reference-validity score; do not report those three responses as perfectly
cited. Further work should separately address citation coverage, complete
multi-document attribution and false refusals. These examples alone do not
justify a full-benchmark answer-correctness percentage.
