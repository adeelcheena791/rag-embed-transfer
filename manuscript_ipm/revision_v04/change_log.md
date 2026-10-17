# Change log: v0.3 (manuscript_ipm/latex, commit f3c573a) → v0.4 (manuscript_ipm/revision_v04/latex)

The v0.3 manuscript, raw data, experiment outputs and original scripts are unchanged. v0.4 is a separate copy. "Scientific" changes alter what a claim says; "editorial" changes alter scope labels, wording or layout only. No reported v0.3 number changed value.

## Scientific corrections

| # | Where | v0.3 | v0.4 | Why | Evidence |
|---|---|---|---|---|---|
| S1 | Results §5.6 (Encoding cost) | throughput ranges "did not overlap for any pair of encoders except bge-base and e5-base-v2" | no two encoders' ranges overlap; bge-base 1.8% faster than e5-base-v2 at the median | the five-run ranges (501.0–502.7 vs 490.6–495.9 passages/s) do not overlap; the analysis flag excluded this pair a priori and the prose misread the exclusion | outputs/throughput_ranges.csv |
| S2 | Discussion §6.3 | "Approximate search can replace exact search at no measurable cost in answers" | approximate search returned nearly the same top five at a fraction of the latency; effect on answers not measured | no reader was run on HNSW results | paper_inputs/v03_nq_s4/raw/hnsw.jsonl; no HNSW answer files |
| S3 | Results §5.6 (Sub-corpus) | BM25-mined negatives "may make the sub-corpus easier for BM25 relative to the dense encoders" | the construction favours the dense encoders relative to BM25; the E9 drops agree | the sentence reversed the direction implied by src/data/subcorpus.py and by E9 (BM25 −0.002 vs dense −0.083 to −0.086) | E9 in v03_analysis.json |
| S4 | Results §5.4 | pooled GEE "gave the same picture" | registered pooled model kept; retriever × dataset interaction reported (χ²₆ = 179, p = 5.2 × 10⁻³⁶) with dataset-specific odds ratios; reader interaction not detected (p = 0.55) | the pooled odds ratio averages opposite effects on the two datasets | outputs/audit_results.json gee; Table gee (rebuilt) |
| S5 | Results §5.5; Discussion §6.2 | "Random passages were harmless"; "random passages did not hurt" | no detectable change, with difference +0.006 [−0.016, 0.026] and an explicit statement that this is not equivalence | non-significance is not evidence of no effect; no margin was registered | mechanism.gold_only_vs_MR |
| S6 | Method §3.1, Results §5.3, Discussion §6.1, Abstract | decomposition share 48–70% "exact"; ordering not stated | formulation named; symmetric range 44–64% and bootstrap intervals (26–76%) added; stated as accounting identity, not causal; reason for not reporting NQ shares given | the split is not unique and the NQ shares are unstable | outputs/transfer_decomposition_audit.csv; Table share (new) |
| S7 | Method §3.1, Results §5.2, Discussion §6.4, Limitations, Conclusion, Abstract | "ratios against BM25 would differ" (no numbers) | BM25-baseline sensitivity analysis: NQ 0.29–0.35, HotpotQA 0.25–0.75; dataset contrast holds for 4 of 5 encoders | resolves open review item D1; T describes a retriever pair | outputs/transfer_ratio_alt_baselines.csv; Table altbase (new) |
| S8 | Introduction | "A ratio of 1 means every extra question with a relevant passage … also gains a correct answer" | a ratio of 1 means EM rises by as many points as Hit@5; the ratio compares averages, not questions | the old sentence implied a query-level reading that the method section rules out | — |
| S10 | Abstract, highlight 2, Results §5.3, Discussion §6.1, Conclusion, Table altbase | HotpotQA "transfer was larger" (0.49–0.93) presented as a dataset finding; denominator artefact only hypothesised | post hoc check added: with Both@5 as the denominator, HotpotQA dense-encoder ratios are 0.29–0.35, much closer to NQ; the large Hit@5-based ratios "largely reflect the denominator"; conclusion rewritten accordingly | the headline contrast between datasets depends on the retrieval metric; reporting only the Hit@5 version would overstate it | outputs/transfer_ratio_both5_hotpotqa.csv; scripts/both5_ratio.py |
| S11 | Conclusion | "Qwen3-Embedding-0.6B gave lower EM" | "… gave lower EM on HotpotQA" | on NQ no difference was detected (Holm p ≥ 0.27; GEE OR 0.952 [0.906, 1.001]) | pairwise_em; audit_results.json |
| S9 | Related work | — | added Hitl et al. 2026 (Data Technologies and Applications), which concludes that retrieval quality was the main determinant of end-to-end RAG quality | closest new work found in the 2026-10-17 search; counter-evidence to a strong gap claim | publisher abstract page (DOI 10.1108/DTA-07-2025-0609), abstract-level only |

## Scope and wording (editorial)

| # | Where | Change |
|---|---|---|
| E1 | Abstract, highlight 1, Discussion §6.1, Conclusion | the NQ range 0.16–0.27 and "at most 0.27" now say "dense encoders" (BM25's 0.37/0.36 is a ratio of losses) |
| E2 | Abstract | "one upper bound reached 1.10" → "three of five upper bounds reached 1" (Llama–HotpotQA: 1.000, 1.000, 1.096) |
| E3 | Abstract | judge agreement "with human labels" → "with annotator labels" (one annotator) |
| E4 | Highlights 2–3 | "its intervals reached 1" → "some upper bounds reached 1"; highlight 3 prefixed "Post hoc" |
| E5 | Introduction contributions | "larger transfer driven by the second supporting passage" → "that goes with retrieving the second supporting passage" (not causal); first contribution mentions the second baseline |
| E6 | Method §3.2 | sub-corpus wording: "top-d BM25 passages of every judged query"; all retrievers search the same sub-corpus; use of evaluation labels stated |
| E7 | Method §3.6 | cluster structure of the GEE stated; post hoc interaction test described |
| E8 | Results §5.6 (HNSW), Conclusion | retrieval-only scope of the HNSW check stated; RQ4 sentence limited to what was tested |
| E9 | Limitations | baseline and ordering dependence, post hoc status of the new analyses, GEE assumption, HNSW answers not measured, random-passage control not equivalence, sub-corpus bias towards dense encoders |
| E10 | Conclusion | recommendation adds "with the baseline stated"; future work adds readers on approximate-search results |
| E11 | Appendix | Tables share and altbase added; Table gee replaced by a version with pooled and dataset-specific columns |
| E12 | Abstract | trimmed to 250 words to absorb the additions (no finding removed) |
| E13 | Layout | footer "Page n of N" showed a wrong total in v0.3 (the class records it before deferred appendix floats); v0.4 uses the lastpage package |

## New files

- `latex/numbers_v04.tex`, `latex/tables/tab_share.tex`, `latex/tables/tab_alt_baseline.tex`, `latex/tables/tab_gee.tex`: generated by `scripts/make_v04_tex.py` from audit outputs.
- `latex/references.bib`: one entry added (hitl2026comparative). The generator `manuscript_ipm/tools/make_bib.py` was not updated; port the entry if v0.4 replaces v0.3.
