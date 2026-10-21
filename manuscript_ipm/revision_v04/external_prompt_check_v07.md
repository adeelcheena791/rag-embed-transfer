# Check of the external review prompt (21 October 2026)

An external review prompt listed eight issues. Each was checked against the current
manuscript (revised_manuscript.pdf) and the committed outputs; the checks below were rerun on this date.

| # | Issue raised | Current manuscript | Evidence (rerun today) | Action |
|---|---|---|---|---|
| A | Transfer ratio definition; 48-70% decomposition share | Ratio defined as descriptive, baseline- and metric-dependent (Introduction, Method). The 48-70% share appears only in Results for HotpotQA, with the symmetric form (44-64%) and Table 27; NQ shares are withheld because their intervals are wider than 1.4. Not in abstract or highlights. | independent_check_v05: identity residual 5.6e-17 over 22 ratios; ratio-rule decisions agree for 24 pairs | none |
| B1 | NQ range 0.16-0.27 vs BM25 0.37 | Abstract and highlights state the range for the five dense encoders; Results explains that the BM25 ratio relates an EM loss to a Hit@5 loss. | outputs/transfer_ratio_audit.csv, 0 differences | none |
| B2 | bge-base and e5-base-v2 throughput "overlap" | Text now says no two encoders had overlapping ranges (502 [501-503] vs 493 [491-496]). | outputs/throughput_ranges.csv: no pair overlaps | none |
| C | Pooled GEE without interactions | Wald tests of retriever x dataset and retriever x reader terms reported; dataset-specific odds ratios from the interaction model (post hoc, labelled). | Table 10; statistical_audit.md | none |
| D | HNSW "no measurable cost in answers" | Wording removed; Table 15 note and text say no reader was run on HNSW results and the check covers retrieval and latency only. | Results 5.8, Limitations | none |
| E | Random passages "harmless" (p = 0.71) | Text gives the difference with its CI, states that no equivalence margin was registered, and that the interval is not evidence of no effect. | Table 12 | none |
| F | Full-corpus scope | Body, Discussion and Limitations name the four retrievers of Phase 3. Abstract, one highlight and one contribution line spoke of "the dense ratios"/"NQ ratios" in general. | v04_full_analysis.json | **fixed**: abstract, highlight and Introduction now name the two dense encoders rerun on the full corpus |
| G | Novelty | No "first" claim; scoped "we found no study that reports ... for one controlled setting". All 61 cited references checked (reference_check_v06.md). | check_revised: novelty wording none | none |
| H | LLM judge, kappa = 0.127 | Judge results excluded under the registered rule (kappa < 0.2); kappa with CI and raw agreement reported; Limitations repeat it. | Table 17 | none |

Commands rerun: tools/independent_check.py, scripts/independent_check_v05.py, scripts/independent_check_v06.py
(0 mismatches), scripts/numeric_report.py latex (320 macros, 0 mismatches), scripts/check_revised.py (no hard failures;
abstract 249 words; highlights <= 81 characters), pytest (15 passed), latexmk, make_docx_revised.py.
These are AI-run checks, not human peer review.
