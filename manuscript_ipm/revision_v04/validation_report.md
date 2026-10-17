# Validation report (2026-10-17)

Environment: Linux container, Python 3.11.15, numpy/scipy 1.17.1/pandas 3.0.2/statsmodels 0.15.0, pdfTeX (TeX Live 2023), pandoc 3.1.3. CPU only; no GPU, no network access from the scripts. Repository state before the audit: commit f3c573a, clean tree.

## Commands and outcomes

| # | Command | Outcome |
|---|---|---|
| 1 | `python3 -m src.analysis.v03_analysis --out <scratch>/stats --tables <scratch>/tables` | PASS. 92 s. Output JSON equal to committed `v03_analysis.json` in every numeric field (\|Δ\| ≤ 1e-12); judge block absent by design (read from output dir); 9/9 CSV tables byte-identical |
| 2 | `python3 manuscript_ipm/revision_v04/scripts/audit_v04.py` | PASS. 19 s. Assertions: no duplicate query ids; decomposition identities residual < 1e-12 (both orderings, 24 comparisons); GEE models A–D converged with no warnings; pooled ORs equal original; κ equal original |
| 3 | rerun of #2 and `both5_ratio.py`, then `cmp` of all outputs | PASS. All six output files byte-identical (deterministic) |
| 4 | `python3 manuscript_ipm/revision_v04/scripts/both5_ratio.py` | PASS. One RuntimeWarning (division by zero in a few BM25 bootstrap resamples where ΔBoth@5 = 0); handled by excluding non-finite resamples; BM25 not reported |
| 5 | `python3 manuscript_ipm/revision_v04/scripts/make_v04_tex.py` | PASS. 38 macros, 3 tables |
| 6 | `python3 manuscript_ipm/revision_v04/scripts/numeric_report.py` (v0.3 text) | PASS. 201 macros; 69 independently recomputed, 0 mismatches |
| 7 | same on `revision_v04/latex` | PASS. 231 macros; 68 independently recomputed, 0 mismatches, 0 missing; 38 new macros traced to audit outputs |
| 8 | `latexmk -pdf manuscript.tex` in `revision_v04/latex` | PASS. 29 pages; 0 undefined references or citations; 0 Type 3 fonts; 1 overfull box (117 pt, pre-existing in v0.3, a wide appendix table) |
| 9 | `python3 manuscript_ipm/revision_v04/scripts/check_revised.py` | PASS. No hard failures: abstract 248 words (≤ 250), 7 keywords, 5 highlights ≤ 85 characters, no banned words, no "first"/novelty wording, 61 of 77 bibliography entries cited, all keys resolve |
| 10 | `python3 manuscript_ipm/revision_v04/scripts/make_docx_revised.py` | PASS. Word copy written |
| 11 | `git status` outside `manuscript_ipm/revision_v04/` | PASS. No changes to raw data, outputs, original scripts or the v0.3 manuscript |
| 12 | visual check of rendered pages 12 and 26 (new GEE, share and baseline tables) | PASS. Tables render; footer total corrected (v0.3 showed "Page 26 of 23") |

## What was not executed

- No GPU experiment (full-corpus readers, HNSW readers, HotpotQA full corpus). See `experiment_plan.md`.
- No independent re-implementation of the bootstrap, F1/containment or GEE: they were rerun with the same libraries. The new audit code reuses the original data loader and resampling index.
- The answer-string check for E12 hard negatives (O8): passage text not in the repository.
- Literature: two web searches on 2026-10-17 only; the earlier 2026-10-16 protocol was not repeated in full. Hitl et al. 2026 verified from the publisher's abstract page only (full text paywalled).
- No human or external review. The skeptical review is an AI pass.

## Acceptance criteria

| Criterion | Status | Evidence |
|---|---|---|
| Original manuscript and raw data preserved | ✔ verified | #11 |
| Study design matches artifacts | ✔ verified (two items U, listed) | study_specification.md |
| Every reported transfer ratio traceable | ✔ verified | transfer_ratio_audit.csv |
| Conditional decomposition correct and verified | ✔ verified (identities exact; formulation dependence reported) | transfer_decomposition_audit.csv |
| NQ transfer-range discrepancy resolved | ✔ resolved (scope) | M1 |
| Throughput inconsistency resolved | ✔ resolved (prose error corrected) | M2; throughput_ranges.csv |
| Statistical model and inference units audited | ✔ verified | statistical_audit.md §3–4 |
| Dataset and reader heterogeneity investigated | ✔ done (dataset: yes; reader: not detected) | Table gee; audit_results.json |
| HNSW claims restricted to measured outcomes | ✔ done | M3 |
| Random-passage conclusions do not confuse non-significance with equivalence | ✔ done | M4 |
| Corpus and generalisation claims scoped | ✔ done (direction error corrected) | M5; statistical_audit.md §7 |
| LLM judge represented accurately | ✔ verified (κ recomputed) | statistical_audit.md §8 |
| Citations and novelty verified to the extent possible | ◐ partly: earlier full protocol + two new searches; one new abstract-level source | audit_report.md §5 |
| Abstract, highlights, tables, figures, conclusion agree | ✔ verified by numeric report and reading | #7, #9 |
| New analyses reproducible | ✔ verified (deterministic rerun) | #3 |
| Unresolved issues disclosed | ✔ | skeptical_reviewer_report.md; experiment_plan.md |
| Final manuscript independently reviewed | ◐ AI review pass only, not independent | skeptical_reviewer_report.md |
| Validation report accurate | ✔ | this file |
