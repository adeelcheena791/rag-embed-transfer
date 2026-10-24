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

---

# Change log: v0.4 → v0.5 (2026-10-17)

Snapshot of v0.4 before these changes: git tag `v0.4-audit` (local) and branch `snapshot/v0.4-audit` on GitHub (commit 56ef394). No raw output, registered design document or v0.3 file was changed.

## Errors corrected (confirmed against unrounded outputs)

| # | Where | Before | After | Cause | Evidence |
|---|---|---|---|---|---|
| R1 | Abstract, Results §5.2 | "no NQ upper bound exceeded 0.45" | "largest NQ upper bound 0.455" | reporting/rounding: the bound is 0.4545 (Qwen3-Emb, Qwen), printed as 0.45 | v03_analysis.json; independent_check_v05.md |
| R2 | Decomposition wording (abstract, highlights, results, discussion, conclusion, Table share) | 48–70% "of the HotpotQA transfer to higher EM given a hit" | the share belongs to the two conditional terms together; hits and misses terms now reported separately (hits term alone 48–71%; misses term −0.003 to +0.000) | reporting: the share is h_R Δa + (1 − h_R) Δb, not h_R Δa | decomposition_components.csv; tab_components |
| R3 | Results §5.7, Table hnsw | prose cited top-5 overlap, table showed only top-100 overlap without saying so | table shows both (Ov@5, Ov@100) with definitions; prose gives both; caption says Hit@5 here is over all 3,452 judged queries | reporting: both quantities were computed (hnsw.jsonl) | paper_inputs/v03_nq_s4/raw/hnsw.jsonl |
| R4 | Appendix prompts | A3 prompt missing; user-message layout approximated | exact A1, A2, A3 system prompts and user-message layout from src/generation/prompts.py; all 131,855 generation records carry a matching prompt hash | documentation gap | prompts.py; generation records (prompt_hash) |
| R5 | Table data | "Unique gold passages per judged query: 1.86" for HotpotQA | each HotpotQA query has exactly 2 distinct gold passages; 13,783 distinct passages over 7,405 queries because passages are shared; NQ 1.22 (range 1–4) | reporting: a corpus-level count was labelled as a per-query count | hp_joined.jsonl; NQ qrels |
| R6 | Table predictions | NQ P1 rows: "6 ratios" (BM25 counted) | "5 dense-encoder ratios", as registered; verdicts unchanged | implementation scope: v03_analysis.py checked all reportable ratios incl. BM25; the registered P1 covers dense encoders (addendum v0.3) | design/addendum_v0.3.md; independent_check_v05.md |
| R7 | Setup, Appendix | "the readers ran with the same stack in every session"; one software line | adds that the two encoding sessions used PyTorch 2.10.0/2.11.0, CUDA 12.8, Transformers 5.0.0; new Table env from the environment logs | documentation gap | paper_inputs/**/env*.json |
| R8 | Results, Method, Discussion | "gold bound(s)", "two reference conditions bound each cell" | "gold-passage reference"; references are not hard bounds | wording | — |
| R9 | Results, Discussion | "cross-encoder … hurt e5-base-v2"; "helped only the weak BM25" | lowered e5-base-v2's Hit@5 (p = 2.6e-12); no EM difference detected (p = 0.069) | wording vs non-significant EM change | v03_analysis.json reranker |
| R10 | Title, abstract, RQ4, Results, Discussion | "cost" | "retrieval-stage cost" (only encoding, search and index were measured; reader tokens only for k) | scope wording | — |
| R11 | Predictions throughout | "held/failed" | "criterion met/not met"; P1 notes that a bound of exactly 1.000 does not meet it | wording | — |

## Concerns investigated and not errors

- P1 decisions: unrounded Llama–HotpotQA bounds are exactly 1.0 (bge-base, e5-base-v2) and 1.096 (bge-large); "not met" is correct. Qwen–HotpotQA largest bound 0.9962 < 1: "met" is correct.
- Bootstrap: no resample had a zero or reversed Hit@5 difference (0 of 220,000), so dropping non-finite ratios never mattered.
- Numerator and denominator use identical queries in every cell (same query list, asserted in the independent check).
- Both@5 implementation: gold set ⊆ top-5 passage ids actually given to the reader; no query with fewer or more than two gold passages.
- Model revisions in the paper match configs/resolved_revisions.json and the generation records; package versions for readers match the logs.
- Total generations 131,855 = 122,545 (A1) + 8,510 (A2) + 800 (A3).
- Generation-set vs all-query metrics: the main tables use generation queries; HNSW and full-corpus tables use all 3,452 judged NQ queries (now stated in both captions).
- Overfull box (117 pt) comes from the journal class's \maketitle frontmatter; nothing overflows on the page.

## New analyses (post hoc, descriptive; all from existing outputs)

| Analysis | Script | Output | Manuscript |
|---|---|---|---|
| Query-level hit/miss transitions, exact reconstruction of ΔEM, bootstrap intervals | scripts/transitions_v05.py | outputs/transitions_hit5.csv | §5.4, Tables transitions, transdetail |
| HotpotQA transitions by number of supporting passages (0/1/2) | same | outputs/transitions_hotpotqa_evidence.csv | §5.4, Table hpevidence |
| Decomposition components with intervals | scripts/components_v05.py | outputs/decomposition_components.csv | §5.4, Table components |
| Sensitivity summary (EM/F1/AC; MiniLM/BM25; Hit@5/Both@5; k) | scripts/make_v05_tex.py | Table sensitivity | §5.3 |
| Independent reimplementation of Hit@5/Both@5, Hit@5 tests and Holm, bootstrap bounds, P1 | scripts/independent_check_v05.py | outputs/independent_check_v05.md | — |

## Restructuring

Abstract shortened to 221 words and refocused (question, design, paired gains, evidence completeness, scoped cost). Highlights rewritten. Introduction repositioned: prior work already shows that retrieval gains need not reach answers; the contribution is the paired size, its sensitivity, and the query-level accounting. Results reorganised: §5.3 sensitivity, §5.4 where the gain sits. CPU deployment cost moved to Appendix (app:deploy). Conclusion rewritten around the full-corpus open question.

---

# Change log: v0.5 → v0.6 (Phase 3: full-corpus NQ readers, addendum v0.4)

v0.6 adds the results of the registered Phase 3 session (S5-NQ, run 17 October 2026 with code commit 80b647e and notebook commit efadb53; outputs committed under `paper_inputs/v04_s5nq/`). No earlier number changed value. Every new number is a `\ev{V6.*}` macro generated by `scripts/make_v06_tex.py` from `paper_outputs/v04/v04_full_analysis.json` (`python -m src.analysis.v04_full`).

## Scientific changes

| # | Where | v0.5 | v0.6 | Evidence |
|---|---|---|---|---|
| F1 | Abstract, highlight 2, Intro, Results §5.4 (new), Discussion §6.1/6.4, Conclusion | whether the NQ ratios hold on the full corpus "was not tested" | readers on full-corpus retrieval: dense T 0.15–0.23; e5-base-v2 T 0.233 (Qwen) and 0.195 (Llama), inside the sub-corpus intervals; P7 and P8 met for both readers; the addendum's decision rule ("the paper states that the NQ result held on the full corpus") applied | v04_full_analysis.json; Table fullreader; Fig. 13 |
| F2 | Results §5.4 | — | full corpus lowered dense Hit@5 by 0.089–0.100 and EM by 0.018–0.028 (Holm p ≤ 0.014); BM25 unchanged | Table fullreader |
| F3 | Results §5.5 (new paragraph), Discussion §6.1, Abstract | "many misses probably contained an answer-bearing passage" (untested) | measured: 0.41–0.55 of dense-encoder misses (sub) and 0.44–0.58 (full) have an answer string in the top five; lexical limits stated | Table answerstr; passage_answers.jsonl |
| F4 | Results §5.7 (context), Limitations | O8: hard negatives might contain the answer; "could not check" | 167 of 500 E12 questions (0.33) have an answer-bearing hard negative; P5 holds on the other 333 with a larger drop (−0.102 vs −0.074) | e12_hardneg_answers.jsonl; v04_full_analysis.json O8 |
| F5 | Method §3.7, Table predictions, Results summary | six registered predictions | eight; P7, P8 added with their registration (addendum v0.4) | design/addendum_v0.4.md |
| F6 | Setup §4 | — | Phase 3 environment (same as Phase 2), E9 retrieval reproduced exactly by re-encoding, own independent check (0 disagreements in 18,040 re-scored answers) | outputs/independent_check_v06.md |
| F7 | Deviations | — | item 8: the registered human audit of 100 misses was not complete for this version | annotation/v04_miss_audit/ |
| F8 | Limitations | sub-corpus answer results untested; passage texts not stored | narrowed to what Phase 3 did not cover (bge-base, bge-large, Qwen3-Emb, HotpotQA); answer-string check is lexical; P7/P8 replicate known results | — |

## Editorial
- "two phases" → "three phases" where all phases are meant; cross-phase determinism text now names Phases 1 and 2.
- Highlight "HotpotQA EM gains sat in queries that gained supporting passages" removed to make room for the full-corpus highlight (the finding stays in the abstract and results).
- Fig. 8 (phases) regenerated with a Phase 3 row; new Fig. 13 (sub vs full ratios); new Tables fullreader and answerstr; Table env gains the S5 row.
- Abstract 247 words (limit 250).

## Tooling
- `make_v05_tex.py`: P7/P8 rows in Table predictions; S5 row in Table env; prompt counts exclude Phase 3 (counted and hash-verified by `make_v06_tex.py`).
- `numeric_report.py`: reads the v0.6 ledger and the independent v0.6 check; range dash no longer parsed as a minus sign.
- `check_revised.py`, `make_docx_revised.py`: include `numbers_v06`.

## v0.6 proofreading pass (2026-10-18)
- Table env: the "Phase 1, NQ retrieval and encoding" row appeared twice (the Phase 3 archive contains a restored copy of that log); the generator now skips restored logs.
- Setup: cross-phase sentence now says "the software change between Phase 1 and the later phases" (Phase 3 runs used the Phase 2 image).
- Results §5.4 and Discussion §6.4: the BM25-baseline full-corpus ranges are scoped to bge-small and e5-base-v2 (the sub-corpus range for all five encoders is 0.29–0.35).
- References: BEIR cited as published (NeurIPS 2021 Datasets and Benchmarks track; title spelling "Heterogeneous", verified on the proceedings page); "arXiv" no longer capitalised as "ArXiv" in notes.

## v0.6 human audit (2026-10-18)
- The registered audit of 100 seeded e5-base-v2 sub-corpus misses was labelled by the author (labels: `annotation/v04_miss_audit/labels_author.csv`, sha256 6ef274f6…; results: `outputs/audit_v04_results.json`).
- Results §5.5 (new paragraph and Table humanaudit), Method §3.8 (audit procedure), Abstract, Discussion §6.1/6.4, Limitations, Conclusion: A = yes 41/100 [0.32, 0.51]; B = yes 41/41; C = yes 59/100 [0.49, 0.68] vs EM 24; of 35 correct answers with EM 0, 21 alias/format. Not pre-specified: automatic flag agreed on 92 of 94 yes/no items (κ = 0.96).
- Deviation 8 rewritten: audit done; labelling tool changed from spreadsheet to web page (same items and order; gold and flag hidden; Screen 2 locked until Screen 1 complete; only the last change time per item is recorded).
- Abstract trimmed to 249 words (no claim changed).

## v0.7 (21 October 2026): check of an external review prompt
- Abstract, highlight 2 and the Introduction contribution list now name the two dense encoders (bge-small, e5-base-v2)
  that were rerun on the full NQ corpus, instead of "the dense ratios"/"NQ ratios"; the body already stated this scope.
- The other seven issues of the prompt were already resolved in v0.4-v0.6; see external_prompt_check_v07.md.
- Data availability (Sec. 9): no longer says the repository will be made public only on acceptance, because the
  editor receives its link on the title page; the archived DOI on acceptance stays. See ipm_presubmission_audit_v07.md.

## Submission package (24 October 2026)
- Title page: ORCID 0009-0006-8541-4422 (checksum valid); no funding, no competing interests confirmed by the author.
- Ethics statement: now also names the 100-miss human audit (same annotator, blinding in Section 5.5).
- scripts/make_submission_package.py builds manuscript_ipm/submission_ipm/: cover letter, title page, anonymised
  manuscript PDF, compile-tested LaTeX source zip, highlights, competing-interest declaration, form text, guide.
