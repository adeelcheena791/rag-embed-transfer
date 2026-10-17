# Scientific and reproducibility audit — RAG embedding-transfer manuscript

Date: 2026-10-17. Audited version: `manuscript_ipm/latex` at commit f3c573a (v0.3, 27 pages). Revised version: `manuscript_ipm/revision_v04/latex` (v0.4, 29 pages). Status: **REVISION REQUIRED, close to SUBMISSION CANDIDATE** (see §8).

## 1. Evidence inventory (inspected, not just listed)

| Artifact | Content (inspected) | Role |
|---|---|---|
| `paper_inputs/results/processed/answers/` (24 files) | Phase 1 per-query EM/F1/containment, Qwen–NQ, all conditions incl. k ablation and A3 | Qwen–NQ cell, E4, abstention check |
| `paper_inputs/v03_nq/processed/answers/` (23) | Phase 2 Llama–NQ and Qwen–NQ k ∈ {1,3,10} for R2/R3/R5/R6 | Llama–NQ cell, E4 |
| `paper_inputs/v03_hp/processed/answers/` (20) and `raw/generations/` (18) | HotpotQA answers and the exact top-5 passage ids given to each reader | HotpotQA cells; Hit@5/Both@5 per query |
| `paper_inputs/v03_nq_s4/` | E5 reranker runs and answers, E12 context answers and runs, E6 `hnsw.jsonl`, E7 `throughput.jsonl` (5 reps), E9 summaries | RQ3, RQ4 |
| `paper_inputs/results/raw/top10/` | NQ per-query top-10 for R0–R6 | NQ Hit@5, nDCG@10 |
| `paper_inputs/data/` | NQ qrels, joined NQ set (2,255), seeded order, join stats | populations |
| `annotation/` | 200 human labels, sample key with judge verdicts and inclusion probabilities | κ |
| `configs/*.yaml`, `configs/resolved_revisions.json` | all run settings and model revisions | specification |
| `design/` | v0.2 design and addenda v0.2.1–v0.3.2 with dates | registration, deviations |
| `src/analysis/v03_analysis.py`, `v03_extras.py`, `stats.py` | every statistic of the paper | analysis |
| `manuscript_ipm/tools/` | ledger, macros, tables, figures, checks, independent standard-library check | manuscript build |
| `manuscript_ipm/records/` | evidence ledger (1,016 rows), literature matrix and search log, review log, open issues | provenance |

Not committed (affects re-analysis, not reproduction): full-corpus E9 top-k lists, HNSW top-k lists, passage texts for E12 hard negatives.

## 2. Reproducibility audit

`python -m src.analysis.v03_analysis` was rerun into a scratch directory. The output JSON equals the committed `paper_outputs/v03/statistics/v03_analysis.json` in every numeric field (tolerance 1e-12); the only difference is the judge-agreement block, which the script reads from its output directory by design. All nine CSV tables are byte-identical. Run time 92 s on CPU. See `validation_report.md`.

## 3. Numerical consistency audit

`outputs/numerical_consistency_report.csv` lists all 231 `\ev{}` values printed in the revised manuscript and tables (201 in v0.3: `outputs/numerical_consistency_report_v03_original.csv`). 68 are recomputed from per-query files by new code and match at printed precision; the others are checked by the identical rerun; the 38 new v0.4 values are generated from audit outputs. **No printed number was wrong.** The manual review found thirteen scope, wording or interpretation problems (`outputs/numerical_issues_manual.csv`):

| ID | Finding | Verdict | Action |
|---|---|---|---|
| M1 | NQ "0.16–0.27" and "at most 0.27" vs BM25 0.37/0.36 | numbers right; scope (dense encoders) unlabelled; BM25's ratio is a ratio of losses | scope labelled everywhere |
| M2 | "throughput ranges did not overlap except bge-base and e5-base-v2" vs table 501–503 and 491–496 | **prose wrong**: no pair overlaps; the code excluded the pair a priori | corrected; 1.8% gap reported |
| M3 | HNSW "no measurable cost in answers" | **unsupported**: no reader run on HNSW results | narrowed to retrieval and latency |
| M4 | random passages "harmless" (p = 0.71) | overstated; no equivalence margin | difference +0.006 [−0.016, 0.026] reported; "not evidence of no effect" |
| M5 | sub-corpus "easier for BM25" | **direction reversed** relative to construction and E9 | corrected |
| M6 | pooled GEE "same picture" | interaction rejected (p = 5.2 × 10⁻³⁶) | dataset-specific ORs added |
| M7 | decomposition 48–70% "exact" | correct for its formulation; formulation not named | symmetric 44–64% and intervals 26–76% added |
| M8 | abstract: "one upper bound reached 1.10" | incomplete (three of five ≥ 1) | reworded |
| M9–M12 | NQ upper bound, throughput table, typed counts, prompt growth | verified | none |
| M13 | baseline dependence stated without numbers | — | BM25-baseline sensitivity added |

## 4. Priority audits A–H (details in `statistical_audit.md`)

- **A. Transfer ratio and decomposition.** All 24 ratios recomputed exactly (`transfer_ratio_audit.csv`). Both decomposition identities hold to < 1e-12 for all 24 comparisons (`transfer_decomposition_audit.csv`). New: BM25 baseline (`transfer_ratio_alt_baselines.csv`) and Both@5 denominator (`transfer_ratio_both5_hotpotqa.csv`). The Both@5 result (HotpotQA dense T = 0.29–0.35 instead of 0.49–0.93) is the most consequential finding of this audit: the "larger HotpotQA transfer" is mostly a property of the saturated Hit@5 denominator. The revision reports it in the abstract, highlights, results, discussion and conclusion.
- **B. Numerical discrepancies.** See §3. The NQ range was a scope problem; the throughput statement was a prose error.
- **C. GEE.** Original model reproduced exactly; the common-effect assumption fails for dataset (not reader); dataset-specific estimates added; per-cell exact tests remain primary.
- **D. HNSW.** Only retrieval overlap, Hit@5, nDCG@10 and latency were measured; claim narrowed.
- **E. Random passages.** Reworded; no post hoc equivalence margin chosen.
- **F. Corpus.** Construction documented; evaluation labels used only to insert gold passages (declared design, no trained model); the BM25 mining favours dense encoders; answer results all rest on the sub-corpus.
- **G. Literature.** See §5.
- **H. Judge.** κ = 0.127 recomputed from the label files; exclusion correctly reported.

## 5. Literature and novelty audit

Existing verification (2026-10-16, `manuscript_ipm/records/literature-matrix.md`, `literature-search-log.md`) covered 20 closest works at full-text (FT) or abstract (AB) level. A new search on 2026-10-17 (two web queries) found one additional relevant journal article, added at abstract level.

### Literature audit (claims that carry the gap statement)

| Manuscript claim | Citation | Verified? | What the source supports | Gap or correction |
|---|---|---|---|---|
| A ratio of answer F1 to recall falls as recall rises across RAG methods | Peng et al. 2026 (NeocorRAG) | FT | RCR = F1/Recall@5, level ratio across pipelines | none; distinction (level vs paired difference) stated |
| Gold-document recall matters more than ANN search recall | Leto et al. 2024 | FT | yes, one dense retriever, answers measured | v0.4 says they measured answers and we did not |
| Relevance-label metrics correlate weakly with downstream RAG | Salemi & Zamani 2024 | AB | yes (abstract) | none |
| Coverage metrics correlate positively with generation quality | Samuel et al. 2026 | FT | yes; counter-evidence discussed | none |
| High-scoring non-answer passages lower accuracy; random can help | Cuconasu et al. 2024 | FT | yes | none |
| Position effects marginal in realistic retrieval | Cuconasu et al. 2025 | AB | yes (abstract) | none |
| Lexical EM undercounts correct answers | Kamalloo et al. 2023 | FT | yes | none |
| Retrieval quality is the main determinant of end-to-end RAG quality (embedding × generator benchmark) | Hitl et al. 2026, *Data Technologies and Applications*, DOI 10.1108/DTA-07-2025-0609 | AB (publisher page) | yes, abstract; full text paywalled | **added in v0.4** as counter-evidence; claim limited to what the abstract states |

### Novelty comparison

| Closest prior work | Question | Datasets | Retrieval models | Downstream evaluation | Efficiency | Difference from this study |
|---|---|---|---|---|---|---|
| Peng et al. 2026 (NeocorRAG) | does answer quality keep up with recall across RAG methods | HotpotQA, 2Wiki, MuSiQue, NQ | graph/structure RAG pipelines | F1; RCR level ratio | no | paired gain ratio of interchangeable encoders vs one baseline, per reader, with intervals and cost |
| Leto et al. 2024 | how retrieval accuracy affects RAG | ASQA, QAMPARI, NQ | one dense retriever + ColBERTv2, gold-share and ANN manipulations | correctness, citation | ANN speed | many encoders, measured cost, paired ratio |
| Salemi & Zamani 2024 (eRAG) | evaluate retrieval for RAG | KILT | several | correlation with downstream | no | estimates the size of transfer, not a new label scheme |
| Samuel et al. 2026 | which retrieval metrics predict report quality | NeuCLIR, RAG 24, WikiVideo | 15 stacks | nugget coverage, correlation | no | magnitude (ratio) vs ranking (correlation) |
| Hitl et al. 2026 | best embedding × generator combinations | SQuAD, HotpotQA (+ primary set) | several embedders | EM, F1, ROUGE-L, BERTScore, faithfulness | latency, storage, HNSW | paired ratio with intervals, decomposition, registered predictions; their abstract reports levels, not a transfer ratio |
| El Assadi et al. 2026 | cost of embedders vs LLMs | MTEB tasks | 26 embedders | none (no reader) | yes | answer quality measured |

**Revised contribution statement (unchanged in substance, already bounded in v0.3):** for interchangeable open encoders, the paired ratio of answer gain to retrieval gain with uncertainty, per reader and dataset, next to measured cost, in one setting. No "first", "novel" or "no prior work" wording is used. The new sensitivity results narrow the contribution further: the ratio is a property of a retriever pair and a retrieval metric.

## 6. Experiments rerun or added

Rerun: the full original analysis (CPU). Added (CPU, from committed per-query files): transfer-ratio recomputation; decomposition in two orderings with bootstrap; BM25 and bge-small baselines; Both@5 denominator; GEE interaction models; throughput ranges; κ from labels; numeric consistency for every printed macro. No GPU experiment was run. Proposed GPU runs are in `experiment_plan.md`.

## 7. Deliverables

All in `manuscript_ipm/revision_v04/`: this report; `study_specification.md`; `statistical_audit.md`; `experiment_plan.md`; `skeptical_reviewer_report.md`; `validation_report.md`; `change_log.md`; `claim_evidence_ledger.csv`; `outputs/numerical_consistency_report.csv`; `outputs/transfer_ratio_audit.csv`; `outputs/transfer_decomposition_audit.csv`; supporting CSV/JSON in `outputs/`; `latex/` (revised source); `revised_manuscript.pdf` and `.docx`; `original_manuscript_v03.pdf`; `scripts/` and `README.md` (reproducibility package).

## 8. Readiness

No printed number was wrong. Five prose claims exceeded the evidence and are corrected. Two new checks (baseline, Both@5) weaken the original HotpotQA headline, and the revision says so. The largest remaining risk is that all answer results rest on a sub-corpus that favours dense encoders, with no full-corpus reader run. **REVISION REQUIRED** if that experiment is to be run before submission; **SUBMISSION CANDIDATE** if the author accepts it as a stated limitation. The author-side blockers from earlier sessions remain: repository public or anonymised mirror (O1), title-page facts (O5), wording of the AI-use declaration (O6). None of this predicts acceptance or a Q1 ranking.
