# Statistical audit (revision v0.4, 2026-10-17)

All numbers below were produced by `scripts/audit_v04.py` from the committed per-query files (`outputs/audit_results.json`, the CSV files in `outputs/`). Nothing here was typed from the manuscript.

## 1. Transfer ratio

**Definition (verified in `src/analysis/stats.py::transfer_ratio`).** T(R) = (ē_R − ē_R1)/(h̄_R − h̄_R1), with ē the mean EM and h̄ the mean Hit@5 over the same generation queries, the same reader, prompt A1, k = 5 and greedy decoding. The baseline R1 is MiniLM-L6 for every reported ratio. Both differences are proportions (0–1), so T is unit-free. The ratio is computed from per-query arrays, not from rounded aggregates. It is reported only if |Δh| > 0.02 and the Hit@5 difference is Holm-significant (rule fixed in design v0.2). Bootstrap intervals resample queries and recompute numerator and denominator together (B = 10,000, seed 0).

**Recomputation.** All 24 ratios (4 cells × 6 retrievers) were recomputed. The 22 estimable ratios equal the committed values to machine precision (difference 0.0e+00), and the two non-estimable ones (BM25 on HotpotQA, |Δh| = 0.016) are flagged identically (`outputs/transfer_ratio_audit.csv`).

**Behaviour at small or negative denominators.** The rule removes |Δh| ≤ 0.02. The remaining NQ bge-small ratios (Δh = 0.065) still have intervals that include zero, which the manuscript states. BM25 on NQ has Δh = −0.30 and ΔEM ≈ −0.11; its positive ratio relates a loss to a loss, which the table footnote states.

**Scope of the reported NQ range (M1).** "0.16–0.27" covers the five dense encoders × two readers (10 ratios). BM25's 0.37/0.36 is outside it by construction. No value was wrong; the scope was unlabelled in the abstract, highlights and conclusion. Corrected in the revision.

**Baseline dependence (new, post hoc).** `outputs/transfer_ratio_alt_baselines.csv` repeats every ratio against BM25 and against bge-small under the same rule. Against BM25: NQ dense 0.29–0.35, HotpotQA dense 0.25–0.75. The HotpotQA ratio exceeds the NQ ratio in 8 of 10 encoder–reader pairs; Qwen3-Emb reverses (HotpotQA 0.25/0.27 vs NQ 0.35/0.33). Against bge-small, most HotpotQA ratios are not estimable (Δh ≤ 0.02). Conclusion: T is a property of a retriever pair; the dataset contrast is clear against MiniLM-L6 and holds for four of five encoders against BM25. Ranking encoders by T alone is not supported; the manuscript does not do so.

**Retrieval metric in the denominator (new, post hoc).** HotpotQA Hit@5 is near saturation for the dense encoders (0.945–0.961), so ΔHit@5 understates the retrieval gain that matters for two-passage questions. With Both@5 as the denominator (same baseline, rule and bootstrap; `scripts/both5_ratio.py`), the dense-encoder ratios are 0.29–0.35 for both readers (intervals 0.21–0.41), against 0.49–0.93 with Hit@5. BM25's Both@5 gain is only 0.034, so its ratio is unstable (1.0 [0.5, 2.5]) and is not reported. The contrast "HotpotQA transfer larger than NQ" therefore depends mostly on the denominator; the revision says so in the abstract, highlights, results, discussion and conclusion.

**Interpretation guard.** The method section already stated that T is not the probability that an extra hit yields an extra correct answer. The introduction's sentence "a ratio of 1 means every extra question with a relevant passage … also gains a correct answer" implied a query-level reading and was rewritten.

## 2. Conditional decomposition

With p = P(H = 1), a = P(EM = 1 | H = 1), b = P(EM = 1 | H = 0), EM = p·a + (1 − p)·b. Two exact splits of ΔEM were computed for every cell × retriever against MiniLM-L6:

- Baseline-conditioned (the manuscript's Eq. 2, multiplied by Δp): ΔEM = Δp·(a0 − b0) + p1·Δa + (1 − p1)·Δb.
- Symmetric: each term averages the two orderings, Δp·[(a0 − b0) + (a1 − b1)]/2 + Δa·(p0 + p1)/2 + Δb·[(1 − p0) + (1 − p1)]/2.

Both identities hold with residual < 1e-12 for all 24 comparisons (`outputs/transfer_decomposition_audit.csv`). The manuscript's "share" equals [p1Δa + (1 − p1)Δb]/ΔEM, which is the second term of Eq. 2 divided by T.

| Quantity, HotpotQA (10 estimable ratios) | Range |
|---|---|
| Baseline-conditioned share | 48%–70% (reproduces the manuscript) |
| Symmetric share | 44%–64% |
| Share of the EM-given-hit term alone | 48%–71% |
| Bootstrap 95% intervals of the baseline-conditioned share | lower bounds from 26% (Qwen3-Emb), upper bounds to 76% |

**Verdict.** The 48–70% range is arithmetically correct for the formulation used, but formulation-dependent, and the abstract called it "exact" without naming the formulation. The revision names it, adds the symmetric range and the intervals (Table share), and calls the split an accounting identity. It is not causal: the sets of hit and miss queries differ between encoders, so Δa mixes "a hit is more useful" with "different questions are hits".

**NQ.** The EM gains are small (0.012–0.038), so the share is unstable: every dense-encoder interval is at least 1.4 wide on the share scale, and bge-small's spans about ±11. The manuscript reported NQ terms (DR), not shares, in the text; the revision says why. The share column in the ledger (DS.qnq.*, DS.lnq.*) is not printed anywhere in the manuscript.

## 3. Pooled GEE model

**Original model (verified).** `smf.gee("em ~ C(retriever, Treatment('R5')) + C(dataset) + C(reader)")`, binomial/logit, exchangeable working correlation (0.60), groups = dataset:qid (4,255 clusters; each cluster holds 7 retrievers × 2 readers = 14 binary outcomes), sandwich SE, 59,570 rows, no missing data, converged without warnings. The refit reproduces OR 1.038 [1.001, 1.076] (bge-large) and 0.909 [0.875, 0.943] (Qwen3-Emb) exactly.

**What it estimates.** A single, cell-averaged log-odds contrast per retriever, assuming the contrast is the same on NQ and HotpotQA and for both readers. The manuscript said the pooled model "gave the same picture" as the per-cell tests.

**Heterogeneity check (post hoc).**

| Model | Test | Result |
|---|---|---|
| B: retriever × dataset + reader | Wald, 6 interaction terms | χ²₆ = 179, p = 5.2 × 10⁻³⁶ |
| C: B + retriever × reader | Wald, 6 terms | χ²₆ = 5.0, p = 0.55 |
| B, bge-large vs e5 | NQ / HotpotQA | 0.981 [0.936, 1.028] / 1.109 [1.049, 1.172] |
| B, Qwen3-Emb vs e5 | NQ / HotpotQA | 0.952 [0.906, 1.001] / 0.859 [0.811, 0.909] |
| D: saturated retriever × dataset × reader | fit | converged; equivalent to per-cell estimates |

**Verdict.** The common-effect assumption is rejected for dataset and not for reader. The pooled bge-large OR of 1.038 averages no detected effect on NQ with a gain on HotpotQA; "same picture" overstated agreement. The revision keeps the registered model, labels it, and adds the dataset-specific estimates (Table gee). The per-cell exact McNemar tests with Holm correction remain the primary inference and are unchanged.

**Other GEE notes.** The pooled model mixes Phase 1 (Qwen–NQ, Python 3.12 image) and Phase 2 cells; the cross-phase determinism check (396/397 identical prompts) limits but does not remove this concern. Three-way interactions were not tested separately because the saturated model is equivalent to the per-cell analysis already reported.

## 4. Unit of inference and multiplicity

The query is the unit throughout. Retrievers are compared within a cell on the same queries (paired). The GEE clusters by query across retrievers and readers. No analysis treats repeated outcomes of one query as independent. Holm correction is applied per family (21 pairs × metric × cell); there is no correction across families, which the limitations state. The E12 position tests (3 pairs) are Holm-corrected; the P5 test is a single registered test. The exploratory checks (bge-large k=1 vs BM25 k=10; transfer by k; 5% cost ties) are labelled exploratory and uncorrected.

## 5. Approximate search (HNSW)

**Measured** (`paper_inputs/v03_nq_s4/raw/hnsw.jsonl`): for MiniLM-L6, bge-small and e5-base-v2 on the NQ sub-corpus (3,452 judged queries), at efSearch 64 and 128: Hit@5, nDCG@10, top-5 and top-100 overlap with exact search, and latency. **Not measured:** reader answers under HNSW (no answer files exist). At efSearch 128, top-5 overlap ≥ 0.9956 and |ΔHit@5| ≤ 0.0023.

**Verdict.** "Approximate search can replace exact search at no measurable cost in answers" (Discussion) was unsupported and is removed; the revision limits the claim to retrieval and latency and lists a reader run on HNSW results as future work. No equivalence margin is claimed even for retrieval.

## 6. Random-passage control

Gold@1 + 4 random (MR) vs gold only (MG), 500 queries: EM 0.654 vs 0.648 (ledger MC.MR, MC.MG); MR − MG = +0.006, 95% CI [−0.016, 0.026], exact McNemar p = 0.71 (13 vs 16 discordant). No equivalence or non-inferiority margin was registered, and choosing one now would be post hoc. **Verdict:** "harmless" is replaced by "changed EM by no detectable amount", with the interval and an explicit statement that this is not evidence of no effect. The registered P5 comparison (M1 vs MR, −0.074 [−0.108, −0.040], p = 2.2 × 10⁻⁵) is unaffected.

## 7. Sub-corpus and generalisation

Construction (src/data/subcorpus.py): all gold passages of all judged queries; the top-d BM25 passages of every judged query (d = 42 NQ, 19 HotpotQA); seeded random fill to 300,000. Every retriever searches the same id list. The gold insertion uses evaluation labels by design (declared, not leakage into any trained model). The BM25 mining keeps BM25's strongest competitors and not the dense encoders', which favours the dense encoders relative to BM25; the E9 drops (BM25 −0.002, dense −0.083 to −0.086 in nDCG@10) agree. **The original Results sentence stated the opposite direction ("easier for BM25") and is corrected.**

Claims that rest on the sub-corpus only: all answer results (EM, T, decomposition, E4, E5, E12), all HotpotQA results, and every comparison with BM25. The full-corpus check covers retrieval order for four retrievers on NQ. A full-corpus reader evaluation is listed in the experiment plan.

## 8. LLM judge

Recomputed from `annotation/judge_validation_labels.json` and `annotation/sample_v1_key.json`: n = 200, Cohen's κ = 0.127 (identical to the committed value), raw agreement 0.565, human SUPPORTED rate 0.575, judge 0.51 (sample); population-weighted agreement 0.620. One annotator (the author), stratified sample with inverse-probability weights for population rates. Prevalence and raw agreement are reported in Table judge. Under the registered rule (κ < 0.2) the judge is excluded from all conclusions; the manuscript says so in Results, Limitations and the abstract. No change needed.

## 9. What was not re-implemented independently

Bootstrap intervals, F1 and answer containment, and the GEE are recomputed by the same libraries and loader as the original (a rerun, not an independent implementation). The transfer ratios, the decomposition and the judge κ were recomputed in new code written for this audit, but it reuses the original data loader and resampling index (so it shares their assumptions). The earlier independent standard-library check (`manuscript_ipm/tools/independent_check.py`) covers EM re-scoring, Hit@5/EM means, T point estimates, 84 pairwise EM tests with Holm, and κ.
