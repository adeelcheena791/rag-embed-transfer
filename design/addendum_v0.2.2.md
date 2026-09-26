# Design addendum v0.2.2 (2026-09-26) — Deviation D1: NQ join criterion

Status: PLANNED — NOT YET EXECUTED, except the pilot step reported below.
This is a **deviation from the pre-registered design**. It is recorded **before any retrieval or answer outcome was observed**. Only pilot steps P1 (download) and P3 (join) had run when the decision was made.

## What happened (pilot P3, Kaggle, 2026-09-26)

| | Value |
|---|---|
| BEIR-NQ test queries with ≥1 relevant passage | 3,452 |
| Joined to NQ-open validation (exact normalised text, identical answer lists) | **2,255** |
| Unmatched | 1,197 |
| Ambiguous | 0 |
| NQ-open validation questions | 3,610 |
| **Join rate** | **0.653** |

The v0.2 §2 rule was "join rate < 0.70 → fall back to HotpotQA". It triggered, and the pipeline stopped as designed.

## Decision: keep Natural Questions and use the 2,255 joined queries (Option B)

**Why the pre-registered fallback is not taken**
1. **The 0.70 threshold was not grounded in any property of the data.** I set it in v0.1 without knowing the expected overlap.
   - NQ-open keeps only questions whose short answer has ≤ 5 tokens (verified: NQ-open dataset card).
   - BEIR-NQ keeps questions with a relevant Wikipedia passage.
   - The two are independently filtered subsets of Natural Questions, so a join rate well below 1 is expected by construction.
   - **Hypothesis (to be tested by D1a below):** most of the 1,197 unmatched queries are questions *without* a ≤5-token short answer. That would make them absent from NQ-open, not missed by the text matching. If so, they could not be scored with EM against a short gold answer anyway.
2. **HotpotQA would change the primary question.** It is multi-hop, with ≈2 gold passages per query (BEIR README). The Hit@k → answer-accuracy decomposition (RQ2) is defined most cleanly for single-hop questions answerable from one passage.
3. **Power is adequate**, but not by the margin v0.2 allowed for (see §7.1 below).
4. **No new matching logic is introduced.** Option C (fuzzy or embedding matching) would add an unregistered, tunable step that could itself bias the sample.

**New pre-registered gate (replaces "join rate ≥ 0.70"):** proceed with NQ only if **n_joined ≥ 1,000**.
- Pilot: 2,255 ≥ 1,000, so it passes.
- The join rate is still computed and reported, together with a `v0_2_rule_would_pass` field.

## D1a — descriptive diagnostic of the unmatched queries (new; runs in pilot Cell 4b)

`src/data/join_diagnostics.py` never modifies the joined set. For each not-joined query it records:
1. whether its normalised text is in **NQ-open train**, a different split;
2. its highest token-Jaccard similarity to any NQ-open *validation* question. Many values near 1 would mean a normalisation miss rather than true absence;
3. mean question length and gold passages per query, joined vs not joined;
4. the 10 most similar not-joined pairs, for manual inspection.

**Interpretation rules (fixed now):**
- If ≥ 10% of not-joined queries have max-Jaccard ≥ 0.9 to a validation question, the "structural absence" hypothesis is **rejected**. The Limitations section will then say that some queries were lost to text mismatch.
- Matching is **still not changed** in that case, because changing it after seeing which questions it adds would be a data-dependent choice.
- If < 10%, the hypothesis is supported, and the paper describes the evaluated population as "BEIR-NQ test queries that are part of NQ-open".

**D1b — retrieval-difficulty comparison** (in the analysis stage, Batch 3):
- Retrieval metrics need only relevance labels, so they are computed for **all 3,452** queries as well as the 2,255 joined ones.
- We report whether the joined subset is easier or harder for retrieval than the full set: per-retriever Hit@5 and nDCG@10 with bootstrap CIs, and the rank order of retrievers on both sets.
- This bounds how much the selection could affect RQ1.

## Changes to v0.2 / v0.2.1

**§2 Data**
- Evaluated population = the **2,255 joined queries**. RQ1 is reported on this set (primary, the same queries as RQ2) and on all 3,452 (secondary, D1b).
- NQ-open **train** is also downloaded (4.5 MB), for D1a only. It is never used as gold answers.
- The sub-corpus rule is unchanged: gold passages of all *joined* queries, plus BM25 negatives, plus random fill.

**§7.1 Power / N_gen rule.** v0.2 said "p_d ≤ 0.10 → 1,000; else min(joined, 3,000)". Now:
- **p_d ≤ 0.10 → N_gen = 1,000**
- **p_d > 0.10 → N_gen = 2,255** (all joined queries)

Updated MDD at 80% power:

| n | p_d = 0.05 | 0.10 | 0.15 | 0.20 | 0.30 |
|---|---|---|---|---|---|
| 2,255, α = 0.05 | 1.4 | 2.0 | 2.4 | 2.7 | 3.3 pp |
| 2,255, α = 0.05/21 | 1.9 | 2.7 | 3.2 | 3.8 | 4.5 pp |

Compared with v0.2's 3,000-query branch, the worst-case MDD rises by at most 0.6 pp (e.g. 3.2 → 3.8 pp at p_d = 0.20).

**Cost:** with N_gen = 2,255, generation is 9 × 2,255 = 20,295 main runs plus 9 × 2,255 for A1. That is below v0.2's 3,000-query budget, so the F4 stopping rules are unchanged.

**Power script:** `design/power_mcnemar.py` now computes the exact rejection region in closed form. The previous loop timed out for n = 2,255. The new version reproduces the v0.2 table exactly and matches `scipy.stats.binomtest` in a cross-check.

## Contingency: HotpotQA (asked with this decision)

HotpotQA is no longer the primary fallback. It stays an **optional robustness check**, cut first under F4.
- If it is run, it is joined by **query id**, not text. That BEIR-HotpotQA ids equal the original HotpotQA ids is UNVERIFIED; it is checked before use.
- The same floor applies: **n_joined ≥ 1,000**, otherwise the check is **dropped** and reported as not done. There is no further fallback dataset, because the robustness check is optional and adding datasets is on the "cut first" list.

**If NQ itself had fallen below the floor**, the pre-registered action would have been: stop, report, and choose a replacement dataset with the user before any retrieval run. It is not needed at 2,255.

## Paper disclosure (to appear in Data and Limitations)

- The evaluated queries are the 2,255 of 3,452 BEIR-NQ test queries whose text exactly matches an NQ-open validation question (65.3%).
- The v0.2 threshold of 0.70 was replaced before any outcome was observed; the reasons are above.
- D1a and D1b results are reported. Any conclusion about retrieval is qualified by the D1b comparison.
