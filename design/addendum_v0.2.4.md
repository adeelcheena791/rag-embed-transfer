# Design addendum v0.2.4 (2026-09-27) — Pilot Part B verified; main-run execution plan

Status: pilot Part B **EXECUTED and VERIFIED**. The main run is PLANNED — NOT YET EXECUTED.
Pilot numbers are never reported as results.

## 1. Pilot Part B: verification (facts, recomputed from the returned zip)

| Check | Result | Gate | Verdict |
|---|---|---|---|
| Determinism vs Part A (different session) | BM25 full-corpus run, R1 pilot run and R4 pilot run: top-100 identical for 200/200 queries each; pilot sub-corpus sha256 identical | — | reproducible |
| P5 reader | Qwen2.5-7B-Instruct-AWQ @ `b2503754`, vLLM 0.30.0, default CUDA graphs; 4 × 200 outputs; 0 empty; one GOLD output stopped at the token limit (a list of film titles → unparsed) | ≤ 1% empty | PASS |
| P5 sanity | EM(GOLD) 0.655 > EM(CB) 0.155 | GOLD > CB | PASS |
| P6 judge | Mistral-7B-Instruct-v0.3 @ `c170c708`, TP = 2; 0 / 300 parse failures | F2: > 2% → few-shot | zero-shot kept |
| P7 | p_d = 33/200 = 0.165 (R1-only correct 16, R4-only correct 17) | ≤ 0.10 → 1,000, else all joined | **N_gen = 2,255** |
| P7 projections | S3 6.98 T4-h; S4 0.83 T4-h; S3 + S4 7.81 | F4.2 8 h; F4.3 6 h; F4.4 14 h | none triggered |
| P8 | `requirements-retrieval.lock` (934 lines), `requirements-generation.lock` (1,014 lines); now committed | — | done |

**Engine facts from the logs:**
- FlashAttention-2 is unavailable on sm_75, so vLLM chose the `TRITON_ATTN` backend.
- The AWQ layers used `MarlinLinearKernel`.
- Outputs are unaffected; this is recorded for the Reproducibility section.

**Judge parsing.** One judge output was `SUPPORTED (assuming the`. It parses as SUPPORTED under the pre-registered regex (anchored prefix match), so it is correct per the rule and not re-coded.

## 2. Qualifications (interpretation)

- **Pilot EM figures are not reported or interpreted.** In particular, "R1 captures 70% of the GOLD–CB room" is not a pre-registered quantity. It is based on 200 queries, and the R1 vs R4 difference there is one query. The pre-registered analysis is the Hit@k decomposition of §5.3.
- **GOLD is not an upper bound on retrieval conditions.** It is the reader's accuracy when given only the gold passages. Retrieved non-gold passages can also contain the answer, and a retrieval condition can in principle exceed GOLD.
- **Zero parse failures say nothing about judge accuracy.** The high SUPPORTED rates (0.89–0.97) may indicate leniency. Only the human validation (§7) can show that.
- **The S3 projection (6.98 h) is within about 1 h of F4.2 (8 h).** Measured S3a time is therefore checked before S3b (§5).

## 3. Main-run counts

| Session | Content | Items |
|---|---|---|
| S1 | M1 sub-corpus; R0 + 6 dense retrievers on the main corpus for all 3,452 judged queries; GPU and CPU efficiency | — |
| S3a | 9 conditions (R0–R6, CB, GOLD) × 2,255 | **20,295** generations |
| S3b | F1: R0, R1, R4, GOLD × 200 (prompt A3) | 800 |
| | A2: R3 × 100 (re-run into a separate file) | 100 |
| | A1: R0, R1, R4 × k ∈ {1, 3, 10} × 2,255 | 20,295 |
| | S3 total | **41,490** generations |
| S4 | 8 conditions (7 retrievers + GOLD) × first 500 queries | **4,000** judgements |

S4 has 8 conditions, not 7. v0.2 §5.4 includes GOLD "as a reference", and P7's projection already counted 8.

## 4. Execution mode (implementation decisions)

1. **The code is fixed at commit `22f161a`.** It is published as branch `main-v1`, which is never pushed to again.
   - Every main notebook clones that branch and stops unless HEAD is exactly `22f161a6e48a188ce8114bc8f6d402bcf7fdf327`.
   - A tag was intended, but the session's git access refused to push tags; branches push normally.
2. **Pinned environments.** Sessions run with "Save & Run All (Commit)". Every command goes through a fail-fast helper, so a failure stops the notebook instead of the next cell silently running on missing inputs.
   - Environments are pinned to the pilot versions (`requirements-*-pinned.txt`).
   - `src.check_env` stops the session if a key package differs from the pilot lock. This is expected to pass; if Kaggle has changed its image, we decide before anything runs.
3. **Hand-over between sessions.**
   - Each session saves `<name>.zip` as its output.
   - The next session attaches that output as an input, and `src.restore` copies `results/`, `data/processed/` and the pinned revisions. It refuses to overwrite a differing file.
   - Chains: S1 → S3a → S4, and S1 → S3b.
4. **`--auto-eager-retry`** implements the first pre-registered fallback (same model, no CUDA graphs) for unattended sessions. The pilot did not need it. Any use is logged in `failures.jsonl`.
5. **CPU efficiency (M4) moves from a separate CPU session into S1.** All CPU numbers (BM25 latency, FAISS search latency, CPU encoding and query latency) then come from one CPU. Cost: about 1 extra GPU-session hour.
6. **Clarification of F4.2 for N_gen = 2,255.** Its wording "cut N_gen to 1,000 (if it was 3,000)" assumed the old 3,000-query branch. With 2,255, the rule is now:
   - If measured S3a plus projected S3b exceeds 8 T4-h, A1 is run on the first 1,000 queries only (a nested prefix).
   - If it still exceeds 8 h, A1 at k = 10 is dropped.
   - The 9 main conditions are never cut.
   - This clarification is fixed before S3a runs.

## 5. Checkpoints between sessions

**After S1**, before any generation, I verify from the zip:
- sub-corpus size = 300,000 exactly, and gold passages of all 3,452 judged queries included;
- every run has 3,452 queries × 100 passages;
- efficiency records exist for all 6 models, GPU and CPU;
- no failures.

**After S3a**: counts (20,295), empty-output rate, and measured T4-hours → the F4.2 decision for S3b.

**After S4**: counts (4,000), parse failures → then the human validation sample.

## 6. GPU budget (UNVERIFIED estimates from pilot speeds)

| Session | Estimate |
|---|---|
| S1 | ≈ 4.5 session-hours: encoding ≈ 2.5 h, GPU benchmark ≈ 0.6 h, CPU benchmark ≈ 1 h, BM25 + download ≈ 0.3 h |
| S3a | ≈ 3.3 h |
| S3b | ≈ 3.5 h |
| S4 | ≈ 1 h |
| **Total** | **≈ 12–13 h** |

Kaggle's weekly GPU quota must have at least this remaining. If not, S3b and S4 move to the next quota week, and nothing else changes.

## 7. Human validation of the judge (v0.2 §5.4)

- **When:** after S4, because it samples judged items. It overlaps with the offline analysis.
- **Sample:** 200 items, stratified over the 7 retrievers (≈ 28–29 each) and balanced across judge labels where possible (seed 7).
- **Interface:** I will build a small annotation page that exports a CSV.
- **Needs 2 annotators** for inter-annotator κ. With only 1, human–judge κ is still reported, but inter-annotator κ is not, and that is stated as a limitation.
