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

## 8. Decisions recorded 2026-09-27 (user)

- **Kaggle GPU quota remaining this week: 27 h 09 min.** The main run (≈ 12–13 h estimated) fits in this week's quota.
- **One annotator only.** The human validation reports human–judge agreement (κ with 95% CI) from a single annotator. No inter-annotator κ is available, and this is stated as a limitation.

## 9. S1 execution record (from the Kaggle log, 2026-09-27; zip not yet received)

**Verified from the log:**
- code at `22f161a` (commit check passed); 10/10 key packages match the pilot lock; 152 tests passed;
- P1 all shas match; P3 2,255 / 3,452 joined;
- M1 sub-corpus: 300,000 = 4,201 gold + 127,791 BM25 (depth 42, 3,452 queries) + 168,008 random;
- R0 main run written; all 6 dense models encoded 300,000 rows (dims 384/384/768/768/1024/1024);
- 6 dense runs written for the judged query set; no FAILED line, no traceback.

**Not yet verified (needs `main_s1_results.zip`):**
- run files (3,452 queries × 100 passages each, 7 retrievers);
- that the gold passages of all judged queries are in the id file;
- the latency records.

**Encoding throughput (passages/s) — pilot P4 vs main S1, same protocol, same GPU type (M3 variability):**

| Model | Pilot | Main | Change |
|---|---|---|---|
| R1 | 1,957 | 1,943 | −0.7% |
| R2 | 1,205 | 1,198 | −0.6% |
| R3 | 489 | 504 | +3.1% |
| R5 | 485 | 496 | +2.3% |
| R4 | 164 | 169 | +3.0% |
| R6 | 64 | 66 | +3.1% |

**CPU throughput (4 vCPU Xeon 2.0 GHz; passages/s):**

| Model | Throughput |
|---|---|
| R1 | 38.7 |
| R2 | 20.0 |
| R3 | 6.39 |
| R5 | 6.19 |
| R4 | 1.92 |
| R6 | 0.305 |

The log line "F3: R6 CPU fallback … drop_R6" is **not** an F3 decision. F3 applies only if R6 fails on the GPU, and it did not. The CPU benchmark computes that rule unconditionally, and the frozen code prints it. **R6 is not dropped.** The implied figure (≈ 273 CPU-hours to encode 300k passages with R6 on this CPU) is reported only as an efficiency measurement.

**Cost deviation — my estimation error.** I estimated the CPU benchmark (Cell 8) at ≈ 1 h; it took **8 h 24 min** (R6 6 h 32 min, R4 1 h 03 min).
- S1 therefore used 11 h 20 min of GPU-session time, against the planned ≈ 4.5 h, and came within 40 min of Kaggle's 12-hour session limit.
- Data validity is not affected, but it consumed quota: 15 h 44 min remains, versus ≈ 9 h needed for S3a + S3b + S4.
- **Lesson applied from now on:** every session gets a wall-time estimate computed from measured per-item speed, with ≥ 30% headroom under the 12-hour limit.

## 10. S1 file-level verification (zip received 2026-09-28)

- The sub-corpus id file has 300,000 unique ids.
- All 2,816 gold passages of the 2,255 joined queries are in it.
- All 7 main runs (R0–R6) have 3,452 unique queries × 100 passages, 0 ids outside the corpus, 0 duplicates, scores non-increasing, and query set `judged`.
- Batch-1 search latency (median, ms, on the S1 CPU):

| Retriever | Latency |
|---|---|
| R0 (BM25) | 21.9 |
| R1 | 39.3 |
| R2 | 38.8 |
| R3 | 81.2 |
| R5 | 81.5 |
| R4 | 143.1 |
| R6 | 142.0 |

- A restore rehearsal on the real zip with the frozen code (`22f161a`) resolves runs for all 7 retrievers covering all 2,255 generation queries.
- **Pending:** gold coverage for the 1,197 non-joined queries. It is guaranteed by construction (gold is added first; n_gold = 4,201 is logged), but it is not yet checked on the file, because the BEIR qrels are not in the S1 zip. The S3a/S3b/S4 packaging now also includes `data/raw/nq_qrels/test.tsv` and the BEIR query file, so the check and all retrieval metrics can run offline. This concerns only the secondary D1b analysis; S3a uses joined queries only.
- R6's CPU record is complete, not truncated: a warm-up plus 3 × 2,000 timed passages at 0.305 p/s ≈ 6 h 32 min, matching the log.

## 11. S3a attempt 1 failed — step-order bug in the notebook (2026-09-28)

- **What happened.** Cell 1 ran `src.env_report`, which appends to `results/raw/env.jsonl`. Cell 2's `src.restore` then found a different `results/raw/env.jsonl` in S1's zip and refused to overwrite it, as designed. The notebook stopped before any generation; no data was produced or altered.
- **Cause (my error).** The restore rehearsal used a clean checkout and never ran Cell 1 first.
- **Fix.** In every restoring notebook (S3a, S3b, S4), the restore runs before anything writes to `results/`, and `env_report` runs right after it. The code (`22f161a`) is unchanged; only the notebook cell order changed.
- **Guards.**
  - A new test asserts that no `src` command except the read-only `src.check_env` runs before a restore.
  - The notebooks' real command order was replayed on the real S1 zip with the frozen code: the old order reproduces the Kaggle error, and the new order passes.

## 12. S3a execution record (log, 2026-09-28; zip not yet received) and the F4.2 decision

**From the log:**
- code `22f161a`; 11/11 packages match the generation lock; 152 tests passed;
- restore ran before `env_report`, as fixed in §11 (31 files copied, 1 identical skipped); P1 shas match;
- reader Qwen2.5-7B-Instruct-AWQ @ `b2503754`, vLLM 0.30.0, CUDA graphs on (no eager retry needed);
- all 9 conditions × 2,255 completed; no FAILED line, no traceback.

**Measured generation time:** 10,856 s = 3.02 h (session ≈ 3.1 h). Seconds per item at k = 5:

| R0 | R1 | R2 | R3 | R4 | R5 | R6 | CB | GOLD |
|---|---|---|---|---|---|---|---|---|
| 0.557 | 0.601 | 0.650 | 0.676 | 0.693 | 0.743 | 0.691 | 0.022 | 0.181 |

**S3b projection from these measurements** (prefill-dominated model: 0.022 s fixed per item + 0.954 ms per prompt token, ≈ 124 tokens per passage):

| Part | Estimate |
|---|---|
| A1 | 3.54 h |
| F1 | 7 min |
| A2 | 1 min |
| setup | 15 min |
| **S3b total** | **≈ 3.9 h** (5.1 h with 30% headroom; under the 12 h session limit) |

**F4.2 decision (pre-registered rule, clarified in §4.6):** S3a 3.1 h + projected S3b 3.9 h = 7.0 h ≤ 8 h, so **not triggered**. S3b runs in full: F1, A2, then A1 at k = 1, 3, 10 on all 2,255 queries. The margin is about 1 h.

**Quota:** 12 h 40 min remaining (user report) versus ≈ 3.9 h (S3b) + ≈ 1.3 h (S4).

**Main-run EM values reported by the user from the S3a summary:** these are **not yet verified** against the raw files, and **no comparison between retrievers is made** until the pre-registered statistics (§7: paired tests, Holm, bootstrap CIs; §5.3 decomposition) are computed from the raw generations.

## 13. S3b verified (zip, 2026-09-28)

**File checks:**
- All 14 generation files have the expected query sets: A1 2,255; F1 200; A2 100.
- Each item received exactly its run's top-k (or the gold passages); the prompt hashes are unaltered; one reader revision throughout.
- `failures.jsonl` is empty. The 16 S1 run files are byte-identical.

**A2 (determinism) RESULT.** R3 (k = 5, A1) was re-run on the first 100 queries in a different Kaggle session and on a different day than S3a. **100 / 100 outputs and 100 / 100 prompt token counts are identical to S3a's**, so EM is identical (0.510 vs 0.510).

**A1 (top-k) RESULT.** EM % with 95% CI; adjacent-k exact McNemar tests, Holm-corrected over 9 tests:

| Retriever | k = 1 | k = 3 | k = 5 | k = 10 |
|---|---|---|---|---|
| R0 BM25 | 23.3 | 31.8 | 36.0 | 40.0 |
| R1 MiniLM | 35.2 | 44.2 | 47.1 | 50.2 |
| R4 bge-large | 42.1 | 49.3 | 50.3 | 52.0 |

- Every adjacent step is significant, except R4 k = 3 → 5 (+1.1 pp [−0.3, +2.5], n.s.).
- The k = 5 → 10 gains are: R0 +4.0, R1 +3.1, R4 +1.6 pp (p_Holm = 0.044).
- Mean prompt tokens roughly double from k = 5 to k = 10.
- *Exploratory, not pre-registered:* R4 at k = 1 vs R0 at k = 10: +2.2 pp [+0.1, +4.2], p = 0.044, a single post-hoc test.

**F1 ("I don't know" prompt A3, first 200 queries) RESULT.**
- IDK rates: R0 21.5%, R1 11.5%, R4 4.5%, GOLD 6.0%. The R4 and GOLD CIs overlap heavily: [2.0, 7.5] vs [3.0, 9.5].
- **IDK is concentrated on retrieval misses.** IDK | gold in top-5 is 1.3–2.4%; IDK | gold not in top-5 is R0 34.7%, R1 27.6%, R4 13.9%.
- EM change from A1 to A3: −1.0 to −3.0 pp; none significant at 0.05 (R4 p = 0.07).

**A3 (R6 Matryoshka dimensions 256 / 512) was NOT run. This is my omission.**
- v0.2.1 §6 marks A3 "cut first, keep if time permits".
- When I wrote the main-run plan (§3), I left A3 out of S3b without recording a decision. F4 did not require the cut.
- **Proposed:** run A3 as a separate, small session after S4. It needs only the saved S1 embeddings, truncated to 256 / 512 and renormalised, so no re-encoding is required. Estimated cost ≈ 1–1.5 GPU-hours.
- If not run, the Limitations text of v0.2.1 §6 applies. **Decision pending (user).**

**RQ3 / RQ4 analysis:** not yet done. I said I would do it during S3b and did not. It is next.

## 14. S4 verified (zip + log, 2026-09-29) and next steps

**S4 checks:**
- code `22f161a`; 11/11 packages match; 152 tests passed; restore ran first;
- judge Mistral-7B-Instruct-v0.3 @ `c170c708`, TP = 2, zero-shot;
- 3,999 judgements: 7 × 500, plus GOLD 499 (one GOLD output had stopped at the token limit and is not judgeable); **0 parse failures**;
- every verdict re-parses identically; S3a's generation files are byte-identical in the S4 zip.

**Human-validation sample drawn** (`src/annotation/sample.py`, seed 7, commit `1ddb17d`):
- 200 items: R0–R3 29 each, R4–R6 28 each; 14 NOT_SUPPORTED per retriever and the rest SUPPORTED.
- Each item carries its inclusion probability, so population-level agreement can be estimated with inverse-probability weights.
- The annotator file shows only the question, the answer and the passages.
- The key (judge verdicts) is kept git-ignored and is joined only after annotation.
- Passage texts come from a CPU-only Kaggle notebook (`notebooks/annotation_passages.ipynb`).

**Analysis tooling added:** `src/analysis/rq34.py` (RQ3 efficiency, RQ4 Pareto frontiers annotated with Holm significance).

**Sensitivity view (post hoc).** Cost differences ≤ 5% are treated as ties, because the measured run-to-run variability is ≤ 3.1%. This is labelled as not pre-registered wherever it is reported.
