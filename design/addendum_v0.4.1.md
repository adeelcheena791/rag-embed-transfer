# Design addendum v0.4.1 (2026-10-18) — execution record of session S5-NQ

Record only; it changes nothing in addendum v0.4.

- Session S5-NQ ran on 17 October 2026 (Kaggle, 2 × T4) from notebook commit efadb53 with code commit 80b647e; Cell 1 checked the
  embedded tree hash. Start 11:04 UTC (restore, environment, downloads), encoding 11:04–12:40, search 12:41–12:44, passage-text
  checks 12:45–12:46, readers 12:46–14:32 (Qwen GPU 0, Llama GPU 1), scoring after.
- Every step completed; no step was rerun; Llama did not fail. Software as Phase 2 (NumPy 2.1.3 vs locked 2.0.2, as before).
- The output zip (20.2 MB) was uploaded by the author to the repository and moved to `paper_inputs/v04_s5nq/` with its sha256.
- The Phase 3 analysis was run as specified with `python -m src.analysis.v04_full`. Rounding of the P8 target intervals: the
  addendum quotes them to 3 decimals; the analysis uses the unrounded committed values; the verdicts agree under both.
- Analyses not listed in addendum v0.4 and reported as not pre-specified: EM loss per Hit@5 loss between corpora; M1 vs MR on the
  E12 questions that have an answer-bearing hard negative; T_full − T_sub for bge-small.
- Human audit: workbook generated with the registered seed (`annotation/v04_miss_audit/`, 414 e5-base-v2 sub-corpus misses, 100
  sampled). Labelling by the author had not started on 18 October 2026.
