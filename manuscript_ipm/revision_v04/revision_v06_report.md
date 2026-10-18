# Revision v0.6 report — Phase 3 (full-corpus NQ readers)

Date: 2026-10-18. Design: `design/addendum_v0.4.md` (committed 8d7bd2c before the session). Session: Kaggle, 2 × T4, 17 October 2026, code commit 80b647e (tree hash checked in Cell 1), notebook commit efadb53. Outputs: `paper_inputs/v04_s5nq/` (the session zip, with the four full-corpus top-100 run files, is committed as `v04_s5nq_results.zip`, sha256 in the `.sha256` file).

## Session integrity
- Environment: the generation lock matched except NumPy (2.1.3 vs 2.0.2), the same single difference as Phase 2; Python 3.13.15, PyTorch 2.13.0, CUDA 13.0, Transformers 5.17.0, vLLM 0.30.0; same reader revisions as Phase 2.
- Data revisions: BEIR-NQ corpus and qrels resolved to the pinned revisions (`download.jsonl`, sha_match true).
- Encoding: 2,681,468 passages each; MiniLM-L6 28 min, bge-small 42 min, e5-base-v2 95 min. Exact search, top 100, 3,452 judged queries.
- Readers: both completed all four conditions on 2,255 queries (18,040 outputs); unparsed outputs ≤ 1.0%.
- Checks: readers saw exactly the top five of the stored lists; re-encoding reproduced E9 Hit@5 and nDCG@10 exactly; the restored BM25 full run is byte-identical to the local copy; sub-corpus intervals recomputed equal the committed v0.3 ones.

## Pre-specified results
| Prediction | Qwen | Llama |
|---|---|---|
| P7: every reportable dense full-corpus T has upper bound < 1 | met (bge-small 0.399, e5 0.334) | met (bge-small 0.483, e5 0.300) |
| P8: T_full(e5) inside sub-corpus interval | met: 0.233 in [0.147, 0.356] | met: 0.195 in [0.053, 0.265] |
| T_full − T_sub (e5), paired bootstrap | −0.017 [−0.123, 0.087] | +0.035 [−0.065, 0.137] |

Decision rule of the addendum applied: the manuscript states that the NQ result held on the full corpus for these retrievers.

Secondary (pre-specified, descriptive): EM full − sub −0.018 to −0.028 for the dense retrievers (Holm over 4, p ≤ 0.014), BM25 −0.001/−0.005 (n.s.); dense Hit@5 −0.089 to −0.100, BM25 +0.000. Ratios against BM25: full 0.33–0.38, sub 0.29–0.34. Transitions keep their shape. Answer strings: 0.41–0.55 of dense misses (sub), 0.44–0.58 (full), BM25 0.33; hits 0.92–0.94; gold rule check 0.93. O8: 167/500 E12 questions have an answer-bearing hard negative; P5 on the other 333: −0.102 [−0.144, −0.060], p = 3.4 × 10⁻⁶ (registered P5 on all 500: −0.074).

Not pre-specified (labelled in the text): EM loss per Hit@5 loss on the full corpus (0.20–0.28); EM of M1 and MR on the 167 answer-bearing questions (0.671 vs 0.689); bge-small T_full − T_sub.

## Verification
- `python -m src.analysis.v04_full` (33 s): all internal asserts pass.
- `python3 -I scripts/independent_check_v06.py`: own normaliser, McNemar, Holm, quantiles, containment; 18,040 answers re-scored with 0 disagreements; 0 mismatches in means, ratios, intervals, P7/P8, transitions, answer-string shares (R1, R5), P5 subset.
- `numeric_report.py`: 306 macros; 71 independently recomputed, 0 mismatches.
- `check_revised.py`: no hard failures; abstract 247 words; highlights ≤ 78 characters.
- Tests: `pytest manuscript_ipm/revision_v04/tests` 14 passed; full suite see validation_report.md.

## Open
- **Human audit (registered):** `annotation/v04_miss_audit/miss_audit_v04.xlsx`, 100 seeded e5-base-v2 misses. The author labels Screen1 first, then Screen2; do not open `key.csv` before. Until done, the manuscript reports it as a deviation (item 8).
- O1 repository public/anonymised mirror; O5 title-page facts; O6 AI-use declaration wording (unchanged).

## Human audit (done 2026-10-18)
Labels by the author on the labeling page (`annotation/v04_miss_audit/miss_audit_page.html`; same 100 items and order as the workbook; the page ran from a downloaded copy, so labels were exported with `export_my_labels.html`). All 100 complete and internally consistent. A yes 41 [0.32, 0.51], no 53, unsure 6; B yes 41/41; C yes 59 [0.49, 0.68], no 31, unsure 10; EM 24; D yes 21 of 35. Sample re-drawn from the seed and recounted by `independent_check_v06.py`: 0 mismatches. Limits: one annotator; the export keeps only each item's last change time, so the order of Screen 1 and Screen 2 work is guaranteed by the page lock, not by the record.
