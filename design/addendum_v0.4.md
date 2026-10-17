# Design addendum v0.4 (2026-10-17) — full-corpus validation and passage-text checks (session S5-NQ)

Status: written and committed **before** session S5-NQ runs, after the v0.3 analysis and the v0.4/v0.5 manuscript audits.
It adds one session and its analysis. It does not change any earlier result, prediction or analysis. Everything below is
fixed now; anything not listed here that is computed later is post hoc and will be labelled so.

## Why
All answer results so far use 300,000-passage sub-corpora that contain every gold passage and favour the dense encoders
relative to BM25 (E9: dense nDCG@10 −0.083 to −0.086 on the full corpus, BM25 −0.002). Whether the NQ transfer ratios hold
on the full corpus is the main open validity question. Second, the explanation that many NQ Hit@5 "misses" still contain
the answer, and the E12 hard negatives possibly containing the answer (open issue O8), need the passage texts, which earlier
sessions did not store.

## Session S5-NQ (one Kaggle session, 2 × T4, frozen code commit, no other changes)
1. Restore the v0.2 S1 output (NQ sub-corpus, joined queries, main runs R0–R6, full-corpus BM25 run R0__full); download the NQ
   corpus and qrels at the pinned revisions.
2. Encode the full BEIR-NQ corpus (2,681,468 passages) with MiniLM-L6 (R1), bge-small (R2) and e5-base-v2 (R5): same revisions,
   prefixes, max length 256, fp16 as before. GPU 0: R5; GPU 1: R1, R2.
3. Exact inner-product search (FAISS flat), top 100, all 3,452 judged queries → run files `R1/R2/R5__full__*.jsonl`. These run
   files, and the restored `R0__full__*.jsonl`, are packaged and committed this time.
4. Passage-text checks (`src.analysis.passage_answers`, CPU): for the 2,255 NQ generation queries and the runs main:R0–R6 and
   full:R0,R1,R2,R5, at k = 5: Hit@5 and whether any top-5 passage contains a reference answer (SQuAD-normalised alias as a
   whole-token span of the normalised "title text"; the same rule as answer containment); the same flag for each gold passage;
   for the 500 E12 queries, how many of h1..h4 contain an answer; texts of the audit passages (step 7).
5. Readers on the full-corpus runs, k = 5, prompt A1, greedy, the 2,255 NQ generation queries: GPU 0 Qwen2.5-7B-AWQ
   (`configs/v03_nq_qwen.yaml`), GPU 1 Llama-3.1-8B-AWQ (`configs/v03_nq_llama.yaml`), conditions R0, R1, R2, R5. Same model
   revisions and stack as Phase 2 (addendum v0.3.1). If Llama fails, Qwen results are still analysed (recorded as a deviation).
6. Score answers with `src.evaluation.answer` (unchanged).

## Pre-specified analysis (Phase 3)
Notation: "sub" = the existing main sub-corpus results; "full" = this session. Same query set (2,255), same reporting rule
(|ΔHit@5| > 0.02 and Holm-significant Hit@5 difference over the pairs among R0, R1, R2, R5 within reader × corpus), same
bootstrap (B = 10,000, seed 0, `numpy.random.default_rng(0)`, one index for all quantities, so sub and full are paired).

**P7 (full-corpus transfer below 1).** For each reader, every reportable full-corpus ratio T_full(R) of a dense encoder
(R2, R5) against MiniLM-L6 has an upper 95% bound below 1 (unrounded; a bound of exactly 1 does not meet it).

**P8 (full-corpus transfer within the sub-corpus interval).** For each reader, the point estimate T_full(e5-base-v2) lies inside the
sub-corpus 95% interval of T_sub(e5-base-v2): Qwen [0.147, 0.356], Llama [0.053, 0.265]. We also report the paired difference
T_full − T_sub with its bootstrap interval; P8 is met only if the point estimate is inside the interval.

Secondary (descriptive, pre-specified):
- EM and Hit@5 on the full vs the sub-corpus per retriever and reader; paired exact McNemar for EM, Holm over the 4 retrievers per
  reader.
- T_full for BM25 vs MiniLM-L6 if reportable; full-corpus ratios against BM25.
- The query-level accounting (miss→hit, hit→hit, hit→miss, miss→miss) on the full corpus, as in the v0.5 manuscript.
- Answer strings in passages: for each run, the share of Hit@5 misses and of hits whose top 5 contains an answer string, with
  bootstrap intervals; the share of queries whose gold passage(s) contain an answer string (a check of the rule).
- O8: share of E12 queries with ≥ 1 hard negative containing an answer string; P5 (M1 vs MR, exact McNemar) repeated on the
  E12 queries with no such hard negative. This repeats a registered test on a subset; it is a sensitivity analysis.

## Human audit of NQ misses (after the session; author only)
Sample: 100 queries drawn with `numpy.random.default_rng(20261017)` from the NQ generation queries that are Hit@5 misses for
e5-base-v2 on the sub-corpus. The annotator (the author) sees the question, the reference answers and the five passages; not the
reader's answer, not the automatic answer-string flag. Labels, in this order:
(A) Does any passage contain a correct answer to the question? (yes / no / unsure) — and which passage;
(B) If yes, is that passage about the question's subject such that it should count as relevant? (yes / no).
Then, on a second screen that shows the Qwen reader's answer: (C) is the answer factually correct (yes / no / unsure);
(D) if (C) is yes but EM = 0, is it an alias or formatting difference (yes / no).
Reported as one annotator's labels, with counts and Wilson 95% intervals. No AI labels are used; an AI pre-screen is not used.

## Use in the manuscript (fixed now)
- If P7 and P8 are met for both readers, the paper states that the NQ result held on the full corpus for these retrievers.
- If either is not met, the full-corpus values become the headline NQ numbers next to the sub-corpus ones, and the abstract,
  discussion and conclusion say that the sub-corpus changed the size of the transfer.
- Either way the full-corpus results are reported in full, including a failure of Llama.

## Out of scope (stays a limitation)
Readers on HNSW results; HotpotQA on the full corpus; other readers. No result from this session will be rerun to change it.
