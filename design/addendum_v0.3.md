# Design addendum v0.3 (2026-10-12) — Extension study for the journal version

Status: **PLANNED — NOT YET EXECUTED.** Written and committed before any v0.3 run.
Everything in v0.2 and addenda v0.2.1–v0.2.4 stays in force unless this file changes it.
The v0.2 main-run results are final and are not re-analysed under new rules; v0.3 adds new runs only.

## 0. Purpose

The v0.2 study has one dataset, one reader, an unvalidated judge, a partial top-k ablation, no reranker,
no approximate search, single cost measurements, an EM-only answer metric, a sub-corpus whose effect on the
ranking is unknown, low power for dense-vs-dense EM tests and two untested mechanism hypotheses.
v0.3 adds experiments, run by us on free Kaggle hardware, that address each of these. No data from other
papers is used as a result; other papers are cited only as comparison.

## 1. Fixed elements (unchanged from v0.2)

- Retrievers R0–R6, their revisions, max length 256, fp16, exact FAISS IndexFlatIP (except E6), BM25 k1 = 1.2, b = 0.75.
- Reader prompt A1 (and CB prompt A2, GOLD condition), greedy decoding, max 32 new tokens, stop at newline, k = 5 unless stated.
- SQuAD-style normalisation for EM and token F1.
- Statistics: paired tests (exact McNemar for binary, Wilcoxon signed-rank for graded), Holm within each pre-declared family,
  percentile bootstrap over queries (10,000 resamples, seed 0) in the seeded query order, transfer ratio T(R) with the v0.2 reporting rule
  (|ΔHit@5| > 2 pp and Holm-significant Hit@5 difference).
- Hardware: Kaggle sessions with two Tesla T4 GPUs and a 4-vCPU CPU. One GPU per process; two processes may run in parallel.

## 2. New experiments

### E1 — Second dataset: BEIR-HotpotQA (addresses: single dataset)
- Queries: BEIR-HotpotQA test queries (7,405). Reference answers come from `hotpotqa/hotpot_qa` (fullwiki, validation), joined by id.
  Verified on 2026-10-12 that BEIR query id `5a8b57f25542995d1e6f1371` equals the HotpotQA validation id with answer "yes".
  Gate: the join must cover ≥ 95% of queries; otherwise stop and record.
- Generation sample: the first **2,000** queries of a seeded order (seed 20260926).
- Corpus: a 300,000-passage sub-corpus built with the v0.2 rule (all gold passages of the 7,405 queries + BM25 negatives + random passages).
- Retrieval metrics: Hit@5 = at least one gold passage in the top 5 (primary, same as NQ); **Both@5** = both supporting passages in the top 5 (secondary); nDCG@10.
- HotpotQA has yes/no (comparison) questions. Results are reported overall and split by question type (bridge / comparison).
- Instructions: Qwen3-Embedding task text "Given a multi-hop question, retrieve Wikipedia passages that help answer the question".

### E2 — Second reader: Llama-3.1-8B-Instruct (addresses: single reader)
- Model: `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` (revision pinned at first download), vLLM, one T4, same prompts and decoding.
- Cells (reader × dataset): Qwen–NQ (exists), **Llama–NQ**, **Qwen–HotpotQA**, **Llama–HotpotQA**.
  Each new cell runs 9 conditions (R0–R6, CB, GOLD).
- Fallback if Llama cannot run (licence, loading, or > 2% empty outputs): `google/gemma-2-9b-it` in 4-bit; recorded as a deviation.

### E3 — Judge validation (addresses: unvalidated judge)
- The author labels the existing 200-item blind sample (`annotation/sample_v1_blind.json`) without seeing judge verdicts.
- Cohen's κ against the judge; gate as pre-registered in v0.2 (κ ≥ 0.4 use; 0.2–0.4 sensitivity analysis only; < 0.2 exclude).

### E4 — Full top-k ablation (addresses: k tested for 3 methods only)
- NQ, Qwen reader, k ∈ {1, 3, 10} for the four retrievers not yet run (R2, R3, R5, R6), 2,255 queries each.
- Transfer ratio at each k for all six dense encoders (exploratory in v0.2, **confirmatory** here for the prediction below).

### E5 — Reranker (addresses: no reranker)
- `BAAI/bge-reranker-base` (cross-encoder) reranks the top 50 of R0 (BM25) and R5 (e5-base-v2) on NQ.
- Outcomes: Hit@5, nDCG@10, reader EM at k = 5 (Qwen), reranking latency per query on T4.

### E6 — Approximate search (addresses: exact search only)
- FAISS IndexHNSWFlat (M = 32, efConstruction = 200, efSearch ∈ {64, 128}) for R1–R6 on the NQ sub-corpus, CPU.
- Outcomes: overlap of the top 100 with exact search, Hit@5 / nDCG@10 change, CPU latency. No reader run.

### E7 — Repeated cost measurement (addresses: single cost measurements)
- Encoding throughput (20,000 passages) and batch-1 query latency for all methods, **5 repetitions**, in one session.
- Reported as median with min–max; Pareto conclusions are checked against all five repetitions.

### E8 — Lenient answer metric (addresses: EM only)
- **Answer containment (AC):** 1 if a normalised reference answer is a substring of the normalised output. Computed from saved outputs, all cells.
- Transfer ratios are recomputed with AC and F1 in place of EM.

### E9 — Full-corpus check (addresses: sub-corpus bias)
- Full BEIR-NQ corpus (2,681,468 passages) for R0, R1, R2, R5; all 3,452 judged queries; Hit@5 and nDCG@10.
- Outcome: Kendall's τ between the sub-corpus and full-corpus method orderings, and per-method score drops.

### E10 — Pooled analysis (addresses: low power for dense-vs-dense tests)
- Per query and condition, EM is modelled with a GEE logistic regression (exchangeable correlation, clustered by query):
  EM ~ retriever + dataset + reader. Contrasts e5-base-v2 vs bge-large and e5-base-v2 vs Qwen3-Embedding-0.6B are tested pooled over the four cells.

### E12 — Mechanism tests (addresses: untested hypotheses in §6.1)
- NQ, Qwen reader, first 500 joined queries that have a gold passage. Five passages are always shown.
  - **Distraction:** gold + 4 hard negatives (top BM25 non-gold) vs gold + 4 random passages.
  - **Position:** gold at position 1, 3 or 5 among 4 hard negatives.
- Gold-only (k = 1 gold) is included as the reference.

## 3. Pre-registered predictions (confirmatory)

| ID | Prediction | Test |
|---|---|---|
| P1 | In each new cell, T(R) for the dense encoders is below 1 (upper 95% CI bound < 1 for every reportable ratio) | bootstrap CI |
| P2 | In each new cell, e5-base-v2 has EM ≥ bge-large and ≥ Qwen3-Embedding-0.6B (no Holm-significant EM gain for the two larger encoders) | exact McNemar, Holm over 21 pairs per cell |
| P3 | The sub-corpus ordering of R0, R1, R2, R5 on nDCG@10 is preserved on the full corpus (τ = 1) | descriptive |
| P4 | For each method, EM at k = 1 is lower than at k = 5 (all 7 methods) | exact McNemar, Holm over 7 |
| P5 | Gold + 4 hard negatives gives lower EM than gold + 4 random passages | exact McNemar, one family |
| P6 | Position of the gold passage changes EM (two-sided, any pair of 1/3/5) | exact McNemar, Holm over 3 |

Everything else (reranker, HNSW, cost repeats, AC metric, type split, pooled GEE) is **descriptive / secondary** and labelled so.
A failed prediction is reported as failed.

## 4. Budget, order and stopping rules

- Estimated GPU time: E1 retrieval ≈ 4 h; E2 readers ≈ 9 h; E4 ≈ 4 h; E5 ≈ 1.5 h; E7 ≈ 1 h; E9 ≈ 3 h; E12 ≈ 1.5 h; judge on new cells (first 500, optional) ≈ 1 h. Total ≈ 25 T4-hours; Kaggle weekly quota ≈ 30 h.
- Priority if the quota runs short: **E1 + E2 > E3 > E9 > E12 > E4 > E5 > E7 > E6.** Dropped experiments are reported as not run.
- Code for each session is frozen at a commit recorded in its notebook; the notebook stops if HEAD differs.
- Every failure, rerun or deviation is logged in `results/v03/failures.jsonl` and listed in the paper.

## 5. Reporting

- New results are labelled v0.3 and kept separate from v0.2 results.
- Numbers in the paper are generated from result files by the existing scripts (`make_numbers`, `audit_numbers`); no number is typed by hand.
