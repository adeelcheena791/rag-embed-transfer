# Experiment Design — v0.1 (Phase 3 draft, awaiting approval)

Date: 2026-09-26 · Status: **PLANNED — NOT YET EXECUTED**. No data downloaded, no model run.
Budget: $0 (free Kaggle GPU, open weights, public data).

---

## 1. Research questions (final wording pending approval)

- **RQ1 — Retrieval.** Under identical conditions, how much do open-source embedders of different size and family differ in retrieval quality on open-domain QA? Are the differences statistically distinguishable from BM25 and from each other?
- **RQ2 — Transfer.** For a fixed reader LLM, what fraction of each embedder's retrieval advantage appears as answer-accuracy gain? Is the reader's accuracy *given that the gold passage was retrieved* the same across embedders?
- **RQ3 — Cost.** How do encoding throughput, query latency, memory and index size scale with parameters and dimension, on fixed free hardware (one T4 GPU; one CPU)?
- **RQ4 — Trade-off.** Which retrievers are Pareto-optimal for answer accuracy vs measured cost? Does the frontier look the same when drawn with retrieval metrics instead of answer metrics?

Claimed contribution type: **controlled empirical benchmark**, not a new method.
Closest prior work, to be cited as such:
- BERGEN (2024) — closed-book / oracle bounds.
- Leto et al. (2024) — gold-document recall vs QA accuracy.
- Cirillo et al. (2026) — embedder latency.

---

## 2. Data

### D1 — primary: Natural Questions (open-domain QA)

| Item | Detail | Status |
|---|---|---|
| Queries & relevance labels | BEIR-NQ: 3,452 test queries, 2.68M-passage Wikipedia corpus, ≈1.2 relevant passages/query | V: BEIR README |
| Gold answers | NQ-open validation: 3,610 questions, answer alias lists, CC BY-SA 3.0 | V: HF dataset card |
| Join | BEIR-NQ queries matched to NQ-open by normalised question text | Join rate **UNKNOWN**, measured in the pilot |

**Query sample.** From successfully joined queries, draw **1,000**, stratified by nothing, seed `20260926`. Unjoined queries are dropped and the count is reported.

**Corpus — pre-registered decision rule.**
- If the pilot shows that encoding the full 2.68M corpus with *all* models costs ≤ 10 T4-hours, use the full corpus. This means no subsampling bias.
- Otherwise use a pooled sub-corpus: all gold passages for the 1,000 queries, plus BM25 top-100 per query, plus 200k uniformly random passages (seed fixed).
- The sub-corpus is disclosed as inflating absolute scores and as biasing against BM25, because it contains BM25's hard negatives. Only relative comparisons would be claimed.

**Chunking.** Chunking is **fixed by the dataset**: pre-segmented Wikipedia passages. No re-chunking, so chunking is a controlled variable and not ablated. This removes a confounder rather than adding a factor.

**Fallback if the join rate is < 70%.** Switch to HotpotQA (BEIR: 7,405 test queries, 5.23M corpus, 2.0 relevant/query), whose answers come from the original release. That is decided only after the pilot.

### D2 — retrieval-only sanity check: SciFact (BEIR)

- 300 test queries, ~5K docs.
- Purpose: reproduce published nDCG@10 for our models within a small tolerance, which checks that the pipeline (prefixes, pooling, normalisation) is correct.
- No generation.
- Licence UNVERIFIED (believed CC BY-NC), to be checked before use.

---

## 3. Retrievers (independent variable)

| ID | Model | Params | Dim | Max len | Licence | Query/passage handling (per model card) | Why |
|---|---|---|---|---|---|---|---|
| R0 | BM25 (bm25s, k1=0.9, b=0.4) | — | — | — | — | Lower-cased, same tokeniser for all queries | Sparse baseline |
| R1 | sentence-transformers/all-MiniLM-L6-v2 | 22.7M | 384 | 256 | Apache-2.0 | None; mean pool + L2 | De-facto "cheap default" |
| R2 | BAAI/bge-small-en-v1.5 | 33.4M | 384 | 512 | MIT | Query instruction "Represent this sentence for searching relevant passages: " | BGE size ladder, small |
| R3 | BAAI/bge-base-en-v1.5 | ~0.1B | 768 | 512 | MIT | Same instruction | Ladder, base |
| R4 | BAAI/bge-large-en-v1.5 | ~0.3B | 1024 | 512 | MIT | Same instruction | Ladder, large |
| R5 | intfloat/e5-base-v2 | ~0.1B | 768 | 512 | MIT | "query: " / "passage: " prefixes; mean pool | Second family at matched size with R3 |
| R6 | Qwen/Qwen3-Embedding-0.6B | 0.6B | 1024 (MRL 32–1024) | 32k | Apache-2.0 | "Instruct: {task}\nQuery:{q}" on queries; needs transformers ≥ 4.51 | Recent LLM-based embedder |

All specifications above were read from the Hugging Face model cards on 2026-09-26. Exact revision hashes will be pinned at download.

**Declared unavoidable differences.** Each model uses its model-card prefix or instruction. Using a single prefix for all would handicap some models. This is a documented deviation from "identical preprocessing".

**Held equal across all dense models**
- max input length capped at 256 tokens for passages and queries (NQ passages are ~100 words, so truncation is rare — the pilot will measure the truncation rate)
- fp16 on GPU
- L2-normalised embeddings
- exact inner-product search (FAISS `IndexFlatIP`), so no approximate-search error

**Excluded on purpose**
- rerankers, hybrid retrieval, fine-tuning
- multilingual models (English data)
- MRL dimension truncation (optional ablation A3 only)

---

## 4. Generation (fixed)

**Reader.** Qwen/Qwen2.5-7B-Instruct-AWQ.
- Apache-2.0, 7.61B params, AWQ 4-bit, ungated (V: model card).
- Served with vLLM on one T4.
- **Fallback if vLLM or AWQ fails on T4:** Qwen2.5-3B-Instruct in fp16 via `transformers`. The switch would be applied to *all* conditions and reported.

**Prompt.** A fixed template: system instruction, then the top-k passages in rank order, then the question, then "Answer with a short phrase only."
- Greedy decoding (temperature 0), `max_new_tokens` = 32.
- The same prompt is used for every condition.

**Conditions**
- **Main (k = 5):** R0–R6 — 7 conditions.
- **Bounds:**
  - **CB (closed-book):** no passages. This measures what the reader answers from memory alone.
  - **GOLD:** only the gold passage(s). This measures reader accuracy when retrieval is perfect.
- **Total:** 9 conditions × 1,000 queries = **9,000 generations**.

---

## 5. Metrics (dependent variables)

**Retrieval** (per query; with ≈1.2 relevant passages/query, hit@k is the natural unit)
- Hit@k (≥1 gold passage in the top k) for k ∈ {1, 3, 5, 10}
- nDCG@10
- MRR@10

**Answer** (deterministic, primary)
- **EM:** SQuAD-style normalisation, matched against any gold alias.
- **Token-F1.**
- **Containment:** a gold alias appears in the output. It is robust to verbosity; reported as secondary.

**Transfer decomposition** (per retriever, from per-query data)
- Accuracy = Hit@5 · P(correct | hit) + (1 − Hit@5) · P(correct | miss)
- **Transfer ratio** vs reference R1: (EM_m − EM_R1) / (Hit@5_m − Hit@5_R1), with a bootstrap CI. It is reported only when the retrieval difference is itself significant, since the ratio is unstable when the denominator is near 0.
- CB and GOLD give the floor and ceiling for P(correct | miss) and P(correct | hit) respectively.

**Faithfulness** (secondary, LLM-judged)
- **Judge:** mistralai/Mistral-7B-Instruct-v0.3 (Apache-2.0, ungated — V: model card).
- It is a different model family from the reader, which reduces self-preference bias.
- **Task:** is the answer supported by the provided passages? (yes / no), with a fixed prompt.
- **Scope:** only the 7 main conditions × 500 queries, to bound GPU time.
- **Tooling:** RAGAS faithfulness is used *if* the pilot shows its local-LLM path works without parse failures above 5%. Otherwise we use our own single-prompt judge, documented.
- **Validation:** 150 stratified items labelled by 2 humans; Cohen's κ for human–human and human–judge agreement.
- **If human–judge κ < 0.4**, faithfulness results are reported as unreliable and excluded from conclusions.

**Efficiency** (single fixed Kaggle T4 plus the Kaggle CPU; exact models of both are logged)

| Metric | Protocol |
|---|---|
| Corpus encoding throughput (passages/s) | fp16, batch size 128, max_len 256, 3 timed runs on a 20k-passage slice after warm-up |
| Query encoding latency (ms) | Batch size 1, 50 warm-up + 500 timed queries; median and p95; on GPU **and** CPU |
| Search latency (ms) | FAISS flat on CPU, per query, median and p95 |
| Peak GPU memory (MiB) | `torch.cuda.max_memory_allocated` during encoding |
| Index size (bytes) | Measured; equals N × dim × 4 for flat fp32 |
| Parameters, dimension | Counted from the loaded model |
| End-to-end latency share | Retrieval vs generation time per query, reported descriptively |

**No monetary cost** is computed. All "cost" is expressed as these measured proxies.

---

## 6. Ablations (kept small)

| ID | Factor | Levels | Retrievers | Extra generations |
|---|---|---|---|---|
| A1 | top-k | 1, 3, 10 (5 is the main run) | R0, R1, R4 | 9 × 1,000 |
| A2 | Determinism check | Re-run 100 queries of R3 main | R3 | 100 |
| A3 (optional, cut first) | Qwen3-Embedding dimension | 256, 512 (MRL) | R6 | 2,000 + re-index |

---

## 7. Statistical analysis plan (pre-registered)

**Unit of analysis:** the query. All conditions share the same 1,000 queries, so every test is **paired**.

| Question | Measure | Test | Why this test |
|---|---|---|---|
| Retriever A vs B, binary per-query outcome (Hit@5, EM) | Difference in proportions | **Exact McNemar** | Paired binary outcomes; uses only the discordant pairs |
| Retriever A vs B, graded outcome (nDCG@10, F1) | Mean difference | **Paired bootstrap** (10,000 resamples, seed 0) plus Wilcoxon signed-rank | No normality assumption; bounded, skewed scores |
| Uncertainty for every point estimate | 95% CI | Percentile bootstrap over queries | — |
| Transfer ratio | Ratio with CI | Bootstrap over queries, applied jointly to numerator and denominator | — |
| Retrieval-vs-answer association across retrievers | Spearman ρ | Descriptive only | n = 7 points is too few for inference |

**Multiple comparisons.** All 21 pairwise retriever comparisons per metric, with Holm–Bonferroni correction within each metric family.

**Effect sizes.**
- Absolute difference in percentage points, with CI.
- Odds ratio from the discordant pairs for McNemar.

**Power.** The minimum detectable difference depends on the discordant-pair rate, which the pilot will estimate. It is reported before the main run. If it exceeds 5 points, the sample goes up to all joined queries.

**Randomness and seeds**
- Query sampling: seed `20260926`.
- Bootstrap: seed 0.
- Generation is greedy; residual GPU non-determinism is quantified by ablation A2.
- Only single runs are done beyond that, because of compute limits. This is declared as a limitation.

---

## 8. Compute budget (estimates, UNVERIFIED until the pilot)

| Stage | Estimate on a T4 |
|---|---|
| Full-corpus encoding, 6 dense models × 2.68M passages | Unknown — **the pilot decides between full corpus and sub-corpus** |
| 9,000 main + 9,000 ablation generations (~900 prompt tokens, ≤32 output tokens) | ~1–3 h |
| Judge, 3,500 items | ~1–2 h |
| Efficiency micro-benchmarks | < 1 h |

The target is ≤ 25 GPU-hours total, within about one weekly Kaggle quota (≈30 h, per a secondary source).

---

## 9. Pilot (Day 5) — go/no-go checks

1. BEIR-NQ ↔ NQ-open join rate.
2. Encoding throughput per model, which decides full corpus vs sub-corpus.
3. vLLM + AWQ runs on the T4; otherwise switch to the fallback reader.
4. SciFact nDCG@10 for R1–R6 lies within ±0.02 of the model cards' reported values where they report one. If it doesn't, the preprocessing is wrong: fix it before going further.
5. RAGAS local-judge parse-failure rate.
6. Discordant-pair rate, used for the power estimate.
7. Lock file generated with `pip freeze`, plus the Python, CUDA, driver and GPU record.

---

## 10. Threats to validity (to appear in Limitations)

- **Training contamination:** several embedders and the reader likely saw NQ or Wikipedia in training. CB accuracy measures the reader side; the embedder side cannot be measured and will be declared.
- **Single reader:** transfer may depend on reader strength. Only a 7B-class model is tested (3B if the fallback is used).
- **Single primary dataset** (factoid, Wikipedia): no claim about other domains.
- **EM strictness:** it under-counts correct paraphrases. Containment and human spot checks mitigate this.
- **Prefix/instruction differences** between models, declared in Section 3.
- **Kaggle hardware drift:** the GPU model and driver are logged per run; efficiency runs use one session.

---

## 11. Output layout

```
research/
  configs/        one YAML per experiment (seeds, model revisions, k, paths)
  src/{data,retrieval,generation,evaluation,analysis}/
  results/raw/    append-only JSONL per run (never overwritten; run_id = timestamp+hash)
  results/processed/  results/statistics/
  figures/  tables/  logs/  paper/  literature/  design/
  requirements.txt  requirements.lock  README.md  run_experiment.py
```
