# Study specification (verified against artifacts, 2026-10-17)

Status codes: **V** = verified in a committed artifact named in the row; **D** = stated in a dated design document only (not re-derived from run records); **U** = could not be verified.

## Research questions and hypotheses

| Item | Content | Status | Source |
|---|---|---|---|
| RQ1 | How much of a Hit@5 gain over MiniLM-L6 reaches EM, by reader and dataset | V | design/addendum_v0.3.md; manuscript §1 |
| RQ2 | Do bge-large / Qwen3-Emb-0.6B give higher EM than e5-base-v2 | V | same |
| RQ3 | Context: k, composition, gold position, reranking | V | same |
| RQ4 | Cost on free hardware; effect of sub-corpus and exact search | V | same |
| P1–P6 | Six predictions registered in v0.3 (12 Oct 2026) before any Phase 2 run | V (git date) | design/addendum_v0.3.md, git log |
| Non-confirmatory verdicts | P1/P2 for Qwen–NQ and P4 for R0/R1/R4 were observed before registration | V | addendum v0.3; Table predictions |

## Datasets and query counts

| Item | Value | Status | Source |
|---|---|---|---|
| NQ judged test queries (BEIR) | 3,452 | V | paper_inputs/data/processed/nq_join_stats.json |
| NQ joined to NQ-open answers | 2,255 (rate 0.653); 1,197 unmatched; 0 ambiguous | V | same |
| NQ generation set | all 2,255 joined queries, seeded order (seed 20260926) | V | Data loader; nq_query_order.txt |
| HotpotQA judged queries (BEIR) | 7,405 | V | paper_inputs/v03_hp/raw/subcorpus.jsonl (n_queries) |
| HotpotQA generation set | leading 2,000 of the seeded order; bridge/comparison split in ledger | V | hp_query_order.txt; v03_analysis.json by_type |
| Duplicate query ids in any generation set | none | V | audit_v04.py assertion |
| Missing answers for any query × condition | none | V | v03_analysis.py raises on any missing query; full rerun passed |

## Corpus construction

| Item | Value | Status | Source |
|---|---|---|---|
| Full corpora | NQ 2,681,468; HotpotQA 5,233,329 passages | V | ledger N.*; download records |
| Sub-corpus size | 300,000 per dataset | V | subcorpus.jsonl records |
| Rule | gold passages of all judged queries + top-d BM25 passages of every judged query + seeded uniform random fill | V | src/data/subcorpus.py |
| d | min(100, floor(0.5·(size − |gold|)/n_queries)): NQ 42, HotpotQA 19 | V | subcorpus.py; subcorpus.jsonl |
| NQ composition | 4,201 gold, 127,791 BM25-added, 168,008 random | V | ledger N.sc.nq.* (results/raw/subcorpus.jsonl) |
| HotpotQA composition | 13,783 gold, 118,726 BM25-added, 167,491 random | V | paper_inputs/v03_hp/raw/subcorpus.jsonl |
| Same corpus for all retrievers | yes (one id list per dataset) | V | subcorpus.py; data/processed/subcorpus_main_ids.txt; HNSW records carry one selection hash |
| Use of evaluation labels | gold labels of evaluation queries used to insert gold passages; evaluation queries' BM25 rankings used to mine negatives; no model trained | V | subcorpus.py |
| Consequence | BM25's own top competitors are kept, dense encoders' are mostly not → construction favours dense encoders relative to BM25; E9: BM25 nDCG@10 drop 0.002 vs dense 0.083–0.086 | V (data) / interpretation | E9 in v03_analysis.json |
| Full-corpus check | BM25, MiniLM-L6, bge-small, e5-base-v2; NQ; all 3,452 judged queries; retrieval only | V | v03_nq_full_summary.json |

## Retrievers

| Item | Value | Status | Source |
|---|---|---|---|
| BM25 | bm25s 0.3.11, Lucene variant, k1 1.2, b 0.75, lower-case, no stemming/stop words | V | configs/main.yaml; env records |
| Dense encoders | all-MiniLM-L6-v2, bge-small/base/large-en-v1.5, e5-base-v2, Qwen3-Embedding-0.6B at pinned revisions | V | configs/resolved_revisions.json; Table revisions |
| Encoding | fp16, max 256 tokens (MiniLM truncation rate 5.2% of passages), batch 128, documented prefixes | V | configs; throughput.jsonl |
| Search | exact inner product, FAISS flat; top 100 | V | configs/main.yaml |
| HNSW (E6) | M 32, efConstruction 200, efSearch 64/128; R1, R2, R5; NQ sub-corpus; 3,452 judged queries; 4 FAISS threads | V | paper_inputs/v03_nq_s4/raw/hnsw.jsonl |
| Reranker (E5) | bge-reranker-base over top 50 of BM25 and e5-base-v2; NQ; Qwen reader | V | rerank.jsonl; X0/X5 answer files |

## Readers, prompts, generation

| Item | Value | Status | Source |
|---|---|---|---|
| Readers | Qwen2.5-7B-Instruct-AWQ; Meta-Llama-3.1-8B-Instruct-AWQ-INT4 | V | configs/v03_*.yaml; revisions table |
| Serving | vLLM 0.30.0, one T4 each, half precision, max_model_len 4,096 | V | configs; env records |
| Decoding | greedy, max 32 new tokens, stop at newline | V | configs (generation.max_tokens 32, stop ["\n"]) |
| Prompt A1 | top k passages in rank order, short-answer instruction | V | Appendix prompts; src/generation |
| Conditions | CB (A2, no passages), R0–R6 at k = 5, GOLD (A1 with gold only) | V | answer files |
| k ablation (E4) | k ∈ {1,3,5,10}, Qwen, NQ, all retrievers | V | answer files |
| Context experiment (E12) | 500 NQ queries; M1/M3/M5 (gold at 1/3/5 + 4 top BM25 non-gold), MR (gold@1 + 4 random), MG (gold only) | V | v03_nq_s4 answer and run files |

## Metrics

| Metric | Definition | Status |
|---|---|---|
| Hit@5 | 1 if ≥1 gold passage in top 5 | V (src/common/ir_metrics.py) |
| Both@5 (HotpotQA) | 1 if both gold passages in top 5 | V (v03_analysis.py) |
| nDCG@10 | binary gains | V |
| EM | SQuAD normalisation + typographic quotes, en/em dashes, ellipsis | V (src/evaluation); a plain SQuAD normaliser would change 91 of 131,855 scored answers, 0.07% (ledger IC.*) |
| Token F1, answer containment | secondary | V |
| Transfer ratio T(R) | (mean EM_R − mean EM_R1)/(mean Hit@5_R − mean Hit@5_R1), same queries, reported only if \|ΔHit@5\| > 0.02 and Holm-significant | V (stats.transfer_ratio) |
| Units | proportions (0–1) in both numerator and denominator; the ratio is unit-free ("EM points per Hit@5 point") | V |

## Statistical analysis plan

| Item | Value | Status |
|---|---|---|
| Unit of inference | query (paired across retrievers within a cell) | V |
| Binary paired tests | exact McNemar; Holm over the 21 retriever pairs within a cell and metric | V |
| Intervals | percentile bootstrap over queries, B = 10,000, seed 0; numerator and denominator of T resampled jointly | V |
| Pooled model | GEE logistic, EM ~ retriever + dataset + reader, exchangeable within query cluster (dataset:qid), sandwich SE; 59,570 rows, 4,255 clusters | V (refit identical) |
| Post hoc (v0.4) | decomposition shares (two orderings), BM25-baseline ratios, GEE with retriever × dataset and × reader interactions | new, labelled post hoc |
| Judge validation | Cohen's κ, one annotator, 200 stratified items, IPW for population rates; κ < 0.2 → exclude | V (recomputed κ = 0.127) |

## Hardware, runtime, repetitions

| Item | Value | Status |
|---|---|---|
| Hardware | Kaggle, 2 × Tesla T4 (15 GB), 4 vCPU, 31 GB RAM | V (env records) |
| Software | PyTorch 2.13.0/CUDA 13.0, vLLM 0.30.0, Transformers 5.17.0; Python 3.12.13 → 3.13.15 and NumPy 2.0.2 → 2.1.3 between phases | V |
| Repetitions | answer conditions run once (deterministic within session; 100/100 within-session rerun identical; 396/397 identical gold-only prompts across phases); E7 throughput 5 repetitions | V |
| Seeds | query order 20260926; bootstrap 0; decoding greedy; random passages seeded | V |

## Exclusions, missing observations, failed runs

All seven deviations listed in manuscript §4.1 are confirmed in design addenda v0.2.2–v0.3.2 (failed initial Phase 2 session, torchaudio mismatch, reduced E6 scope, no RAGAS, no Matryoshka ablation, one annotator, 499/500 judged gold items, CPU benchmark 8 h 24 min instead of 1 h, no failures.jsonl). No answer was dropped from any analysis.

## Items that could not be verified (U)

- Whether E12 hard negatives contain the answer string (passage text is not in the repository) — open issue O8.
- Training-data contamination of readers and encoders with NQ/HotpotQA — not measurable here.
