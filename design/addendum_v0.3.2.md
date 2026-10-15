# Design addendum v0.3.2 (2026-10-15) — implementation of E5, E6, E7, E9, E12 (session S4-NQ)

Status: written and committed **before** session S4-NQ runs. It fixes the details that addendum v0.3 left open.
Nothing here changes E1–E4 (already run) or the predictions P1–P6.

## E12 — mechanism tests (`src/generation/mechanism.py`)
- Queries: the first 500 of the seeded NQ order (all have at least one gold passage).
- g = the query's first gold passage in sorted id order. Hard negatives h1..h4 = the four highest-ranked non-gold passages
  of the v0.2 BM25 run on the main sub-corpus. Random negatives r1..r4 = four non-gold passages drawn uniformly from the
  main sub-corpus with `numpy.random.default_rng(20260926)`, in query order.
- Conditions (5 passages unless stated): M1 = [g,h1,h2,h3,h4]; M3 = [h1,h2,g,h3,h4]; M5 = [h1,h2,h3,h4,g];
  MR = [g,r1,r2,r3,r4]; MG = [g] (gold only, reference).
- P5 (distraction) is tested as **M1 vs MR** (gold at position 1 in both; only the negatives differ): two-sided exact McNemar
  at alpha = 0.05; P5 holds only if the difference is significant AND M1 has the lower EM.
- P6 (position) is tested on **M1, M3, M5**, three pairs, two-sided exact McNemar, Holm over 3.
- Reader: Qwen2.5-7B-AWQ, prompt A1, same decoding.

## E5 — reranker (`src/retrieval/rerank.py`)
- `BAAI/bge-reranker-base`, revision pinned at first use, fp16, max length 512, input "title text" as for the reader.
- Reranks the top 50 of the v0.2 runs R0 (BM25) and R5 (e5-base-v2) on the main sub-corpus for all 3,452 judged queries.
  Run names: X0 = reranked R0, X5 = reranked R5.
- Outcomes: Hit@5 and nDCG@10 (all judged queries), reader EM at k = 5 (Qwen, 2,255 queries), reranking latency per query
  (50 pairs in one batch, median and p95 over the first 500 queries after one warm-up query). Descriptive.

## E9 — full corpus
- R1, R2, R5 are encoded on the full BEIR-NQ corpus (2,681,468 passages) in this session; R0 uses the v0.2 full-corpus BM25 run
  (built for all judged queries in v0.2 S1). Exact search, top 100, all judged queries.
- Because this session runs a newer software stack (addendum v0.3.1), the **primary** comparison uses sub-corpus scores computed
  from the **same** new embeddings (rows of the main sub-corpus selected out of the full-corpus embeddings; the "exact" block of
  the E6 output). The v0.2 sub-corpus numbers are reported next to them. P3 (Kendall's tau = 1 on nDCG@10 over R0, R1, R2, R5)
  is evaluated on the primary comparison; the secondary one is reported.

## E6 — HNSW (scope reduced, deviation from v0.3)
- v0.3 named R1–R6. The v0.2 sub-corpus embeddings were not kept, and re-encoding all six models costs about 2.5 T4-hours.
  E6 therefore covers **R1, R2, R5 only**, using the sub-corpus rows of the E9 full-corpus embeddings. R3, R4, R6 are reported as not run.
- IndexHNSWFlat, M = 32, inner product, efConstruction = 200, efSearch ∈ {64, 128}; outcomes as in v0.3 plus build time.

## E7 — repeated cost measurement
- Five repetitions, one after the other, at the END of the session with nothing else running: `src.retrieval.throughput` for all six
  dense models (20,000 passages, batch 128, GPU; batch-1 query-encoding latency) and BM25 batch-1 query latency on the main
  sub-corpus (CPU). Dense FAISS search latency is not repeated (the sub-corpus embeddings of R3, R4, R6 do not exist in this session).
- Reported as median with min–max over the five repetitions. All five come from the same (v0.3.1) software stack.

## Order in the session
1. Parallel: GPU 0 encodes the full corpus (E9); GPU 1 reranks (E5), builds the E12 contexts and runs the reader for X0, X5 and M1–MG.
2. CPU: full-corpus search (E9), retrieval summaries, HNSW (E6).
3. E7 alone.
