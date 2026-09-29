# Handoff: RAG embedding-transfer paper (state as of 2026-09-29)

Repo: github.com/adeelcheena791/rag-embed-transfer (main, latest commit a9de1bd). Frozen experiment code: 22f161a (branch main-v1). Pre-registration: `design/experiment_design_v0.2.md` + addenda v0.2.1–v0.2.4 (§15 records the A3 cut). Budget: $0 (Kaggle free T4 only; ~10.5 GPU-h left). Rules: never fabricate; label FACT / RESULT / OBSERVATION / INTERPRETATION / HYPOTHESIS / LIMITATION; never overwrite raw results; report failures.

## 1. Title, RQs, lead finding

**Working title:** How Much Retrieval Quality Reaches the Answer? Accuracy–Efficiency Trade-offs of Open-Source Embedding Models in RAG

- **RQ1:** How do open embedding models differ in retrieval quality on one fixed corpus?
- **RQ2:** How much of a retrieval gain transfers to end-to-end answer accuracy (EM)?
- **RQ3:** What does each model cost (encoding time, latency, index size, memory) on free hardware?
- **RQ4:** Which models are Pareto-optimal on quality vs cost?

**Lead finding (within our setup):** Only about 20–25% of a dense retriever's Hit@5 gain appears as EM gain. The two largest encoders (bge-large, Qwen3-Emb-0.6B) cost 3–7.5× more to encode than e5-base-v2 and are dominated by it; they give no detectable EM gain.

## 2. Setup (facts)

- Data: BEIR-NQ test (3,452 judged queries) joined to NQ-open answers → 2,255 joined queries (65.3%). Corpus: 300,000-passage sub-corpus = 4,201 gold + 127,791 BM25 top-42 negatives + 168,008 random.
- Retrievers: R0 BM25 (bm25s, k1 1.2, b 0.75); R1 all-MiniLM-L6-v2; R2 bge-small-en-v1.5; R3 bge-base-en-v1.5; R4 bge-large-en-v1.5; R5 e5-base-v2; R6 Qwen3-Embedding-0.6B. Exact FAISS flat IP, max_len 256, fp16. Revisions pinned in `configs/resolved_revisions.json`.
- Reader: Qwen2.5-7B-Instruct-AWQ, vLLM 0.30.0, one T4, greedy, k = 5, max 32 tokens. Conditions: CB (no passages), R0–R6, GOLD.
- Judge: Mistral-7B-Instruct-v0.3, zero-shot SUPPORTED / NOT_SUPPORTED (Appendix B), first 500 queries.
- Stats: paired; exact McNemar (binary), Wilcoxon (graded); percentile bootstrap 10,000, seed 0; Holm over 21 pairs. Transfer ratio ΔEM/ΔHit@5 vs R1, reported only if |ΔHit@5| > 2 pp and Holm-significant. MDE for EM ≈ 3.4 pp.

## 3. Key numbers

**Sanity:** SciFact nDCG@10 within 0.0014 of model cards (R2–R5). Pilot runs deterministic across sessions.

**S1 / RQ1 (joined, n = 2,255):** Hit@5 / nDCG@10
BM25 36.3 / 27.7 · MiniLM 66.4 / 52.0 · bge-s 72.9 / 58.0 · Qwen3 74.4 / 60.0 · bge-b 77.0 / 62.6 · bge-L 79.6 / 64.0 · e5-b 81.6 / 66.7.
nDCG: 21/21 pairs significant. Hit@5: 20/21 (bge-s vs Qwen3 n.s., −1.5 [−3.3, 0.3]). Same order on all 3,452 queries; joined vs not-joined Hit@5 CIs all include 0. e5-b vs bge-b (same size): +4.1 pp nDCG, significant.

**S3a / RQ2 EM (n = 2,255):** CB 14.9 · BM25 36.0 · MiniLM 47.1 · bge-s 48.4 · Qwen3 49.2 · bge-b 49.3 · bge-L 50.3 · e5-b 51.0 · GOLD 65.3.
Dense vs BM25 +11.2 to +15.0 pp (all significant). Dense pairs: 3/15 significant (MiniLM < bge-L −3.2; MiniLM < e5 −3.8; bge-s < e5 −2.6). e5 vs bge-L +0.6 [−0.9, 2.1]; e5 vs Qwen3 +1.8, p_Holm 0.27.
P(EM | hit@5) dense 56–58%, BM25 63.2%. P(EM | miss) dense 24–26%, BM25 20.4%.
Transfer ratio vs R1: bge-s 0.19 [−0.07, 0.44]; bge-b 0.20 [0.04, 0.36]; bge-L 0.24 [0.11, 0.36]; e5-b 0.25 [0.15, 0.36]; Qwen3 0.25 [0.05, 0.45]; BM25 0.37 [0.31, 0.43]. Spearman(Hit@5, EM) = 1.0 (descriptive, 7 points).

**S3b ablations:** A2 determinism 100/100 identical. A1 EM at k = 1/3/5/10: R0 23.3/31.8/36.0/40.0; R1 35.2/44.2/47.1/50.2; R4 42.1/49.3/50.3/52.0. All adjacent steps significant except R4 3→5; R4 5→10 +1.6 (p_Holm 0.044). Exploratory: R4@k1 vs R0@k10 +2.2 [0.1, 4.2]. Prompt tokens ≈ double from k 5→10. F1 (n = 200) IDK rate: R0 21.5%, R1 11.5%, R4 4.5%, GOLD 6.0%; IDK | miss R0 34.7, R1 27.6, R4 13.9. Prompt A3 variant: EM −1 to −3 pp, n.s.

**RQ3 (T4 + 4-vCPU Xeon; order R1, R2, R3, R5, R4, R6):**
Encode 300k on T4 (s): 158 / 250 / 584 / 592 / 1,753 / 4,431. CPU encode (p/s): 38.7 / 20.0 / 6.39 / 6.19 / 1.92 / 0.31. Query encode GPU/CPU (ms): 5.6/14.3, 9.7/35.3, 9.6/66.5, 9.6/67.8, 17.4/259.4, 46.9/719.7. Exact search 39 / 81 / 142 ms at 384 / 768 / 1024 d; BM25 21.9 ms. Index 439 / 879 / 1,172 MB. Peak GPU mem R6 7,548 MiB vs R1 352. Pilot→main throughput variability −0.7 to +3.1%. Params: 22.7M, 33.4M, 109.5M, 109.5M, 335.1M, 595.8M.

**RQ4:** bge-L and Qwen3 dominated by e5-b on every cost axis and quality metric (retrieval significantly; EM n.s.). Frontier (encode time or latency): BM25 → MiniLM → bge-s → bge-b → e5-b. Dense EM steps along it n.s. bge-b's place rests on a 1.4% cost gap (within noise); post-hoc ≤5% tie rule → e5-b dominates bge-b.

**S4 faithfulness — UNVALIDATED (n = 500):** SUPPORTED BM25 79.0 · MiniLM 88.6 · bge-s 89.8 · bge-b 90.2 · bge-L 91.2 · e5-b 91.6 · Qwen3 89.6 · GOLD 96.6. 0/3,999 parse failures. Only BM25 vs dense significant. Interpret only after κ: ≥ 0.4 use; 0.2–0.4 report with sensitivity; < 0.2 exclude. Pilot hinted the judge is lenient.

## 4. Source papers (from chat notes; PDFs were never read here — verify every detail before citing)

1. **MTEB** (Muennighoff et al., EACL 2023). A large benchmark of text-embedding models across many tasks, retrieval included. It showed no single model wins every task. We use it as the standard source of retrieval scores and model-card numbers. It measures embedding quality in isolation, not downstream RAG answers. Our post-hoc SciFact check matched MTEB references (R1 0.645 vs 0.655 ours; R6 0.697 vs 0.699).
2. **Baidya 2026, "Choosing a Text Embedding Model"** (arXiv 2607.23507 — ID and exact title UNVERIFIED; check). Practical benchmarking of embedding models on quality and cost. We cite it only as consistent with our observation that model family and training matter more than size. Do not attribute specific numbers until verified.
3. **RAGAs** (Es et al., EACL 2024 demo). Reference-free LLM-based RAG metrics, including faithfulness. We planned to run it and dropped it (D2): local deployment needs infrastructure untested on our hardware. Our own single-prompt judge replaces it.
4. **ARES** (Saad-Falcon et al., NAACL 2024). Automated RAG evaluation with trained LLM judges plus a small human-labelled set to calibrate them (prediction-powered inference). It motivates our human check of the judge on 200 items.

**Close prior work to position against (verify each):** BERGEN (RAG benchmarking library); Leto et al. 2024 (distracting passages hurt readers); Cirillo et al. 2026; RAGPerf; Oro et al. 2025; eRAG (retrieval evaluation via downstream LLM output). Novelty claim wording: "We are not aware of work that measures retrieval-to-answer transfer ratios jointly with measured cost on free hardware." Never "first".

## 5. Style rules

Active voice. Short sentences. One claim per sentence. Numbers with CIs and n. "Within our setup" on every general claim. "No difference detected" rather than "no difference". No hype, no "delve", "crucial", "landscape", "notably", "furthermore", no em-dash chains, no rule-of-three flourishes. Hypotheses labelled as such.

## 6. Figures (8)

F1 system architecture · F2 methodology/pipeline (stages S1–S4) · F3 embedding comparison (RQ1 bars with CIs) · F4 retrieval quality vs latency (scatter) · F5 EM vs efficiency (scatter) · F6 Pareto frontier with significance marks · F7 error/failure analysis (P(EM|hit/miss), IDK) · F8 top-k ablation (A1 lines).

## 7. Tables (6)

T1 model characteristics (params, dim, revision) · T2 dataset/sub-corpus · T3 retrieval (RQ1) · T4 efficiency (RQ3) · T5 RAG evaluation (EM, F1, conditional EM, faithfulness) · T6 statistical comparisons (pairwise, Holm, transfer ratios).

## 8. Limitations (all must appear)

One dataset (NQ), one reader (7B AWQ), one hardware setting. BM25 disadvantaged: sub-corpus holds its own top-42 negatives. Sub-corpus inflates absolute scores; claims are relative. EM is strict. Possible training contamination (not measurable). One human annotator. Judge unvalidated and possibly lenient. A3 cut (user's verbatim text in addendum §15). RAGAS dropped (user's verbatim text: "RAGAS was not run because its local deployment requires additional infrastructure untested on our hardware; our own single-prompt judge (Appendix B) is the primary faithfulness measure and is validated against human labels." — final clause only if κ ≥ 0.4). Exact search only, no ANN/quantisation. Joined subset is 65.3% of queries. Pre-registration deviations: D1 (join gate), M1 (sub-corpus), F4.2 clarification; S1 CPU-time estimate error (1 h → 8 h 24 m); S3a first attempt failed (restore order).

## 9. Exact next steps

1. User annotates 200 items (blind page; key in `results/annotation/sample_v1_key.json`, git-ignored, never shown to annotator). Export JSON/CSV.
2. Compute Cohen's κ (and weighted by inclusion probability) judge vs human; per-retriever agreement; decide faithfulness use per the bands.
3. Generate F1–F8 and T1–T6 from `results/statistics/*.json` and `tables/*.csv`.
4. Write paper: Abstract + Introduction drafted (awaiting review) → Related Work → Method → Results → Discussion → Limitations → Appendix B (judge prompt).
5. Hostile self-review by a separate agent; check every number against JSON.
6. Release: tag/branch, README reproduction steps, data cards, Zenodo-style archive if free.
