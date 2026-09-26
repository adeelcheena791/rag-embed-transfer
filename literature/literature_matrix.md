# Literature Matrix — Phase 2, pass 1 (2026-09-26)

Status legend: **V** = checked on the arXiv/ACL/publisher page this session · **U** = UNVERIFIED (from memory or incomplete page) · "—" = not found in what was read.
Scope of this pass: abstracts / HTML pages via web search + fetch. Full texts not yet read. Google Scholar not searched directly (no browser in session).

## 1. Overlap table

| # | Paper | Embedder comparison | End-to-end RAG | Retrieval metrics | Answer metrics | Efficiency measured | Stat. tests | Data |
|---|---|---|---|---|---|---|---|---|
| 1 | Caspari et al. 2024, Beyond Benchmarks (arXiv 2407.08275; CEUR-WS Vol-3784) | ✓ (similarity, open + proprietary) | — | overlap/similarity only (CKA, Jaccard, rank sim.) | — | — | U | 5 BEIR sets |
| 2 | IRSC (arXiv 2409.15763; Springer chapter) | ✓ (Snowflake-Arctic, BGE, GTE, M3E) | — | ✓ (new SSCI, RCCI) | — | — | U | own, multilingual |
| 3 | Strich et al., T²-RAGBench, EACL 2026 | ✓ (ablation) | ✓ | ✓ | ✓ | partial (corpus size) | U | 23,088 text+table fin. QA, CC-BY-4.0 |
| 4 | Es et al., RAGAs, EACL 2024 demo | — | method | context relevance | faithfulness, answer relevance | — | — | WikiEval |
| 5 | Saad-Falcon et al., ARES, NAACL 2024 | — | method | context relevance | faithfulness, answer relevance | — | PPI conf. intervals (U) | KILT/SuperGLUE (U) |
| 6 | Muennighoff et al., MTEB, EACL 2023 | ✓ (many) | — | ✓ | — | partial: speed vs quality (U) | — | 56 datasets / 8 tasks (U) |
| 7 | Thakur et al., BEIR, NeurIPS D&B 2021 | ✓ | — | ✓ nDCG@10 | — | ✓ retrieval latency & index size (U) | — | 18 datasets |
| 8 | Rau et al., BERGEN, Findings EMNLP 2024 (arXiv 2407.01102) | ✓ (BM25, SPLADE, RetroMAE, BGE/E5/GTE) + rerankers | ✓ | ✓ | ✓ EM/F1 + LLM-eval | not systematically (notes 20 vs 6 docs ≈6× time) | U | 10+ QA sets (KILT NQ, HotpotQA, TriviaQA …) |
| 9 | Salemi & Zamani, eRAG, SIGIR 2024 (arXiv 2404.13781) | — | ✓ | ✓ | ✓ downstream | ✓ (eval cost: ~50× less GPU memory) | Kendall τ | several (U) |
| 10 | Xiong et al., MIRAGE, Findings ACL 2024 (arXiv 2402.13178) | ✓ retrievers (U: BM25, Contriever, SPECTER, MedCPT) | ✓ | — | accuracy | — | U | 7,663 medical Qs |
| 11 | Abdallah et al., From Retrieval to Generation (arXiv 2502.20245) | ✓ (BM25, DPR, MSS, Contriever …) | ✓ | ✓ | EM | minimal | U | NQ, TriviaQA, WebQ, BEIR |
| 12 | Oro, Granata, Ruffolo 2025, BDCC 9(5):141 | ✓ (12 embedders) | ✓ (4 LLMs) | ✓ nDCG, MAP, R@k, P@k | ✓ ROUGE, F1, BERTScore, LLM-judge | — (not in visible text) | U | 7 sets EN/IT |
| 13 | Alinejad et al. 2024, LLM-retEval (arXiv 2406.06458) | — (e5-large-v2 only) | ✓ | ✓ | ✓ vs gold-doc answers | — | U | NQ-based (U) |
| 14 | Cirillo et al. 2026, Google Embeddings 2 vs open-source (arXiv 2605.23618) | ✓ (BGE-M3, E5-L, mE5-L, LaBSE, mMPNet + commercial) | — (explicitly retrieval only) | ✓ R@k, MRR, nDCG@10 | — | ✓ CPU query latency (median/p95), storage | — (single run, stated) | 4 BEIR + IT-RAG-Bench |
| 15 | Baidya 2026, Choosing a Text Embedding Model (arXiv 2607.23507) | ✓ | — | ✓ | — | ✓ query latency | — | same 4 BEIR sets as #14 |
| 16 | Li et al. 2026, RAGPerf (arXiv 2603.10765) | ✓ (MiniLM, MPNet, GTE-large, ColPali, CLIP) | ✓ | context recall | factual consistency, query accuracy | ✓ **systems-level** latency breakdown, memory | U | Wikipedia, arXiv PDFs, code, audio |
| 17 | El Assadi, Muennighoff, Lee 2026, The Embedder's Dilemma (arXiv 2608.12875) | ✓ (26 embedders vs 10 LLMs) | — | ✓ | — | ✓ throughput on H100, $ per pass | ✓ | 37 tasks |
| 18 | Mathur et al. 2026, Retrieval Improvements Do Not Guarantee Better Answers (arXiv 2603.24580) | — (one ColBERT retriever, fine-tuned vs not) | ✓ | ✓ | ✓ relevance, faithfulness | — | U | AI-policy QA |
| 19 | Muthyam 2026, Sensitivity/Robustness/Stability in RAG (arXiv 2606.28337) | — (MiniLM only) | ✓ | Hit@k | EM, F1 | partial (tokens, latency) | U | SQuAD |
| 20 | Tamber et al. 2025, FaithJudge (arXiv 2505.04847) | — | judge method | — | faithfulness | — | U | summarisation, QA, data-to-text |
| 21 | Leto et al. 2024, Toward Optimal Search and Retrieval for RAG (arXiv 2411.07396; Intel Labs / CU Boulder) | partial (BGE-base vs ColBERTv2) | ✓ (Llama-2-7b-chat, Mistral-7B-Instruct-v0.3) | ✓ gold-doc recall, ANN search recall | ✓ QA correctness | speed/memory of ANN search (qualitative) | U | ASQA, QAMPARI, KILT-NQ |

Pass 2 note (2026-09-26): #21 varies *search* accuracy (ANN approximation), not the embedder. It reports that gold-document recall matters far more for QA than ANN search recall. **Our transfer question is closest to this paper; cite it prominently.** The Semantic Scholar API returned HTTP 429 (rate limit) and was not searched.

Found but not yet read: arXiv 2604.01733 (BM25→Corrective RAG, text+table), 2609.17709 (dynamic retriever/generator selection), 2408.08067 (RAGChecker), 2502.15025, 2506.06339 (Arabic RAG, RAGAS metrics, BGE-M3 vs mE5-L).

## 2. What the evidence shows (FACT, from the pages read)

- Embedder comparisons on **retrieval + efficiency** without answer quality exist (#14, #15, #17).
- Embedder comparisons on **retrieval + answer quality** without efficiency exist (#8, #12; also #10, #11 for older retrievers).
- **Systems-level end-to-end efficiency** with a few embedders exists (#16). It reports generation dominating latency.
- The claim "better retrieval metrics do not reliably yield better answers" is **already reported** in several settings (#9, #13, #18, #19). #8 reports the opposite direction for strong reranking on some QA sets. So RQ2 as a general question is **not new**.

## 3. Integrity note

\#14 and #15 report the same headline number for a commercial model (nDCG@10 = 0.638 on BEIR), the same 4 BEIR datasets and overlapping model lists. Their relationship is **not determined**. We should cite measured numbers from one only after reading both full texts.

## 4. Gap status: HYPOTHESIS (narrowed), not a claim

Not yet found in the 20 papers above: one controlled setup that measures, across a **size ladder of open embedders on fixed free hardware**:
(a) retrieval quality,
(b) end-to-end answer quality with a fixed reader, bounded by no-retrieval and gold-context runs,
(c) encoding/query cost and memory,
(d) per-query paired significance tests,
and then quantifies **how much of each embedder's retrieval advantage survives into answer quality per unit of compute**.

Full-text check of the three closest papers (2026-09-26, via HTML full text; details marked V where the page stated them):

- **#16 RAGPerf** (V)
  - Compares 3 text embedders (MiniLM-L6, MPNet-base, GTE-large) with RAGAS quality metrics.
  - Provides **no per-embedder efficiency breakdown**.
  - Reports **no statistical tests**.
  - Focus is on vector DB and generator.
- **#8 BERGEN** (V)
  - Reports answer metrics per retriever with a fixed generator, plus closed-book and oracle bounds.
  - Its main tables cover RetroMAE, BM25 and SPLADE-v3 (+ reranking), **not a ladder of open sentence-embedders**.
  - Has **no efficiency measurement** and **no significance tests**.
  - Its closed-book/oracle bounding is prior art for our design and must be cited as such.
- **#12 Oro et al.** (V)
  - Compares embedders on **retrieval only**.
  - End-to-end RAG uses **one** embedder (Cohere embed-multilingual-v3.0, commercial) while LLMs vary.
  - Reports **no efficiency** and **no significance tests**.

Result of this pass: the narrowed gap in (a)–(d) is **supported by the 20 papers read**. It is **not proven**: the search was not exhaustive (no direct Google Scholar / Semantic Scholar sweep).

## 5. Bibliography records (fields left blank where not verified)

Each record will be completed with authors/venue/DOI from the publisher page before entering the .bib file.
