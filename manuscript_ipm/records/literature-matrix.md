# Literature matrix and novelty assessment (Gate 4) — 2026-10-16

Verification level: **FT** = full-text passages read for the cited claim; **AB** = abstract read (claims limited to what the
abstract states); **MD** = metadata only (cited for existence of a dataset/model/tool). Source findings are paraphrased from the
source; "Relation" is our interpretation.

## 1. Closest work
| ID | Work | Type | Setting | Source-reported finding | Relation to this manuscript | Level |
|---|---|---|---|---|---|---|
| L01 | Peng et al. 2026, NeocorRAG (WWW '26) | conf. | HotpotQA, 2Wiki, MuSiQue, NQ; graph/structure RAG methods; 3B and 70B readers | Defines Recall Conversion Rate RCR = F1 / Recall@5 over a query set; across RAG methods RCR decays almost linearly as Recall@5 rises | Closest metric. RCR is a **level ratio** across different pipelines; our T(R) is a **ratio of paired differences** against a fixed baseline encoder on the same queries, with bootstrap CIs and a pre-registered reporting rule. Their decay is consistent with our low NQ ratios, but a level ratio falls mechanically when F1 saturates, so the two are not interchangeable | FT |
| L02 | Leto et al. 2024 (arXiv) | preprint | ASQA, QAMPARI, NQ (KILT); BGE-base, ColBERTv2; Mistral, Llama readers | One gold passage raises correctness most; gold-document recall matters more than ANN search recall; lowering ANN recall has minor QA cost | Supports our E6 reading (HNSW overlap ≥0.995 → no answer-relevant change expected). Their retriever axis is one dense model; ours is seven retrievers | FT |
| L03 | Salemi & Zamani 2024, eRAG (SIGIR) | conf. | KILT tasks | Relevance-label retrieval metrics show small correlation with downstream RAG performance; per-document LLM utility labels correlate better | Motivates measuring answer effect directly rather than inferring it from Hit@k | AB |
| L04 | Samuel et al. 2026 (ICTIR '26) | conf. | TREC NeuCLIR 24, RAG 24, WikiVideo; 15 text retrieval stacks; report generation | Coverage-based retrieval metrics correlate positively with nugget coverage at topic (r 0.26–0.56) and system level; relevance nDCG near zero on NeuCLIR | **Counter-evidence to a strong "retrieval does not matter" reading.** Correlation of rankings ≠ size of the transfer; our data also show the EM order mostly follows the Hit@5 order (P3) while the magnitude is small on NQ | FT |
| L05 | Trappolini et al. 2025, UDCG (arXiv) | preprint | 5 datasets, 6 LLMs | Classical IR metrics predict RAG accuracy poorly; utility- and distraction-aware metric improves correlation up to 36% | Same motivation; we do not propose a metric | AB |
| L06 | Cuconasu et al. 2024, Power of Noise (SIGIR) | conf. | NQ-open; LLM readers | NQ-open, gold passage fixed: adding high-scoring non-answer passages lowers accuracy (one such passage: up to −0.24); random passages placed before the gold one can raise accuracy | Consistent with P5 (top-ranked non-gold passages lower EM vs random passages) | FT |
| L07 | Cuconasu et al. 2025 (EMNLP) | conf. | 3 benchmarks | Retrieval pipelines bring highly distracting passages to top ranks; positional bias is marginal in realistic settings; reordering ≈ random shuffling | Consistent with P6 null (gold position 1/3/5 → no difference detected) | AB |
| L08 | Amiraz et al. 2025 (ACL) | conf. | QA; several LLMs | Quantifies the distracting effect of a passage; hard distractors lower accuracy | Consistent with P5 | AB |
| L09 | Hsia et al. 2025, RAGGED (ICML) | conf. | open-domain, multi-hop, specialised QA; many retriever–reader pairs | Reader robustness to noise determines RAG stability; retrievers matter but do not change reader-driven trends | Consistent with reader-dependent T (Qwen vs Llama) | AB |
| L10 | Mathur et al. 2026; Medrano et al. 2026; Muthyam 2026 (arXiv) | preprints | AI-policy QA; enterprise KB; 500-question QA | Retrieval gains (fine-tuned retriever / fusion / broader retrieval) do not reliably raise end-to-end quality | Same phenomenon in single domains/pipelines; no cross-encoder transfer estimate, no cost | AB |
| L11 | Bala 2026 (arXiv) | preprint | 3 multi-hop datasets, budgeted context | Recall@k on the retrieved set differs from answer-in-context after packing; adds ΔR² 0.17–0.27 | Offers a mechanism (what reaches the reader); our k=5 setting packs all retrieved passages, so this gap is smaller here | AB |
| L12 | Ghosh et al. 2026 (arXiv) | preprint | 5 benchmarks + TREC RAG 2025 | Evidence-aware retrieval signals change rankings but do not robustly predict answer quality on unseen topics | Consistent with low NQ transfer; warns against over-reading any retrieval metric | AB |
| L13 | Ning et al. 2025 (arXiv) | preprint | open-domain QA; corpus size × generator size | Corpus scaling helps mainly via coverage of answer-bearing passages; utilisation efficiency unchanged | Related to E9 (full corpus) | AB |
| L14 | El Assadi et al. 2026 (arXiv; COLM 2026 per arXiv comment) | preprint | MTEB-style tasks; 26 embedders, 10 LLMs | Cost-aware comparison: embedders vs LLMs | Cost axis for embedders, but no RAG answer quality | AB |
| L15 | Zhao et al. 2026 (TOSEM) | journal | 3 LLMs, 6 QA/code datasets | QA: 5–10 documents roughly optimal; failures even with perfect documents | Context for E4 (top-k) | AB |
| L16 | Repantis et al. 2026 (arXiv) | preprint | SQuAD; Phi-2, Qwen2.5-1.5B | Replacing hard competitors with less competitive passages at fixed length recovers EM | Same design logic as our P5 (fixed k, gold fixed, negatives varied) | AB |
| L17 | Kamalloo et al. 2023 (ACL) | conf. | NQ-open, TQA | Manual re-evaluation on an NQ-open subset: lexical matching underestimates true accuracy of all models, InstructGPT zero-shot by nearly 60 points; >50% of lexical failures are semantically equivalent answers; regex matching keeps model ranking | Limitation of EM; supports reporting AC/F1 and caution about absolute EM | FT |
| L18 | Rau et al. 2024, BERGEN (Findings EMNLP) | conf. | RAG benchmarking library | — (cited as a tool/benchmark framework) | Related reproducible benchmarking | MD |
| L19 | Jin et al. 2024 (arXiv) | preprint | long-context LLMs | Quality first rises then falls with more passages; hard negatives a key cause | Context for E4 and P5 | AB |
| L20 | Shaikh 2026 (arXiv) | preprint | HotpotQA distractor, Qwen2.5-7B local | Component ablation of agentic RAG on local hardware | Similar reader and hardware budget; different question | AB |

## 2. Contribution in one falsifiable sentence
For off-the-shelf open embedding models in a fixed retrieve-then-read pipeline, the gain in exact match per unit gain in
Hit@5 over a small baseline encoder (MiniLM-L6) is well below 1 on single-hop NQ and larger, with wider uncertainty, on
multi-hop HotpotQA; the ratio depends on reader and dataset; and the encoder with the best retrieval is not always the one with the
best answers, while costs differ by more than an order of magnitude.

## 3. Comparison with the closest papers
- Task/data: NQ (single-hop) and HotpotQA (multi-hop) with BEIR qrels and gold answers; L01 uses the same datasets but
  compares RAG methods, not encoders; L02 uses NQ with one dense encoder.
- Models: 6 open encoders (22M–600M parameters) + BM25; 2 quantised 7–8B readers. No paper in Part 1 crosses ≥6 encoders with
  2 readers on 2 datasets with paired tests.
- Metric: T(R)=ΔEM/ΔHit@5 vs MiniLM-L6 with bootstrap CI and a pre-registered reporting rule. L01's RCR is a level ratio;
  L03/L04/L05 report correlations.
- Controls: closed-book and gold-passage bounds per cell; k ablation; reranker; HNSW; full-corpus check; controlled
  passage-composition test (P5/P6); cross-session determinism check.
- Cost: measured encoding throughput (5 repetitions) and query latency on free Kaggle T4 GPUs. L14 measures embedder cost without
  RAG answers.

## 4. Novelty verdict
Type: **new empirical insight within a defined setting + improved evaluation practice** (paired difference ratio with CIs,
pre-registration, measured cost). Not a new method. Confidence: moderate. The search is targeted, not systematic; 2026
preprints appear quickly and a closer paper may exist.

Bounded gap statement (for the Introduction, not the abstract):
"Earlier studies relate retrieval quality to answer quality through correlations across systems or topics
(Salemi & Zamani 2024; Samuel et al. 2026; Trappolini et al. 2025), through a level ratio of answer F1 to recall across RAG
methods (Peng et al. 2026), or through manipulations of a single retriever (Leto et al. 2024). They do not report, for a set
of interchangeable open encoders, the paired ratio of answer gain to retrieval gain with uncertainty, per reader and dataset,
next to measured encoding cost."
(Each clause is checked against L01–L05 above; the sentence makes no claim about all prior work.)

Unsupported claims to remove from earlier drafts: "first", "novel metric", "no prior work", "we are not aware of work …" in the
abstract.

## 5. Counter-evidence and how the manuscript handles it
| Counter-evidence | Handling |
|---|---|
| L04: retrieval metrics correlate with generation coverage | State that our results agree on order (better retrieval → mostly better EM) and differ on magnitude; T is a magnitude, not a correlation |
| L01: conversion falls as recall rises | Agrees with the NQ pattern; note the level-vs-difference distinction |
| HotpotQA ratios near 1 for bge-large/Llama (CI up to 1.096) | Report per cell; do not claim universal low transfer (P1 fails in llama_hp) |
| L17: EM undervalues LLM answers | Report AC and F1 in a table; limitation paragraph |
| L06/L19: hard negatives hurt | P5 replicates the direction within our setup; describe negatives as "top-ranked non-gold BM25 passages" (S9) |

## 6. Open questions
- Author spot-check of L01 (NeocorRAG) definitions before submission is recommended because it is the closest metric.
- No search was run in Scopus/Web of Science (no access); IP&M back-issues were not searched separately.
