# Literature search log (Gate 4) — 2026-10-16

Scope: targeted literature search for an empirical benchmark paper. This is **not** a systematic review and is not reported as
exhaustive. AI-assisted search; the author should spot-check the closest works (literature-matrix.md, rows L01–L12).

Honesty note: exploratory web searches were run earlier in the project (2026-10-14/15) to find candidate close work. Their exact
query strings were not logged and are not reproduced here. The searches below were re-run and logged on 2026-10-16; every
paper cited in the manuscript was verified through the metadata sources in Part B, not through search snippets.

## A. Logged searches (Firecrawl research index: arXiv + other scientific sources; ranked, top-k shown)
| # | Date (UTC) | Tool / index | Exact query | Filters | Shown | Screening outcome |
|---|---|---|---|---|---|---|
| S1 | 2026-10-16 | firecrawl_research_search_papers | does better retrieval quality translate into better answer accuracy in retrieval-augmented generation, embedding model comparison | from 2023-01-01, k=15 | 15 | kept 2608.26379, 2604.27852, 2603.24580, 2606.28337, 2404.13781 (=eRAG); others off-topic (domain systems, GraphRAG) |
| S2 | 2026-10-16 | same | retrieval metrics as proxy for end-to-end RAG question answering performance correlation | from 2023-01-01, k=15 | 15 | kept 2510.21440 (UDCG), 2604.01733, 2511.09545, 2601.21803; 2404.13781 again |
| S3 | 2026-10-16 | same | cost-quality trade-off of open-source text embedding models for RAG on consumer or free GPU hardware | from 2023-01-01, k=15 | 15 | kept 2608.12875 (cost of embedders vs LLMs); 2605.23618 noted (CPU latency of encoders, not RAG answers); rest off-topic |
| S4 | 2026-10-16 | same | hard negative distracting passages reduce LLM reader exact match in open-domain QA | from 2023-01-01, k=15 | 15 | kept 2505.06914, 2605.27294, 2410.05983, 2310.18077 |
| S5 | 2026-10-16 | same | multi-hop HotpotQA retrieval recall versus generation accuracy dense retrievers reader comparison | from 2023-01-01, k=15 | 15 | kept 2606.21553 (local Qwen2.5-7B on HotpotQA); method papers excluded (new pipelines, not encoder comparisons) |
| S6 | 2026-10-16 | firecrawl_search (web) | Holm 1979 "A simple sequentially rejective multiple test procedure" Scandinavian Journal of Statistics 6 65-70 | — | 3 | JSTOR stable 4615733 confirms vol. 6, no. 2, pp. 65–70 |

Citation chaining: from Leto et al. (2411.07396) reference list → Hsia et al. RAGGED (2403.09040), BehnamGhader et al. (2212.09146).

Full-text passages read (firecrawl_research_read_paper), to check the exact nearby claim:
- 2604.27852 (NeocorRAG): definition of RCR (Eq. 1: F1 / Recall@5 per query set) and the "near-linear decay" statement.
- 2411.07396 (Leto et al.): gold-document recall vs QA correctness; ANN search recall has minor effect; BGE-base and ColBERTv2 only.
- 2603.08819 (Samuel et al.): topic- and system-level Pearson correlations; ICTIR '26 reference format and DOI.
- 2401.14887 (Cuconasu et al. 2024, Power of Noise): distracting vs random passages, Tables 1–2.
- 2305.06984 (Kamalloo et al. 2023): size of lexical-match underestimation on NQ-open.

## B. Metadata verification
| Source | Request | Result | Saved as |
|---|---|---|---|
| Crossref REST API | works?filter=doi:… (30 DOIs) | 30/30 found; titles, authors, venues, pages match | records/lit/crossref_2026-10-16.json |
| arXiv API | id_list= 26 ids | 26/26 found | records/lit/arxiv_2026-10-16.json |
| arXiv API | id_list= 12 ids | 12/12 found | records/lit/arxiv2_2026-10-16.json |
| DBLP search API | 8 title queries | published venues: Lewis 2020 NeurIPS; MiniLM NeurIPS 2020; Shi ICML 2023 (PMLR 202, 31210–31227); Yoran ICLR 2024; Zheng NeurIPS 2023 D&B; AWQ MLSys 2024; Cuconasu EMNLP 2025 (28022–28036); BEIR: only CoRR record → cited as arXiv | this log |

Known metadata limits:
- Crossref gives only the first page (159) for Landis & Koch 1977; the bib entry keeps the Crossref value.
- BEIR, E5, Qwen3-Embedding, Qwen2.5, Llama 3, Mistral 7B, bm25s, Nogueira & Cho, Gao et al. survey and the 2025–2026 preprints
  without a venue record are cited as arXiv preprints.
- Venues for RAGGED (ICML 2025), BehnamGhader (Findings EMNLP 2023), Oh & Thorne (Findings EMNLP 2023) come from arXiv
  journal_ref/comment fields only, not from the proceedings page.

## C. Generated bibliography
manuscript_ipm/tools/make_bib.py → manuscript_ipm/latex/references.bib (69 entries, built only from the files above).

## D. Additions after the author's earlier draft (2026-10-16, afternoon)
References used in the author's earlier draft were checked before reuse: arXiv API (2603.10765 RAGPerf, 2205.13147 Matryoshka,
1803.05449 SentEval, 2104.06979 TSDAE, 2605.23618 Cirillo et al., 2607.23507 Baidya) → records/lit/arxiv3_2026-10-16.json;
DBLP for the NeurIPS 2022 venue of Matryoshka; Crossref for Wilcoxon 1945 (10.2307/3001968) → records/lit/crossref2_2026-10-16.json.
Direct quotations in the earlier draft (Baidya; Cirillo; Salemi & Zamani; Leto) are replaced by paraphrases of what the
retrieved abstracts state, because the full texts were not re-read here.
