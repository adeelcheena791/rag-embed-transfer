# Methods and statistics audit (Gate 3) — 2026-10-16

Checks marked RUN were executed on the committed artifacts; outputs are in this folder. AI self-audit, not human review.

## Metric definitions vs code
| Metric | Definition (manuscript) | Code | Check | Status |
|---|---|---|---|---|
| Hit@k | 1 if ≥1 qrels-relevant passage among the top k | src/common/ir_metrics.py hit_at_k | own re-implementation, all cells (independent_check.md) | VERIFIED |
| Both@5 (HotpotQA) | 1 if all (both) gold passages are in the top 5 | v03_analysis.py, retrieval_summary.py | own re-implementation; equals Kaggle summary | VERIFIED |
| nDCG@10 | binary gain, log2 discount, ideal from qrels | ir_metrics.ndcg_at_k (cross-checked against ranx in v0.2 tests) | v0.2 test suite | VERIFIED (v0.2) |
| EM | max over reference answers of exact string equality after normalisation (lowercase; drop ASCII punctuation and ‘ ’ “ ” – — …; drop a/an/the; collapse spaces) | src/evaluation/answer.py | re-scored 131,855 answers from raw text: 0 disagreements | VERIFIED |
| F1, AC | SQuAD token F1; AC = normalised reference is a substring (whole-token) of the normalised output | answer.py | AC/F1 not independently re-implemented | PARTIALLY VERIFIED |
| T(R) | (EM_R − EM_R1)/(Hit@5_R − Hit@5_R1) on the same generation queries; reported only if |ΔHit@5| > 2 pp and ΔHit@5 Holm-significant | stats.transfer_ratio | point estimates re-derived independently (identical) | VERIFIED |

## Issues
| ID | Severity | Issue | Affected claims | Check performed | Outcome | Action |
|---|---|---|---|---|---|---|
| S1 | MAJOR | T(R) is a ratio of mean differences; numerator and denominator change on different queries; not the probability that an extra hit yields a correct answer | all T(R) | definition review; decomposition P(EM|hit), P(EM|miss) available per retriever | RUN | define precisely; add decomposition; forbid causal wording |
| S2 | MAJOR | NQ CIs for bge-small and Qwen3-Emb T include 0 (lower −0.071 / −0.083 for bge-small) | NQ T for R2 | read CIs | confirmed | report as uncertain; no directional claim for R2 alone |
| S3 | MAJOR | HotpotQA Hit@5 is near ceiling for dense encoders (0.945–0.961); ΔHit vs MiniLM ≈ 13–14 pp, so T is estimable, but the ratio is sensitive to the small spread among dense models | HP T, P1 | CI widths 0.28–0.32 (recomputed) | confirmed | report Both@5 alongside; state sensitivity |
| S4 | MAJOR | P1 result is mixed: holds in 3 cells, fails in Llama-HotpotQA (bge-large upper bound 1.096); Qwen-HotpotQA bound 0.996 holds narrowly | P1 | exact values read | confirmed | report per cell, no pooled "P1 holds" |
| S5 | MINOR | P2 wording has two readings (point estimates vs no significant gain). Verdict uses the test reading pre-registered in parentheses; under both readings the HotpotQA result fails (bge-large significantly higher) | P2 | both readings computed | consistent | state operationalisation |
| S6 | MAJOR→MITIGATED | Cross-session comparisons (v0.2 Qwen-NQ vs v0.3; k ablation k=5 from v0.2; reranker X vs R) | P4, E5, GEE | reader stacks compared (identical torch/vLLM/transformers; Python 3.12→3.13, numpy 2.0.2→2.1.3); identical-input rerun: 396/397 outputs identical across sessions; 100/100 within session | RUN (cross_session_check.json) | report the check; keep caveat |
| S7 | MINOR | Multiplicity: Holm within families (21 pairs per metric per cell; 7 for P4; 3 for P6); no correction across the four cells or across predictions | P1–P6 | design read | as pre-registered | state families explicitly |
| S8 | CRITICAL→RESOLVED | LLM judge reliability | faithfulness results | κ = 0.127 [−0.006, 0.261] (single annotator, blind) | pre-registered rule: exclude | faithfulness not used for conclusions; appendix only |
| S9 | MAJOR | E12 hard negatives are BM25 top non-gold passages under sparse NQ qrels; some may contain the answer (unjudged positives). This would raise M1 EM, so the observed drop is if anything conservative, but "hard negative" is a label, not a verified property | P5 | answer-containment check on passage text NOT RUN (corpus text not in the repository) | NOT RUN | state as limitation; describe as "top-ranked non-gold BM25 passages" |
| S10 | MINOR | E9: full-corpus scores from new embeddings; primary comparison uses sub-corpus rows of the same embeddings (same session) | P3 | v0.2 sub-corpus nDCG equals same-embedding subset to ≤0.0001 | RUN | report both |
| S11 | MINOR | E7 timing: GPU encoding throughput (fp16, batch 128, 20k passages, warm-up, CUDA sync) and batch-1 query-encoding latency; FAISS search latency and generation latency not repeated | cost claims | read throughput.py and records | as designed | keep offline vs online cost separate |
| S12 | MINOR | GEE pools four cells from two sessions; exchangeable working correlation 0.60; robust SE; secondary analysis | E10 | model refit deterministic | RUN | label secondary |
| S13 | MINOR | NQ generation set is the joined subset (2,255 of 3,452 judged); retrieval reported on both | NQ retrieval | v0.2 D1b analysis exists | RUN (v0.2) | state population per table |
| S14 | MINOR | Reranker result (e5 Hit@5 and EM lower after reranking) is a single-configuration observation (one reranker, depth 50, max length 512) | E5 | score sanity (no NaN/inf, ranges 3.7e-05–1.0) | RUN | report as observation within setup |
| S15 | EDITORIAL | Exact-match normalisation includes typographic punctuation; a plain SQuAD normaliser would change 83 of 117,425 answers (0.07%) | EM everywhere | RUN | negligible | mention in Methods |

## Not verified
- F1 and AC were not independently re-implemented (only the shared pipeline + second-machine reproduction).
- Bootstrap CIs were reproduced on a second machine with the same code, not with an independent implementation.
- E12 answer presence in hard negatives (S9).
