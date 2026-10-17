# Skeptical reviewer report on the revised manuscript (v0.4)

**Nature of this review.** This is an AI review pass written after the audit, by the same assistant that made the revision. It tries to argue against the paper, but it is not independent and it is not human peer review. It must not be described as peer review.

## Summary of the submission

The paper measures how much of a retrieval gain (Hit@5) reaches the answer (EM) when one open embedding model replaces another in a RAG pipeline: six encoders plus BM25, two 4-bit 7–8B readers, NQ (2,255 queries) and HotpotQA (2,000 queries), 300,000-passage sub-corpora, free T4 GPUs. The main quantity is T = ΔEM/ΔHit@5 against MiniLM-L6 with bootstrap intervals. Secondary analyses cover context composition, reranking, top-k, HNSW, a full-corpus retrieval check and encoder cost. Six predictions were registered before Phase 2.

## Answers to the twenty questions

| # | Question | Assessment after revision |
|---|---|---|
| 1 | Question important and precise? | Yes. Practitioners pick encoders by retrieval benchmarks; the question of how much survives the reader is concrete. |
| 2 | Distinguishable from closest work? | Moderately. NeocorRAG's RCR (level ratio across pipelines) and Hitl et al. 2026 (embedding × generator benchmark with EM and latency) are close. The paired-gain ratio with intervals, the decomposition and the baseline/metric sensitivity are the distinct parts. A reviewer could still call the contribution incremental. |
| 3 | Fair comparisons? | Within a cell, yes (same queries, prompt, decoding). Across retrievers, the sub-corpus favours dense encoders over BM25; now stated. |
| 4 | Data and corpus documented? | Yes, including the construction bias and the use of evaluation labels. |
| 5 | Metrics correct? | Yes. EM is strict (stated; F1 and containment reported). Hit@5 saturates on HotpotQA; the revision now quantifies this with Both@5. |
| 6 | Statistical assumptions? | Paired exact tests and bootstrap are appropriate. The pooled GEE's common-effect assumption is rejected; now reported. |
| 7 | Pairing and repeated measures? | Correct: query is the unit; GEE clusters by query across retrievers and readers. |
| 8 | Transfer ratio valid? | Arithmetic verified exactly. Interpretation is now limited to a ratio of mean differences that depends on the baseline and on the retrieval metric. |
| 9 | Numbers reproducible? | Yes from committed files: full rerun identical; 68 printed values independently recomputed with no mismatch. |
| 10 | HNSW conclusions match measurements? | Now yes (retrieval and latency only). |
| 11 | Random-passage interpretation justified? | Now yes (no detected difference, interval reported, not equivalence). |
| 12 | Full-corpus claims supported? | Only for retrieval order of four retrievers on NQ; stated. No reader run on the full corpus. |
| 13 | Failed predictions reported honestly? | Yes: P1 (one cell), P2 (HotpotQA), P6 failed and are in the abstract and highlights. |
| 14 | Conclusions follow? | Yes after scoping; see major concern M2 for the HotpotQA wording. |
| 15 | Reproducible by others? | Code, prompts, outputs and revisions are committed; the E9 full-corpus and HNSW top-k lists were not saved, so those two checks can be re-run but not re-analysed. |
| 16 | Limitations transparent? | Yes, now including baseline, metric and decomposition dependence. |
| 17 | References accurate? | Verified earlier at full-text or abstract level (literature-matrix.md); one new abstract-level reference. No fabricated entries found. |
| 18 | Abstract reflects results? | Yes after revision. |
| 19 | Main text vs appendix contradictions? | None found by the numeric report; Table gee now matches the text. |
| 20 | Strongest legitimate reason to reject | See below. |

## Major concerns

**M1. The central estimate rests on a biased sub-corpus with no full-corpus reader run.** Every answer result uses a 300,000-passage sub-corpus that contains every gold passage and BM25's own hard negatives. The E9 check shows the order of four retrievers is kept on the full NQ corpus, but absolute dense scores drop by 0.08 nDCG@10 and BM25's by almost nothing. T could change if both EM and Hit@5 drop on the full corpus. *Status: disclosed, not resolved.* The full-corpus reader run in experiment_plan.md (≈ 4–4.5 T4-hours) would resolve it.

**M2. The dataset contrast is a property of the denominator.** With Hit@5, HotpotQA T is 0.49–0.93; with Both@5, 0.29–0.35, close to NQ against BM25 (0.29–0.35) and somewhat above NQ against MiniLM-L6 (0.16–0.27). A reviewer may argue that the paper's own headline ("HotpotQA transfer was larger") is mostly an artefact that the authors then explain away. *Status: the revision now leads with both numbers and calls the Hit@5 contrast largely a denominator effect.* Authors should consider making the Both@5 ratio the primary HotpotQA quantity in a future version; it is post hoc here and cannot be promoted to confirmatory.

**M3. The ratio depends on the baseline.** Against BM25 the NQ ratios rise to 0.29–0.35, and Qwen3-Emb reverses the dataset contrast. A ratio that changes with the reference system is harder to use as an encoder property. *Status: reported as a sensitivity analysis; the discussion now says T describes a pair of retrievers.*

**M4. Generality.** Two 4-bit 7–8B readers, one prompt, k = 5, English Wikipedia, encoders ≤ 0.6B. Larger readers might convert hits into answers at different rates. *Status: disclosed.*

## Minor concerns

- The pre-registration lives in the author's own repository, not a public registry (disclosed).
- The judge was validated by one annotator, the author (disclosed; judge excluded).
- Phase 1 and Phase 2 used different Python/NumPy images; the pooled GEE mixes them (disclosed; cross-phase determinism 396/397).
- Hard negatives in E12 may contain the answer (O8, disclosed; check not run).
- Holm correction is per family, not across the paper (disclosed).
- The decomposition, BM25 baseline, Both@5 ratios and GEE interactions are post hoc (labelled). Their number has grown; a reviewer may see a garden of forking paths. Each one was added to test a stated weakness, not to find a significant result, and each is reported whatever its direction (two of them weaken the original headline).

## Corrected in this revision

Throughput overlap prose; HNSW "no measurable cost in answers"; sub-corpus direction; pooled GEE "same picture"; "harmless" random passages; unlabelled scope of the NQ range; incomplete Llama–HotpotQA bounds in the abstract; query-level reading of T in the introduction; formulation of the decomposition; missing baseline sensitivity; Qwen3-Emb conclusion not scoped to HotpotQA; footer page total; one new related paper.

## Unresolved

M1 (full-corpus readers), the HNSW answer check, O8 (answer strings in hard negatives), second annotator (optional), title-page facts and data-availability decision (author).

## Additional experiments that are genuinely necessary

Only the full-corpus NQ reader run (M1) is needed to defend the central number against the largest known bias. Everything else is optional or already disclosed.

## Recommendation on scientific readiness

**REVISION REQUIRED, close to SUBMISSION CANDIDATE.** No numerical error was found in any reported value. Five prose claims went beyond the evidence and are now corrected, and two new post hoc checks weaken the original HotpotQA headline; the revised text reports them. The remaining scientific risk is M1. The paper can be submitted with M1 as a stated limitation, but a reviewer is likely to ask for that experiment. Acceptance and journal ranking cannot be predicted from this review.
