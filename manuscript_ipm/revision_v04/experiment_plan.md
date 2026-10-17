# Experiment plan (revision v0.4)

Budget: $0, Kaggle free T4 quota only. Nothing in Tiers 3–4 was started. No existing output was modified.

## Decision table

| Issue | Existing data sufficient? | Additional analysis? | New inference? | Tier | Estimated compute | Scientific reason |
|---|---|---|---|---|---|---|
| NQ range scope (M1) | yes | no | no | 1 — done | — | label already-correct numbers |
| Throughput overlap prose (M2) | yes | recompute ranges — done | no | 1 — done | — | prose contradicted the table |
| HNSW answer claim (M3) | no answers exist | — | yes, to restore the claim | 1 (narrow claim) — done; 3 (experiment) — open | ≈ 1.5–2 T4-hours: re-encode the NQ sub-corpus for 3 encoders (≈ 0.3 h; HNSW top-k lists were not saved), then 3 reader conditions at ≈ 0.4 h each (Phase 2 timing: 1,410 s for 2,255 NQ queries at k = 5) | answer quality under ANN was asserted, never measured |
| Random-passage wording (M4) | yes | CI reported — done | no | 1 — done | — | non-significance ≠ equivalence |
| Sub-corpus direction (M5) | yes | — | no | 1 — done | — | sentence contradicted construction and E9 |
| Pooled GEE heterogeneity (M6) | yes | interaction models — done | no | 2 — done | CPU, < 1 min | common-effect assumption rejected (p ≈ 5e-36) |
| Decomposition formulation (M7) | yes | symmetric form + bootstrap — done | no | 2 — done | CPU | share depends on ordering |
| Baseline dependence of T (M13) | yes | BM25 and bge-small baselines — done | no | 2 — done | CPU | open review item D1 |
| Full-corpus reader evaluation | — | — | yes | 3 — **done** (Phase 3, 17 Oct 2026; addendum v0.4) | MiniLM-L6 28 min, bge-small 42 min, e5-base-v2 95 min encoding; readers 92 and 102 min in parallel | P7 and P8 met for both readers; see revision_v06_report.md |
| HotpotQA full-corpus check | no | — | yes (retrieval only) | 3 — open | encoding 5.2 M passages × 4 encoders exceeds the free quota for bge-large/Qwen3; MiniLM/bge-small/e5 ≈ 4–6 T4-hours | sub-corpus effect on HotpotQA is untested |
| Hard negatives containing the answer (O8) | — | answer-string search | no GPU | 2 — **done** (Phase 3) | — | 167/500 affected; P5 holds without them (−0.102) |
| Second annotator for the judge | no | — | human time | 3 — optional | ≈ 2–3 h of annotation | the judge is already excluded; affects only the appendix |
| Equivalence test for random passages | yes | possible, but margin would be post hoc | no | 4 — not recommended | — | no registered margin; any margin chosen now is post hoc |
| Larger / proprietary readers, other domains | no | — | yes | 4 — future work | beyond free budget | generality, stated in limitations |

## Smallest defensible next experiment

**Reader on full-corpus NQ retrieval (Tier 3).** Rebuild the E9 full-corpus top-k lists for BM25, MiniLM-L6, bge-small and e5-base-v2 with the frozen code (the original lists were summarised but not committed), save them this time, and run Qwen2.5-7B-AWQ with prompt A1, k = 5, greedy, on the 2,255 NQ generation queries for these four retrievers, in a new results directory with the frozen code commit and a dated addendum (v0.4) written first. Pre-specify: (i) T of bge-small and e5-base-v2 vs MiniLM-L6 with the existing rule; (ii) prediction that the full-corpus T lies inside the sub-corpus 95% interval. This is the experiment that most directly tests the main claim against the largest known design bias.

**Reader on HNSW results (Tier 3).** Same design, efSearch 128, three encoders. Pre-specify any non-inferiority margin before running. Design v0.2 computed a minimum detectable EM difference of about 3.4 points for 2,255 paired queries, so a margin much smaller than that would probably end inconclusive; if no margin is registered, report only the difference and its interval.

Update 2026-10-18: the full-corpus reader run was done as Phase 3 (addendum v0.4); results in revision_v06_report.md. The HNSW reader run was not started.
