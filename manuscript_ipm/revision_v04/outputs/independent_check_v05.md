# Independent check, part 2 (v0.5)

Script: manuscript_ipm/revision_v04/scripts/independent_check_v05.py (no project imports).

| Quantity | Largest absolute difference vs analysis |
|---|---|
| Hit@5 means, 4 cells × 7 retrievers | 0.0e+00 |
| Both@5 means, 2 HotpotQA cells × 7 retrievers | 0.0e+00 |
| Hit@5 McNemar p, 84 pairs (own exact binomial) | 4.2e-14 |
| Hit@5 Holm p, 84 pairs (own Holm) | 4.2e-14 |
| Transfer-ratio point estimates, 22 estimable | 0.0e+00 |
| Transfer-ratio bootstrap bounds, 44 (own ratio and quantile) | 1.1e-16 |
| Decomposition identity residual, 22 ratios | 5.6e-17 |

Ratio-rule decisions (estimable or not) agree for all 24 retriever × cell pairs.

## Bootstrap denominators

Resamples with a zero Hit@5 difference (dropped): 0 of 220000; resamples whose Hit@5 difference has the opposite sign to the observed one: 0.

## P1 with unrounded bounds and the registered scope (dense encoders only)

| Cell | Dense ratios | Largest upper bound (unrounded) | Criterion met (dense only) | Analysis verdict | Ratios counted by the analysis |
|---|---|---|---|---|---|
| qwen_nq (not confirmatory) | 5 | 0.45454545454545453 | met | met | 6 |
| llama_nq | 5 | 0.4274235261401556 | met | met | 6 |
| qwen_hp | 5 | 0.9961689479687637 | met | met | 5 |
| llama_hp | 5 | 1.0958916746701715 | not met | not met | 5 |

HotpotQA generation queries: distinct gold passages per query = [2] (Both@5 requires both).

