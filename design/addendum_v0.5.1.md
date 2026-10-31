# Design addendum v0.5.1 (2026-10-31) — two additions to Phase 4, before any run

Status: written and committed **before** any Phase 4 generation (no model has been run on the laptop yet). It adds readers to
Phase 4 and leaves every item of addendum v0.5 unchanged, including Q4, P9, P10 and P11, which keep their registered scope
(qwen3 1.7B, 4B, 8B).

## Why
Addendum v0.5 varies reader size inside one family (Qwen3). A finding about "reader capacity" from one family could be a
property of that family. A second, independently trained family tests whether the pattern generalises, and a smaller Qwen3
size extends the curve at low cost.

## Additional readers (same machine, stack, inputs, decoding and scoring as v0.5)
| Reader (Ollama tag) | Plan | Sample | Conditions | Prompts | Part |
|---|---|---|---|---|---|
| qwen3:0.6b-q4_K_M | core | all | R0–R6, GOLD, CB | 38,295 | A |
| gemma3:1b-it-q4_K_M | core | all | R0–R6, GOLD, CB | 38,295 | B |
| gemma3:4b-it-q4_K_M | core | all | R0–R6, GOLD, CB | 38,295 | B |
| gemma3:12b-it-q4_K_M | large (= the v0.5 14b plan) | first1000, fallback first500 at median > 8 s | R1, R4, R5, GOLD, CB | 10,000 | B |

Part A (v0.5 readers plus qwen3:0.6b) runs first. Part B runs after Part A as time allows; if Part B is not completed, the
paper says so and reports only what finished.

Gemma 3 has no thinking mode: the `think` field is sent only to models that report the "thinking" capability (Qwen3); the
model capabilities are recorded. Ollama's Gemma 3 chat template has no separate system role and places the system text in the
first user turn; this is recorded as a property of that family's template.

## Additional pre-specified analysis (secondary)
- **Q4-G (second family, two-sided).** For each dataset, T̄_gemma4B − T̄_gemma1B with its 95% bootstrap interval, read with the
  same rule as Q4 ("rises" / "falls" / "no change detected").
- **P9-G.** For gemma3 1B and 4B, every reportable NQ dense ratio has an upper 95% bound below 1.
- Descriptive: qwen3:0.6b alongside 1.7B–8B in every Phase 4 table and figure; gemma3:12b on its sample like qwen3:14b.

## Integrity of the kit
`KIT_SHA256.txt` holds the sha256 of `v05.py` at this commit. Every run records the sha256 of the `v05.py` it used; a
difference is logged and reported in the paper. The helper instructions given to the person running the kit forbid changes to
`v05.py`, `inputs/` and all run parameters.
