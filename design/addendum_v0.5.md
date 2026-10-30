# Design addendum v0.5 (2026-10-30) — reader scaling of the transfer ratio (Phase 4)

Status: written and committed **before** any Phase 4 generation. It adds one experiment and its analysis and does not change
any earlier result, prediction or analysis. Everything below is fixed now; anything computed later that is not listed here is
post hoc and will be labelled so.

## Why
Phases 1–3 used two 7–8B readers of the 2024 generation. The open question, raised by the IP&M desk decision
(IPM-D-26-08758: weak link to recent advances; claims not validated beyond one setting), is whether the low single-hop transfer
is a property of retrieval or of reader capacity. Phase 4 holds retrieval fixed and varies only the reader within one recent
model family. Because the reader input is fixed by the stored passage ids, no retrieval is rerun.

## Inputs (frozen)
`experiments/v05_reader_scaling/kit/inputs/`, built by `src/v05/make_inputs.py` from committed files only:
- the 2,255 NQ and 2,000 HotpotQA generation queries of Phases 1–2, in their seeded query order;
- for each query and condition R0–R6 (k = 5), GOLD and CB, the exact passage ids the Phase 2 readers saw (Qwen and Llama
  records agree for every query; asserted);
- prompts A1 (with passages) and A2 (closed book), verbatim from `src/generation/prompts.py` (PROMPT_HASH recorded);
- passage texts are read on the run machine from the pinned BEIR corpora (BeIR/nq b7253e6…, BeIR/hotpotqa a7e8bab…); 500 NQ
  passages are checked against committed texts by sha256 before any generation, and every needed id must be present.

Samples: `*_all` (2,255 / 2,000), `*_first1000`, `*_first500` = prefixes of the seeded order.

## Readers and runs (one machine: laptop, RTX 4070 Laptop GPU, 16 GB RAM; Ollama, GGUF Q4_K_M)
| Reader (Ollama tag) | Sample | Conditions | Prompts |
|---|---|---|---|
| qwen3:1.7b-q4_K_M | all, NQ + HotpotQA | R0–R6, GOLD, CB | 38,295 |
| qwen3:4b-q4_K_M | all | R0–R6, GOLD, CB | 38,295 |
| qwen3:8b-q4_K_M | all | R0–R6, GOLD, CB | 38,295 |
| qwen3:14b-q4_K_M | first1000 per dataset | R1, R4, R5, GOLD, CB | 10,000 |
| qwen2.5:7b-instruct-q4_K_M (bridge) | NQ first500 | R1, R5 | 1,000 |

Decoding: chat endpoint, system + user message exactly as A1/A2, `think: false`, temperature 0, top_k 1, top_p 1,
repeat_penalty 1.0, seed 0, num_ctx 4096, num_predict 32, stop "\n" (the Phase 1–3 limits). One request at a time. The model
digest, Ollama version, GPU name and per-item latency are recorded. Any `<think>…</think>` text left in an output is removed
before scoring and counted. If the tag of a size is unavailable, the closest Qwen3 Q4_K_M tag of that size is used and recorded
as a deviation; all sizes must be the same Qwen3 release.

Fallback fixed now: if the median latency of qwen3:14b over its first 200 prompts exceeds 8 s, its sample becomes first500 per
dataset (5,000 prompts). Determinism check: at the end, the first 100 NQ R5 prompts are rerun with qwen3:8b and the share of
identical outputs is reported.

Anchor (optional, descriptive): one Gemini Flash-class model on the free API tier, model id recorded, temperature 0, thinking
disabled where the API allows, first500 per dataset, conditions R1 and R5 required, then R4, GOLD, CB in that order as the
quota allows. It is not part of any prediction, and the paper states that it cannot be rerun exactly.

## Scoring
`src.evaluation.answer` unchanged (SQuAD-normalised EM, F1, containment; unparsed outputs stay in the denominator).

## Pre-specified analysis (Phase 4)
Notation as before: for reader s and dataset d, ΔEM_s(R) and ΔHit@5(R) against MiniLM-L6 (R1) on the same queries, k = 5,
T_s(R) = ΔEM_s(R) / ΔHit@5(R). Hit@5 does not depend on the reader, so the reporting rule (|ΔHit@5| > 0.02 and Holm-significant
Hit@5 difference over the 21 pairs) gives the same estimable set as Phase 2. Bootstrap: B = 10,000, `numpy.random.default_rng(0)`,
one resampling index per dataset shared by all readers, so readers are paired. HotpotQA ratios are reported with Hit@5 and
with Both@5.

Pooled dense ratio: T̄_s,d = Σ_R ΔEM_s(R) / Σ_R ΔHit@5(R) over the five dense encoders R2–R6.

**Q4 (scaling question, two-sided, no directional prediction).** For each dataset, the paired difference
T̄_8B,d − T̄_1.7B,d with its 95% bootstrap interval. Reading fixed now: "rises with reader size" if the interval lies above 0,
"falls" if below 0, otherwise "no change detected" (not evidence of equality). T̄ of 4B is reported between them; no trend test.

**P9 (NQ transfer below 1 at every size).** For each of qwen3 1.7B, 4B and 8B, every reportable NQ ratio T_s(R) of a dense
encoder has an unrounded upper 95% bound below 1.

**P10 (complete-evidence transfer below 1 on HotpotQA).** For each of the three sizes, every dense HotpotQA ratio with Both@5 as
denominator has an upper 95% bound below 1.

**P11 (parametric knowledge grows with size).** Closed-book EM (CB) of qwen3:8b exceeds that of qwen3:1.7b on both datasets
(exact McNemar, two-sided p < 0.05 in each).

Descriptive, fixed now:
- per size: EM per condition with bootstrap intervals; the gold gap EM(GOLD) − EM(R) for R1 and R5;
- per size: the three-term decomposition of ΔEM (hit prevalence, EM among hits, EM among misses) for R5 and bge-large (R4),
  and EM given a hit / a miss under R1 and R5;
- qwen3:14b on its sample: T(R4), T(R5) and T̄ over R4, R5, with the same quantities for 1.7B–8B on the same sample;
- bridge: for qwen2.5:7b on Ollama versus the Phase 1 vLLM AWQ outputs on NQ first500: EM per condition, paired EM difference,
  share of identical normalised answers, and T(R5) under both stacks. Comparisons between Qwen3 and the Phase 1–3 readers cross
  model family and software stack and are labelled so;
- Gemini anchor: EM per condition and T(R5) with intervals on its sample.

## What would change the paper's claims
If P9 fails for a size, the claim "NQ transfer well below 1" is restricted to the readers where it holds. If Q4 reads "rises",
the paper reports that retrieval gains reach the answer more with larger readers and quantifies by how much; if "falls" or
"no change detected", it reports that reader capacity within 1.7–8B did not raise transfer. Either outcome is reported.
