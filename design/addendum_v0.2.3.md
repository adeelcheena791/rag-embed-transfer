# Design addendum v0.2.3 (2026-09-27) — Pilot Part A verified; main-run adjustments; Batch 2 decisions

Status: pilot Part A **EXECUTED and VERIFIED**. Everything else is PLANNED — NOT YET EXECUTED.
Pilot numbers are never reported as results.

## 1. Pilot Part A: verification (facts from the raw files, rechecked on 2026-09-27)

Every figure in the Kaggle summary was recomputed from `results/raw` in the returned zip, and all match.

| Check | Result | Criterion | Verdict |
|---|---|---|---|
| P1 | all 5 dataset shas match the pins | exact match | PASS |
| P3 | 2,255 / 3,452 joined (0.653), 0 ambiguous | n_joined ≥ 1,000 (v0.2.2) | PASS |
| D1a | 0 / 1,197 not-joined queries have max-Jaccard ≥ 0.9 to an NQ-open validation question; 29 (2.4%) in [0.7, 0.9); 0 are in NQ-open train | < 10% at ≥ 0.9 → structural-absence hypothesis supported | supported |
| P2 | SciFact nDCG@10: R2 0.71276 (card 0.71275), R3 0.74174 (0.74039), R4 0.74617 (0.74607), R5 0.72048 (0.71944) | ±0.02 of the model card | PASS (max abs. diff 0.0014) |
| P4c | R6 loads and encodes on the T4 (SDPA attention) | no failure | PASS; F3 not triggered |
| P4 | projected full-corpus encoding 20.24 T4-h | ≤ 6.0 → full, else sub-corpus | **sub-corpus** |
| P4b | R1 / R4 pilot runs: 200 queries × top-100, all ids in the pilot corpus | structural | PASS |
| failures | `failures.jsonl` never created | — | 0 failures |

**Qualifications (interpretation, not fact):**
- **D1a** supports, but does not prove, that the not-joined queries are absent from NQ-open by construction.
  - One of the 10 most-similar pairs is a paraphrase of the same question: "where is the hotel **in** the movie the shining" vs "…hotel **used in**…". So exact-text matching misses at least some re-worded duplicates.
  - The matching stays unchanged, as pre-registered.
- **P2** validates the encode → exact-search path for R2–R5 on SciFact. It does not test BM25, the NQ join, sub-corpus effects or generation.
- **P4 efficiency figures are pilot numbers.** They are not reported as results (see §2, M3).

**Supplementary SciFact references for R1 and R6** (post-hoc; not a gate).
- Source: the public MTEB results repository (`embeddings-benchmark/results`), read 2026-09-27.
  - R1 all-MiniLM-L6-v2: MTEB 0.64508 (model revision `8b3219a9…`, mteb 1.12.75); ours 0.65511 (revision `1110a243…`); Δ +0.010.
  - R6 Qwen3-Embedding-0.6B: MTEB 0.69718 (revision `b22da495…`, mteb 1.38.9); ours 0.69949 (revision `97b0c614…`); Δ +0.002.
- The revisions differ, and MTEB's R6 instruction text is unknown, so this agreement is corroborating only.

**Pinned retriever revisions** (from the pilot; now committed in `configs/resolved_revisions.json`):
- MiniLM `1110a243`
- bge-small `5c38ec7c`
- bge-base `a5beb1e3`
- bge-large `d4aa6901`
- e5-base-v2 `f52bf8ec`
- Qwen3-Embedding-0.6B `97b0c614`

## 2. Main-run adjustments (pre-registered now, before any main-run data)

**M1 — sub-corpus membership (fixes an inconsistency I introduced in v0.2.2).**
- The problem: v0.2.2 D1b reports retrieval on all 3,452 judged queries. But the v0.2 sub-corpus contained gold passages only for the *joined* queries, and BM25 negatives only for the *generation* queries. Retrieval for the other queries would therefore be meaningless or easier.
- **New rule:** the 300,000-passage main sub-corpus contains:
  1. the gold passages of **all 3,452 judged BEIR-NQ test queries**;
  2. BM25 top-d negatives for **all 3,452** queries, with d = min(100, ⌊0.5 × (300,000 − |gold|) / 3,452⌋). That is ≈ 42 for |gold| ≈ 4,200; the exact value is computed and logged;
  3. a seeded random fill to exactly 300,000.
- As a result, every query gets identical treatment, and the corpus no longer depends on N_gen. It can be built before P7.
- Implementation (a `judged` query set in bm25 / encode / search / subcorpus) comes with Batch 3, before the main run.

**M2 — encoding budget.**
- Using the pilot throughput (same protocol and GPU type), 300k passages need ≈ 2.3 T4-hours for R1–R6: R6 ≈ 1.3 h, R4 ≈ 0.5 h, the others < 0.2 h each.
- The main-run numbers replace this estimate.

**M3 — efficiency results come from the main S1 run.**
- S1 re-runs the P4 benchmark on the same protocol.
- The pilot P4 numbers are used only in an appendix, as a run-to-run variability estimate (pilot vs main, same GPU type).

**M4 — CPU efficiency.** In S2, `throughput --device cpu --n 2000` for all six dense models gives CPU encoding throughput and batch-1 CPU query latency on the Kaggle CPU (4 × Xeon 2.0 GHz, as recorded in Part A).

**M5 — N_gen** is not pre-decided. It follows the pre-registered P7 rule (p_d ≤ 0.10 → 1,000; else 2,255).

## 3. Batch 2 implementation decisions (made before any generation output exists)

1. **Judge `max_tokens` is 8, not 5 (deviation from Appendix B).** "NOT_SUPPORTED" can take more than 5 tokens in a 32k vocabulary, and truncation would create artificial parse failures. The parsing regex is unchanged.
2. **GOLD passage order is sorted passage-id order.** Appendix A said "qrels order", but the join stores the gold ids sorted. The order is deterministic either way; ≈ 1.25 gold passages per query.
3. **"unparsed"** means: empty after normalisation, generation stopped by the token limit (`length`), or the prompt was too long to run (`prompt_too_long`). Unparsed items score 0 and stay in the denominator. The P5 gate (≤ 1% empty outputs) is reported as specified.
4. **Prompts are token ids.**
   - Messages are rendered with each model's **own Hugging Face chat template** from the pinned tokenizer, and passed to vLLM as token ids.
   - `tokenizer_mode: hf` is set explicitly; it exists in vLLM 0.30.0, checked in the wheel's source.
5. **Weights** are downloaded at the pinned revision into `/tmp/models` using file patterns that exclude `consolidated.safetensors`. The Mistral repository ships both formats, and a plain download would fetch ≈ 29 GB instead of 14.5 GB. A unit test checks the patterns against the real file lists.
6. **Hardware settings:**
   - `dtype: half` (T4 has no bf16).
   - Judge tensor-parallel over both T4s with `disable_custom_all_reduce: true` (PCIe, no NVLink).
   - `--enforce-eager` (no CUDA graphs) is the first fallback, **before** any model fallback. It changes kernels, not the model, and is recorded.
7. **The judge skips** reader outputs that are not answers (unparsed or idk). They are counted, and CB is never judged.
8. **p_d** is the share of the 200 pilot queries on which R1 and R4 (k = 5, prompt A1) disagree on EM.
9. **Projections for F4** (formulas in `src/analysis/pilot_power.py`):
   - S3 = pilot sec/item × [9 N_gen + 9 N_gen × 14/15 (A1, length-scaled) + 900 (F1, A2)]
   - S4 = pilot judge sec/item × 2 GPUs × 8 conditions × 500
10. **Fallback reader licence:** Qwen2.5-3B-Instruct is licensed "other" (Qwen research licence; HF card). If this fallback is ever used, its licence is disclosed.
11. **Checks of the real libraries in the dev workspace:**
    - The vLLM 0.30.0 wheel was downloaded and its source checked for every API the backend uses: the `LLM` arguments (incl. `max_model_len` via kwargs), `generate(prompts, sampling_params, use_tqdm)`, `TokensPrompt.prompt_token_ids`, the `SamplingParams` fields, `CompletionOutput.text/token_ids/finish_reason`, and `RequestOutput.prompt_token_ids`.
    - vLLM's docs list compute capability 7.5 (T4) as the minimum, and AWQ as supported on Turing.
    - The `transformers` fallback was run end-to-end on a tiny local model. That run found and fixed two API issues: `device_map` requires `accelerate`, and `torch_dtype` is deprecated.

## 4. Deviation D2 (PROPOSED — needs approval): drop the RAGAS cross-check

v0.2 §5.4 kept RAGAS faithfulness as a secondary cross-check, conditional on the pilot. **Proposed: do not run it.**
- **Primary measure unaffected.** The primary faithfulness measure is our fully specified single-prompt judge (App. B), validated against 200 human labels. RAGAS would add no human-validated evidence.
- **Unverified stack.** Running RAGAS locally needs an OpenAI-compatible vLLM server plus LiteLLM plus LangChain (ragas 0.4.3 depends on `langchain`, `langchain-community`, `langchain_openai` and `openai`, per PyPI metadata). None of this has been tested on a T4.
- **Budget.** RAGAS faithfulness makes several LLM calls per item (claim extraction, then verification). That would compete with the S4 budget, which is capped at 6 T4-hours by F4.3.
- **Cut list.** It falls under "unnecessary APIs / dependencies", which is on the pre-registered cut-first list.

RAGAS stays in Related Work as a method. If D2 is rejected, it is added as a separate step after the main S4 run, and the pilot Part B does not depend on it.

## 5. Pilot Part B plan (notebook `notebooks/pilot_part_b.ipynb`)

Cells 3–6 deterministically rebuild the Part A pilot corpus and runs.
- Cell 5 checks the pilot sub-corpus id file against Part A's sha256: `9d1eb76d…d4a4`.

Then:

| Step | Detail | Gate |
|---|---|---|
| P8a / P8b | lock files for both environments | — |
| P5 | reader on R1, R4, CB, GOLD × 200 queries | ≤ 1% empty outputs; vLLM starts |
| P5 sanity (non-gating) | EM(GOLD) > EM(CB) expected | a failure means a bug is suspected; stop and inspect |
| P6 | judge zero-shot on R1, R4, GOLD × first 100 queries | parse-failure rate → F2 (> 2% → few-shot for the main run; the pilot then also checks few-shot) |
| P7 | p_d → N_gen; S3/S4 projections | F4 thresholds evaluated |

## 6. D2 decision (2026-09-27): ACCEPTED

RAGAS is not run. Limitations wording (agreed with the user):
> "RAGAS was not run because its local deployment requires additional infrastructure untested on our hardware; our own single-prompt judge (Appendix B) is the primary faithfulness measure and is validated against human labels."

The final clause is **conditional**. It becomes true only after the 200-item human validation (§5.4 of v0.2), and it is reworded if the result falls in the pre-registered κ bands:
- κ < 0.4: "validated against human labels, with fair agreement (κ = …)".
- κ < 0.2: judge results excluded from conclusions.
