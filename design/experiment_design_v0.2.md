# Experiment Design — v0.2 (Phase 3, revised, awaiting approval)

Date: 2026-09-26 · Status: **PLANNED — NOT YET EXECUTED**. No data downloaded, no model run.
Budget: $0. Cost is measured as **GPU-hours, not dollars**; Kaggle's ≈30 h/week free GPU quota is treated as the binding constraint (quota figure UNVERIFIED — secondary source only; confirm on your account).

---

## CHANGE LOG (v0.1 → v0.2)

| ID | Request | What changed | Where |
|---|---|---|---|
| C1 | GPU session strategy | Five-session plan. Each session has one heavy model family and a fixed cell order. VRAM checkpoints are logged after every load and unload. One vLLM model per kernel lifetime. Embedders, reader and judge never share a GPU session. | §8 |
| C2 | Corpus cap | Full-corpus threshold lowered to **≤ 6 T4-hours total** for all 6 dense models. The sub-corpus is now exactly **300,000 unique passages**, with the construction order specified. | §2.1 |
| C3 | BM25 | Changed to **k1 = 1.2, b = 0.75** (Lucene/Elasticsearch convention; see §3.2 for why v0.1 used 0.9/0.4). Tokenisation: lowercase, **no stemming, no stopwords**. Library `bm25s`, method `lucene`, exact version pinned in the lock. **Found in bm25s source:** its default is k1 = 1.5, stopwords = "english", and the default token pattern drops 1-character tokens. All three are therefore set explicitly. | §3.2, App. C |
| C4 | Transfer-ratio threshold | Reported only if \|ΔHit@5\| > 2 pp **and** the difference is significant after Holm; otherwise "not estimable". | §5.3 |
| C5 | Faithfulness validation | Raised to **200 items**. κ reported with a 95% bootstrap CI. Exclusion only if κ < 0.2; 0.2 ≤ κ < 0.4 results are kept and flagged, with a sensitivity analysis. **Label note:** Landis & Koch call 0.21–0.40 "fair" and 0.41–0.60 "moderate", so the standard labels are used; the substance of the request is adopted unchanged. | §5.4 |
| C6 | Exact prompts | Added Appendix A (reader prompts), Appendix B (judge prompt) and Appendix C (BM25 configuration). | Appendices |
| C7 | Failure protocol | New §3.1: OOM batch-halving, drop rule, R6 CPU fallback on a shared reduced corpus, and a fallback for every other component. | §3.1 |
| M1 | SciFact licence | **Not adopted as requested; the facts differ.** The upstream SciFact repository states: claims CC BY 4.0, corpus abstracts ODC-By 1.0, code Apache 2.0. The HF BeIR mirror card tags cc-by-sa-4.0. **It is not CC BY-NC 2.0** (my v0.1 guess was also wrong). No non-commercial restriction applies. | §10 |
| M2 | NQ licence | Licence column added. **NQ-open = CC BY-SA 3.0** (V: HF card). **BEIR-NQ:** the HF mirror card tags cc-by-sa-4.0, while upstream Natural Questions is CC BY-SA 3.0 per the NQ-open card. Both are recorded; our derived data would be released under CC BY-SA. | §2, §10 |
| M3 | Revision pinning | Dataset commit hashes (retrieved today via the HF API) added to §2. Model revision hashes are pinned at first download and written to configs. | §2, §3 |
| M4 | A3 ablation | Kept if time permits; Limitations text written for the case where it is cut. | §6 |
| M5 | Cost framing | Sentence added (header, §5.5, §9). | — |
| R1 | Pilot checklist | Copy-paste Kaggle cells added. The environment cells run today; step cells call scripts written in Phase 4, with interfaces fixed now. | §11 |
| R2 | Power analysis | Exact McNemar minimum-detectable-difference table added, computed analytically (not experimental data). The query-count rule changed as a result. | §7.1 |
| R3 | Reproducibility statement | Added. | §12 |
| R4 | Data statement | Added. | §10 |
| R5 | RAGAS local judge | Answered in §5.4. RAGAS documents local and custom LLMs via `llm_factory` with a LiteLLM provider, but not vLLM explicitly. **Design change:** our fully specified single-prompt judge (App. B) becomes the **primary** faithfulness measure; RAGAS faithfulness is a secondary cross-check run only if the pilot passes. | §5.4 |

---

## 1. Research questions (unchanged from v0.1)

- **RQ1 — Retrieval.** Differences in retrieval quality across open embedders and BM25, with significance.
- **RQ2 — Transfer.** Fraction of each embedder's retrieval advantage that appears as answer-accuracy gain for a fixed reader; P(correct | gold retrieved) across embedders.
- **RQ3 — Cost.** Encoding throughput, query latency, memory and index size vs parameters and dimension, on fixed free hardware.
- **RQ4 — Trade-off.** Pareto frontier of answer accuracy vs measured GPU and CPU cost, compared with the frontier drawn from retrieval metrics.

Contribution type: controlled empirical benchmark. Closest prior work: BERGEN (2024), Leto et al. (2024), Cirillo et al. (2026).

---

## 2. Data

| Dataset | HF repo | Pinned revision (sha, retrieved 2026-09-26) | Licence (HF card tag) | Upstream licence | Role |
|---|---|---|---|---|---|
| BEIR-NQ corpus + queries | `BeIR/nq` | `b7253e6c379163d024ddb1d6948152a91a2e3b46` | cc-by-sa-4.0 | Natural Questions: CC BY-SA 3.0 (per the NQ-open card) | Passages (2.68M) and test queries (3,452) |
| BEIR-NQ relevance labels | `BeIR/nq-qrels` | `519acd4e48bb3e5da22b2b888ce36c614f4f2bc9` | cc-by-sa-4.0 | as above | Relevance labels (≈1.2 per query) |
| NQ-open | `google-research-datasets/nq_open` | `5dd9790a83002ad084ddeb7c420dc716852c6f28` | cc-by-sa-3.0 | CC BY-SA 3.0 | Gold answer aliases (validation split: 3,610) |
| SciFact corpus + queries | `BeIR/scifact` | `b3b5335604bf5ee3c4447671af975ea25143d4f5` | cc-by-sa-4.0 | Abstracts ODC-By 1.0; claims CC BY 4.0 | Pipeline sanity check only |
| SciFact relevance labels | `BeIR/scifact-qrels` | `2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029` | cc-by-sa-4.0 | as above | as above |

`BeIR/nq` was last modified on 2026-04-09, so pinning its hash matters. The downloader refuses to run without these hashes.

**Join.** BEIR-NQ test queries are matched to NQ-open validation questions by normalised text (lowercase, strip punctuation, collapse whitespace). The join rate is UNKNOWN and is measured in pilot step P3.
- If the join rate is < 70%, fall back to HotpotQA, with that decision recorded.
- Unjoined queries are dropped and counted.

**Query sets**
- **Retrieval evaluation:** all joined queries (cheap).
- **Generation:** a seeded sample. The size rule is in §7.1 (1,000, or up to 3,000).

**Passage text.** Every retriever sees `title + " " + text`. Passage segmentation is taken as given by BEIR, with no re-chunking.

### 2.1 Corpus decision rule (pre-registered)

- Pilot step P4 measures corpus-encoding throughput for each model on a fixed 20,000-passage slice.
- Projected total = Σ over R1–R6 of (2.68M / throughput).
- **If the projected total is ≤ 6.0 T4-hours → use the full corpus.**
- **Otherwise use the sub-corpus**, exactly 300,000 unique passages, built in this order:
  1. All gold passages for all joined queries.
  2. The BM25 (App. C) top-100 for each generation-sample query, de-duplicated.
  3. Passages drawn uniformly at random (seed `20260926`) from the rest of the corpus, until the total is exactly 300,000.
- **Declared bias:** step 2 injects BM25's hard negatives. This makes BM25's task harder than the dense models', so BM25 differences on the sub-corpus are a lower bound. Absolute scores are inflated relative to the full corpus; only relative comparisons are claimed.

---

## 3. Retrievers

| ID | Model | Params | Dim | Licence | Prefix / instruction (from the model card) | Revision |
|---|---|---|---|---|---|---|
| R0 | BM25 (`bm25s`) | — | — | MIT (UNVERIFIED) | App. C | pinned in lock |
| R1 | sentence-transformers/all-MiniLM-L6-v2 | 22.7M | 384 | Apache-2.0 | none | pinned at download |
| R2 | BAAI/bge-small-en-v1.5 | 33.4M | 384 | MIT | query: "Represent this sentence for searching relevant passages: " | " |
| R3 | BAAI/bge-base-en-v1.5 | ~0.1B | 768 | MIT | same | " |
| R4 | BAAI/bge-large-en-v1.5 | ~0.3B | 1024 | MIT | same | " |
| R5 | intfloat/e5-base-v2 | ~0.1B | 768 | MIT | "query: " / "passage: " | " |
| R6 | Qwen/Qwen3-Embedding-0.6B | 0.6B | 1024 | Apache-2.0 | query: "Instruct: Given a question, retrieve Wikipedia passages that answer the question\nQuery:{q}"; passages: none; needs transformers ≥ 4.51 | " |

Model facts were read from the Hugging Face model cards on 2026-09-26.

**R6 attention.** The card recommends FlashAttention-2, which, to my knowledge, does not run on Turing GPUs such as the T4 (UNVERIFIED). R6 therefore uses PyTorch SDPA attention, and this is declared.

**Held equal for all dense models**
- max length 256 tokens (truncation rate logged)
- fp16
- L2-normalised embeddings
- FAISS `IndexFlatIP` exact search on CPU, top-100 retrieved
- the same device

### 3.1 Failure protocol (new)

| Failure | Action | Recorded as |
|---|---|---|
| Encoding OOM | Halve the batch size: 128 → 64 → … → 1. Log every attempt. | `failures.jsonl`: model, batch size, error |
| Any model needs a batch size below 128 | Re-run the throughput benchmark for **all** models at that common batch size, so efficiency stays comparable | both sets reported |
| OOM at batch size 1 | Drop the model; report it | Limitations |
| R6 cannot run on the T4 | Encode R6 on CPU over the **300k sub-corpus**. Re-evaluate **all** retrievers on that same sub-corpus for any comparison involving R6. This is cheap, because the sub-corpus is a subset of the existing indices. R6's CPU efficiency is reported separately and never compared with GPU numbers. If the projected CPU encoding time exceeds 8 CPU-hours, drop R6. | " |
| vLLM/AWQ fails for the reader | Qwen2.5-7B-Instruct-AWQ via `transformers`; if that fails, Qwen2.5-3B-Instruct in fp16 via `transformers`, applied to **all** conditions | " |
| Judge fails with TP = 2 in fp16 | Community AWQ build `solidrust/Mistral-7B-Instruct-v0.3-AWQ` on one T4 (hash pinned; not an official Mistral release, declared). If that also fails, faithfulness is dropped; EM/F1 remain primary. | " |
| Kaggle session timeout | Every script checkpoints per model and condition, and resumes from the last completed unit. Raw files are append-only. | `logs/` |

Every failure that changes a model, the corpus or a batch size is reported in the paper.

### 3.2 BM25 parameters

v0.1's k1 = 0.9, b = 0.4 are, to my knowledge, the Anserini/Pyserini defaults tuned for MS MARCO (UNVERIFIED). That is not a good reason to use them on NQ. v0.2 uses **k1 = 1.2, b = 0.75**:
- These are the Lucene/Elasticsearch convention. BEIR's BM25 baseline ran on Elasticsearch (UNVERIFIED that it used exactly these values).
- They lie within the range the bm25s README cites from the IR textbook: k1 between 1.2 and 2.0, b = 0.75.

A retrieval-only sensitivity check with (1.5, 0.75) — the bm25s default — and (0.9, 0.4) is reported in the appendix. It costs no GPU time.

---

## 4. Generation (fixed)

- **Reader:** Qwen/Qwen2.5-7B-Instruct-AWQ (Apache-2.0, AWQ 4-bit, ungated), served with vLLM on one T4.
- **Settings:** `max_model_len = 4096`, `gpu_memory_utilization = 0.85`, greedy decoding, `max_tokens = 32`. Exact prompts are in App. A.

**Conditions**
- **Main (k = 5):** R0–R6, plus CB (closed-book) and GOLD (gold passages only) — 9 conditions × N_gen queries.
- **Ablation A1 (§6).**

---

## 5. Metrics

### 5.1 Retrieval

Computed over all joined queries: Hit@{1, 3, 5, 10}, nDCG@10 and MRR@10, via `ranx`.

### 5.2 Answer (primary; deterministic)

- EM (SQuAD normalisation, against any alias), token-F1, and containment.

### 5.3 Transfer decomposition

- Accuracy = Hit@5 · P(correct | hit) + (1 − Hit@5) · P(correct | miss).
- **Transfer ratio vs R1** = ΔEM / ΔHit@5, with a 95% bootstrap CI (both deltas resampled jointly).
- **Reporting rule (pre-registered):** the ratio is reported only if **|ΔHit@5| > 2 pp and the Hit@5 difference is significant after Holm correction** (§7). Otherwise the table shows "not estimable".
- GOLD gives the ceiling for P(correct | hit), and CB the floor for P(correct | miss).

### 5.4 Faithfulness (secondary)

**R5 answer — can RAGAS use a local judge?**
- The RAGAS stable docs configure custom or local LLMs through `llm_factory(..., provider="litellm", client=litellm.completion)`. LiteLLM can target OpenAI-compatible endpoints, which vLLM serves.
- The docs show **no explicit vLLM example**, and a GitHub issue asks about local-model support. So support is **partially verified**; whether it works with a 7B judge at an acceptable parse-failure rate is **UNKNOWN until pilot step P6**.
- An OpenAI-compatible endpoint takes a placeholder key string. That is not a paid key.

**Design choice.** The **primary faithfulness measure is our own judge**:
- Exactly the prompt in App. B, in one call, with a fixed decision rule. RAGAS runs several internal prompts that change between versions, which makes it harder to reproduce.
- **Implementation:** vLLM offline `LLM.generate`, greedy, `max_tokens = 5`. The output is parsed with regex `^\s*(SUPPORTED|NOT_SUPPORTED)\b` on the first line.
- **Parse failures** are counted and reported per condition. There are no retries, and a failed item is treated as missing, not as NOT_SUPPORTED. If the pinned vLLM version supports guided choice decoding over the two labels, it is used and recorded.
- **RAGAS faithfulness** (pinned version) runs on the same items as a secondary cross-check, **only if** its pilot parse-failure rate is ≤ 5%.

**Judge model:** mistralai/Mistral-7B-Instruct-v0.3 (Apache-2.0, ungated), fp16, tensor-parallel over 2×T4. The fallback is in §3.1.

**Scope:** the 7 main retrievers × 500 queries = 3,500 judgements. CB is not judged because it has no passages. GOLD is judged as a reference.

**Human validation**
- **200 items**, stratified: ≈28–29 per main retriever, balanced across judge labels where possible (seed 7).
- 2 annotators, same instructions as App. B. Disagreements are adjudicated by discussion into a final human label.
- **Reported:**
  - κ between the two humans
  - κ between the adjudicated human label and the judge
  - 95% CIs by bootstrap over items (10,000 resamples)
- **Decision rule for human–judge κ**

| κ | Label (Landis & Koch) | Action |
|---|---|---|
| ≥ 0.4 | moderate or better | Use as secondary evidence |
| 0.2 ≤ κ < 0.4 | fair | Report as limited reliability; include in conclusions only alongside the sensitivity analysis |
| < 0.2 | slight | Exclude from conclusions; report in the appendix |

- **Sensitivity analysis:** repeat the faithfulness comparisons (a) on the 200 human-labelled items using human labels only, and (b) excluding items where judge and human disagree. Report whether the ranking of retrievers changes.
- **Estimated effort:** 200 items × ~2 min × 2 annotators ≈ 7 h each, plus about 1 h of calibration.

### 5.5 Efficiency

Protocol unchanged from v0.1, with these additions:
- **Recorded per session:** GPU model, driver, CUDA, CPU model and core count.
- **Unit:** all "cost" results are reported in **GPU-seconds / GPU-hours and CPU-seconds**, never dollars.

---

## 6. Ablations

| ID | Factor | Levels | Retrievers | Generations |
|---|---|---|---|---|
| A1 | top-k | 1, 3, 10 | R0, R1, R4 | 9 × N_gen |
| A2 | Determinism | Re-run 100 queries of R3 main | R3 | 100 |
| A3 (cut first) | R6 MRL dimension | 256, 512 | R6 | 2 × N_gen + re-index |

**If A3 is cut, the Limitations section states:** "We did not evaluate Matryoshka dimension truncation for R6, so our dimension–cost conclusions come only from comparing different models. Those comparisons confound dimension with architecture and training."

---

## 7. Statistical analysis plan

Unchanged from v0.1: paired, per-query; exact McNemar for binary outcomes; paired bootstrap (10,000 resamples, seed 0) plus Wilcoxon for graded outcomes; Holm across all 21 pairwise comparisons within each metric family; effect sizes in pp with CIs and the discordant-pair odds ratio; Spearman across retrievers as descriptive only.

### 7.1 Power analysis (new — analytic, not experimental)

Minimum detectable difference (MDD) at 80% power for the exact two-sided McNemar test, as a function of the discordant-pair rate p_d. Computed by `design/power_mcnemar.py`, with n_d = round(n·p_d).

| n queries | p_d | MDD, α = 0.05 | MDD, α = 0.05/21 (Holm worst case) |
|---|---|---|---|
| 1,000 | 0.05 | 2.1 pp | 2.8 pp |
| 1,000 | 0.10 | 3.0 pp | 3.9 pp |
| 1,000 | 0.15 | 3.6 pp | 4.9 pp |
| 1,000 | 0.20 | 4.1 pp | 5.5 pp |
| 1,000 | 0.30 | 5.0 pp | 6.8 pp |
| 3,000 | 0.05 | 1.2 pp | 1.7 pp |
| 3,000 | 0.10 | 1.7 pp | 2.3 pp |
| 3,000 | 0.15 | 2.1 pp | 2.8 pp |
| 3,000 | 0.20 | 2.4 pp | 3.2 pp |
| 3,000 | 0.30 | 2.9 pp | 4.0 pp |

The expected p_d between embedders is **UNKNOWN**. Pilot step P7 estimates it from the pilot EM outcomes of R1 vs R4.

**Pre-registered N_gen rule**
- **If the pilot p_d ≤ 0.10:** N_gen = 1,000 (worst-case MDD ≈ 4 pp).
- **Otherwise:** N_gen = min(joined queries, 3,000). This roughly triples generation cost, estimated at +2–4 T4-hours (UNVERIFIED).

**Interpretation rule:** a non-significant difference is reported as "no difference detected at MDD ≈ X pp", never as "no difference".

---

## 8. GPU session plan (C1)

Principle: **one heavy model family per Kaggle session.** vLLM does not reliably free GPU memory inside one Python process, so every vLLM model gets a fresh kernel. Embedders, reader and judge never coexist on a GPU; a coexistence test is therefore unnecessary.

| Session | Accelerator | Visible GPUs | Content | Input | Output (saved as a Kaggle Dataset) |
|---|---|---|---|---|---|
| S0 | CPU (no GPU quota) | — | Download pinned data, join, BM25 index and retrieval, sub-corpus construction if §2.1 requires it | HF | `rag-data-v1` |
| S1 | GPU T4 ×2 | `CUDA_VISIBLE_DEVICES=0` (one device, for comparable timings) | For R1, R2, R3, R5, R4, R6 in that order (small to large, so failures show up early): throughput benchmark, GPU query-latency benchmark, corpus and query encoding | `rag-data-v1` | `rag-emb-v1` (fp16 `.npy` per model) |
| S2 | CPU | — | FAISS exact search for all models, CPU query-latency benchmark, retrieval metrics | `rag-emb-v1` | `rag-runs-v1` |
| S3 | GPU T4 ×2 | `CUDA_VISIBLE_DEVICES=0` | Reader (vLLM): 9 main conditions, then A1, then A2 | `rag-runs-v1` | `rag-gen-v1` |
| S4 | GPU T4 ×2 | both (TP = 2) | Judge (vLLM): 3,500 main judgements, plus RAGAS cross-check if the pilot passed | `rag-gen-v1` | `rag-judge-v1` |

**VRAM checkpoints** (written to `logs/vram_<session>.jsonl`; S1 cell order shown)
1. `env_report` — GPU name, driver, CUDA, total and free memory.
2. For each model:
   - **CP-a (before load):** `torch.cuda.memory_allocated` must be ≤ 256 MiB, otherwise stop and restart the kernel.
   - Load the model. **CP-b:** allocated, reserved, and `nvidia-smi` used memory.
   - Throughput benchmark. **CP-c:** `max_memory_allocated` — this is the reported peak.
   - Encode and save.
   - `del model; gc.collect(); torch.cuda.empty_cache()`. **CP-d:** allocated must be ≤ 256 MiB. If not, save progress and restart the kernel before the next model.

**S3 / S4:** `env_report` → one vLLM initialisation → **CP-b** (`nvidia-smi`) → generation loop with a checkpoint every 500 prompts → end of session.

**Expected peaks** (UNVERIFIED; the pilot measures them)
- R6 in fp16: ≈1.2 GB of weights plus activations, well within 15 GB.
- Reader AWQ: a few GB of weights; vLLM then pre-allocates up to 85% of the GPU.
- Judge fp16 split over 2 GPUs: ≈7.3 GB of weights per GPU.

---

## 9. Compute budget (UNVERIFIED — the pilot replaces these with measurements)

| Item | T4-hours |
|---|---|
| Pilot | ~1.5 |
| S1 encoding | ≤ 6.0 (cap, §2.1) |
| S3 generation | ~2–6, depending on N_gen |
| S4 judge (2 GPUs count double) | ~3–5 |
| **Total target** | **≤ 20** |

This is within one weekly free quota, with a re-run buffer. Cost is reported as GPU-hours against that quota, not in dollars.

---

## 10. Data statement (R4)

**Provenance**
- NQ questions are real, anonymised, aggregated Google search queries (per the NQ-open card).
- NQ passages are English Wikipedia, segmented by BEIR.
- SciFact abstracts come from S2ORC; its claims were written by annotators.

**Licences** — the full table is in §2.
- We follow **upstream** licences where they differ from the HF mirror tags: NQ CC BY-SA 3.0; SciFact abstracts ODC-By 1.0 and claims CC BY 4.0.
- Any released derived data (query sample, join map, sub-corpus ids) is released under **CC BY-SA 4.0**, compatible with the share-alike terms. Only ids and our own annotations are released, not re-hosted corpora.

**Preprocessing:** the exact steps are in §2 and App. C, with no manual edits. Per-dataset counts (joined, dropped, truncated) are reported.

**Personal data:** no collection. Human annotators label only model outputs; no personal information is recorded about them.

**Language and domain:** English, general encyclopedic (NQ), and biomedical abstracts (SciFact, sanity check only).

---

## 11. Pilot checklist (R1) — Kaggle, copy-paste

**Notebook settings:** Accelerator **GPU T4 ×2** (never P100: vLLM needs compute capability ≥ 7.0); Internet **On**. No Hugging Face token is needed, since all models are ungated.

**Cell 1 — environment (runnable today)**
```bash
!nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
!python --version
!nproc && free -g && df -h /kaggle/working
!pip list 2>/dev/null | grep -Ei "^(torch|transformers|sentence-transformers|vllm|faiss|numpy) "
```

**Cell 2 — code (the repository is created in Phase 4; replace `<USER>`)**
```bash
!git clone https://github.com/<USER>/rag-embed-transfer.git /kaggle/working/rag
%cd /kaggle/working/rag
!pip install -q -r requirements-retrieval.txt
!python -m src.env_report --session pilot --out results/raw/env_pilot.json
```

**Cells 3–9 — pilot steps (scripts delivered in Phase 4; interfaces fixed here)**

| Step | Command | Pass criterion |
|---|---|---|
| P1 | `!python -m src.data.download --config configs/pilot.yaml` | All five sha values match §2 |
| P2 | `!python -m src.sanity.scifact --models R1,R2,R3,R4,R5,R6` | nDCG@10 within ±0.02 of the values on the model cards, where a card reports one |
| P3 | `!python -m src.data.join_nq --config configs/pilot.yaml` | Join rate ≥ 70% (else HotpotQA fallback) |
| P4 | `!CUDA_VISIBLE_DEVICES=0 python -m src.retrieval.throughput --models R1,R2,R3,R5,R4,R6 --n 20000 --batch 128` | Writes the projected full-corpus T4-hours; applies the §2.1 rule |
| P5 | `!CUDA_VISIBLE_DEVICES=0 python -m src.generation.run --config configs/pilot.yaml --conditions R1,R4,CB,GOLD --n 200` | vLLM starts; ≤ 1% empty outputs |
| P6 | *(fresh kernel)* `!python -m src.evaluation.judge --config configs/pilot.yaml --n 100 --ragas-check` | Our judge's parse failures ≤ 5%; decide on RAGAS |
| P7 | `!python -m src.analysis.pilot_power --in results/raw/pilot_gen.jsonl` | Reports p_d and applies the §7.1 N_gen rule |
| P8 | `!pip freeze > requirements.lock` | The lock file is committed |

After the pilot, you send me `results/raw/`, `logs/` and `requirements.lock`. I analyse them and report go/no-go on every check before any main run.

---

## 12. Reproducibility statement (R3)

- **Seeds:**
  - query sample and sub-corpus: `20260926`
  - bootstrap: `0`
  - annotation stratification: `7`
  - generation and judging are greedy; residual non-determinism is quantified by A2
- **Versions:**
  - `requirements.lock` from `pip freeze` in each session type
  - Python, CUDA and driver recorded by `env_report`
  - dataset hashes (§2) and model revision hashes stored in `configs/*.yaml`
- **Hardware:** Kaggle T4 (GPU name, driver and memory logged per session); Kaggle CPU (model and core count logged). All efficiency numbers are labelled with the session they came from.
- **Commands:** one command per step (§11 for the pilot; the main runs follow the same interface with `configs/main.yaml`), plus `reproduce.sh`, which chains S0–S4 and the analysis.
- **Result integrity:**
  - `results/raw/` is append-only JSONL, with `run_id` = UTC timestamp + config hash.
  - Processed results, statistics, figures and tables are regenerated only by scripts, never edited by hand.
  - Failed runs are kept and logged (§3.1).
- **Release:** code on GitHub; results and ids on Kaggle Datasets and optionally Zenodo, all free.

---

## Appendix A — Reader prompts (exact)

Chat messages rendered with the reader tokenizer's own `apply_chat_template(..., add_generation_prompt=True)`.

**A.1 Retrieval conditions (R0–R6, GOLD)**

System:
```
You are a question answering assistant. Answer the question using the passages provided. If the passages do not contain the answer, give your best short answer anyway. Reply with the answer only, as a short phrase of a few words, with no explanation.
```
User:
```
Passages:
[1] {title_1}
{text_1}

[2] {title_2}
{text_2}

... (k passages, in retrieval rank order; GOLD uses the gold passages in qrels order)

Question: {question}
Answer:
```

**A.2 Closed-book (CB)**

System:
```
You are a question answering assistant. Reply with the answer only, as a short phrase of a few words, with no explanation.
```
User:
```
Question: {question}
Answer:
```

**Decoding:** temperature 0 (greedy), `max_tokens = 32`, stop at the first newline. The output is stripped of whitespace and trailing punctuation before scoring.

**Declared choice.** "Give your best short answer anyway" allows the reader to answer from memory when retrieval misses. That keeps P(correct | miss) meaningful, and the faithfulness judge reveals it as unsupported answers.

## Appendix B — Judge prompt (exact)

A single user message with no system role. The Mistral chat template may reject a system message (UNVERIFIED), so none is used.

```
You will be given PASSAGES, a QUESTION, and an ANSWER.
Decide whether the ANSWER is fully supported by the PASSAGES.
An answer is SUPPORTED only if the passages state it or directly imply it.
Do not use outside knowledge. Do not judge whether the answer is true in the real world.

PASSAGES:
{passages formatted exactly as in Appendix A.1}

QUESTION: {question}
ANSWER: {answer}

Reply with exactly one word: SUPPORTED or NOT_SUPPORTED.
```

**Decoding:** greedy, `max_tokens = 5`. Parse with regex `^\s*(SUPPORTED|NOT_SUPPORTED)\b`; the first match wins, and no match counts as a parse failure.

**Human annotators** receive the same text, plus: "Label each item SUPPORTED or NOT_SUPPORTED. If unsure, choose NOT_SUPPORTED and add a note."

## Appendix C — BM25 configuration (exact)

```python
import bm25s                      # exact version from requirements.lock (≥ 0.2; 0.3.11 is current on PyPI as of 2026-09-26)

TOKEN_PATTERN = r"(?u)\b\w+\b"    # keeps 1-character tokens (the bm25s default "\b\w\w+\b" drops them)

def tok(texts):
    return bm25s.tokenize(texts,
                          lower=True,               # lowercase
                          token_pattern=TOKEN_PATTERN,
                          stopwords=[],             # NO stopwords (bm25s default is "english")
                          stemmer=None)             # NO stemming

docs = [p["title"] + " " + p["text"] for p in corpus]
retriever = bm25s.BM25(k1=1.2, b=0.75, method="lucene")   # bm25s default would be k1 = 1.5
retriever.index(tok(docs))
results, scores = retriever.retrieve(tok(queries), k=100)
```

- Queries and documents use the identical tokenizer.
- The index is saved with `retriever.save(...)`.
- **Sensitivity (appendix, retrieval only):** (k1, b) = (1.5, 0.75) and (0.9, 0.4).
- **Note:** BEIR's published BM25 used Elasticsearch's analysis chain (UNVERIFIED which filters). Our numbers are not expected to match it and are not compared with it.
- One code detail must be checked in Phase 4: whether bm25s accepts `stopwords=[]` or needs `None` to disable stopwords. The README says to pass an empty list, but the function signature's type is `str | list`. A unit test will assert that the stopword "the" is kept.
