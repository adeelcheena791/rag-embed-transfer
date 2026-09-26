# Design addendum v0.2.1 (2026-09-26) — follow-ups F1–F4 and Batch 1 implementation decisions

Status: PLANNED — NOT YET EXECUTED. This addendum is pre-registered together with v0.2.

## F1 — Reader "I don't know" robustness check
- Second reader prompt (Appendix A.3, below). It is identical to A.1 except that the model may answer exactly `I don't know` when the passages lack the answer.
- **Scope:** the first 200 queries of the seeded order (a nested prefix, so a subset of the main sample); retrievers R0, R1, R4 and GOLD at k = 5; 800 generations.
- **Scoring:**
  - EM / F1 are computed as for the main runs.
  - An "I don't know" (IDK) answer scores as incorrect for EM.
  - IDK rate is reported separately; the rate conditional on a retrieval miss is the informative number.
- **Reported in the appendix only.** The main results use A.1, as pre-registered.

**A.3 system prompt (exact):**
```
You are a question answering assistant. Answer the question using only the passages provided. If the passages do not contain the answer, reply exactly: I don't know. Otherwise reply with the answer only, as a short phrase of a few words, with no explanation.
```
The user message is identical to A.1. IDK detection: after normalisation (lowercase, strip punctuation), the output equals `i don t know` or `i dont know`.

## F2 — Judge few-shot fallback
- **Trigger:** if pilot P6 shows a parse-failure rate > 2% for the zero-shot judge, the few-shot prompt below replaces Appendix B for **all** judgements, and the switch is reported.
- **Examples:** written now, before any data is seen. They use invented passages that do not come from NQ, so they cannot leak test content.

**B.2 few-shot block** (inserted before `PASSAGES:` in Appendix B):
```
Here are three examples.

Example 1
PASSAGES:
[1] Lake Arvel
Lake Arvel is a freshwater lake in the fictional region of Tolmark. Its deepest point is 212 metres.
QUESTION: how deep is lake arvel
ANSWER: 212 metres
Verdict: SUPPORTED

Example 2
PASSAGES:
[1] Orsin Bridge
The Orsin Bridge was completed in 1874 and spans the river Dellet.
QUESTION: who designed the orsin bridge
ANSWER: Marta Keel
Verdict: NOT_SUPPORTED

Example 3
PASSAGES:
[1] Valen Cup
The Valen Cup is an annual rowing race. The 2019 race was won by the Harrow Point crew.
[2] Harrow Point
Harrow Point is a coastal town known for its rowing club.
QUESTION: who won the 2019 valen cup
ANSWER: Harrow Point
Verdict: SUPPORTED

Now the item to judge.
```
Example 2 is a plausible answer that the passages don't support: it tests "no outside knowledge". Example 3 uses a multi-passage context. Decoding and parsing are unchanged.

## F3 — R6 CPU fallback threshold
- The threshold is lowered from 8 to **4 CPU-hours**: projected time to encode the 300k sub-corpus on the Kaggle CPU with R6.
- If exceeded, **R6 is dropped** and the drop is reported in Limitations. R6 is not needed for the core claim, which rests on the BGE size ladder plus R1, R5 and BM25.
- Implemented in `src/retrieval/throughput.py::decide_r6_cpu` and set in `configs/pilot.yaml` as `r6_cpu_fallback.max_cpu_hours: 4.0`.

## F4 — Stopping rules for the main run (pre-registered)
Measured GPU time comes from the session logs (wall-clock × visible GPUs).

| Rule | Condition | Action |
|---|---|---|
| F4.1 | S1+S2 retrieval differences below the MDD | **Still proceed to S3.** The transfer decomposition is the main contribution, and "no retrieval difference detected at MDD ≈ X pp" is itself reportable. |
| F4.2 | S3 projected or measured > 8 T4-hours | Cut N_gen to 1,000 (if it was 3,000); keep all 9 main conditions; cut A1 before main conditions. |
| F4.3 | S4 projected or measured > 6 T4-hours | Cut judge scope from 500 to 300 queries per retriever. The 200 human-validation items are drawn from within those 300. |
| F4.4 | S3 + S4 combined > 14 T4-hours | Stop running. Write up with the completed conditions. List every uncompleted condition in the paper. |

"Projected" = the per-item time measured in the pilot × the planned number of items. The check is made before each session starts, and again every 500 items during the session.

## Batch 1 implementation decisions (new; not in v0.2)

**1. Pilot corpus.**
- Pilot step P5 needs dense retrieval, but the corpus decision is only made in P4. So the pilot uses its own **50,000-passage pilot corpus**, built by the same procedure as the §2.1 sub-corpus: gold passages + BM25 negatives for the 200 pilot queries + random fill.
- Pilot numbers are never reported as results.

**2. BM25 depth rule for sub-corpora.**
- v0.2 fixed the depth at 100. With N_gen = 3,000 that would fill the entire 300k budget with BM25 negatives.
- New rule: depth = min(100, ⌊0.5 × (size − |gold|) / n_queries⌋), so BM25 negatives take at most half of the non-gold budget.
- For N_gen = 1,000 the depth stays 100; for N_gen = 3,000 it is 49.

**3. Nested samples.** One seeded permutation of joined query ids. Every sample is a prefix of it:
- generation: N_gen
- judge: 500 (300 under F4.3)
- F1: 200
- A2: 100

**4. Dense search is sharded exact search** (FAISS IndexFlatIP per 100k-row shard + global top-k merge). The ranking is identical to a single flat index; a unit test checks this. Ties are broken by lower corpus row.

**5. SciFact sanity check** uses the model's native 512-token length (the settings behind the model-card numbers), not the 256 cap. References (from the model card READMEs, read 2026-09-26): R2 0.71275, R3 0.74039, R4 0.74607, R5 0.71944. R1 and R6 have no card value and are reported as "no_reference".

**6. bm25s checks, executed locally on bm25s 0.3.11.**
- `stopwords=[]` together with pattern `\b\w+\b` keeps all tokens, e.g. `the cat is a b 1 dog` → 7 tokens.
- The library defaults would keep only `cat`, `dog`.

## Revised pilot order (replaces the §11 table order)
Commands are exact.

**P1–P3b** can run in a CPU session. **P2 onwards** must run in a GPU T4 ×2 session with `CUDA_VISIBLE_DEVICES=0`.

| Step | Command |
|---|---|
| P1 | `python -m src.data.download --config configs/pilot.yaml` |
| P3 | `python -m src.data.join_nq --config configs/pilot.yaml` |
| P3b | `python -m src.retrieval.bm25 --config configs/pilot.yaml --corpus full --queries order:200` |
|  | `python -m src.data.subcorpus --config configs/pilot.yaml --bm25-run results/raw/runs/R0__full__<RUN_ID>.jsonl --size 50000 --n-queries 200 --name pilot` |
| P2 | `python -m src.sanity.scifact --config configs/pilot.yaml --models R1,R2,R3,R4,R5,R6` |
| P4 | `python -m src.retrieval.throughput --config configs/pilot.yaml --models R1,R2,R3,R5,R4,R6 --n 20000 --batch 128` |
| P4b | `python -m src.retrieval.encode --config configs/pilot.yaml --models R1,R4 --corpus pilot` |
|  | `python -m src.retrieval.search --config configs/pilot.yaml --model R1 --corpus pilot --queries order:200` |
|  | `python -m src.retrieval.search --config configs/pilot.yaml --model R4 --corpus pilot --queries order:200` |
| P5–P8 | As in v0.2 §11. Scripts arrive in Batches 2–3. |
