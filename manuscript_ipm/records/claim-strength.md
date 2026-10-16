# Claim strength, conflicts and allowed wording (Gates 2–3) — 2026-10-16

Central claims of the manuscript. IDs refer to evidence-ledger.csv (number claims) and literature-matrix.md (L-rows).

## Claim-strength review
| # | Central claim | Evidence IDs | Support | Strongest counter-evidence | Allowed wording | Required action |
|---|---|---|---|---|---|---|
| K1 | On NQ, only a small part of a retrieval gain over MiniLM-L6 reaches EM | T.qnq.*, T.lnq.*, T.*.hi | supported in both NQ cells (every upper bound ≤ 0.45) | F1/AC ratios higher (TF.*, TA.*: up to ~0.46 point estimate); EM undervalues LLM answers (L17) | "On NQ, between 0.16 and 0.27 points of EM per point of Hit@5 within our setup" | report F1/AC ratios next to EM |
| K2 | On HotpotQA, transfer is larger and less certain | T.qhp.*, T.lhp.* | supported | Llama bge-large CI reaches 1.10; F1 ratio for Llama bge-large 1.10 [≈0.94, 1.28] | "larger (0.49–0.93), with upper bounds up to 1.10; P1 fails for Llama–HotpotQA" | never pool P1 |
| K3 | On HotpotQA the larger encoder (bge-large) gives higher EM than e5-base-v2; on NQ no difference detected | P2.qhp.R4, P2.lhp.R4, HIT.hp.*, BOTH.hp.* | supported | bge-large also has the highest Hit@5 and Both@5 on HotpotQA, so this is not a retrieval/answer mismatch | "P2 fails on HotpotQA: bge-large gives higher EM than e5-base-v2, in line with its higher Both@5" — do NOT write "the best retriever is not the best for answers" | — |
| K3b | (post hoc, exploratory) On HotpotQA the EM order of the seven retrievers equals their Both@5 order in both reader cells | RHOB.qhp, RHOB.lhp | descriptive, 7 points, not pre-registered | Hit@5 order also close (rho 0.96) | "post hoc observation" only | label exploratory |
| K4 | Qwen3-Emb-0.6B gives lower EM than e5-base-v2 pooled | GEE.or.R6, P2.*.R6 | supported (OR 0.909, p 6e-7; Holm-significant in both HotpotQA cells) | NQ cells: no difference detected | as stated, pooled and per cell | — |
| K5 | Encoder cost differs by more than an order of magnitude | E7.tp.*, E7.ratio.R1R6 | supported (27×) | single GPU type; batch size fixed | "on one T4 at batch 128" | — |
| K6 | Hard non-gold passages lower EM vs random ones; position does not matter detectably | P5.*, P6.* | supported within setup (n=500, one reader) | S9: hard negatives may contain answers (unjudged positives) → bias towards zero | "top-ranked non-gold BM25 passages"; P6: "no difference detected" | limitation S9 |
| K7 | k=1 lowers EM for all retrievers | K1.* | supported | k ablation for R0/R1/R4 known before registration | confirmatory only for R2, R3, R5, R6 | disclose |
| K8 | Sub-corpus preserves ordering | E9.* | supported for 4 retrievers on NQ retrieval | absolute scores drop by ~0.08 nDCG for dense; not checked for answers or HotpotQA | "retrieval ordering of four retrievers on NQ" | — |
| K9 | Reranker helps BM25, not e5 | RR.* | X0 vs R0 EM +0.097 (p<1e-20); X5 vs R5 Hit@5 −0.067 (p 2.6e-12), EM −0.016 (p 0.069, no difference detected) | one reranker config | "one reranker at depth 50" | EM for e5 = no difference detected |
| K10 | LLM judge is unreliable here | JA.* | supported (κ 0.127) | single annotator | exclude judge from conclusions | — |

## Conflict log
| ID | Claim/metric | Source A | Source B | Reconciliation | Decision | Status |
|---|---|---|---|---|---|---|
| X1 | EM normalisation | plain SQuAD normaliser | project normaliser (+ typographic punctuation) | re-scored: 83 of 117,425 answers differ (0.07%) | use project normaliser (documented in v0.2 design); report in Methods | RESOLVED |
| X2 | Hit@5 of BM25 on NQ | 0.363 (2,255 generation queries) | 0.373 (3,452 judged queries) | different populations | always state population | RESOLVED |
| X3 | Encoder throughput | v0.2 single measurement (e.g. MiniLM 1,943/s) | v0.3 five repetitions (median 1,807/s) | different sessions; v0.3 is the pre-registered repeated measurement | report v0.3 median and range | RESOLVED |
| X4 | P1/P2 for Qwen–NQ | listed as predictions in analysis output | addendum v0.3 restricts P1/P2 to "each new cell" | the Qwen–NQ cell was observed (v0.2) before registration | report Qwen–NQ verdicts as non-confirmatory | RESOLVED |
| X5 | P4 | registered for all 7 retrievers | k ablation for R0, R1, R4 ran in v0.2 | already observed | confirmatory for R2, R3, R5, R6 only | RESOLVED |
| X6 | Failure log | addendum v0.3 §4 promised results/v03/failures.jsonl | file does not exist; failures recorded in addenda v0.3.1/v0.3.2 and notebooks | deviation | state in Deviations | RESOLVED (disclosed) |
| X7 | NQ join gate | v0.2 rule: join rate ≥ 0.70 else fall back | observed 0.653; rule replaced by n_joined ≥ 1,000 (addendum v0.2.2, before outcomes) | deviation | state in Deviations | RESOLVED (disclosed) |
