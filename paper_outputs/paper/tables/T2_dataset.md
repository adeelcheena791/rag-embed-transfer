**Table 2. Data and sub-corpus.**

| item | value | note |
|---|---|---|
| Source corpus | BEIR-NQ (BeIR/nq) | 2,681,468 passages |
| Judged test queries | 3,452 | BeIR/nq-qrels test |
| Joined to NQ-open answers | 2,255 | join rate 65.3%; primary query set |
| Sub-corpus size | 300,000 | exact search over this set |
|   gold passages | 4,201 | all judged gold passages |
|   BM25 negatives | 127,791 | BM25 top-42 per query |
|   random passages | 168,008 | uniform fill |
| Generation queries | 2,255 | all conditions, k = 5 |
| Top-k ablation queries | 2,255 | R0, R1, R4 at k = 1, 3, 5, 10 |
| Abstention prompt (A3) queries | 200 | R0, R1, R4, gold |
| Faithfulness-judged queries | 500 | first of the seeded order (gold: 499) |
| Human-validation items | 200 | stratified by retriever and verdict |

