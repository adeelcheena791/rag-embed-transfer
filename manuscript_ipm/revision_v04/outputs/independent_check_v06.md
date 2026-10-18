# Independent check v0.6 (Phase 3)

## 1. Phase 3 EM re-scored from raw reader outputs; reader passages vs ranked lists
- 18040 answers re-scored, 0 disagreements; every reader saw exactly the top five of the stored ranked list
## 2. Means and full - sub
- checked; mismatches so far: 0
## 3. Reporting rule, transfer ratios, P7, P8
- qwen: P7 True, P8 True; T(e5) 0.2331 [0.1298, 0.3342]
- llama: P7 True, P8 True; T(e5) 0.1951 [0.0922, 0.2997]
## 4. Transition contributions (full corpus)
- checked; mismatches so far: 0
## 5. Answer strings among Hit@5 misses (own containment rule, stored passage texts)
- main R1: 307 of 758 misses (0.4050)
- main R5: 207 of 414 misses (0.5000)
- full R1: 433 of 983 misses (0.4405)
- full R5: 358 of 614 misses (0.5831)
## 6. P5 without answer-bearing hard negatives
- 333 clean questions; M1 - MR = -0.1021, p = 3.39e-06
## 7. Human audit of misses
- sample re-drawn from the seed and equal to the audited items; A yes 41, C yes 59, EM 24, D yes 21 of 35

**Result: 0 mismatches.**
