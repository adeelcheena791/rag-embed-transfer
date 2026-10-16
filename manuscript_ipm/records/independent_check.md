# Independent recomputation (Gate 2)

Script: manuscript_ipm/tools/independent_check.py (standard library only, no project imports).

## 1. EM re-scored from raw reader outputs (own normaliser) vs scored answer files
- HotpotQA Qwen: 18000 answers re-scored, 0 disagreements
- HotpotQA Llama: 18000 answers re-scored, 0 disagreements
- NQ Llama: 20295 answers re-scored, 0 disagreements
- NQ Qwen top-k (v0.3): 27060 answers re-scored, 0 disagreements
- NQ Qwen S4: 7010 answers re-scored, 0 disagreements
- NQ Qwen v0.2 (all conditions, k ablation, A3, A2rep): 41490 answers re-scored, 0 disagreements
- TOTAL: 131855 answers, 0 disagreements (0.000%)

## 2-3. Hit@5, EM means and transfer-ratio point estimates per cell (own computation vs analysis file)
- qwen_nq: T(own/analysis) R0 0.372/0.372, R2 0.190/0.190, R3 0.204/0.204, R4 0.242/0.242, R5 0.250/0.250, R6 0.254/0.254
- llama_nq: T(own/analysis) R0 0.358/0.358, R2 0.184/0.184, R3 0.271/0.271, R4 0.158/0.158, R5 0.160/0.160, R6 0.221/0.221
- qwen_hp: T(own/analysis) R0 2.156/n.e., R2 0.659/0.659, R3 0.709/0.709, R4 0.836/0.836, R5 0.689/0.689, R6 0.490/0.490
- llama_hp: T(own/analysis) R0 2.375/n.e., R2 0.692/0.692, R3 0.837/0.837, R4 0.927/0.927, R5 0.831/0.831, R6 0.537/0.537
- largest absolute difference over all Hit@5, EM means and ratios: 0.00e+00

## 4. Exact McNemar p-values (own binomial) vs analysis (unadjusted p)
- qwen_hp e5 vs R4: own p = 0.000785385, analysis p = 0.000785385
- qwen_hp e5 vs R6: own p = 0.000386488, analysis p = 0.000386488
- llama_hp e5 vs R4: own p = 0.0100512, analysis p = 0.0100512
- llama_hp e5 vs R6: own p = 8.50969e-07, analysis p = 8.50969e-07
- P5 M1 vs MR: own p = 2.24296e-05, analysis p = 2.24296e-05

## 5. Cohen's kappa (own) vs analysis
- kappa own = 0.127382, analysis = 0.127382; n = 200

## 6. Other EM means (own mean over the documented query set) vs analysis
- 43 EM means compared; largest absolute difference 0.00e+00

## 7. Pairwise EM tests: own exact McNemar + own Holm (21 pairs per cell) vs analysis
- 84 pairs: largest |p difference| 9.99e-16; largest |Holm p difference| 2.00e-15; pairs whose Holm significance at 0.05 differs: 0
