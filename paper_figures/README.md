# Figures and tables for the journal version (v0.3)

Everything here is generated from data files committed in this repository. No number is typed by hand.

## Run it yourself (VS Code, Windows / macOS / Linux)

1. Open the repository folder in VS Code and open a terminal (Terminal → New Terminal).
2. Create an environment and install the packages (once):

   ```
   python -m venv .venv
   .venv\Scripts\activate          # Windows   (macOS/Linux: source .venv/bin/activate)
   pip install -r paper_figures/requirements.txt
   ```

3. Optional, about 1–2 minutes: recompute all statistics from the raw result files.

   ```
   python -m src.analysis.judge_agreement --labels annotation/judge_validation_labels.json --out paper_outputs/v03/statistics/judge_agreement.json
   python -m src.analysis.v03_analysis
   ```

4. Make the figures and tables (a few seconds):

   ```
   python paper_figures/make_figures_v03.py
   python paper_figures/make_tables_v03.py
   ```

Output goes to `paper_figures/out/` (figures as vector PDF for the manuscript and 300-dpi PNG for checking) and
`paper_figures/out/tables/` (LaTeX, booktabs). The CSV versions of every table are in `paper_outputs/v03/tables/`.

## What each figure shows

| File | Content |
|---|---|
| fig1_transfer_ratios | Transfer ratio T(R) = ΔEM / ΔHit@5 vs MiniLM-L6 with 95% bootstrap CIs, NQ and HotpotQA, both readers |
| fig2_hit_vs_em | Hit@5 against exact match for the seven retrievers in the four reader × dataset cells, with closed-book and gold-passage references |
| fig3_topk | Exact match as a function of the number of passages k (Qwen, NQ) |
| fig4_mechanism | Distraction (hard vs random negatives) and gold-position tests, 500 NQ queries |
| fig5_cost_quality | Encoding throughput on one T4 (median and range of five runs) against mean exact match |
| fig6_full_corpus | nDCG@10 on the 300k sub-corpus vs the full 2.68M NQ corpus |

Style: one fixed colour order (checked for colour-vision deficiency), markers as a second cue, thin marks, light grid,
fonts embedded as TrueType (pdf.fonttype 42) as journals require.
