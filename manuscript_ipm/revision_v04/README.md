# Revision v0.4: audit and revised manuscript

This folder is self-contained and does not modify anything outside it. The v0.3 manuscript stays in `manuscript_ipm/latex/`; its PDF is copied here as `original_manuscript_v03.pdf`.

## Rerun everything (CPU only, about 3 minutes)

From the repository root, with the packages in `requirements-retrieval.txt` (numpy, scipy, pandas, statsmodels) and a TeX Live with latexmk:

```bash
python3 -m src.analysis.v03_analysis --out /tmp/rerun/stats --tables /tmp/rerun/tables   # original analysis, for comparison
python3 manuscript_ipm/revision_v04/scripts/audit_v04.py          # ratios, decomposition, baselines, GEE, throughput, κ
python3 manuscript_ipm/revision_v04/scripts/both5_ratio.py        # HotpotQA ratios with Both@5 denominator
python3 manuscript_ipm/revision_v04/scripts/make_v04_tex.py       # new macros and tables for the revised LaTeX
python3 manuscript_ipm/revision_v04/scripts/numeric_report.py manuscript_ipm/revision_v04/latex   # every printed number
cd manuscript_ipm/revision_v04/latex && latexmk -pdf manuscript.tex && cd -
python3 manuscript_ipm/revision_v04/scripts/check_revised.py     # word limits, banned words, citations, log
python3 manuscript_ipm/revision_v04/scripts/make_docx_revised.py # Word copy (needs pandoc)
```

## Files

| File | What it is |
|---|---|
| `audit_report.md` | full audit: inventory, reproducibility, numbers, priority audits, literature, readiness |
| `study_specification.md` | verified design, with the source of each item |
| `statistical_audit.md` | transfer ratio, decomposition, GEE, HNSW, random passages, corpus, judge |
| `experiment_plan.md` | what still needs GPU time, with estimates |
| `skeptical_reviewer_report.md` | adversarial AI review of v0.4 (not peer review) |
| `validation_report.md` | every command run and its result |
| `change_log.md` | every manuscript change and why |
| `claim_evidence_ledger.csv` | major claims → data, script, table, status |
| `outputs/` | CSV/JSON results of the audit scripts |
| `scripts/` | the audit, generator and check scripts |
| `latex/` | revised manuscript source (sections, tables, `numbers_v04.tex`) |
| `revised_manuscript.pdf`, `revised_manuscript.docx` | revised manuscript |

New numbers in the revised text come from `latex/numbers_v04.tex`, written by `make_v04_tex.py` from the audit outputs; each is listed with its source in `outputs/evidence_ledger_v04_additions.csv`.
