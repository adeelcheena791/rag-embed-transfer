# Building the paper

Every number, table and figure is generated from the frozen result files. Nothing is typed by hand.

```bash
python -m src.paper.make_figures_tables   # figures (paper/figures/*.pdf) + derived statistics (paper/derived_stats.json)
python -m src.paper.make_latex_tables     # paper/latex/tables/*.tex
python -m src.paper.make_numbers          # paper/latex/numbers.tex (\NUM{key}, \CI{key}); an unknown key stops the build
python -m src.paper.audit_numbers         # must print "0 unexplained numbers"
cd paper/latex && latexmk -pdf main.tex   # ACL review mode (anonymous, line numbers)
```

Inputs: `results/statistics/{rq12,rq34}_main.json`, `results/processed/answers`, `results/raw/{judgements,generations,runs,subcorpus.jsonl,bm25.jsonl}`, `data/processed/nq_*`, `data/raw/nq_qrels/test.tsv`, `configs/main.yaml`, `annotation/sample_v1_blind.json`, and `paper/records/verified_records.json`. The last file holds the pilot facts, each with its source line in the design addenda.

Checks performed before delivery: `pdffonts` (all embedded, no Type 3), `aclpubcheck --paper_type long` on a `[final]` build (All Clear), and zero LaTeX warnings, undefined references or undefined citations.
