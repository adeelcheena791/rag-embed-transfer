"""Build the manuscript PDF from committed artifacts only.

  python manuscript_ipm/tools/build_paper.py

Steps: regenerate the ledger/macros (make_ledger.py) and bibliography (make_bib.py); copy the figure PDFs and table .tex files
produced by paper_figures/*.py into latex/figs and latex/tables; compile manuscript.tex and title_page.tex with latexmk;
then run the checks in check_manuscript.py.
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "manuscript_ipm/latex"


def run(cmd, cwd=ROOT):
    print("+", " ".join(cmd))
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode:
        sys.exit(r.returncode)


def main():
    run([sys.executable, "manuscript_ipm/tools/make_ledger.py"])
    run([sys.executable, "manuscript_ipm/tools/make_bib.py"])
    (TEX / "figs").mkdir(exist_ok=True)
    (TEX / "tables").mkdir(exist_ok=True)
    for f in (ROOT / "paper_figures/out").glob("fig*.pdf"):
        shutil.copy2(f, TEX / "figs" / f.name)
    for f in (ROOT / "paper_figures/out/tables").glob("*.tex"):
        shutil.copy2(f, TEX / "tables" / f.name)
    for doc in ("manuscript", "title_page"):
        run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", f"{doc}.tex"], cwd=TEX)
    run([sys.executable, "manuscript_ipm/tools/check_manuscript.py"])


if __name__ == "__main__":
    main()
