"""Copy of manuscript_ipm/tools/make_docx.py pointed at the v0.4 revision (paths and ledger merge only)."""
"""Word copy of the anonymised manuscript (convenience copy; the LaTeX/PDF is the version of record).

Flattens manuscript.tex (\\input files, \\ev macros expanded from the ledger, PNG figures), then runs pandoc with citeproc.
Output: manuscript_ipm/latex/manuscript.docx
"""
import csv
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TEX = ROOT / "manuscript_ipm/revision_v04/latex"
LEDGER = {r["claim_id"]: r["value"] for r in csv.DictReader(open(ROOT / "manuscript_ipm/records/evidence-ledger.csv", encoding="utf-8"))}
for _f in ("v04", "v05", "v06"):
    LEDGER.update({r["claim_id"]: r["value"] for r in csv.DictReader(open(ROOT / f"manuscript_ipm/revision_v04/outputs/evidence_ledger_{_f}_additions.csv", encoding="utf-8"))})


def flatten(path):
    t = path.read_text()
    t = re.sub(r"(?<!\\)%.*", "", t)
    return re.sub(r"\\input\{([^}]*)\}", lambda m: "" if m.group(1) in ("numbers", "numbers_v04", "numbers_v05", "numbers_v06", "glyphtounicode") else flatten(TEX / (m.group(1) + ".tex")), t)


def main():
    t = flatten(TEX / "manuscript.tex")
    t = re.sub(r"\\ev\{([^}]*)\}", lambda m: LEDGER[m.group(1)], t)
    t = t.replace("\\ensuremath{-}", "\u2212")
    t = t.replace("\\title[mode=title]{", "\\title{")
    t = re.sub(r"\\RenewDocumentCommand\\printorcid\{\}\{\}", "", t)
    t = re.sub(r"figs/(fig\d+_[a-z_]+)\.pdf", lambda m: str(ROOT / "paper_figures/out" / (m.group(1) + ".png")), t)
    t = re.sub(r"\\begin\{highlights\}.*?\\end\{highlights\}", "", t, flags=re.S)
    t = re.sub(r"\\begin\{keywords\}(.*?)\\end\{keywords\}",
               lambda m: "\\paragraph{Keywords} " + "; ".join(x.strip() for x in m.group(1).split("\\sep")), t, flags=re.S)
    t = re.sub(r"\\par\\smallskip\\parbox\{\\linewidth\}\{\\footnotesize (.*?)\}\n\\end\{table", r"\\end{tabular}\n\n\\emph{\1}\n\\end{table", t)
    t = t.replace("\\end{tabular}\n\\end{tabular}", "\\end{tabular}")
    title = re.search(r"\\title\{(.*?)\}\n", t).group(1)
    flat = TEX / "manuscript_flat.tex"
    flat.write_text(t)
    subprocess.run(["pandoc", str(flat), "-o", str(TEX / "manuscript.docx"), "--citeproc", f"--bibliography={TEX / 'references.bib'}",
                    "--resource-path", str(TEX), "-M", "link-citations=true", "-M", f"title={title}"], check=True)
    flat.unlink()
    print("wrote", TEX / "manuscript.docx")


if __name__ == "__main__":
    main()
