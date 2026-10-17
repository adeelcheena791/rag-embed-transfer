"""Copy of manuscript_ipm/tools/check_manuscript.py pointed at the v0.4 revision (paths and ledger merge only)."""
"""Gate 8 lexical checks on the manuscript sources and build log. A pass is a screen, not proof of correctness.

Checks: banned words; novelty wording; anonymity; abstract length (<= 250 words, macros expanded from the ledger);
keywords (1-7); highlights (<= 85 characters, macros expanded); every \\ev ID exists; every citation key exists in
references.bib; no undefined references or citations in the LaTeX log; numbers typed in prose outside \\ev (listed for review).
Writes manuscript_ipm/records/manuscript-checks.md and exits 1 on any hard failure.
"""
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TEX = ROOT / "manuscript_ipm/revision_v04/latex"
LEDGER = {r["claim_id"]: r["value"] for r in csv.DictReader(open(ROOT / "manuscript_ipm/records/evidence-ledger.csv", encoding="utf-8"))}
for _f in ("v04", "v05"):
    LEDGER.update({r["claim_id"]: r["value"] for r in csv.DictReader(open(ROOT / f"manuscript_ipm/revision_v04/outputs/evidence_ledger_{_f}_additions.csv", encoding="utf-8"))})
BANNED = ["delve", "crucial", "landscape", "notably", "furthermore", "moreover", "pivotal", "groundbreaking", "seamless", "robust",
          "additionally"]
NOVELTY = [r"\bthe first\b", r"\bnovel\b", r"\bunprecedented\b", r"\bno prior work\b", r"\bnot aware\b", r"\bstate-of-the-art\b"]
IDENT = ["adeel", "zhengzhou", "zzu", "adeelcheena", "cheena"]
SECTIONS = sorted((TEX / "sections").glob("*.tex"))
fails, notes = [], []


def strip_comments(t):
    return re.sub(r"(?<!\\)%.*", "", t)


def expand(t):
    return re.sub(r"\\ev\{([^}]*)\}", lambda m: LEDGER.get(m.group(1), "??"), t)


def plain(t):
    t = expand(t)
    t = re.sub(r"\\(textminus)\s*", "-", t)
    t = re.sub(r"\$[^$]*\$", " X ", t)
    t = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", t)
    return re.sub(r"[{}~]", " ", t)


text = {p.name: strip_comments(p.read_text()) for p in SECTIONS}
main = strip_comments((TEX / "manuscript.tex").read_text())

# banned words and novelty wording (prose only; bibliography titles are not checked)
for name, t in text.items():
    low = plain(t).lower()
    for w in BANNED:
        for m in re.finditer(rf"\b{w}\w*", low):
            fails.append(f"banned word '{m.group(0)}' in {name}")
    for pat in NOVELTY:
        for m in re.finditer(pat, low):
            fails.append(f"novelty wording '{m.group(0)}' in {name}")
# anonymity of the anonymised manuscript
for name, t in list(text.items()) + [("manuscript.tex", main)]:
    for w in IDENT:
        if re.search(rf"\b{w}", t.lower()):
            fails.append(f"identifying string '{w}' in {name}")
# \ev ids
for name, t in list(text.items()) + [("manuscript.tex", main)]:
    for i in re.findall(r"\\ev\{([^}]*)\}", t):
        if i not in LEDGER:
            fails.append(f"unknown evidence id {i} in {name}")
# abstract, keywords, highlights
ab = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text["00_abstract.tex"], re.S).group(1)
n_words = len(plain(ab).split())
(notes if n_words <= 250 else fails).append(f"abstract words (macros expanded): {n_words} (limit 250)")
kw = re.search(r"\\begin\{keywords\}(.*?)\\end\{keywords\}", text["00_abstract.tex"], re.S).group(1).split("\\sep")
(notes if 1 <= len(kw) <= 7 else fails).append(f"keywords: {len(kw)} (limit 1-7)")
for h in re.findall(r"\\item (.*)", re.search(r"\\begin\{highlights\}(.*?)\\end\{highlights\}", text["00_abstract.tex"], re.S).group(1)):
    hl = " ".join(plain(h).split())
    (notes if len(hl) <= 85 else fails).append(f"highlight {len(hl)} chars: {hl}")
# citations
bib = set(re.findall(r"@\w+\{([^,]+),", (TEX / "references.bib").read_text()))
cited = set()
for t in text.values():
    for grp in re.findall(r"\\cite[pt]?\*?(?:\[[^\]]*\])?\{([^}]*)\}|\\citealp\{([^}]*)\}", t):
        for g in grp:
            cited |= {k.strip() for k in g.split(",") if k.strip()}
for k in sorted(cited - bib):
    fails.append(f"citation key not in references.bib: {k}")
notes.append(f"cited references: {len(cited)} of {len(bib)} in references.bib")
# build log
log = TEX / "manuscript.log"
if log.exists():
    lg = log.read_text(errors="ignore")
    for pat in (r"Citation `[^']*' on page \d+ undefined", r"Reference `[^']*' on page \d+ undefined", r"There were undefined references"):
        for m in re.findall(pat, lg):
            fails.append(f"LaTeX: {m}")
    over = re.findall(r"Overfull \\hbox \((\d+\.\d+)pt too wide\)", lg)
    big = [float(x) for x in over if float(x) > 5]
    notes.append(f"overfull hboxes > 5pt: {len(big)}" + (f" (max {max(big):.0f}pt)" if big else ""))
else:
    fails.append("manuscript.log not found (build first)")
# typed numbers in prose (screen)
typed = []
for name, t in text.items():
    if name == "10_appendix.tex":
        continue
    body = re.sub(r"\\ev\{[^}]*\}|\\(label|ref|cite[pt]?|citealp|input|includegraphics|eqref)(\[[^\]]*\])?\{[^}]*\}|\$[^$]*\$", " ", t)
    body = re.sub(r"\\begin\{equation\}.*?\\end\{equation\}", " ", body, flags=re.S)
    for m in re.finditer(r"(?<![\w.-])\d[\d,.]*\d*", body):
        typed.append(f"{name}: {m.group(0)}")
notes.append(f"numbers typed in prose (design constants etc., review manually): {len(typed)}")

out = ["# Manuscript checks (generated by tools/check_manuscript.py)", "", "## Hard failures", ""]
out += [f"- {x}" for x in fails] or ["- none"]
out += ["", "## Notes", ""] + [f"- {x}" for x in notes] + ["", "## Typed numbers (lexical screen)", ""] + [f"- {x}" for x in typed]
(ROOT / "manuscript_ipm/revision_v04/outputs/manuscript_checks_revised.md").write_text("\n".join(out) + "\n")
print("\n".join(out[:40]))
sys.exit(1 if fails else 0)
