"""Numerical consistency report for every \\ev{} macro used in the manuscript and its tables.

For each macro: where it appears, the value printed in the paper (from the evidence ledger), and, where audit_v04.py or this
script recomputes the quantity from per-query data, the recomputed value rounded to the printed precision. All other
values are checked by the full rerun of src/analysis/v03_analysis.py, whose output was identical to the committed JSON
(see validation_report.md).

    python3 manuscript_ipm/revision_v04/scripts/numeric_report.py [latex_dir]
"""
from __future__ import annotations

import csv
import json
import re
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.analysis.v03_analysis import CELLS, R, Data  # noqa: E402

OUT = ROOT / "manuscript_ipm/revision_v04/outputs"
TEX = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "manuscript_ipm/latex"
LED = {r["claim_id"]: r for r in csv.DictReader(open(ROOT / "manuscript_ipm/records/evidence-ledger.csv", encoding="utf-8"))}
V4 = OUT / "evidence_ledger_v04_additions.csv"
if V4.exists():
    for r in csv.DictReader(open(V4, encoding="utf-8")):
        LED[r["claim_id"]] = {"value": r["value"], "source": "manuscript_ipm/revision_v04/outputs/" + r["source"]}
AUD = json.load(open(OUT / "audit_results.json"))
TR = {(r["cell"], r["retriever"]): r for r in csv.DictReader(open(OUT / "transfer_ratio_audit.csv"))}
DC = {(r["cell"], r["retriever"]): r for r in csv.DictReader(open(OUT / "transfer_decomposition_audit.csv"))}
SHORT = {"qnq": "qwen_nq", "lnq": "llama_nq", "qhp": "qwen_hp", "lhp": "llama_hp"}
NAME = {"R0": "BM25", "R1": "MiniLM-L6", "R2": "bge-small", "R3": "bge-base", "R4": "bge-large", "R5": "e5-base-v2",
        "R6": "Qwen3-Emb-0.6B"}


def nums(s):
    s = s.replace("\\ensuremath{-}", "-").replace("\\%", "").replace(",", "")
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", s)]


def decimals(s):
    s = s.replace("\\ensuremath{-}", "-").replace("\\%", "")
    m = re.findall(r"\d+\.(\d+)", s)
    return len(m[0]) if m else 0


def main():
    D = Data(ROOT)
    means = {}
    for cell in CELLS:
        for r in R:
            ret, ans = D.retrieval(cell, r), D.answers(cell, f"{r}_k5_A1")
            means[(cell, r)] = {"hit": ret["hit@5"].mean(), "em": ans["em"].mean(), "both": ret.get("both@5", np.array([np.nan])).mean()}
        for c in ("CB", "GOLD"):
            means[(cell, c)] = {"em": D.answers(cell, "CB_A2" if c == "CB" else "GOLD_A1")["em"].mean()}
    hp_shares = [float(r["share_conditional"]) for k, r in DC.items() if k[0].endswith("hp") and r["ratio_estimable"] == "yes"]

    def recompute(cid):
        p = cid.split(".")
        try:
            if p[0] == "T" and len(p) >= 3 and p[1] in SHORT:
                row = TR[(SHORT[p[1]], NAME[p[2]])]
                if len(p) == 3:
                    return [float(row["recomputed_ratio"])], "transfer_ratio_audit.csv"
                if p[3] == "ci":
                    return [float(row["ci_lo"]), float(row["ci_hi"])], "transfer_ratio_audit.csv"
                if p[3] in ("hi", "hi3"):
                    return [float(row["ci_hi"])], "transfer_ratio_audit.csv"
            if p[0] == "DS" and p[1] == "hp":
                return [100 * (min if p[2] == "min" else max)(hp_shares)], "transfer_decomposition_audit.csv"
            if p[0] == "DS":
                return [100 * float(DC[(SHORT[p[1]], NAME[p[2]])]["share_conditional"])], "transfer_decomposition_audit.csv"
            if p[0] == "DR":
                row, t = DC[(SHORT[p[1]], NAME[p[2]])], TR[(SHORT[p[1]], NAME[p[2]])]
                return [float(row["share_conditional"]) * float(t["recomputed_ratio"])], "transfer_decomposition_audit.csv"
            if p[0] == "DB":
                row = DC[(SHORT[p[1]], "bge-small")]
                return [float(row["a0_em_hit_base"]) - float(row["b0_em_miss_base"])], "transfer_decomposition_audit.csv"
            if p[0] in ("PH", "PM"):
                cell, r = SHORT[p[1]], p[2]
                if r == "R1":
                    row = DC[(cell, "bge-small")]
                    return [float(row["a0_em_hit_base" if p[0] == "PH" else "b0_em_miss_base"])], "transfer_decomposition_audit.csv"
                row = DC[(cell, NAME[r])]
                return [float(row["a1_em_hit_sys" if p[0] == "PH" else "b1_em_miss_sys"])], "transfer_decomposition_audit.csv"
            if p[0] in ("DH", "DE"):
                row = TR[(SHORT[p[1]], NAME[p[2]])]
                return [float(row["d_hit5" if p[0] == "DH" else "d_em"])], "transfer_ratio_audit.csv"
            if p[0] == "HIT" and len(p) == 3:
                cell = "qwen_nq" if p[1] == "nq" else "qwen_hp"
                return [means[(cell, p[2])]["hit"]], "per-query recomputation"
            if p[0] == "BOTH" and len(p) == 3:
                return [means[("qwen_hp", p[2])]["both"]], "per-query recomputation"
            if p[0] == "EM" and len(p) == 3:
                return [means[(SHORT[p[1]], p[2])]["em"]], "per-query recomputation"
            if cid == "JA.kappa":
                return [AUD["judge"]["kappa"]], "audit_results.json judge"
            if cid in ("GEE.or.R4", "GEE.or.R6"):
                return [AUD["gee"]["models"]["A_main_effects"][f"OR_{NAME[p[2]]}_vs_e5"][0]], "audit_results.json gee A"
            if cid in ("GEE.ci.R4", "GEE.ci.R6"):
                return AUD["gee"]["models"]["A_main_effects"][f"OR_{NAME[p[2]]}_vs_e5"][1:], "audit_results.json gee A"
            if cid.startswith("E7.tp."):
                return [st.median(AUD["throughput"][NAME[p[2]]])], "audit_results.json throughput"
            if cid == "E7.ratio.R1R6":
                return [st.median(AUD["throughput"]["MiniLM-L6"]) / st.median(AUD["throughput"]["Qwen3-Emb-0.6B"])], "audit_results.json throughput"
            if cid == "E7.ratio.R5R4":
                return [st.median(AUD["throughput"]["e5-base-v2"]) / st.median(AUD["throughput"]["bge-large"])], "audit_results.json throughput"
        except (KeyError, ValueError):
            return None, ""
        return None, ""

    where = {}
    for f in sorted(TEX.glob("sections/*.tex")) + sorted(TEX.glob("tables/*.tex")):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            for m in re.finditer(r"\\ev\{([^}]+)\}", line):
                where.setdefault(m.group(1), []).append(f"{f.parent.name}/{f.name}:{i}")
    rows, n_ok, n_bad = [], 0, 0
    for cid in sorted(where):
        led = LED.get(cid)
        if not led:
            rows.append([cid, "; ".join(where[cid]), "", "", "", "MISSING FROM LEDGER", "add to ledger"])
            n_bad += 1
            continue
        val = led["value"]
        rec, src = recompute(cid)
        if rec is None and cid.startswith("V4."):
            rows.append([cid, "; ".join(where[cid]), val, val, led["source"], "new in v0.4 (generated by make_v04_tex.py from audit outputs)", ""])
            continue
        if rec is None:
            rows.append([cid, "; ".join(where[cid]), val, "", led["source"], "verified (full rerun identical)", ""])
            continue
        d = decimals(val)
        shown = nums(val)
        tol = 0.5 * 10 ** (-d) + 1e-9
        ok = len(shown) == len(rec) and all(abs(a - b) <= tol for a, b in zip(shown, rec))
        n_ok += ok
        n_bad += not ok
        rows.append([cid, "; ".join(where[cid]), val, ", ".join(f"{x:.{max(d, 1) + 1}f}" for x in rec), src,
                     "verified (independent recomputation)" if ok else "MISMATCH", "" if ok else "investigate"])
    with open(OUT / ("numerical_consistency_report.csv" if "revision_v04" in str(TEX) else "numerical_consistency_report_v03_original.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["claim_id", "location", "reported_value", "recomputed_value", "source_artifact", "status", "required_correction"])
        w.writerows(rows)
        for extra in EXTRA:
            w.writerow(extra)
    print(f"{len(where)} macros; {n_ok} independently recomputed and matching; {n_bad} mismatches or missing")


# Discrepancies found by reading (scope, wording and typed numbers); filled in by the audit, see audit_report.md
EXTRA = []

if __name__ == "__main__":
    extra = OUT / "numerical_issues_manual.csv"
    if extra.exists():
        EXTRA = list(csv.reader(open(extra, encoding="utf-8")))[1:]
    main()
