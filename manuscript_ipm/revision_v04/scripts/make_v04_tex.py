"""Generate the LaTeX macros and tables that the v0.4 revision adds, from audit_v04.py outputs only.

    python3 manuscript_ipm/revision_v04/scripts/make_v04_tex.py

Writes manuscript_ipm/revision_v04/latex/numbers_v04.tex, three table files in .../latex/tables/, and
manuscript_ipm/revision_v04/outputs/evidence_ledger_v04_additions.csv (one row per new macro, with its source).
"""
from __future__ import annotations

import csv
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "manuscript_ipm/revision_v04/outputs"
TEX = ROOT / "manuscript_ipm/revision_v04/latex"
AUD = json.load(open(OUT / "audit_results.json"))
TR = list(csv.DictReader(open(OUT / "transfer_ratio_audit.csv")))
AB = list(csv.DictReader(open(OUT / "transfer_ratio_alt_baselines.csv")))
DC = list(csv.DictReader(open(OUT / "transfer_decomposition_audit.csv")))
CELLS = ["qwen_nq", "llama_nq", "qwen_hp", "llama_hp"]
CELLNAME = {"qwen_nq": "NQ Qwen", "llama_nq": "NQ Llama", "qwen_hp": "HotpotQA Qwen", "llama_hp": "HotpotQA Llama"}
DENSE = ["bge-small", "bge-base", "bge-large", "e5-base-v2", "Qwen3-Emb-0.6B"]
MAC, LED = [], []


def neg(s):
    return s.replace("-", "\\ensuremath{-}", 1) if s.startswith("-") else s


def add(cid, value, desc, source):
    MAC.append(f"\\expandafter\\def\\csname ev@{cid}\\endcsname{{{value}}}")
    LED.append([cid, desc, value, source, "revision v0.4 audit (audit_v04.py)", "VERIFIED (recomputed from per-query files)"])


def MAC_VAL(cid):
    return next(l for l in MAC if "ev@" + cid + "\\endcsname" in l).split("\\endcsname{", 1)[1][:-1]


def f(x, d=2, sign=False):
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return neg(s)


def pv(p):
    if p < 0.001:
        import math
        e = math.floor(math.log10(p))
        return f"${p / 10 ** e:.1f}\\times10^{{{e}}}$"
    return f"{p:.2f}" if p >= 0.01 else f"{p:.3f}"


def main():
    (TEX / "tables").mkdir(parents=True, exist_ok=True)
    # ---------------- decomposition shares (HotpotQA, estimable ratios)
    hp = [r for r in DC if r["dataset"] == "HotpotQA" and r["ratio_estimable"] == "yes"]
    sh = [float(r["share_conditional"]) for r in hp]
    sym = [float(r["sym_share_conditional"]) for r in hp]
    hit = [float(r["hit_term_p1_da"]) / float(r["d_em"]) for r in hp]
    lo = [float(r["share_ci_lo"]) for r in hp]
    hi = [float(r["share_ci_hi"]) for r in hp]
    src = "transfer_decomposition_audit.csv"
    add("V4.sym.hp.min", f"{100 * min(sym):.0f}\\%", "smallest HotpotQA share under the symmetric decomposition", src)
    add("V4.sym.hp.max", f"{100 * max(sym):.0f}\\%", "largest HotpotQA share under the symmetric decomposition", src)
    add("V4.hit.hp.min", f"{100 * min(hit):.0f}\\%", "smallest HotpotQA share of the EM-given-hit term alone", src)
    add("V4.hit.hp.max", f"{100 * max(hit):.0f}\\%", "largest HotpotQA share of the EM-given-hit term alone", src)
    add("V4.shareci.hp.lo", f"{100 * min(lo):.0f}\\%", "lowest bootstrap lower bound of a HotpotQA share", src)
    add("V4.shareci.hp.hi", f"{100 * max(hi):.0f}\\%", "highest bootstrap upper bound of a HotpotQA share", src)
    nq = [r for r in DC if r["dataset"] == "NQ" and r["retriever"] in DENSE]
    add("V4.nq.share.width", f"{max(float(r['share_ci_hi']) - float(r['share_ci_lo']) for r in nq):.0f}",
        "widest bootstrap interval of an NQ share (in units of T)", src)
    add("V4.nq.share.minwidth", f"{min(float(r['share_ci_hi']) - float(r['share_ci_lo']) for r in nq):.1f}",
        "narrowest bootstrap interval of an NQ dense-encoder share (share scale, 1 = whole EM gain)", src)
    rows = []
    for name in DENSE:
        cells = []
        for c in ("qwen_hp", "llama_hp"):
            r = next(x for x in DC if x["cell"] == c and x["retriever"] == name)
            cells.append(f"{100 * float(r['share_conditional']):.0f} [{100 * float(r['share_ci_lo']):.0f}, "
                         f"{100 * float(r['share_ci_hi']):.0f}] & {100 * float(r['sym_share_conditional']):.0f}")
        rows.append(f"{name} & " + " & ".join(cells) + " \\\\")
    (TEX / "tables/tab_share.tex").write_text(
        "\\begin{table}[t]\n\\centering\n\\small\n"
        "\\caption{Share of the HotpotQA EM gain over MiniLM-L6 that comes from changed conditional EM (percent), under the "
        "baseline-conditioned decomposition of Eq.~\\ref{eq:decomp} with a 95\\% bootstrap interval, and under the symmetric "
        "decomposition. Post hoc.}\n\\label{tab:share}\n\\begin{tabular}{lcccc}\n\\toprule\n"
        " & \\multicolumn{2}{c}{Qwen reader} & \\multicolumn{2}{c}{Llama reader} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\nEncoder & Baseline-cond. & Symmetric & Baseline-cond. & Symmetric \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
        "\\par\\smallskip\\parbox{\\linewidth}{\\footnotesize Baseline-conditioned: $\\Delta\\mathrm{EM}=\\Delta h\\,(a_{R_1}-b_{R_1})+h_R"
        "\\,\\Delta a+(1-h_R)\\,\\Delta b$. Symmetric: each term averages the two orderings of the baseline and the encoder. "
        "Both identities hold exactly (residual $<10^{-12}$). The intervals resample queries ($B=10{,}000$, seed 0).}\n\\end{table}\n")

    # ---------------- alternative baseline (BM25)
    def ab(cell, name, base="BM25"):
        return next(x for x in AB if x["cell"] == cell and x["retriever"] == name and x["baseline"] == base)
    bnq = [float(ab(c, n)["ratio"]) for c in ("qwen_nq", "llama_nq") for n in DENSE]
    bhp = [float(ab(c, n)["ratio"]) for c in ("qwen_hp", "llama_hp") for n in DENSE]
    s2 = "transfer_ratio_alt_baselines.csv"
    add("V4.b0.nq.min", f(min(bnq)), "smallest NQ dense-encoder ratio against BM25", s2)
    add("V4.b0.nq.max", f(max(bnq)), "largest NQ dense-encoder ratio against BM25", s2)
    add("V4.b0.hp.min", f(min(bhp)), "smallest HotpotQA dense-encoder ratio against BM25", s2)
    add("V4.b0.hp.max", f(max(bhp)), "largest HotpotQA dense-encoder ratio against BM25", s2)
    for c in CELLS:
        add(f"V4.b0.{c}.R6", f(float(ab(c, "Qwen3-Emb-0.6B")["ratio"])), f"Qwen3-Emb ratio against BM25, {c}", s2)
    hp_gt = sum(float(ab(c.replace("nq", "hp"), n)["ratio"]) > float(ab(c, n)["ratio"]) for c in ("qwen_nq", "llama_nq") for n in DENSE)
    add("V4.b0.hpgt", f"{hp_gt}", "encoder-reader pairs (of 10) whose HotpotQA ratio exceeds the NQ ratio against BM25", s2)
    trnq = [float(x["recomputed_ratio"]) for x in TR if x["dataset"] == "NQ" and x["retriever"] in DENSE]
    add("V4.r1.nq.dense.n", f"{len(trnq)}", "number of NQ dense-encoder ratios against MiniLM-L6", "transfer_ratio_audit.csv")
    rows = []
    for name in DENSE:
        cells = []
        for c in CELLS:
            x = ab(c, name)
            cells.append(f"{f(float(x['ratio']))} [{f(float(x['ci_lo']))}, {f(float(x['ci_hi']))}]" if x["status"] == "estimable" else "n.e.")
        rows.append(f"{name} & " + " & ".join(cells) + " \\\\")
    BO = {(r["cell"], r["retriever"]): r for r in csv.DictReader(open(OUT / "transfer_ratio_both5_hotpotqa.csv"))}
    bo = [float(BO[(c, n)]["ratio_em_per_both5"]) for c in ("qwen_hp", "llama_hp") for n in DENSE]
    s3 = "transfer_ratio_both5_hotpotqa.csv"
    add("V4.both.hp.min", f(min(bo)), "smallest HotpotQA dense-encoder ratio with Both@5 denominator vs MiniLM-L6", s3)
    add("V4.both.hp.max", f(max(bo)), "largest HotpotQA dense-encoder ratio with Both@5 denominator vs MiniLM-L6", s3)
    add("V4.both.hp.R0", f(float(BO[("qwen_hp", "BM25")]["ratio_em_per_both5"]), 1) + " [" + f(float(BO[("qwen_hp", "BM25")]["ci_lo"]), 1) + ", "
        + f(float(BO[("qwen_hp", "BM25")]["ci_hi"]), 1) + "]", "BM25 ratio with Both@5 denominator, Qwen (unstable)", s3)
    rows_b = []
    for name in DENSE:
        cells = []
        for c in ("qwen_hp", "llama_hp"):
            x = BO[(c, name)]
            cells.append(f"{f(float(x['ratio_em_per_both5']))} [{f(float(x['ci_lo']))}, {f(float(x['ci_hi']))}]")
        rows_b.append(f"{name} & -- & -- & " + " & ".join(cells) + " \\\\")
    (TEX / "tables/tab_alt_baseline.tex").write_text(
        "\\begin{table*}[t]\n\\centering\n\\small\n"
        "\\caption{Sensitivity of the transfer ratio to the baseline and to the retrieval metric, with 95\\% bootstrap intervals. "
        "Upper block: $\\Delta\\mathrm{EM}/\\Delta\\mathrm{Hit@5}$ against BM25 instead of MiniLM-L6. Lower block: HotpotQA, "
        "$\\Delta\\mathrm{EM}/\\Delta\\mathrm{Both@5}$ against MiniLM-L6. Same queries, readers and reporting rule. Post hoc.}\n"
        "\\label{tab:altbase}\n\\begin{tabular}{lcccc}\n\\toprule\n"
        "Encoder & NQ Qwen & NQ Llama & HotpotQA Qwen & HotpotQA Llama \\\\\n\\midrule\n"
        "\\multicolumn{5}{l}{\\emph{Baseline BM25, denominator Hit@5}} \\\\\n" + "\n".join(rows) +
        "\n\\midrule\n\\multicolumn{5}{l}{\\emph{Baseline MiniLM-L6, denominator Both@5}} \\\\\n" + "\n".join(rows_b) +
        "\n\\bottomrule\n\\end{tabular}\n\\par\\smallskip\\parbox{\\linewidth}{\\footnotesize BM25 with Both@5 is omitted: its "
        "Both@5 gain over MiniLM-L6 is " + f(float(BO[("qwen_hp", "BM25")]["d_both5"]), 3) + " and its ratio is unstable (Qwen: " + MAC_VAL("V4.both.hp.R0") + ").}\n\\end{table*}\n")

    # ---------------- GEE with interactions
    g = AUD["gee"]["models"]
    A, B, C = g["A_main_effects"], g["B_retr_x_dataset"], g["C_retr_x_dataset_reader"]
    sg = "audit_results.json gee"
    add("V4.gee.int.chi", f"{B['wald_retr_x_dataset']['chi2']:.0f}", "Wald chi-square, retriever x dataset", sg)
    add("V4.gee.int.df", f"{B['wald_retr_x_dataset']['df']}", "df, retriever x dataset", sg)
    add("V4.gee.int.p", pv(B["wald_retr_x_dataset"]["p"]), "Wald p, retriever x dataset", sg)
    add("V4.gee.rxr.chi", f"{C['wald_retr_x_reader']['chi2']:.1f}", "Wald chi-square, retriever x reader", sg)
    add("V4.gee.rxr.p", pv(C["wald_retr_x_reader"]["p"]), "Wald p, retriever x reader", sg)
    for r, nm in (("R4", "bge-large"), ("R6", "Qwen3-Emb-0.6B")):
        for ds, key in (("hp", "HotpotQA"), ("nq", "NQ")):
            o = B[f"OR_{nm}_vs_e5_{key}"]
            add(f"V4.or.{r}.{ds}", f"{o[0]:.3f}", f"odds ratio vs e5-base-v2 on {key}, model B", sg)
            add(f"V4.or.{r}.{ds}.ci", f"[{o[1]:.3f}, {o[2]:.3f}]", f"95% CI, odds ratio vs e5-base-v2 on {key}", sg)
    rows = []
    for r, nm in (("R4", "bge-large"), ("R6", "Qwen3-Emb-0.6B")):
        a = A[f"OR_{nm}_vs_e5"]
        h = B[f"OR_{nm}_vs_e5_HotpotQA"]
        n_ = B[f"OR_{nm}_vs_e5_NQ"]
        rows.append(f"{nm} vs e5-base-v2 & {a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}] & {n_[0]:.3f} [{n_[1]:.3f}, {n_[2]:.3f}] & "
                    f"{h[0]:.3f} [{h[1]:.3f}, {h[2]:.3f}] \\\\")
    (TEX / "tables/tab_gee.tex").write_text(
        "\\begin{table*}[t]\n\\centering\n\\small\n"
        "\\caption{GEE logistic models of EM over the four reader--dataset cells: odds ratios against e5-base-v2 [95\\% CI].}\n"
        "\\label{tab:gee}\n\\begin{tabular}{lccc}\n\\toprule\n"
        " & Pooled (main effects) & \\multicolumn{2}{c}{With retriever $\\times$ dataset interaction} \\\\\n"
        "\\cmidrule(lr){3-4}\nContrast & both datasets & NQ & HotpotQA \\\\\n\\midrule\n" + "\n".join(rows) +
        "\n\\bottomrule\n\\end{tabular}\n"
        f"\\par\\smallskip\\parbox{{\\linewidth}}{{\\footnotesize Pooled: EM $\\sim$ retriever + dataset + reader (registered secondary model). "
        f"Interaction model: EM $\\sim$ retriever $\\times$ dataset + reader (post hoc). Retriever $\\times$ dataset: Wald "
        f"$\\chi^2_{{{B['wald_retr_x_dataset']['df']}}}$ = {B['wald_retr_x_dataset']['chi2']:.0f}, $p$ = {pv(B['wald_retr_x_dataset']['p'])}; "
        f"adding retriever $\\times$ reader: $\\chi^2_{{{C['wald_retr_x_reader']['df']}}}$ = {C['wald_retr_x_reader']['chi2']:.1f}, "
        f"$p$ = {pv(C['wald_retr_x_reader']['p'])}. Exchangeable working correlation ({A['working_corr']:.2f}) within query; "
        f"{AUD['gee']['n_rows']:,} answers in {AUD['gee']['n_clusters']:,} query clusters; empirical (sandwich) standard errors. "
        "All models converged without warnings.}\n\\end{table*}\n")

    # ---------------- throughput
    tp = AUD["throughput"]
    add("V4.tp.R3R5", f"{100 * (st.median(tp['bge-base']) / st.median(tp['e5-base-v2']) - 1):.1f}\\%",
        "bge-base median throughput above e5-base-v2", "audit_results.json throughput")
    # ---------------- random passages
    rp = AUD["random_passages"]["MR_minus_MG"]
    add("V4.mrmg.diff", f(rp["diff"], 3, sign=True), "EM of gold@1 + 4 random minus gold only", "audit_results.json random_passages")
    add("V4.mrmg.ci", f"[{f(rp['ci_lo'], 3)}, {f(rp['ci_hi'], 3)}]", "95% CI of that difference", "audit_results.json random_passages")
    add("V4.mrmg.p", f"{rp['p']:.2f}", "exact McNemar p of that difference", "audit_results.json random_passages")

    (TEX / "numbers_v04.tex").write_text("% Generated by revision_v04/scripts/make_v04_tex.py from audit outputs. Do not edit.\n" + "\n".join(MAC) + "\n")
    with open(OUT / "evidence_ledger_v04_additions.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["claim_id", "claim", "value", "source", "produced_by", "status"])
        w.writerows(LED)
    print(f"{len(MAC)} macros; tables tab_share, tab_alt_baseline, tab_gee")


if __name__ == "__main__":
    main()
