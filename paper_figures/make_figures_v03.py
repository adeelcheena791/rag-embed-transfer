"""Figures for the journal version (v0.3). Reads ONE file produced by the analysis:

    paper_outputs/v03/statistics/v03_analysis.json

and writes vector PDFs (for the manuscript) and 300-dpi PNGs (for checking) into paper_figures/out/.
Run from the repository root:

    python paper_figures/make_figures_v03.py

Needs only numpy and matplotlib (see paper_figures/requirements.txt). No number is typed by hand: every plotted
value comes from the JSON file, so the figures always match the tables and the text.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STATS = ROOT / "paper_outputs/v03/statistics/v03_analysis.json"
OUT = ROOT / "paper_figures/out"

# ---- style: one place, applied to every figure ------------------------------------------------------------
INK, INK2, INK3, GRID = "#1a1a1a", "#4d4d4d", "#8a8a8a", "#e6e6e6"
BLUE, ORANGE = "#2a78d6", "#eb6834"                                     # readers: Qwen, Llama (validated pair)
SEVEN = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]   # fixed order, validated
MARKERS = {"R0": "s", "R1": "o", "R2": "^", "R3": "v", "R4": "D", "R5": "P", "R6": "X"}
NAME = {"R0": "BM25", "R1": "MiniLM-L6", "R2": "bge-small", "R3": "bge-base", "R4": "bge-large", "R5": "e5-base-v2",
        "R6": "Qwen3-Emb-0.6B"}
SINGLE, DOUBLE = 3.45, 7.0                                               # column widths in inches

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8.5, "axes.titleweight": "regular", "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": INK3, "axes.linewidth": 0.6, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 150,
    "lines.linewidth": 1.4, "lines.markersize": 4.5, "errorbar.capsize": 0,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def ygrid(ax):
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


# ---- Figure 1: transfer ratios with 95% CIs, by dataset and reader ----------------------------------------
def fig_transfer(S):
    order = ["R0", "R2", "R3", "R4", "R5", "R6"]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.5), sharey=True)
    for ax, ds in zip(axes, ("NQ", "HotpotQA")):
        cells = [(c, v) for c, v in S["cells"].items() if v["dataset"] == ds]
        for j, (c, v) in enumerate(cells):
            col, off = (BLUE, -0.13) if v["reader"].startswith("Qwen") else (ORANGE, 0.13)
            tr = {t["retriever"]: t for t in v["transfer_vs_R1"]["em"]}
            for i, r in enumerate(order):
                t = tr[r]
                y = len(order) - 1 - i + off
                if t["estimable"]:
                    ax.plot([t["ci_lo"], t["ci_hi"]], [y, y], color=col, lw=1.2, solid_capstyle="round")
                    ax.plot(t["ratio"], y, "o", color=col, mec="white", mew=0.8, ms=5,
                            label=v["reader"] if i == 0 else None)
                else:
                    ax.text(0.02, y, "n.e.", color=col, fontsize=6.5, va="center", ha="left")
        ax.axvline(1.0, color=INK2, lw=0.8, ls=(0, (3, 2)))
        ax.axvline(0.0, color=INK3, lw=0.6)
        ax.set_xlim(-0.15, 1.2)
        ax.set_title(ds, loc="left")
        ax.set_xlabel(r"Transfer ratio $T(R)=\Delta\mathrm{EM}/\Delta\mathrm{Hit@5}$ vs MiniLM-L6")
        ax.xaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([NAME[r] for r in reversed(order)])
    axes[0].spines["left"].set_visible(False)
    axes[1].spines["left"].set_visible(False)
    for ax in axes:
        ax.text(1.0, 1.0, " T = 1", transform=ax.get_xaxis_transform(), color=INK2, fontsize=6.5, ha="left", va="top")
    axes[0].legend(loc="lower right", handletextpad=0.3)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig1_transfer_ratios")


# ---- Figure 2: retrieval quality vs answer quality, four cells --------------------------------------------
def fig_hit_vs_em(S):
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 4.6))
    layout = {("NQ", "Qwen"): (0, 0), ("NQ", "Llama"): (0, 1), ("HotpotQA", "Qwen"): (1, 0), ("HotpotQA", "Llama"): (1, 1)}
    for c, v in S["cells"].items():
        ax = axes[layout[(v["dataset"], v["reader"].split("2.5")[0].split("-")[0])]]
        col = BLUE if v["reader"].startswith("Qwen") else ORANGE
        x = [v["retrieval"][r]["hit@5"]["mean"] for r in NAME]
        y = [v["answers"][r]["em"]["mean"] for r in NAME]
        lo = [v["answers"][r]["em"]["mean"] - v["answers"][r]["em"]["ci_lo"] for r in NAME]
        hi = [v["answers"][r]["em"]["ci_hi"] - v["answers"][r]["em"]["mean"] for r in NAME]
        ax.errorbar(x, y, yerr=[lo, hi], fmt="none", ecolor=col, elinewidth=0.8, alpha=0.6)
        for r, xi, yi in zip(NAME, x, y):
            ax.plot(xi, yi, MARKERS[r], color=col, mec="white", mew=0.6, ms=5.5)
        for ref, lab in (("GOLD", "gold passages"), ("CB", "closed book")):
            e = v["answers"][ref]["em"]["mean"]
            ax.axhline(e, color=INK3, lw=0.7, ls=(0, (3, 2)))
            ax.text(0.995, e, f"{lab} {e:.3f}", transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                    fontsize=6.5, color=INK2)
        ax.set_title(f"{v['dataset']} · {v['reader']}", loc="left")
        ax.set_xlabel("Hit@5 (generation queries)")
        ax.set_ylabel("Exact match")
        ygrid(ax)
        ax.set_ylim(0.1, 0.72)
    handles = [mpl.lines.Line2D([], [], marker=MARKERS[r], ls="", color=INK2, mec="white", ms=5.5, label=NAME[r]) for r in NAME]
    fig.legend(handles=handles, loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.01), handletextpad=0.2, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=1.2, w_pad=1.5)
    save(fig, "fig2_hit_vs_em")


# ---- Figure 3: number of passages k (Qwen, NQ) ------------------------------------------------------------
def fig_topk(S):
    t = S["topk"]["em_by_k"]
    ks = [1, 3, 5, 10]
    fig, ax = plt.subplots(figsize=(DOUBLE * 0.62, 2.6))
    for r, col in zip(NAME, SEVEN):
        y = [t[r][str(k)]["mean"] for k in ks]
        ax.plot(ks, y, color=col, marker=MARKERS[r], mec="white", mew=0.6, ms=4.5, lw=1.2, label=NAME[r])
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=1, handlelength=1.6, handletextpad=0.4, labelspacing=0.35)
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.minorticks_off()
    ax.set_xlim(0.9, 11)
    ax.set_xlabel("Passages given to the reader (k)")
    ax.set_ylabel("Exact match (Qwen, NQ)")
    ygrid(ax)
    fig.tight_layout()
    save(fig, "fig3_topk")


# ---- Figure 4: mechanism tests (distraction and position) --------------------------------------------------
def fig_mechanism(S):
    m = S["mechanism"]["answers"]
    rows = [("MG", "gold only"), ("MR", "gold@1 + 4 random"), ("M1", "gold@1 + 4 hard"), ("M3", "gold@3 + 4 hard"),
            ("M5", "gold@5 + 4 hard")]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.0))
    for i, (c, lab) in enumerate(rows):
        y = len(rows) - 1 - i
        e = m[c]["em"]
        col = INK2 if c in ("MG", "MR") else BLUE
        ax.plot([e["ci_lo"], e["ci_hi"]], [y, y], color=col, lw=1.4, solid_capstyle="round")
        ax.plot(e["mean"], y, "o", color=col, mec="white", mew=0.8, ms=5.5)
        ax.text(e["ci_hi"] + 0.006, y, f"{e['mean']:.3f}", va="center", fontsize=6.5, color=INK2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([lab for _, lab in reversed(rows)])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Exact match, 500 NQ queries (95% CI)")
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    p5 = S["mechanism"]["P5"]["test"]
    ax.text(0.0, 1.03, f"hard vs random negatives: ΔEM = {p5['diff']:+.3f}, exact McNemar p = {p5['p']:.1e}",
            transform=ax.transAxes, fontsize=6.5, color=INK2)
    fig.tight_layout()
    save(fig, "fig4_mechanism")


# ---- Figure 5: cost vs answer quality (Pareto view) -------------------------------------------------------
def fig_cost(S):
    c = S["corpus_hnsw_cost"]["E7"]
    cells = S["cells"]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.7))
    pts = []
    for r in ["R1", "R2", "R3", "R4", "R5", "R6"]:
        x = c[r]["passages_per_s"]["median"]
        xl, xh = c[r]["passages_per_s"]["min"], c[r]["passages_per_s"]["max"]
        em = np.mean([cells[k]["answers"][r]["em"]["mean"] for k in cells])
        pts.append((x, em, r))
        ax.plot([xl, xh], [em, em], color=BLUE, lw=1.2)
        ax.plot(x, em, MARKERS[r], color=BLUE, mec="white", mew=0.6, ms=6)
        off, ha = {"R3": ((5, 5), "left"), "R5": ((-5, -9), "right"), "R1": ((-5, 4), "right"),
                   "R6": ((5, 4), "left"), "R4": ((6, 0), "left"), "R2": ((6, 2), "left")}[r]
        ax.annotate(NAME[r], (x, em), textcoords="offset points", xytext=off, ha=ha, fontsize=6.5, color=INK2)
    front, best = [], -1
    for x, em, r in sorted(pts, key=lambda p: -p[0]):          # fastest first: keep points that raise EM
        if em > best:
            front.append((x, em)); best = em
    fx, fy = zip(*sorted(front))
    ax.plot(fx, fy, color=INK3, lw=0.8, ls=(0, (3, 2)), zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel("Encoding throughput (passages/s, log)")
    ax.set_ylabel("Mean EM over the four cells")
    ygrid(ax)
    fig.tight_layout()
    save(fig, "fig5_cost_quality")


# ---- Figure 6: sub-corpus vs full corpus (NQ) -------------------------------------------------------------
def fig_full_corpus(S):
    e = S["corpus_hnsw_cost"]["E9"]
    rs = e["retrievers"]
    fig, ax = plt.subplots(figsize=(SINGLE, 1.9))
    for i, r in enumerate(rs):
        y = len(rs) - 1 - i
        a, b = e["subcorpus_same_embeddings"][r]["ndcg@10"], e["full_corpus"][r]["ndcg@10"]
        ax.plot([b, a], [y, y], color=INK3, lw=1.0, zorder=0)
        ax.plot(a, y, "o", color=BLUE, mec="white", mew=0.8, ms=6, label="300k sub-corpus" if i == 0 else None)
        ax.plot(b, y, "o", color=ORANGE, mec="white", mew=0.8, ms=6, label="full corpus (2.68M)" if i == 0 else None)
    ax.set_yticks(range(len(rs)))
    ax.set_yticklabels([NAME[r] for r in reversed(rs)])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("nDCG@10, all 3,452 judged NQ queries")
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", handletextpad=0.2)
    fig.tight_layout()
    save(fig, "fig6_full_corpus")


def main():
    S = json.load(open(STATS))
    fig_transfer(S)
    fig_hit_vs_em(S)
    fig_topk(S)
    fig_mechanism(S)
    fig_cost(S)
    fig_full_corpus(S)


if __name__ == "__main__":
    main()
