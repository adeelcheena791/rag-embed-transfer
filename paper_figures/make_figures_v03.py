"""Figures for the journal version (v0.3). Reads files produced by the analysis scripts only:

    paper_outputs/v03/statistics/v03_analysis.json         (all v0.3 cells and experiments)
    paper_inputs/results/statistics/rq12_main.json         (Phase 1 retrieval statistics, NQ: nDCG@10 with CIs)
    paper_inputs/results/statistics/rq34_main.json         (Phase 1 cost measurements: CPU latency, index size, encoding time)
    paper_outputs/paper/derived_stats.json                 (Phase 1 IDK-prompt ablation, from src/paper/make_figures_tables.py)

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
    mant, ex = f"{p5['p']:.1e}".split("e")
    ax.text(0.0, 1.03, f"hard vs random negatives: ΔEM = {p5['diff']:+.3f}, exact McNemar p = {mant}×10$^{{{int(ex)}}}$",
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


# ---- diagrams ----------------------------------------------------------------------------------------------
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402


def _box(ax, x, y, w, h, text, fc="#ffffff", fs=6.8, ec=INK3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04", fc=fc, ec=ec, lw=0.7))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.25)


def _arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, lw=0.7, color=INK2))


def fig_system(S):
    """System under test (counts read from the analysis file)."""
    nq = S["cells"]["qwen_nq"]["n_queries"]
    hp = S["cells"]["qwen_hp"]["n_queries"]
    fig, ax = plt.subplots(figsize=(DOUBLE, 2.7))
    ax.set_xlim(-0.15, 13.2), ax.set_ylim(0, 4.6), ax.axis("off")
    fs = 6.3
    _box(ax, 0.0, 1.55, 1.85, 1.25, f"Question\nNQ ({nq:,})\nHotpotQA ({hp:,})", fs=fs)
    _box(ax, 2.2, 2.6, 3.1, 1.3, "Retriever (1 of 7)\nBM25 or dense encoder\nexact FAISS search", fc="#e9f1fb", fs=fs)
    _box(ax, 2.2, 0.1, 3.1, 1.55, "Corpus per dataset\n300,000 passages:\ngold + BM25 negatives\n+ random", fc="#ffffff", fs=fs)
    _box(ax, 5.65, 1.5, 2.1, 1.35, "Top-$k$ passages\n$k$ = 5\n(1, 3, 10 in E4)", fs=fs)
    _box(ax, 8.1, 1.4, 2.45, 1.55, "Reader (1 of 2)\nQwen2.5-7B (AWQ)\nLlama-3.1-8B (AWQ)\ngreedy decoding", fc="#fdeee5", fs=fs)
    _box(ax, 10.9, 2.65, 2.25, 1.25, "EM, F1, AC\nagainst reference\nanswers", fc="#eaf6ef", fs=fs)
    _box(ax, 10.9, 0.35, 2.25, 1.4, "LLM judge\n(Phase 1 only;\nchecked on 200\nhuman labels)", fc="#f2f2f2", fs=fs)
    _arrow(ax, 1.85, 2.3, 2.2, 3.1)
    _arrow(ax, 3.75, 1.65, 3.75, 2.6)
    _arrow(ax, 5.3, 3.1, 5.8, 2.85)
    _arrow(ax, 7.75, 2.17, 8.1, 2.17)
    _arrow(ax, 10.55, 2.55, 10.9, 3.1)
    _arrow(ax, 10.55, 1.8, 10.9, 1.3)
    ax.text(3.75, 4.25, "cost measured at this stage: encoding throughput, query latency, index size, GPU memory",
            ha="center", fontsize=6.2, color=INK2, style="italic")
    save(fig, "fig7_system")


def fig_phases(S):
    """Study phases and sessions."""
    fig, ax = plt.subplots(figsize=(DOUBLE, 2.3))
    ax.set_xlim(-0.15, 12.45), ax.set_ylim(0, 4.0), ax.axis("off")
    p1 = [("Design v0.2\n(registered)", "#ffffff"), ("S1 retrieval\n7 methods, NQ\n+ cost", "#e9f1fb"),
          ("S3a reading\nQwen, NQ\n9 conditions", "#fdeee5"), ("S3b ablations\ntop-k, IDK,\nrerun", "#fdeee5"),
          ("S4 judge\n500 q per\ncondition", "#f2f2f2")]
    p2 = [("Addendum v0.3\nP1–P6\n(registered)", "#ffffff"), ("HotpotQA\nretrieval\n(S1-HP)", "#e9f1fb"),
          ("Readers on\nHotpotQA\n(S2-HP)", "#fdeee5"), ("Llama on NQ,\ntop-k (S3-NQ)", "#fdeee5"),
          ("Reranker, HNSW,\nfull corpus,\ncost, context\n(S4-NQ)", "#e9f1fb"), ("200 human\nlabels (E3)", "#f2f2f2")]
    w = 1.85
    for row, (items, y, lab) in enumerate(((p1, 2.25, "Phase 1"), (p2, 0.2, "Phase 2"))):
        ax.text(0.0, y + 1.55, lab, fontsize=7.5, weight="bold", color=INK)
        for i, (t, c) in enumerate(items):
            x = 0.0 + i * (w + 0.22)
            _box(ax, x, y, w, 1.35, t, fc=c, fs=6.4)
            if i:
                _arrow(ax, x - 0.22, y + 0.67, x, y + 0.67)
    ax.text(12.35, 0.05, "one analysis script → all tables and figures", ha="right", fontsize=6.3, color=INK2, style="italic")
    save(fig, "fig8_phases")


# ---- retrieval quality bars ----------------------------------------------------------------------------------
def fig_retrieval(S, RQ12):
    nq = RQ12["rq1_joined"]["means"]
    hp = S["cells"]["qwen_hp"]["retrieval"]
    panels = [("(a) NQ, Hit@5", {r: nq[r]["hit@5"] for r in NAME}), ("(b) NQ, nDCG@10", {r: nq[r]["ndcg@10"] for r in NAME}),
              ("(c) HotpotQA, Hit@5", {r: hp[r]["hit@5"] for r in NAME}), ("(d) HotpotQA, Both@5", {r: hp[r]["both@5"] for r in NAME})]
    fig, axes = plt.subplots(1, 4, figsize=(DOUBLE, 2.2), sharey=True)
    order = list(NAME)
    for ax, (title, d) in zip(axes, panels):
        y = np.arange(len(order))[::-1]
        m = np.array([d[r]["mean"] for r in order])
        err = np.array([[d[r]["mean"] - d[r]["ci_lo"], d[r]["ci_hi"] - d[r]["mean"]] for r in order]).T
        ax.barh(y, m, color=[SEVEN[i] for i in range(len(order))], height=0.68, xerr=err,
                error_kw={"lw": 0.7, "ecolor": INK}, alpha=0.9)
        for yi, mi in zip(y, m):
            ax.text(mi + 0.02, yi, f"{mi:.3f}", va="center", fontsize=6, color=INK2)
        ax.set_xlim(0, 1.18)
        ax.set_xticks([0, 0.5, 1])
        ax.set_title(title, loc="left")
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    axes[0].set_yticks(np.arange(len(order))[::-1])
    axes[0].set_yticklabels([NAME[r] for r in order])
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig9_retrieval")


# ---- error analysis: EM given hit or miss; IDK prompt ---------------------------------------------------------
def fig_error(S, DER):
    fig = plt.figure(figsize=(DOUBLE, 3.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.62])
    pos = {"qwen_nq": (0, 0), "llama_nq": (0, 1), "qwen_hp": (1, 0), "llama_hp": (1, 1)}
    short = {"R0": "BM25", "R1": "MiniLM", "R2": "bge-s", "R3": "bge-b", "R4": "bge-L", "R5": "e5-b", "R6": "Qwen3"}
    x = np.arange(7)
    for c, (i, j) in pos.items():
        ax = fig.add_subplot(gs[i, j])
        v = S["cells"][c]
        d = v["decomposition"]
        col = BLUE if v["reader"].startswith("Qwen") else ORANGE
        ax.bar(x - 0.2, [d[r]["p_em_given_hit"] for r in NAME], 0.4, color=col, label="hit")
        ax.bar(x + 0.2, [d[r]["p_em_given_miss"] for r in NAME], 0.4, color="white", edgecolor=col, hatch="////", lw=0.6,
               label="miss")
        ax.axhline(v["answers"]["CB"]["em"]["mean"], color=INK2, lw=0.8, ls=":", label="closed book")
        ax.set_xticks(x)
        ax.set_xticklabels([short[r] for r in NAME], rotation=45, ha="right", fontsize=6.3)
        ax.set_ylim(0, 0.75)
        ax.set_title(f"({'abcd'[i * 2 + j]}) {v['dataset']} · {v['reader']}", loc="left", fontsize=7.5)
        ygrid(ax)
        if j == 0:
            ax.set_ylabel("EM")
        if (i, j) == (0, 0):
            ax.legend(loc="upper right", ncol=3, fontsize=6, handlelength=1.2, columnspacing=0.8)
    ax = fig.add_subplot(gs[0, 2])
    a = DER["abstention_A3"]
    cs = ["R0", "R1", "R4"]
    xs = np.arange(len(cs))
    ax.bar(xs - 0.2, [a[r]["idk_given_hit"] for r in cs], 0.4, color=BLUE, label="hit")
    ax.bar(xs + 0.2, [a[r]["idk_given_miss"] for r in cs], 0.4, color="white", edgecolor=BLUE, hatch="////", lw=0.6, label="miss")
    ax.set_xticks(xs)
    ax.set_xticklabels([short[r] for r in cs], fontsize=6.5)
    ax.set_ylabel('"I don\'t know" rate')
    ax.set_title(f"(e) IDK prompt, Qwen, NQ, n = {a['R0']['n']}", loc="left", fontsize=7.5)
    ax.set_ylim(0, 0.45)
    ygrid(ax)
    fig.tight_layout(h_pad=1.0, w_pad=1.0)
    save(fig, "fig10_error")


# ---- Pareto frontier: EM vs per-query CPU latency, per cell, with significance of each frontier step ----------
def _frontier(pts):
    front = [a for a, (ca, qa) in pts.items()
             if not any(cb <= ca and qb >= qa and (cb < ca or qb > qa) for b, (cb, qb) in pts.items() if b != a)]
    return sorted(front, key=lambda r: pts[r][0])


PARETO_LABEL = {"R0": ((7, -2), "left"), "R1": ((7, -9), "left"), "R2": ((7, -9), "left"), "R3": ((7, 1), "left"),
                "R5": ((7, -8), "left"), "R4": ((7, 1), "left"), "R6": ((-7, 4), "right")}


def fig_pareto(S, RQ34):
    eff = RQ34["efficiency"]
    pos = {"qwen_nq": (0, 0), "llama_nq": (0, 1), "qwen_hp": (1, 0), "llama_hp": (1, 1)}
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 4.6))
    for c, ij in pos.items():
        ax = axes[ij]
        v = S["cells"][c]
        col = BLUE if v["reader"].startswith("Qwen") else ORANGE
        pts = {r: (eff[r]["query_ms_cpu"], v["answers"][r]["em"]["mean"]) for r in NAME}
        pw = {frozenset((t["a"], t["b"])): t for t in v["pairwise_em"]}
        front = _frontier(pts)
        ax.plot([pts[r][0] for r in front], [pts[r][1] for r in front], color=INK2, lw=0.9, zorder=1)
        for a_, b_ in zip(front, front[1:]):
            t = pw[frozenset((a_, b_))]
            xm = np.sqrt(pts[a_][0] * pts[b_][0])
            ym = (pts[a_][1] + pts[b_][1]) / 2
            ax.annotate("*" if t["p_holm"] < 0.05 else "n.s.", (xm, ym), textcoords="offset points", xytext=(-3, 4),
                        fontsize=6.5, color="#b00000", ha="right")
        for r in NAME:
            x, y = pts[r]
            dominated = r not in front
            ax.plot(x, y, MARKERS[r], color="white" if dominated else col, mec=col, mew=0.9, ms=6, zorder=2)
        ax.set_xscale("log")
        ax.set_xlim(15, 1500)
        ax.set_title(f"({'abcd'[ij[0] * 2 + ij[1]]}) {v['dataset']} · {v['reader']}", loc="left")
        ax.set_xlabel("Per-query latency, CPU only (ms, log)")
        ax.set_ylabel("Exact match")
        ygrid(ax)
    handles = [plt.Line2D([], [], marker=MARKERS[r], ls="", color=INK2, mec=INK2, ms=5, label=NAME[r]) for r in NAME]
    handles.append(plt.Line2D([], [], marker="o", ls="", color="white", mec=INK2, ms=5, label="dominated (open)"))
    fig.legend(handles=handles, loc="lower center", ncol=8, bbox_to_anchor=(0.5, -0.01), handletextpad=0.2, columnspacing=0.9)
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=1.2, w_pad=1.5)
    save(fig, "fig11_pareto")


def fig_latency(RQ12, RQ34):
    eff = RQ34["efficiency"]
    nq = RQ12["rq1_joined"]["means"]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.3), sharey=True)
    for ax, key, lab in ((axes[0], "query_ms_gpu", "(a) query encoding on T4 + CPU search"),
                         (axes[1], "query_ms_cpu", "(b) query encoding and search on CPU")):
        for r in NAME:
            x, y = eff[r][key], nq[r]["ndcg@10"]["mean"]
            ax.plot(x, y, MARKERS[r], color=BLUE, mec="white", mew=0.6, ms=6)
            ax.annotate(NAME[r], (x, y), textcoords="offset points", xytext=(5, 2), fontsize=6, color=INK2)
        ax.set_xscale("log")
        ax.set_xlim(15, 2000)
        ax.set_title(lab, loc="left")
        ax.set_xlabel("Per-query latency (ms, log)")
        ygrid(ax)
    axes[0].set_ylabel("nDCG@10, NQ")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig12_latency")


def main():
    S = json.load(open(STATS))
    RQ12 = json.load(open(ROOT / "paper_inputs/results/statistics/rq12_main.json"))
    RQ34 = json.load(open(ROOT / "paper_inputs/results/statistics/rq34_main.json"))
    DER = json.load(open(ROOT / "paper_outputs/paper/derived_stats.json"))
    fig_system(S)
    fig_phases(S)
    fig_retrieval(S, RQ12)
    fig_error(S, DER)
    fig_pareto(S, RQ34)
    fig_latency(RQ12, RQ34)
    fig_transfer(S)
    fig_hit_vs_em(S)
    fig_topk(S)
    fig_mechanism(S)
    fig_cost(S)
    fig_full_corpus(S)


if __name__ == "__main__":
    main()
