"""v0.6 figures: study phases with Phase 3 (replaces fig8_phases) and sub- vs full-corpus transfer ratios (fig13_fullcorpus_T).

    python3 manuscript_ipm/revision_v04/scripts/make_v06_figs.py

Uses the style helpers of paper_figures/make_figures_v03.py and writes PDF + PNG to paper_figures/out/ and the PDF to
manuscript_ipm/revision_v04/latex/figs/.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "paper_figures"))
import make_figures_v03 as M  # noqa: E402

FIGS = ROOT / "manuscript_ipm/revision_v04/latex/figs"
F = json.load(open(ROOT / "paper_outputs/v04/v04_full_analysis.json"))
S = json.load(open(ROOT / "paper_outputs/v03/statistics/v03_analysis.json"))


def phases():
    fig, ax = plt.subplots(figsize=(M.DOUBLE, 3.25))
    ax.set_xlim(-0.15, 12.45), ax.set_ylim(0, 5.95), ax.axis("off")
    p1 = [("Design v0.2\n(registered)", "#ffffff"), ("S1 retrieval\n7 methods, NQ\n+ cost", "#e9f1fb"),
          ("S3a reading\nQwen, NQ\n9 conditions", "#fdeee5"), ("S3b ablations\ntop-k, IDK,\nrerun", "#fdeee5"),
          ("S4 judge\n500 q per\ncondition", "#f2f2f2")]
    p2 = [("Addendum v0.3\nP1–P6\n(registered)", "#ffffff"), ("HotpotQA\nretrieval\n(S1-HP)", "#e9f1fb"),
          ("Readers on\nHotpotQA\n(S2-HP)", "#fdeee5"), ("Llama on NQ,\ntop-k (S3-NQ)", "#fdeee5"),
          ("Reranker, HNSW,\nfull corpus,\ncost, context\n(S4-NQ)", "#e9f1fb"), ("200 human\nlabels (E3)", "#f2f2f2")]
    p3 = [("Addendum v0.4\nP7–P8\n(registered)", "#ffffff"), ("Full NQ corpus\n3 encoders\n(S5-NQ)", "#e9f1fb"),
          ("Both readers,\nfull-corpus\nretrieval", "#fdeee5"), ("Answer strings\nin passages", "#f2f2f2")]
    w = 1.85
    for items, y, lab in ((p1, 4.2, "Phase 1"), (p2, 2.15, "Phase 2"), (p3, 0.1, "Phase 3")):
        ax.text(0.0, y + 1.5, lab, fontsize=7.5, weight="bold", color=M.INK)
        for i, (t, c) in enumerate(items):
            x = 0.0 + i * (w + 0.22)
            M._box(ax, x, y, w, 1.3, t, fc=c, fs=6.4)
            if i:
                M._arrow(ax, x - 0.22, y + 0.65, x, y + 0.65)
    ax.text(12.35, 0.7, "one analysis script per phase → all tables and figures", ha="right", fontsize=6.3, color=M.INK2,
            style="italic")
    M.save(fig, "fig8_phases")


def transfer_full():
    fig, ax = plt.subplots(figsize=(M.SINGLE, 2.05))
    labels, y = [], 0
    for rd, cell, col, lab in (("qwen", "qwen_nq", M.BLUE, "Qwen"), ("llama", "llama_nq", M.ORANGE, "Llama")):
        sub = {t["retriever"]: t for t in S["cells"][cell]["transfer_vs_R1"]["em"]}
        full = {t["retriever"]: t for t in F["readers"][rd]["transfer_full_vs_R1"]["em"]}
        for r, name in (("R2", "bge-small"), ("R5", "e5-base-v2")):
            for t, off, mk, fc in ((sub[r], 0.17, "o", "white"), (full[r], -0.17, "s", col)):
                ax.errorbar(t["ratio"], y + off, xerr=[[t["ratio"] - t["ci_lo"]], [t["ci_hi"] - t["ratio"]]], fmt=mk,
                            color=col, mfc=fc, ms=4, lw=0.9, capsize=2)
            labels.append((y, f"{name}, {lab}"))
            y -= 1
    ax.axvline(1, ls="--", lw=0.7, color=M.INK3)
    ax.axvline(0, lw=0.5, color=M.GRID)
    ax.set_yticks([p for p, _ in labels]), ax.set_yticklabels([l for _, l in labels])
    ax.set_xlim(-0.2, 1.08)
    ax.set_xlabel("$T$ vs MiniLM-L6 ($\\Delta$EM / $\\Delta$Hit@5)")
    ax.plot([], [], "o", color=M.INK2, mfc="white", ms=4, label="sub-corpus")
    ax.plot([], [], "s", color=M.INK2, ms=4, label="full corpus")
    ax.legend(loc="lower right", frameon=False, handletextpad=0.3)
    ax.spines["left"].set_visible(False), ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    M.save(fig, "fig13_fullcorpus_T")


if __name__ == "__main__":
    phases()
    transfer_full()
    for n in ("fig8_phases", "fig13_fullcorpus_T"):
        shutil.copy(M.OUT / f"{n}.pdf", FIGS / f"{n}.pdf")
