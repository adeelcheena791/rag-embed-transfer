"""Post hoc, descriptive: the three terms of the baseline-conditioned decomposition of the EM difference (vs MiniLM-L6),
dEM = dp (a0 - b0) [hit prevalence] + p1 da [EM among hits] + (1 - p1) db [EM among misses], each with a percentile
bootstrap interval over queries (B = 10,000, seed 0; terms recomputed in every resample).

    python3 manuscript_ipm/revision_v04/scripts/components_v05.py
"""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.analysis import stats  # noqa: E402
from src.analysis.v03_analysis import CELLS, Data  # noqa: E402

OUT = ROOT / "manuscript_ipm/revision_v04/outputs"
NAME = {"R2": "bge-small", "R3": "bge-base", "R4": "bge-large", "R5": "e5-base-v2", "R6": "Qwen3-Emb-0.6B"}


def bm(x, idx):
    return np.concatenate([x[idx[i:i + 1000]].mean(axis=1) for i in range(0, len(idx), 1000)])


def terms(p0, a0, b0, p1, a1, b1):
    return (p1 - p0) * (a0 - b0), p1 * (a1 - a0), (1 - p1) * (b1 - b0)


def main():
    D = Data(ROOT)
    rows = []
    for cell in CELLS:
        q = D.qids(cell)
        idx = stats.resample_index(len(q))
        h0 = D.retrieval(cell, "R1")["hit@5"]; e0 = D.answers(cell, "R1_k5_A1")["em"]
        B0 = (bm(h0, idx), bm(h0 * e0, idx), bm(e0, idx))
        P0 = (h0.mean(), (h0 * e0).sum() / h0.sum(), ((1 - h0) * e0).sum() / (1 - h0).sum())
        bP0 = (B0[0], B0[1] / B0[0], (B0[2] - B0[1]) / (1 - B0[0]))
        for r, name in NAME.items():
            h1 = D.retrieval(cell, r)["hit@5"]; e1 = D.answers(cell, f"{r}_k5_A1")["em"]
            P1 = (h1.mean(), (h1 * e1).sum() / h1.sum(), ((1 - h1) * e1).sum() / (1 - h1).sum())
            B1 = (bm(h1, idx), bm(h1 * e1, idx), bm(e1, idx))
            bP1 = (B1[0], B1[1] / B1[0], (B1[2] - B1[1]) / (1 - B1[0]))
            pt = terms(*P0, *P1)
            bt = terms(*bP0, *bP1)
            dem = e1.mean() - e0.mean()
            assert abs(sum(pt) - dem) < 1e-12
            row = [cell, CELLS[cell]["dataset"], CELLS[cell]["reader"], name, f"{dem:+.4f}"]
            for v, b in zip(pt, bt):
                row += [f"{v:+.4f}", f"{np.percentile(b, 2.5):+.4f}", f"{np.percentile(b, 97.5):+.4f}"]
            rows.append(row)
    with open(OUT / "decomposition_components.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "dataset", "reader", "encoder", "dEM", "prevalence", "prev_lo", "prev_hi", "hits", "hits_lo", "hits_hi",
                    "misses", "miss_lo", "miss_hi"])
        w.writerows(rows)
    for r in rows:
        print(*r)


if __name__ == "__main__":
    main()
