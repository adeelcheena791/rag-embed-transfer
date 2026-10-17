"""Post hoc, exploratory: HotpotQA transfer ratio with Both@5 instead of Hit@5 in the denominator (vs MiniLM-L6).
Writes manuscript_ipm/revision_v04/outputs/transfer_ratio_both5_hotpotqa.csv. Not used in the manuscript."""
import csv, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.analysis import stats
from src.analysis.v03_analysis import Data, R
NAME = {"R0": "BM25", "R1": "MiniLM-L6", "R2": "bge-small", "R3": "bge-base", "R4": "bge-large", "R5": "e5-base-v2", "R6": "Qwen3-Emb-0.6B"}
D = Data(ROOT)
rows = []
for cell in ("qwen_hp", "llama_hp"):
    q = D.qids(cell); idx = stats.resample_index(len(q))
    b = {r: D.retrieval(cell, r)["both@5"] for r in R}; e = {r: D.answers(cell, f"{r}_k5_A1")["em"] for r in R}
    bb = {r: np.concatenate([b[r][idx[i:i+1000]].mean(1) for i in range(0, len(idx), 1000)]) for r in R}
    be = {r: np.concatenate([e[r][idx[i:i+1000]].mean(1) for i in range(0, len(idx), 1000)]) for r in R}
    pw = {frozenset((t["a"], t["b"])): t for t in stats.pairwise(b, R, idx, binary=True)}
    for r in R:
        if r == "R1": continue
        ph = pw[frozenset(("R1", r))]["p_holm"]
        db, de = b[r].mean() - b["R1"].mean(), e[r].mean() - e["R1"].mean()
        ok = abs(db) > 0.02 and ph < 0.05
        with np.errstate(divide="ignore", invalid="ignore"):
            bt = (be[r] - be["R1"]) / (bb[r] - bb["R1"])
        lo, hi = np.percentile(bt[np.isfinite(bt)], [2.5, 97.5])
        rows.append([cell, NAME[r], f"{db:.4f}", f"{de:.4f}", f"{ph:.3g}", "estimable" if ok else "not estimable",
                     f"{de/db:.3f}" if ok else "", f"{lo:.3f}" if ok else "", f"{hi:.3f}" if ok else ""])
w = csv.writer(open(ROOT / "manuscript_ipm/revision_v04/outputs/transfer_ratio_both5_hotpotqa.csv", "w", newline=""))
w.writerow(["cell", "retriever", "d_both5", "d_em", "both5_p_holm", "status", "ratio_em_per_both5", "ci_lo", "ci_hi"]); w.writerows(rows)
for x in rows: print(*x)
