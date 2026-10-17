"""Post hoc, descriptive: query-level retrieval/answer transitions of each dense encoder against MiniLM-L6.

For every cell and dense encoder R, each generation query falls into one stratum by its Hit@5 under MiniLM-L6 and under R:
miss->hit, hit->hit, hit->miss, miss->miss. For each stratum: n, share, EM under MiniLM-L6 and R, paired EM difference
within the stratum, counts of wrong->right and right->wrong answers, and the stratum's contribution to the overall EM
difference, (sum over the stratum of EM_R - EM_R1) / N. The four contributions add up to the overall EM difference exactly.

HotpotQA: the same accounting by the number of supporting passages (0, 1, 2) in the top five under each retriever, grouped
as gained evidence (count up), same count, lost evidence (count down).

Uncertainty: the percentile bootstrap resamples queries with the analysis' index (B = 10,000, seed 0). Each query keeps its
own stratum, so the stratum sizes vary across resamples; the interval is for the contribution of that stratum to the
population EM difference. These are outcome-defined groups: they describe where the EM difference sits, not why.

    python3 manuscript_ipm/revision_v04/scripts/transitions_v05.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.analysis import stats  # noqa: E402
from src.analysis.v03_analysis import CELLS, Data  # noqa: E402

OUT = ROOT / "manuscript_ipm/revision_v04/outputs"
NAME = {"R2": "bge-small", "R3": "bge-base", "R4": "bge-large", "R5": "e5-base-v2", "R6": "Qwen3-Emb-0.6B"}
STRATA = [("miss_to_hit", 0, 1), ("hit_to_hit", 1, 1), ("hit_to_miss", 1, 0), ("miss_to_miss", 0, 0)]


def boot_sum(x, idx):
    return np.concatenate([x[idx[i:i + 1000]].sum(axis=1) for i in range(0, len(idx), 1000)])


def main():
    D = Data(ROOT)
    rows, hp_rows, summary = [], [], {}
    for cell in CELLS:
        q = D.qids(cell)
        n = len(q)
        idx = stats.resample_index(n)
        h1 = D.retrieval(cell, "R1")["hit@5"].astype(int)
        e1 = D.answers(cell, "R1_k5_A1")["em"].astype(int)
        hp = CELLS[cell]["dataset"] == "HotpotQA"
        if hp:
            gold = {x: set(D.hp_joined[x]["gold_pids"]) for x in q}
            g1 = {x["qid"]: len(gold[x["qid"]] & set(x["pids"][:5])) for x in
                  [json.loads(l) for l in open(ROOT / CELLS[cell]["gens"] / f"{CELLS[cell]['prefix']}__R1_k5_A1__main.jsonl")]}
            c1 = np.array([g1[x] for x in q])
        for r, name in NAME.items():
            hr = D.retrieval(cell, r)["hit@5"].astype(int)
            er = D.answers(cell, f"{r}_k5_A1")["em"].astype(int)
            d = (er - e1).astype(float)
            total = d.sum() / n
            rec = 0.0
            for s, a, b in STRATA:
                m = (h1 == a) & (hr == b)
                contrib = d[m].sum() / n
                rec += contrib
                bc = boot_sum(d * m, idx) / n
                rows.append([cell, CELLS[cell]["dataset"], CELLS[cell]["reader"], name, s, int(m.sum()), f"{m.mean():.4f}",
                             f"{e1[m].mean():.4f}" if m.any() else "", f"{er[m].mean():.4f}" if m.any() else "",
                             f"{d[m].mean():+.4f}" if m.any() else "", int(((e1 == 0) & (er == 1) & m).sum()),
                             int(((e1 == 1) & (er == 0) & m).sum()), f"{contrib:+.4f}",
                             f"{np.percentile(bc, 2.5):+.4f}", f"{np.percentile(bc, 97.5):+.4f}",
                             f"{contrib / total:.3f}" if total else "", f"{total:+.4f}"])
            assert abs(rec - total) < 1e-12, (cell, r, rec, total)
            summary.setdefault(cell, {})[name] = {s: float(d[(h1 == a) & (hr == b)].sum() / n) for s, a, b in STRATA}
            summary[cell][name]["total"] = float(total)
            if hp:
                gr = {x["qid"]: len(gold[x["qid"]] & set(x["pids"][:5])) for x in
                      [json.loads(l) for l in open(ROOT / CELLS[cell]["gens"] / f"{CELLS[cell]['prefix']}__{r}_k5_A1__main.jsonl")]}
                cr = np.array([gr[x] for x in q])
                groups = {"evidence_gained": cr > c1, "evidence_same": cr == c1, "evidence_lost": cr < c1}
                rec2 = 0.0
                for gname, m in groups.items():
                    contrib = d[m].sum() / n
                    rec2 += contrib
                    bc = boot_sum(d * m, idx) / n
                    hp_rows.append([cell, CELLS[cell]["reader"], name, gname, int(m.sum()), f"{m.mean():.4f}",
                                    f"{e1[m].mean():.4f}", f"{er[m].mean():.4f}", int(((e1 == 0) & (er == 1) & m).sum()),
                                    int(((e1 == 1) & (er == 0) & m).sum()), f"{contrib:+.4f}", f"{np.percentile(bc, 2.5):+.4f}",
                                    f"{np.percentile(bc, 97.5):+.4f}", f"{contrib / total:.3f}"])
                for a in range(3):
                    for b in range(3):
                        m = (c1 == a) & (cr == b)
                        hp_rows.append([cell, CELLS[cell]["reader"], name, f"count_{a}_to_{b}", int(m.sum()), f"{m.mean():.4f}",
                                        f"{e1[m].mean():.4f}" if m.any() else "", f"{er[m].mean():.4f}" if m.any() else "",
                                        int(((e1 == 0) & (er == 1) & m).sum()), int(((e1 == 1) & (er == 0) & m).sum()),
                                        f"{d[m].sum() / n:+.4f}", "", "", f"{d[m].sum() / n / total:.3f}"])
                assert abs(rec2 - total) < 1e-12
                summary[cell][name].update({g: float(d[m].sum() / n) for g, m in groups.items()})
    hdr = ["cell", "dataset", "reader", "encoder", "stratum", "n", "share", "em_minilm", "em_encoder", "em_diff_within",
           "wrong_to_right", "right_to_wrong", "contribution_to_dEM", "contrib_ci_lo", "contrib_ci_hi", "share_of_dEM", "dEM_total"]
    with open(OUT / "transitions_hit5.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(hdr); w.writerows(rows)
    with open(OUT / "transitions_hotpotqa_evidence.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "reader", "encoder", "group", "n", "share", "em_minilm", "em_encoder", "wrong_to_right", "right_to_wrong",
                    "contribution_to_dEM", "contrib_ci_lo", "contrib_ci_hi", "share_of_dEM"])
        w.writerows(hp_rows)
    (OUT / "transitions_summary.json").write_text(json.dumps(summary, indent=1))
    for cell, v in summary.items():
        for name, s in v.items():
            print(cell, f"{name:15s}", " ".join(f"{k}={x:+.4f}" for k, x in s.items()))


if __name__ == "__main__":
    main()
