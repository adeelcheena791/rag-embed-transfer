"""Independent check, part 2 (v0.5): quantities that manuscript_ipm/tools/independent_check.py does not cover.

No project imports. Reads raw files directly and recomputes:
  A. Hit@5 (NQ, HotpotQA) and Both@5 (HotpotQA) per query from passage ids and gold labels;
  B. the 21 pairwise Hit@5 exact McNemar tests per cell with own Holm correction (they gate the ratio rule);
  C. transfer ratios vs MiniLM-L6 with the 95% percentile bootstrap (same resampling scheme by definition:
     numpy default_rng(0).integers(0, n, (10000, n)); own ratio and quantile code), counting resamples whose
     denominator is zero or reversed in sign;
  D. P1 decisions from unrounded upper bounds, with the registered scope (dense encoders only);
  E. the decomposition terms and their exact reconstruction of the EM difference.
Compares with paper_outputs/v03/statistics/v03_analysis.json and writes outputs/independent_check_v05.md.
"""
import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "manuscript_ipm/revision_v04/outputs"
S = json.load(open(ROOT / "paper_outputs/v03/statistics/v03_analysis.json"))
R = ["R0", "R1", "R2", "R3", "R4", "R5", "R6"]
DENSE = ["R2", "R3", "R4", "R5", "R6"]
REP = []


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def mcnemar(a, b):
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n01 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n10 + n01
    if n == 0:
        return 1.0
    k = min(n10, n01)
    # exact two-sided binomial p, p0 = 0.5: sum of probabilities <= P(k), computed in log space
    logp = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2) for i in range(n + 1)]
    pk = logp[k]
    return min(1.0, sum(math.exp(v) for v in logp if v <= pk + 1e-7))


def holm(ps):
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj, run = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i]))
        adj[i] = run
    return adj


def quantile(x, q):
    """Linear-interpolation quantile (the same definition as numpy's default), written out."""
    s = np.sort(x)
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return float(s[lo] + (s[hi] - s[lo]) * (pos - lo))


def load_cells():
    qrels = {}
    with open(ROOT / "paper_inputs/data/raw/nq_qrels/test.tsv") as f:
        next(f)
        for line in f:
            q, d, g = line.rstrip("\n").split("\t")[:3]
            if int(g) > 0:
                qrels.setdefault(q, set()).add(d)
    nq_join = {r["qid"] for r in jl(ROOT / "paper_inputs/data/processed/nq_joined.jsonl")}
    nq_order = [l.strip() for l in open(ROOT / "paper_inputs/data/processed/nq_query_order.txt") if l.strip() in nq_join][:2255]
    hp = {r["qid"]: r for r in jl(ROOT / "paper_inputs/v03_hp/data/hp_joined.jsonl")}
    hp_order = [l.strip() for l in open(ROOT / "paper_inputs/v03_hp/data/hp_query_order.txt") if l.strip() in hp][:2000]
    top = {r: {x["qid"]: x["pids"] for x in jl(ROOT / f"paper_inputs/results/raw/top10/{r}__main__top10.jsonl")} for r in R}
    cells = {}
    for cell, adir, pre in (("qwen_nq", "paper_inputs/results/processed/answers", "main"),
                            ("llama_nq", "paper_inputs/v03_nq/processed/answers", "v03_nq_llama")):
        em = {}
        for r in R:
            rows = {x["qid"]: x["em"] for x in jl(ROOT / adir / f"{pre}__{r}_k5_A1__main.jsonl")}
            em[r] = [rows[q] for q in nq_order]
        hit = {r: [float(bool(set(top[r][q][:5]) & qrels[q])) for q in nq_order] for r in R}
        cells[cell] = {"q": nq_order, "hit": hit, "em": em}
    for cell, pre in (("qwen_hp", "v03_hp_qwen"), ("llama_hp", "v03_hp_llama")):
        em, hit, both = {}, {}, {}
        for r in R:
            rows = {x["qid"]: x["em"] for x in jl(ROOT / f"paper_inputs/v03_hp/processed/answers/{pre}__{r}_k5_A1__main.jsonl")}
            gens = {x["qid"]: x["pids"][:5] for x in jl(ROOT / f"paper_inputs/v03_hp/raw/generations/{pre}__{r}_k5_A1__main.jsonl")}
            em[r] = [rows[q] for q in hp_order]
            hit[r] = [float(bool(set(gens[q]) & set(hp[q]["gold_pids"]))) for q in hp_order]
            both[r] = [float(set(hp[q]["gold_pids"]) <= set(gens[q])) for q in hp_order]
        cells[cell] = {"q": hp_order, "hit": hit, "em": em, "both": both,
                       "n_gold": sorted({len(set(hp[q]["gold_pids"])) for q in hp_order})}
    return cells


def main():
    cells = load_cells()
    worst = {"hit_mean": 0, "both_mean": 0, "p_hit": 0, "holm_hit": 0, "ratio": 0, "ci": 0}
    p1_rows, dec_worst, nonfinite, reversed_ = [], 0.0, {}, {}
    for cell, C in cells.items():
        A = S["cells"][cell]
        n = len(C["q"])
        # A. means
        for r in R:
            worst["hit_mean"] = max(worst["hit_mean"], abs(np.mean(C["hit"][r]) - A["retrieval"][r]["hit@5"]["mean"]))
            if "both" in C:
                worst["both_mean"] = max(worst["both_mean"], abs(np.mean(C["both"][r]) - A["retrieval"][r]["both@5"]["mean"]))
        # B. Hit@5 pairwise tests + Holm
        pairs = [(a, b) for i, a in enumerate(R) for b in R[i + 1:]]
        ps = [mcnemar(C["hit"][a], C["hit"][b]) for a, b in pairs]
        hs = holm(ps)
        ref = {(t["a"], t["b"]): t for t in A["pairwise_hit5"]}
        holm_own = {}
        for (a, b), p, h in zip(pairs, ps, hs):
            worst["p_hit"] = max(worst["p_hit"], abs(p - ref[(a, b)]["p"]))
            worst["holm_hit"] = max(worst["holm_hit"], abs(h - ref[(a, b)]["p_holm"]))
            holm_own[frozenset((a, b))] = h
        # C. ratios with bootstrap
        idx = np.random.default_rng(0).integers(0, n, size=(10000, n), dtype=np.int32)
        h = {r: np.array(C["hit"][r]) for r in R}
        e = {r: np.array(C["em"][r]) for r in R}
        orig = {t["retriever"]: t for t in A["transfer_vs_R1"]["em"]}
        ub = {}
        for r in R:
            if r == "R1":
                continue
            dh, de = h[r].mean() - h["R1"].mean(), e[r].mean() - e["R1"].mean()
            ok = abs(dh) > 0.02 and holm_own[frozenset(("R1", r))] < 0.05
            assert ok == orig[r]["estimable"], (cell, r)
            if not ok:
                continue
            bh = (h[r] - h["R1"])[idx].mean(axis=1)
            be = (e[r] - e["R1"])[idx].mean(axis=1)
            fin = bh != 0
            nonfinite[(cell, r)] = int((~fin).sum())
            reversed_[(cell, r)] = int((np.sign(bh) == -np.sign(dh)).sum())
            br = be[fin] / bh[fin]
            lo, hi = quantile(br, 0.025), quantile(br, 0.975)
            worst["ratio"] = max(worst["ratio"], abs(de / dh - orig[r]["ratio"]))
            worst["ci"] = max(worst["ci"], abs(lo - orig[r]["ci_lo"]), abs(hi - orig[r]["ci_hi"]))
            ub[r] = hi
            # E. decomposition reconstruction
            p0, p1 = h["R1"].mean(), h[r].mean()
            a0 = e["R1"][h["R1"] == 1].mean(); b0 = e["R1"][h["R1"] == 0].mean()
            a1 = e[r][h[r] == 1].mean(); b1 = e[r][h[r] == 0].mean()
            dec_worst = max(dec_worst, abs((p1 - p0) * (a0 - b0) + p1 * (a1 - a0) + (1 - p1) * (b1 - b0) - de))
        if cell != "qwen_nq":
            dense_ub = {r: ub[r] for r in DENSE if r in ub}
            p1_rows.append((cell, len(dense_ub), max(dense_ub.values()), all(v < 1 for v in dense_ub.values()),
                            S["cells"][cell]["P1"]["holds"], len(ub)))
        else:
            dense_ub = {r: ub[r] for r in DENSE if r in ub}
            p1_rows.append((cell + " (not confirmatory)", len(dense_ub), max(dense_ub.values()), all(v < 1 for v in dense_ub.values()),
                            S["cells"][cell]["P1"]["holds"], len(ub)))
    rep = ["# Independent check, part 2 (v0.5)", "", "Script: manuscript_ipm/revision_v04/scripts/independent_check_v05.py (no project imports).", "",
           "| Quantity | Largest absolute difference vs analysis |", "|---|---|",
           f"| Hit@5 means, 4 cells × 7 retrievers | {worst['hit_mean']:.1e} |",
           f"| Both@5 means, 2 HotpotQA cells × 7 retrievers | {worst['both_mean']:.1e} |",
           f"| Hit@5 McNemar p, 84 pairs (own exact binomial) | {worst['p_hit']:.1e} |",
           f"| Hit@5 Holm p, 84 pairs (own Holm) | {worst['holm_hit']:.1e} |",
           f"| Transfer-ratio point estimates, 22 estimable | {worst['ratio']:.1e} |",
           f"| Transfer-ratio bootstrap bounds, 44 (own ratio and quantile) | {worst['ci']:.1e} |",
           f"| Decomposition identity residual, 22 ratios | {dec_worst:.1e} |", "",
           "Ratio-rule decisions (estimable or not) agree for all 24 retriever × cell pairs.", "",
           "## Bootstrap denominators", "",
           f"Resamples with a zero Hit@5 difference (dropped): {sum(nonfinite.values())} of {10000 * len(nonfinite)}; "
           f"resamples whose Hit@5 difference has the opposite sign to the observed one: {sum(reversed_.values())}.", "",
           "## P1 with unrounded bounds and the registered scope (dense encoders only)", "",
           "| Cell | Dense ratios | Largest upper bound (unrounded) | Criterion met (dense only) | Analysis verdict | Ratios counted by the analysis |",
           "|---|---|---|---|---|---|"]
    for c, k, m, ok, orig, n_all in p1_rows:
        rep.append(f"| {c} | {k} | {m!r} | {'met' if ok else 'not met'} | {'met' if orig else 'not met'} | {n_all} |")
    hp_gold = sorted(set(sum((C.get('n_gold', []) for C in cells.values()), [])))
    rep += ["", f"HotpotQA generation queries: distinct gold passages per query = {hp_gold} (Both@5 requires both).", ""]
    (OUT / "independent_check_v05.md").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))


if __name__ == "__main__":
    main()
