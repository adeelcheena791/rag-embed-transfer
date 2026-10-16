"""Results carried over from the Phase 1 (v0.2) paper into the journal version, plus one exploratory extension.

Reads committed inputs only and writes paper_outputs/v03/statistics/v03_extras.json:
  retrieval_nq      Hit@1/3/5/10, nDCG@10, MRR@10 with 95% CIs on the 2,255 joined queries (Phase 1 pipeline output, rq12_main.json)
  retrieval_hp      the same metrics plus Both@5 on the 2,000 HotpotQA generation queries (v03_hp retrieval summary)
  deployment_cost   Phase 1 single measurements (rq34_main.json): 300k encoding time on T4, peak GPU memory, CPU throughput,
                    batch-1 query encoding on GPU and CPU, exact-search latency on CPU, flat index size
  idk               prompt A3 ("I don't know" allowed) on the first 200 queries of the seeded order: abstention overall,
                    on hits and on misses; EM under A3 vs A1 (exact McNemar). Recomputed here from the scored answers.
  transfer_by_k     EXPLORATORY, post hoc: T_k = dEM@k / dHit@k vs MiniLM-L6 for every retriever at k = 1, 3, 5, 10, with Hit@k
                    matched to the k passages the reader received; paired percentile bootstrap (10,000, seed 0); no reporting rule.

    python -m src.analysis.v03_extras
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from src.analysis import stats

ROOT = Path(__file__).resolve().parents[2]
PI = ROOT / "paper_inputs"
R = ["R0", "R1", "R2", "R3", "R4", "R5", "R6"]


def jl(p):
    return [json.loads(line) for line in open(p, encoding="utf-8") if line.strip()]


def main() -> int:
    rq12 = json.load(open(PI / "results/statistics/rq12_main.json"))
    rq34 = json.load(open(PI / "results/statistics/rq34_main.json"))
    order = [l.strip() for l in open(PI / "data/processed/nq_query_order.txt") if l.strip()]
    joined = {r["qid"] for r in jl(PI / "data/processed/nq_joined.jsonl")}
    order = [q for q in order if q in joined]
    qrels = {}
    with open(PI / "data/raw/nq_qrels/test.tsv") as f:
        next(f)
        for row in csv.reader(f, delimiter="\t"):
            if int(row[2]) > 0:
                qrels.setdefault(row[0], set()).add(row[1])
    top10 = {r: {x["qid"]: x["pids"] for x in jl(PI / f"results/raw/top10/{r}__main__top10.jsonl")} for r in R}

    def hit(r, qs, k):
        return np.array([float(bool(set(top10[r][q][:k]) & qrels[q])) for q in qs])

    out = {"note": __doc__.split("\n\n")[0]}
    # ---- retrieval (NQ from the Phase 1 pipeline output; check our own Hit@5 against it)
    m = rq12["rq1_joined"]["means"]
    out["retrieval_nq"] = {"n": rq12["n_joined"], "n_judged": rq12["n_judged"], "metrics": m,
                           "ndcg10_all_judged": {r: rq12["rq1_judged"]["means"][r]["ndcg@10"]["mean"] for r in R}}
    for r in R:
        assert abs(hit(r, order, 5).mean() - m[r]["hit@5"]["mean"]) < 1e-12, r
    hp = json.load(open(PI / "v03_hp/summary/v03_hp_retrieval_summary.json"))
    out["retrieval_hp"] = {"n": hp["sets"]["gen_first_2000"], "metrics": {r: hp["retrievers"][r]["gen_first_2000"] for r in R}}
    # ---- Phase 1 deployment cost (single measurements)
    out["deployment_cost"] = {"source": "paper_inputs/results/statistics/rq34_main.json (Phase 1, single measurements)",
                              "efficiency": rq34["efficiency"]}
    # ---- IDK prompt (A3)
    ans = lambda tag: {x["qid"]: x for x in jl(PI / f"results/processed/answers/main__{tag}__main.jsonl")}  # noqa: E731
    idk = {}
    for c in ("R0", "R1", "R4", "GOLD"):
        a = ans("GOLD_A3" if c == "GOLD" else f"{c}_k5_A3")
        qs = [q for q in order if q in a]
        v = np.array([a[q]["category"] == "idk" for q in qs], dtype=float)
        row = {"n": len(qs), "idk_rate": float(v.mean())}
        if c != "GOLD":
            h = hit(c, qs, 5).astype(bool)
            a1 = ans(f"{c}_k5_A1")
            e1, e3 = np.array([a1[q]["em"] for q in qs]), np.array([a[q]["em"] for q in qs])
            row.update(idk_given_hit=float(v[h].mean()), idk_given_miss=float(v[~h].mean()), n_hit=int(h.sum()), n_miss=int((~h).sum()),
                       em_A1=float(e1.mean()), em_A3=float(e3.mean()), p_A3_vs_A1=stats.mcnemar_exact(e3, e1)["p"])
        idk[c] = row
    out["idk"] = idk
    # ---- exploratory: transfer ratio with Hit@k matched to k
    v3 = {"R2", "R3", "R5", "R6"}
    idx = stats.resample_index(len(order))
    rows = []
    for k in (1, 3, 5, 10):
        h = {r: hit(r, order, k) for r in R}
        e = {}
        for r in R:
            path = (PI / f"v03_nq/processed/answers/v03_nq_qwen__{r}_k{k}_A1__main.jsonl") if (r in v3 and k != 5) \
                else (PI / f"results/processed/answers/main__{r}_k{k}_A1__main.jsonl")
            a = {x["qid"]: x["em"] for x in jl(path)}
            e[r] = np.array([a[q] for q in order])
        for r in R[0:1] + R[2:]:
            dh, de = float(h[r].mean() - h["R1"].mean()), float(e[r].mean() - e["R1"].mean())
            br = stats.boot_means(e[r] - e["R1"], idx) / stats.boot_means(h[r] - h["R1"], idx)
            lo, hi = stats.ci(br[np.isfinite(br)])
            rows.append({"retriever": r, "k": k, "hit_k": float(h[r].mean()), "hit_k_R1": float(h["R1"].mean()), "d_hit": dh, "d_em": de,
                         "ratio": de / dh if dh else None, "ci_lo": float(lo), "ci_hi": float(hi),
                         "hit_mcnemar_p": stats.mcnemar_exact(h[r], h["R1"])["p"]})
    out["transfer_by_k"] = {"note": "post hoc, exploratory; Qwen reader, NQ; Hit@k matched to the k passages read; no reporting rule",
                            "rows": rows}
    p = ROOT / "paper_outputs/v03/statistics/v03_extras.json"
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
