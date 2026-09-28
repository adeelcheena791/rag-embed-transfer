"""RQ1 (retrieval) + RQ2 (answers, transfer) analysis on the main run (design v0.2 §5, §7;
addenda v0.2.2 D1b, v0.2.3 M1). Reads restored raw results; writes results/statistics/ and tables/.

    python -m src.analysis.rq12 --config configs/main.yaml --qrels data/raw/nq_qrels/test.tsv --tag s3a
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.analysis import stats
from src.analysis.metrics import METRICS, load_qrels_tsv, per_query
from src.common.config import cpath, load_config, root
from src.common.io import read_jsonl, read_lines, write_json_atomic
from src.common.runs import find_latest_run

RETRIEVERS = ["R0", "R1", "R2", "R3", "R4", "R5", "R6"]
NAMES = {"R0": "BM25", "R1": "MiniLM-L6", "R2": "bge-small", "R3": "bge-base", "R4": "bge-large",
         "R5": "e5-base-v2", "R6": "Qwen3-Emb-0.6B", "CB": "closed-book", "GOLD": "gold passages"}


def answer_vectors(cfg, conds, qids):
    out = {}
    for c in conds:
        tag = "CB_A2" if c == "CB" else "GOLD_A1" if c == "GOLD" else f"{c}_k5_A1"
        rows = {r["qid"]: r for r in read_jsonl(cpath(cfg, "results_processed") / "answers" / f"{cfg['experiment']}__{tag}__main.jsonl")}
        out[c] = {m: np.array([rows[q][m] for q in qids], dtype=float) for m in ("em", "f1", "contain")}
    return out


def retrieval_block(cfg, qrels, qids, idx):
    vals = {r: per_query(find_latest_run(cfg, r, "main", want="judged"), qrels, qids) for r in RETRIEVERS}
    means = {r: {m: stats.mean_ci(vals[r][m], idx) for m in METRICS} for r in RETRIEVERS}
    tests = {m: stats.pairwise({r: vals[r][m] for r in RETRIEVERS}, RETRIEVERS, idx, binary=m.startswith("hit"))
             for m in METRICS}
    return vals, means, tests


def decomposition(hit, em, idx):
    h = hit.astype(bool)
    bh, be = hit[idx], em[idx]
    with np.errstate(invalid="ignore", divide="ignore"):
        p_hit = (be * bh).sum(1) / bh.sum(1)
        p_miss = (be * (1 - bh)).sum(1) / (1 - bh).sum(1)
    return {"hit@5": float(h.mean()),
            "p_correct_given_hit": float(em[h].mean()), "p_hit_ci": stats.ci(p_hit[np.isfinite(p_hit)]),
            "p_correct_given_miss": float(em[~h].mean()), "p_miss_ci": stats.ci(p_miss[np.isfinite(p_miss)]),
            "n_hit": int(h.sum()), "n_miss": int((~h).sum()),
            "identity_check": float(h.mean() * em[h].mean() + (1 - h.mean()) * em[~h].mean() - em.mean())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    qrels = load_qrels_tsv(args.qrels)
    proc = cpath(cfg, "data_processed")
    joined = sorted(json.loads(l)["qid"] for l in open(proc / "nq_joined.jsonl", encoding="utf-8"))
    judged = sorted(q for q in qrels if qrels[q])
    order = read_lines(proc / "nq_query_order.txt")
    assert set(order) == set(joined) and set(joined) <= set(judged)
    out = {"n_joined": len(joined), "n_judged": len(judged), "bootstrap": {"B": stats.B, "seed": stats.SEED}}

    # ---------------- RQ1: retrieval, primary (joined) + secondary (judged, D1b) ----------------
    idx_j = stats.resample_index(len(joined))
    vals_j, means_j, tests_j = retrieval_block(cfg, qrels, joined, idx_j)
    idx_a = stats.resample_index(len(judged))
    vals_a, means_a, tests_a = retrieval_block(cfg, qrels, judged, idx_a)
    out["rq1_joined"] = {"means": means_j, "pairwise": tests_j}
    out["rq1_judged"] = {"means": means_a, "pairwise": tests_a}
    # D1b: joined vs non-joined (independent groups, bootstrap within groups)
    nonj = [q for q in judged if q not in set(joined)]
    pos = {q: i for i, q in enumerate(judged)}
    rng = np.random.default_rng(1)
    d1b = {}
    for r in RETRIEVERS:
        for m in ("hit@5", "ndcg@10"):
            a = vals_a[r][m][[pos[q] for q in joined]]
            b = vals_a[r][m][[pos[q] for q in nonj]]
            bs = [a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean() for _ in range(2000)]
            d1b.setdefault(r, {})[m] = {"joined": float(a.mean()), "not_joined": float(b.mean()),
                                        "diff": float(a.mean() - b.mean()), "ci": stats.ci(np.array(bs))}
    rank_j = sorted(RETRIEVERS, key=lambda r: -means_j[r]["ndcg@10"]["mean"])
    rank_a = sorted(RETRIEVERS, key=lambda r: -means_a[r]["ndcg@10"]["mean"])
    out["d1b"] = {"n_not_joined": len(nonj), "per_retriever": d1b, "rank_ndcg_joined": rank_j, "rank_ndcg_judged": rank_a,
                  "spearman_rank_agreement": stats.spearman([means_j[r]["ndcg@10"]["mean"] for r in RETRIEVERS],
                                                            [means_a[r]["ndcg@10"]["mean"] for r in RETRIEVERS])}

    # ---------------- RQ2: answers on the 2,255 generation queries ----------------
    conds = RETRIEVERS + ["CB", "GOLD"]
    idx_o = stats.resample_index(len(order))
    ans = answer_vectors(cfg, conds, order)
    out["rq2_answers"] = {c: {m: stats.mean_ci(ans[c][m], idx_o) for m in ("em", "f1", "contain")} for c in conds}
    out["rq2_pairwise_em"] = stats.pairwise({r: ans[r]["em"] for r in RETRIEVERS}, RETRIEVERS, idx_o, binary=True)
    out["rq2_pairwise_f1"] = stats.pairwise({r: ans[r]["f1"] for r in RETRIEVERS}, RETRIEVERS, idx_o, binary=False)
    posj = {q: i for i, q in enumerate(joined)}
    hit5 = {r: vals_j[r]["hit@5"][[posj[q] for q in order]] for r in RETRIEVERS}
    out["rq2_decomposition"] = {r: decomposition(hit5[r], ans[r]["em"], idx_o) for r in RETRIEVERS}
    h5_tests = {(t["a"], t["b"]): t for t in stats.pairwise(hit5, RETRIEVERS, idx_o, binary=True)}
    out["rq2_hit5_pairwise_generation_set"] = list(h5_tests.values())
    tr = []
    for r in RETRIEVERS:
        if r == "R1":
            continue
        t = h5_tests.get(("R1", r)) or h5_tests.get((r, "R1"))
        tr.append(stats.transfer_ratio(hit5[r], hit5["R1"], ans[r]["em"], ans["R1"]["em"], idx_o, t["p_holm"], r).__dict__)
    out["rq2_transfer_ratio_vs_R1"] = tr
    out["rq2_spearman_hit5_em"] = stats.spearman([hit5[r].mean() for r in RETRIEVERS], [ans[r]["em"].mean() for r in RETRIEVERS])
    out["rq2_spearman_ndcg_em"] = stats.spearman([means_j[r]["ndcg@10"]["mean"] for r in RETRIEVERS],
                                                 [ans[r]["em"].mean() for r in RETRIEVERS])

    sdir = cpath(cfg, "results_raw").parent / "statistics"
    write_json_atomic(sdir / f"rq12_{args.tag}.json", out)
    tdir = root(cfg) / "tables"
    tdir.mkdir(exist_ok=True)
    with open(tdir / f"rq1_retrieval_{args.tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["set", "retriever", "name"] + [f"{m}{s}" for m in METRICS for s in ("", "_lo", "_hi")])
        for setname, means in (("joined", means_j), ("judged", means_a)):
            for r in RETRIEVERS:
                w.writerow([setname, r, NAMES[r]] + [round(means[r][m][k], 4) for m in METRICS for k in ("mean", "ci_lo", "ci_hi")])
    with open(tdir / f"rq2_answers_{args.tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "name", "em", "em_lo", "em_hi", "f1", "f1_lo", "f1_hi", "contain", "hit@5",
                    "p_em_given_hit", "p_em_given_miss"])
        for c in conds:
            a, d = out["rq2_answers"][c], out["rq2_decomposition"].get(c, {})
            w.writerow([c, NAMES[c], round(a["em"]["mean"], 4), round(a["em"]["ci_lo"], 4), round(a["em"]["ci_hi"], 4),
                        round(a["f1"]["mean"], 4), round(a["f1"]["ci_lo"], 4), round(a["f1"]["ci_hi"], 4),
                        round(a["contain"]["mean"], 4), round(d.get("hit@5", float("nan")), 4),
                        round(d.get("p_correct_given_hit", float("nan")), 4), round(d.get("p_correct_given_miss", float("nan")), 4)])
    print(f"wrote {sdir / f'rq12_{args.tag}.json'} and tables/*_{args.tag}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
