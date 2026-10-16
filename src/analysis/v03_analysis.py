"""v0.3 analysis (addenda v0.3, v0.3.1, v0.3.2): every pre-registered test and every number of the journal version.

Reads ONLY committed files under paper_inputs/ (no GPU, no network), so anyone can rerun it:

    python -m src.analysis.v03_analysis      # writes paper_outputs/v03/statistics/*.json and paper_outputs/v03/tables/*.csv

Cells (reader x dataset), generation queries = first n of the seeded order:
    qwen_nq   Qwen2.5-7B-AWQ   NQ        2,255   (v0.2 main run)
    llama_nq  Llama-3.1-8B-AWQ NQ        2,255   (v0.3 S3-NQ)
    qwen_hp   Qwen2.5-7B-AWQ   HotpotQA  2,000   (v0.3 S2-HP)
    llama_hp  Llama-3.1-8B-AWQ HotpotQA  2,000   (v0.3 S2-HP)
Per-query retrieval on the generation queries: NQ from the v0.2 run files (top 10, compact copies), HotpotQA from the
top-5 passages that were actually given to the reader (identical to the run's top 5). Statistics follow
src/analysis/stats.py (exact McNemar, Wilcoxon, Holm over 21 pairs per cell, bootstrap B = 10,000 seed 0,
transfer ratio vs R1 with the v0.2 reporting rule).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.analysis import stats
from src.common.ir_metrics import hit_at_k, ndcg_at_k

R = ["R0", "R1", "R2", "R3", "R4", "R5", "R6"]
NAMES = {"R0": "BM25", "R1": "MiniLM-L6", "R2": "bge-small", "R3": "bge-base", "R4": "bge-large", "R5": "e5-base-v2",
         "R6": "Qwen3-Emb-0.6B", "CB": "closed book", "GOLD": "gold passages", "X0": "BM25 + reranker",
         "X5": "e5-base-v2 + reranker", "M1": "gold@1 + 4 hard", "M3": "gold@3 + 4 hard", "M5": "gold@5 + 4 hard",
         "MR": "gold@1 + 4 random", "MG": "gold only"}
CELLS = {
    "qwen_nq": {"reader": "Qwen2.5-7B", "dataset": "NQ", "answers": "paper_inputs/results/processed/answers", "prefix": "main",
                "n": 2255},
    "llama_nq": {"reader": "Llama-3.1-8B", "dataset": "NQ", "answers": "paper_inputs/v03_nq/processed/answers",
                 "prefix": "v03_nq_llama", "n": 2255},
    "qwen_hp": {"reader": "Qwen2.5-7B", "dataset": "HotpotQA", "answers": "paper_inputs/v03_hp/processed/answers",
                "prefix": "v03_hp_qwen", "gens": "paper_inputs/v03_hp/raw/generations", "n": 2000},
    "llama_hp": {"reader": "Llama-3.1-8B", "dataset": "HotpotQA", "answers": "paper_inputs/v03_hp/processed/answers",
                 "prefix": "v03_hp_llama", "gens": "paper_inputs/v03_hp/raw/generations", "n": 2000},
}
METRICS = ("em", "f1", "contain")


def jl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def lines(path):
    return [l.strip() for l in open(path, encoding="utf-8") if l.strip()]


def load_qrels(path):
    q = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for row in csv.reader(f, delimiter="\t"):
            if len(row) >= 3 and int(row[2]) > 0:
                q.setdefault(row[0], {})[row[1]] = int(row[2])
    return q


# ---------------------------------------------------------------- data
class Data:
    def __init__(self, root: Path):
        self.root = root
        self.nq_qrels = load_qrels(root / "paper_inputs/data/raw/nq_qrels/test.tsv")
        nq_joined = {r["qid"]: r for r in jl(root / "paper_inputs/data/processed/nq_joined.jsonl")}
        self.nq_order = [q for q in lines(root / "paper_inputs/data/processed/nq_query_order.txt") if q in nq_joined]
        hp_joined = {r["qid"]: r for r in jl(root / "paper_inputs/v03_hp/data/hp_joined.jsonl")}
        self.hp_joined = hp_joined
        self.hp_order = [q for q in lines(root / "paper_inputs/v03_hp/data/hp_query_order.txt") if q in hp_joined]
        self.nq_top10 = {r: {x["qid"]: x["pids"] for x in jl(root / f"paper_inputs/results/raw/top10/{r}__main__top10.jsonl")}
                         for r in R}

    def qids(self, cell):
        c = CELLS[cell]
        return (self.nq_order if c["dataset"] == "NQ" else self.hp_order)[: c["n"]]

    def answers(self, cell, tag):
        c = CELLS[cell]
        rows = {r["qid"]: r for r in jl(self.root / c["answers"] / f"{c['prefix']}__{tag}__main.jsonl")}
        q = self.qids(cell)
        missing = [x for x in q if x not in rows]
        if missing:
            raise KeyError(f"{cell} {tag}: {len(missing)} queries missing")
        return {m: np.array([rows[x][m] for x in q], dtype=float) for m in METRICS}

    def retrieval(self, cell, r):
        """Per-query hit@5 (+ ndcg@10 for NQ, both@5 for HotpotQA) on the cell's generation queries."""
        c, q = CELLS[cell], self.qids(cell)
        if c["dataset"] == "NQ":
            top = self.nq_top10[r]
            return {"hit@5": np.array([hit_at_k(top[x], self.nq_qrels[x], 5) for x in q], float),
                    "ndcg@10": np.array([ndcg_at_k(top[x], self.nq_qrels[x], 10) for x in q], float)}
        g = {x["qid"]: x["pids"] for x in jl(self.root / c["gens"] / f"{c['prefix']}__{r}_k5_A1__main.jsonl")}
        gold = {x: set(self.hp_joined[x]["gold_pids"]) for x in q}
        return {"hit@5": np.array([float(bool(gold[x] & set(g[x][:5]))) for x in q]),
                "both@5": np.array([float(gold[x] <= set(g[x][:5])) for x in q])}


# ---------------------------------------------------------------- per cell
def decomposition(hit, em):
    h = hit.astype(bool)
    return {"p_em_given_hit": float(em[h].mean()) if h.any() else None,
            "p_em_given_miss": float(em[~h].mean()) if (~h).any() else None, "n_hit": int(h.sum()), "n_miss": int((~h).sum())}


def analyse_cell(D: Data, cell: str) -> dict:
    q = D.qids(cell)
    idx = stats.resample_index(len(q))
    conds = R + ["CB", "GOLD"]
    ans = {c: D.answers(cell, "CB_A2" if c == "CB" else "GOLD_A1" if c == "GOLD" else f"{c}_k5_A1") for c in conds}
    ret = {r: D.retrieval(cell, r) for r in R}
    out = {"cell": cell, **{k: CELLS[cell][k] for k in ("reader", "dataset")}, "n_queries": len(q),
           "bootstrap": {"B": stats.B, "seed": stats.SEED}}
    out["answers"] = {c: {m: stats.mean_ci(ans[c][m], idx) for m in METRICS} for c in conds}
    out["retrieval"] = {r: {m: stats.mean_ci(v, idx) for m, v in ret[r].items()} for r in R}
    out["pairwise_hit5"] = stats.pairwise({r: ret[r]["hit@5"] for r in R}, R, idx, binary=True)
    out["pairwise_em"] = stats.pairwise({r: ans[r]["em"] for r in R}, R, idx, binary=True)
    out["pairwise_contain"] = stats.pairwise({r: ans[r]["contain"] for r in R}, R, idx, binary=True)
    out["pairwise_f1"] = stats.pairwise({r: ans[r]["f1"] for r in R}, R, idx, binary=False)
    h5 = {(t["a"], t["b"]): t for t in out["pairwise_hit5"]}
    out["transfer_vs_R1"] = {}
    for m in METRICS:                                      # E8: EM (primary), F1 and answer containment
        rows = []
        for r in R:
            if r == "R1":
                continue
            t = h5.get(("R1", r)) or h5.get((r, "R1"))
            rows.append(stats.transfer_ratio(ret[r]["hit@5"], ret["R1"]["hit@5"], ans[r][m], ans["R1"][m], idx,
                                             t["p_holm"], r).__dict__)
        out["transfer_vs_R1"][m] = rows
    out["decomposition"] = {r: decomposition(ret[r]["hit@5"], ans[r]["em"]) for r in R}
    out["spearman_hit5_em"] = stats.spearman([ret[r]["hit@5"].mean() for r in R], [ans[r]["em"].mean() for r in R])
    # P1: every reportable EM ratio has an upper 95% CI bound < 1
    rep = [t for t in out["transfer_vs_R1"]["em"] if t["estimable"]]
    out["P1"] = {"n_reportable": len(rep), "upper_bounds": {t["retriever"]: t["ci_hi"] for t in rep},
                 "holds": bool(rep) and all(t["ci_hi"] < 1 for t in rep)}
    # P2: no Holm-significant EM gain of bge-large (R4) or Qwen3-Emb (R6) over e5-base-v2 (R5)
    em = {(t["a"], t["b"]): t for t in out["pairwise_em"]}
    p2 = {}
    for big in ("R4", "R6"):
        t = em.get(("R5", big)) or em.get((big, "R5"))
        d = ans[big]["em"].mean() - ans["R5"]["em"].mean()
        p2[big] = {"em_diff_minus_e5": float(d), "p_holm": t["p_holm"], "sig_gain_over_e5": bool(d > 0 and t["p_holm"] < 0.05),
                   "point_estimate_e5_ge": bool(d <= 0)}
    out["P2"] = {"per_encoder": p2, "holds": not any(v["sig_gain_over_e5"] for v in p2.values()),
                 "point_estimates_e5_ge_both": all(v["point_estimate_e5_ge"] for v in p2.values())}
    if CELLS[cell]["dataset"] == "HotpotQA":           # E1: split by question type (descriptive)
        types = [D.hp_joined[x]["type"] for x in q]
        out["by_type"] = {}
        for ty in sorted(set(types)):
            m = np.array([t == ty for t in types])
            out["by_type"][ty] = {"n": int(m.sum()), **{c: {"em": float(ans[c]["em"][m].mean())} for c in conds},
                                  **{f"hit5_{r}": float(ret[r]["hit@5"][m].mean()) for r in R},
                                  **{f"both5_{r}": float(ret[r]["both@5"][m].mean()) for r in R}}
    return out


# ---------------------------------------------------------------- E4 / P4 top-k
def analyse_topk(D: Data) -> dict:
    q = D.qids("qwen_nq")
    idx = stats.resample_index(len(q))
    v02 = CELLS["qwen_nq"]
    v03 = {"answers": "paper_inputs/v03_nq/processed/answers", "prefix": "v03_nq_qwen"}

    def em(r, k):
        src = v02 if (r in ("R0", "R1", "R4") or k == 5) else v03
        rows = {x["qid"]: x["em"] for x in jl(D.root / src["answers"] / f"{src['prefix']}__{r}_k{k}_A1__main.jsonl")}
        return np.array([rows[x] for x in q], float), ("v0.2" if src is v02 else "v0.3")
    table, tests = {}, []
    for r in R:
        table[r] = {}
        for k in (1, 3, 5, 10):
            v, origin = em(r, k)
            table[r][k] = {**stats.mean_ci(v, idx), "source": origin}
        a, _ = em(r, 1)
        b, _ = em(r, 5)
        tests.append({"retriever": r, **stats.paired_diff(a, b, idx, binary=True)})
    for t, p in zip(tests, stats.holm([t["p"] for t in tests])):
        t["p_holm"] = p
    return {"n_queries": len(q), "em_by_k": table, "k1_vs_k5": tests,
            "P4": {"holds": all(t["diff"] < 0 and t["p_holm"] < 0.05 for t in tests)},
            "note": "R2, R3, R5, R6 at k = 1, 3, 10 were run in v0.3 (Python 3.13 stack, addendum v0.3.1); their k = 5 "
                    "and all R0, R1, R4 values are v0.2. Same reader, prompt, decoding and model revision."}


# ---------------------------------------------------------------- E12 / P5, P6
def analyse_mechanism(D: Data) -> dict:
    base = D.root / "paper_inputs/v03_nq_s4/processed/answers"
    conds = ["M1", "M3", "M5", "MR", "MG"]
    rows = {c: {x["qid"]: x for x in jl(base / f"v03_nq_qwen__{c}_k5_A1__main.jsonl")} for c in conds}
    q = D.qids("qwen_nq")[:500]
    if any(set(rows[c]) != set(q) for c in conds):
        raise ValueError("E12 answer files do not cover exactly the first 500 queries")
    v = {c: {m: np.array([rows[c][x][m] for x in q], float) for m in METRICS} for c in conds}
    idx = stats.resample_index(len(q))
    out = {"n_queries": len(q), "answers": {c: {m: stats.mean_ci(v[c][m], idx) for m in METRICS} for c in conds}}
    p5 = stats.paired_diff(v["M1"]["em"], v["MR"]["em"], idx, binary=True)
    out["P5"] = {"test": p5, "holds": bool(p5["p"] < 0.05 and p5["diff"] < 0)}
    pairs = [("M1", "M3"), ("M1", "M5"), ("M3", "M5")]
    t6 = [{"a": a, "b": b, **stats.paired_diff(v[a]["em"], v[b]["em"], idx, binary=True)} for a, b in pairs]
    for t, p in zip(t6, stats.holm([t["p"] for t in t6])):
        t["p_holm"] = p
    out["P6"] = {"tests": t6, "holds": any(t["p_holm"] < 0.05 for t in t6)}
    out["gold_only_vs_MR"] = stats.paired_diff(v["MG"]["em"], v["MR"]["em"], idx, binary=True)
    return out


# ---------------------------------------------------------------- E5 reranker
def analyse_reranker(D: Data) -> dict:
    q = D.qids("qwen_nq")
    idx = stats.resample_index(len(q))
    base = D.root / "paper_inputs/v03_nq_s4"
    runs = {}
    for x in ("X0", "X5"):
        f = next((base / "raw/runs").glob(f"{x}__main__*.jsonl"))
        runs[x] = {r["qid"]: r["pids"] for r in jl(f)}
    runs.update({r: D.nq_top10[r] for r in ("R0", "R5")})
    out = {"n_generation_queries": len(q), "retrieval_generation_set": {}, "answers": {}, "pairs": {}}
    em = {}
    for r in ("R0", "R5", "X0", "X5"):
        h = np.array([hit_at_k(runs[r][x], D.nq_qrels[x], 5) for x in q], float)
        n = np.array([ndcg_at_k(runs[r][x], D.nq_qrels[x], 10) for x in q], float)
        out["retrieval_generation_set"][r] = {"hit@5": stats.mean_ci(h, idx), "ndcg@10": stats.mean_ci(n, idx)}
        src = (CELLS["qwen_nq"]["answers"], "main") if r.startswith("R") else ("paper_inputs/v03_nq_s4/processed/answers", "v03_nq_qwen")
        rows = {x["qid"]: x for x in jl(D.root / src[0] / f"{src[1]}__{r}_k5_A1__main.jsonl")}
        em[r] = {m: np.array([rows[x][m] for x in q], float) for m in METRICS}
        em[r]["hit@5"] = h
        out["answers"][r] = {m: stats.mean_ci(em[r][m], idx) for m in METRICS}
    for a, b in (("X0", "R0"), ("X5", "R5")):
        out["pairs"][f"{a}_vs_{b}"] = {"hit@5": stats.paired_diff(em[a]["hit@5"], em[b]["hit@5"], idx, binary=True),
                                       "em": stats.paired_diff(em[a]["em"], em[b]["em"], idx, binary=True)}
    meta = jl(base / "raw/rerank.jsonl")
    out["latency_ms"] = {m["first_stage"]: m["latency"] for m in meta}
    s = json.load(open(base / "summary/v03_nq_main_summary.json"))["retrievers"]
    out["retrieval_all_judged"] = {r: {k: s[r]["judged"][k] for k in ("hit@5", "ndcg@10")} for r in ("R0", "R5", "X0", "X5")}
    out["note"] = "Reader EM for X0/X5 is v0.3 (Python 3.13 stack); R0/R5 EM is v0.2. Same reader revision and prompt."
    return out


# ---------------------------------------------------------------- E9 full corpus, E6 HNSW, E7 cost
def analyse_corpus_hnsw_cost(D: Data) -> dict:
    base = D.root / "paper_inputs/v03_nq_s4"
    full = json.load(open(base / "summary/v03_nq_full_summary.json"))["retrievers"]
    main = json.load(open(base / "summary/v03_nq_main_summary.json"))["retrievers"]
    hn = {r["model"]: r for r in jl(base / "raw/hnsw.jsonl")}
    ks = ["R0", "R1", "R2", "R5"]
    sub_same = {"R0": main["R0"]["judged"]["ndcg@10"], **{m: hn[m]["exact"]["ndcg@10"] for m in ("R1", "R2", "R5")}}
    sub_hit = {"R0": main["R0"]["judged"]["hit@5"], **{m: hn[m]["exact"]["hit@5"] for m in ("R1", "R2", "R5")}}
    from scipy.stats import kendalltau
    e9 = {"retrievers": ks, "n_judged": 3452,
          "subcorpus_same_embeddings": {k: {"ndcg@10": sub_same[k], "hit@5": sub_hit[k]} for k in ks},
          "subcorpus_v02_runs": {k: {m: main[k]["judged"][m] for m in ("ndcg@10", "hit@5")} for k in ks},
          "full_corpus": {k: {m: full[k]["judged"][m] for m in ("ndcg@10", "hit@5")} for k in ks}}
    tau = kendalltau([sub_same[k] for k in ks], [full[k]["judged"]["ndcg@10"] for k in ks]).statistic
    tau2 = kendalltau([main[k]["judged"]["ndcg@10"] for k in ks], [full[k]["judged"]["ndcg@10"] for k in ks]).statistic
    e9["P3"] = {"kendall_tau_primary": float(tau), "kendall_tau_secondary_v02": float(tau2), "holds": bool(tau == 1.0)}
    e9["drop_full_minus_sub"] = {k: {m: full[k]["judged"][m] - e9["subcorpus_same_embeddings"][k][m] for m in ("ndcg@10", "hit@5")} for k in ks}
    e6 = {m: {"build_seconds": hn[m]["hnsw_build_seconds"], "n_rows": hn[m]["n_rows"], "exact": hn[m]["exact"],
              **{f"ef{ef}": hn[m][f"hnsw_ef{ef}"] for ef in (64, 128)}} for m in ("R1", "R2", "R5")}
    import statistics as st
    rep = [r for r in jl(base / "raw/throughput.jsonl") if r.get("run_id", "").startswith("20261015")]
    v02 = {r["model"]: r for r in jl(D.root / "paper_inputs/results/raw/throughput_v02_S1.jsonl") if r["device"] == "cuda"}
    e7 = {}
    for m in ["R1", "R2", "R3", "R5", "R4", "R6"]:
        rs = [r for r in rep if r["model"] == m]
        pps, lat = [r["passages_per_s"] for r in rs], [r["query_latency"]["median_ms"] for r in rs]
        e7[m] = {"n_reps": len(rs), "passages_per_s": {"median": st.median(pps), "min": min(pps), "max": max(pps)},
                 "query_latency_ms": {"median": st.median(lat), "min": min(lat), "max": max(lat)},
                 "v02_single": {"passages_per_s": v02[m]["passages_per_s"], "query_latency_ms": v02[m]["query_latency"]["median_ms"]},
                 "peak_mib": max(r["peak_mib"] for r in rs), "n_params": rs[0]["n_params"], "dim": rs[0]["dim"]}
    bl = [r["latency"]["median_ms"] for r in jl(base / "raw/bm25.jsonl") if r.get("latency") and r["run_id"].startswith("20261015")]
    e7["R0"] = {"n_reps": len(bl), "query_latency_ms": {"median": st.median(bl), "min": min(bl), "max": max(bl)}}
    e7["pareto_order_stable"] = all(e7[a]["passages_per_s"]["max"] < e7[b]["passages_per_s"]["min"] or
                                    e7[a]["passages_per_s"]["min"] > e7[b]["passages_per_s"]["max"]
                                    for i, a in enumerate(["R1", "R2", "R3", "R5", "R4", "R6"])
                                    for b in ["R1", "R2", "R3", "R5", "R4", "R6"][i + 1:] if {a, b} != {"R3", "R5"})
    return {"E9": e9, "E6": e6, "E7": e7}


# ---------------------------------------------------------------- E10 pooled GEE
def analyse_gee(D: Data, cells: dict) -> dict:
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    rows = []
    for cell in CELLS:
        q = D.qids(cell)
        for r in R:
            e = D.answers(cell, f"{r}_k5_A1")["em"]
            rows += [{"em": e[i], "retriever": r, "dataset": CELLS[cell]["dataset"], "reader": CELLS[cell]["reader"],
                      "cluster": f"{CELLS[cell]['dataset']}:{x}"} for i, x in enumerate(q)]
    df = pd.DataFrame(rows)
    df["cluster_id"] = df["cluster"].astype("category").cat.codes
    model = smf.gee("em ~ C(retriever, Treatment('R5')) + C(dataset) + C(reader)", groups="cluster_id", data=df,
                    family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable())
    res = model.fit()
    coef = {}
    for name in res.params.index:
        coef[name] = {"beta": float(res.params[name]), "se_robust": float(res.bse[name]), "p": float(res.pvalues[name]),
                      "odds_ratio": float(np.exp(res.params[name])),
                      "or_ci95": [float(np.exp(res.conf_int().loc[name, 0])), float(np.exp(res.conf_int().loc[name, 1]))]}
    pick = lambda r: coef[f"C(retriever, Treatment('R5'))[T.{r}]"]   # noqa: E731
    return {"n_rows": len(df), "n_clusters": int(df["cluster_id"].nunique()), "family": "binomial (logit)",
            "cov_struct": "exchangeable", "working_correlation": float(res.cov_struct.dep_params),
            "formula": "EM ~ retriever (ref e5-base-v2) + dataset + reader; clusters = query within dataset",
            "coefficients": coef, "contrasts": {"bge-large_vs_e5": pick("R4"), "qwen3emb_vs_e5": pick("R6")},
            "note": "Pools the v0.2 Qwen-NQ cell (older software stack) with three v0.3 cells (addendum v0.3.1)."}


# ---------------------------------------------------------------- tables
def fmt(x, d=3):
    return "" if x is None else f"{x:.{d}f}"


def write_tables(out_dir: Path, res: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    w = lambda name, header, rows: _csv(out_dir / name, header, rows)  # noqa: E731
    rows = []
    for cell, c in res["cells"].items():
        for r in R + ["CB", "GOLD"]:
            a = c["answers"][r]
            ret = c["retrieval"].get(r, {})
            tr = next((t for t in c["transfer_vs_R1"]["em"] if t["retriever"] == r), None)
            rows.append([cell, c["reader"], c["dataset"], r, NAMES[r], fmt(ret.get("hit@5", {}).get("mean")),
                         fmt(ret.get("ndcg@10", ret.get("both@5", {})).get("mean") if ret else None),
                         fmt(a["em"]["mean"]), fmt(a["em"]["ci_lo"]), fmt(a["em"]["ci_hi"]), fmt(a["f1"]["mean"]),
                         fmt(a["contain"]["mean"]), fmt(tr["ratio"] if tr and tr["estimable"] else None, 2),
                         fmt(tr["ci_lo"] if tr and tr["estimable"] else None, 2), fmt(tr["ci_hi"] if tr and tr["estimable"] else None, 2)])
    w("T_main_cells.csv", ["cell", "reader", "dataset", "cond", "name", "hit@5", "ndcg@10_or_both@5", "em", "em_lo", "em_hi",
                           "f1", "ac", "T_em", "T_lo", "T_hi"], rows)
    rows = []
    for cell, c in res["cells"].items():
        for m in METRICS:
            for t in c["transfer_vs_R1"][m]:
                rows.append([cell, m, t["retriever"], fmt(t["d_hit"], 4), fmt(t["d_em"], 4), fmt(t["ratio"], 3) if t["estimable"] else "",
                             fmt(t["ci_lo"], 3) if t["estimable"] else "", fmt(t["ci_hi"], 3) if t["estimable"] else "", t["reason"]])
    w("T_transfer_ratios.csv", ["cell", "metric", "retriever", "d_hit5", "d_metric", "T", "lo", "hi", "status"], rows)
    k = res["topk"]["em_by_k"]
    w("T_topk_em.csv", ["retriever", "name", "k1", "k3", "k5", "k10", "k1_vs_k5_diff", "p_holm"],
      [[r, NAMES[r]] + [fmt(k[r][kk]["mean"]) for kk in (1, 3, 5, 10)] +
       [fmt(t["diff"]), f"{t['p_holm']:.2g}"] for r, t in zip(R, res["topk"]["k1_vs_k5"])])
    m = res["mechanism"]
    w("T_mechanism.csv", ["cond", "name", "em", "em_lo", "em_hi", "f1", "ac"],
      [[c, NAMES[c], fmt(m["answers"][c]["em"]["mean"]), fmt(m["answers"][c]["em"]["ci_lo"]), fmt(m["answers"][c]["em"]["ci_hi"]),
        fmt(m["answers"][c]["f1"]["mean"]), fmt(m["answers"][c]["contain"]["mean"])] for c in ("MG", "MR", "M1", "M3", "M5")])
    rr = res["reranker"]
    w("T_reranker.csv", ["cond", "name", "hit@5_gen", "ndcg@10_gen", "hit@5_judged", "ndcg@10_judged", "em", "f1", "ac", "latency_ms"],
      [[r, NAMES[r], fmt(rr["retrieval_generation_set"][r]["hit@5"]["mean"]), fmt(rr["retrieval_generation_set"][r]["ndcg@10"]["mean"]),
        fmt(rr["retrieval_all_judged"][r]["hit@5"]), fmt(rr["retrieval_all_judged"][r]["ndcg@10"]),
        fmt(rr["answers"][r]["em"]["mean"]), fmt(rr["answers"][r]["f1"]["mean"]), fmt(rr["answers"][r]["contain"]["mean"]),
        fmt(rr["latency_ms"].get({"X0": "R0", "X5": "R5"}.get(r), {}).get("median_ms"), 1) if r.startswith("X") else ""]
       for r in ("R0", "X0", "R5", "X5")])
    e = res["corpus_hnsw_cost"]
    w("T_full_corpus.csv", ["retriever", "name", "ndcg10_sub_same_emb", "ndcg10_sub_v02", "ndcg10_full", "hit5_sub_same_emb", "hit5_full"],
      [[r, NAMES[r], fmt(e["E9"]["subcorpus_same_embeddings"][r]["ndcg@10"], 4), fmt(e["E9"]["subcorpus_v02_runs"][r]["ndcg@10"], 4),
        fmt(e["E9"]["full_corpus"][r]["ndcg@10"], 4), fmt(e["E9"]["subcorpus_same_embeddings"][r]["hit@5"], 4),
        fmt(e["E9"]["full_corpus"][r]["hit@5"], 4)] for r in e["E9"]["retrievers"]])
    w("T_hnsw.csv", ["retriever", "name", "build_s", "exact_hit5", "exact_ms", "ef64_hit5", "ef64_overlap100", "ef64_overlap5", "ef64_ms",
                     "ef128_hit5", "ef128_overlap100", "ef128_overlap5", "ef128_ms"],
      [[r, NAMES[r], fmt(v["build_seconds"], 0), fmt(v["exact"]["hit@5"], 4), fmt(v["exact"]["latency"]["median_ms"], 1)] +
       [x for ef in (64, 128) for x in (fmt(v[f"ef{ef}"]["hit@5"], 4), fmt(v[f"ef{ef}"]["overlap@100"], 4),
                                        fmt(v[f"ef{ef}"]["overlap@5"], 4), fmt(v[f"ef{ef}"]["latency"]["median_ms"], 2))]
       for r, v in e["E6"].items()])
    w("T_cost.csv", ["retriever", "name", "params_M", "dim", "pps_median", "pps_min", "pps_max", "pps_v02", "qlat_ms_median",
                     "qlat_min", "qlat_max", "qlat_v02", "peak_mib"],
      [[r, NAMES[r], fmt(v["n_params"] / 1e6, 1), v["dim"], fmt(v["passages_per_s"]["median"], 0), fmt(v["passages_per_s"]["min"], 0),
        fmt(v["passages_per_s"]["max"], 0), fmt(v["v02_single"]["passages_per_s"], 0), fmt(v["query_latency_ms"]["median"], 2),
        fmt(v["query_latency_ms"]["min"], 2), fmt(v["query_latency_ms"]["max"], 2), fmt(v["v02_single"]["query_latency_ms"], 2),
        fmt(v["peak_mib"], 0)] for r, v in e["E7"].items() if r.startswith("R") and r != "R0"])
    g = res["gee"]["coefficients"]
    w("T_gee.csv", ["term", "beta", "se_robust", "p", "odds_ratio", "or_lo", "or_hi"],
      [[k, fmt(v["beta"], 4), fmt(v["se_robust"], 4), f"{v['p']:.3g}", fmt(v["odds_ratio"], 3), fmt(v["or_ci95"][0], 3),
        fmt(v["or_ci95"][1], 3)] for k, v in g.items()])
    ja = res.get("judge_agreement")
    if ja:
        w("T_judge_agreement.csv", ["quantity", "value"],
          [["n_items", ja["n_items"]], ["cohen_kappa", fmt(ja["cohen_kappa"])], ["kappa_ci_lo", fmt(ja["kappa_ci95"][0])],
           ["kappa_ci_hi", fmt(ja["kappa_ci95"][1])], ["raw_agreement", fmt(ja["raw_agreement"])],
           ["weighted_agreement", fmt(ja["population_weighted"]["agreement"])], ["decision", ja["decision_band"]]] +
          [[k, v] for k, v in ja["confusion_human_x_judge"].items()])
    w("T_predictions.csv", ["prediction", "scope", "holds", "evidence"], res["predictions"])


def _csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(header)
        wr.writerows(rows)


def predictions(res) -> list[list]:
    rows = []
    for cell, c in res["cells"].items():
        ub = ", ".join(f"{k} {v:.2f}" for k, v in c["P1"]["upper_bounds"].items())
        rows.append(["P1", cell, c["P1"]["holds"], f"{c['P1']['n_reportable']} reportable EM ratios; upper CI bounds: {ub}"])
    for cell, c in res["cells"].items():
        p = c["P2"]["per_encoder"]
        rows.append(["P2", cell, c["P2"]["holds"], "; ".join(f"{k}-R5 EM {v['em_diff_minus_e5']:+.3f} (Holm p {v['p_holm']:.3g})"
                                                            for k, v in p.items())])
    e9 = res["corpus_hnsw_cost"]["E9"]["P3"]
    rows.append(["P3", "NQ R0,R1,R2,R5", e9["holds"], f"Kendall tau = {e9['kendall_tau_primary']:.2f} (v0.2 runs: {e9['kendall_tau_secondary_v02']:.2f})"])
    t = res["topk"]
    rows.append(["P4", "qwen_nq", t["P4"]["holds"], "; ".join(f"{x['retriever']} {x['diff']:+.3f} (Holm p {x['p_holm']:.2g})" for x in t["k1_vs_k5"])])
    m = res["mechanism"]
    rows.append(["P5", "qwen_nq 500", m["P5"]["holds"], f"M1-MR EM {m['P5']['test']['diff']:+.3f}, p = {m['P5']['test']['p']:.2g}"])
    rows.append(["P6", "qwen_nq 500", m["P6"]["holds"], "; ".join(f"{x['a']}-{x['b']} {x['diff']:+.3f} (Holm p {x['p_holm']:.2g})" for x in m["P6"]["tests"])])
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="paper_outputs/v03/statistics")
    ap.add_argument("--tables", default="paper_outputs/v03/tables")
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()
    D = Data(root)
    res = {"cells": {c: analyse_cell(D, c) for c in CELLS}}
    res["topk"] = analyse_topk(D)
    res["mechanism"] = analyse_mechanism(D)
    res["reranker"] = analyse_reranker(D)
    res["corpus_hnsw_cost"] = analyse_corpus_hnsw_cost(D)
    res["gee"] = analyse_gee(D, res["cells"])
    ja = root / args.out / "judge_agreement.json"
    if ja.exists():
        res["judge_agreement"] = json.load(open(ja))
    res["predictions"] = predictions(res)
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "v03_analysis.json").write_text(json.dumps(res, indent=1, default=float))
    write_tables(root / args.tables, res)
    for p in res["predictions"]:
        print(f"{p[0]} {p[1]:16s} {'HOLDS' if p[2] else 'FAILS':6s} {p[3]}")
    print(f"-> {out / 'v03_analysis.json'}, tables in {root / args.tables}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
