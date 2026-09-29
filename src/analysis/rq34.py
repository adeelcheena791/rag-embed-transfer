"""RQ3 (efficiency) and RQ4 (quality-efficiency Pareto frontier), design v0.2 §5.5; addenda v0.2.3 M2-M4.

All costs are measured on the S1 session (one Kaggle T4 for GPU numbers; that session's 4-vCPU
Xeon 2.0 GHz for all CPU numbers). Cost is reported in seconds / milliseconds / MB, never dollars.

Cost axes for RQ4 (lower is better):
  encode_gpu_s     measured wall time to encode the 300,000-passage corpus on the T4 (BM25: 0 GPU s)
  query_ms_gpu     batch-1 query encoding on GPU (median) + batch-1 exact FAISS search on CPU (median)
                   (BM25: its batch-1 CPU retrieval latency)
  query_ms_cpu     the same with query encoding on CPU (a CPU-only deployment)
  index_mb         FAISS flat fp32 index = 300,000 x dim x 4 bytes (BM25 index size was not measured -> excluded)
Quality: EM (RQ2, n = 2,255), nDCG@10 and Hit@5 (RQ1 joined set, n = 2,255).
A retriever is Pareto-optimal on (cost, quality) if no other retriever has cost <= and quality >= with at least
one strict. Each step along a frontier is annotated with whether its quality gain is significant after Holm
(taken from the RQ1/RQ2 pairwise tests), because a point-estimate frontier can rest on differences the data
cannot resolve.

    python -m src.analysis.rq34 --config configs/main.yaml --rq12 results/statistics/rq12_s3a.json --tag main
"""
from __future__ import annotations

import argparse
import csv
import json

from src.common.config import cpath, load_config, root
from src.common.io import read_jsonl, write_json_atomic

DENSE = ["R1", "R2", "R3", "R4", "R5", "R6"]
ALL = ["R0"] + DENSE
NAMES = {"R0": "BM25", "R1": "MiniLM-L6", "R2": "bge-small", "R3": "bge-base", "R4": "bge-large",
         "R5": "e5-base-v2", "R6": "Qwen3-Emb-0.6B"}
N_CORPUS = 300_000


def efficiency(raw) -> dict:
    tp = {(r["model"], r["device"]): r for r in read_jsonl(raw / "throughput.jsonl")}
    enc = {r["model"]: r for r in read_jsonl(raw / "encode.jsonl") if r.get("corpus") == "main"}
    srch = {r["model"]: r for r in read_jsonl(raw / "search.jsonl")}
    bm = [r for r in read_jsonl(raw / "bm25.jsonl") if r["n_docs"] == N_CORPUS and r.get("latency")][-1]
    out = {"R0": {"params": None, "dim": None, "gpu_peak_mib": 0.0, "encode_gpu_s": 0.0,
                  "index_build_cpu_s": bm["index_seconds"], "throughput_gpu_pps": None, "throughput_cpu_pps": None,
                  "q_enc_gpu_ms": 0.0, "q_enc_cpu_ms": 0.0, "search_ms": bm["latency"]["median_ms"],
                  "search_p95_ms": bm["latency"]["p95_ms"], "index_mb": None}}
    for m in DENSE:
        g, c, e, s = tp[(m, "cuda")], tp[(m, "cpu")], enc[m], srch[m]
        dim = e["dim"]
        out[m] = {"params": g["n_params"], "dim": dim, "gpu_peak_mib": g["peak_mib"], "encode_gpu_s": e["corpus_seconds"],
                  "index_build_cpu_s": None, "throughput_gpu_pps": g["passages_per_s"], "throughput_cpu_pps": c["passages_per_s"],
                  "cpu_encode_300k_h_projected": N_CORPUS / c["passages_per_s"] / 3600,
                  "q_enc_gpu_ms": g["query_latency"]["median_ms"], "q_enc_gpu_p95_ms": g["query_latency"]["p95_ms"],
                  "q_enc_cpu_ms": c["query_latency"]["median_ms"], "q_enc_cpu_p95_ms": c["query_latency"]["p95_ms"],
                  "search_ms": s["latency"]["median_ms"], "search_p95_ms": s["latency"]["p95_ms"],
                  "index_mb": N_CORPUS * dim * 4 / 2**20, "truncation_rate": g.get("truncation_rate_passages")}
    for m, v in out.items():
        v["query_ms_gpu"] = v["q_enc_gpu_ms"] + v["search_ms"]
        v["query_ms_cpu"] = v["q_enc_cpu_ms"] + v["search_ms"]
    return out


def pareto(points: dict[str, tuple[float, float]]) -> list[str]:
    """points: name -> (cost, quality). Returns Pareto-optimal names sorted by cost."""
    opt = []
    for a, (ca, qa) in points.items():
        dominated = any((cb <= ca and qb >= qa) and (cb < ca or qb > qa) for b, (cb, qb) in points.items() if b != a)
        if not dominated:
            opt.append(a)
    return sorted(opt, key=lambda n: (points[n][0], -points[n][1]))


def sig_lookup(pairwise: list[dict]) -> dict:
    d = {}
    for t in pairwise:
        d[(t["a"], t["b"])] = d[(t["b"], t["a"])] = t["significant_holm_0.05"]
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--rq12", required=True)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    eff = efficiency(cpath(cfg, "results_raw"))
    rq = json.load(open(args.rq12))
    quality = {"em": {r: rq["rq2_answers"][r]["em"]["mean"] for r in ALL},
               "ndcg@10": {r: rq["rq1_joined"]["means"][r]["ndcg@10"]["mean"] for r in ALL},
               "hit@5": {r: rq["rq1_joined"]["means"][r]["hit@5"]["mean"] for r in ALL}}
    sig = {"em": sig_lookup(rq["rq2_pairwise_em"]), "ndcg@10": sig_lookup(rq["rq1_joined"]["pairwise"]["ndcg@10"]),
           "hit@5": sig_lookup(rq["rq1_joined"]["pairwise"]["hit@5"])}
    frontiers = {}
    for axis in ("encode_gpu_s", "query_ms_gpu", "query_ms_cpu", "index_mb"):
        for qname in ("em", "ndcg@10", "hit@5"):
            pts = {r: (eff[r][axis], quality[qname][r]) for r in ALL if eff[r][axis] is not None}
            front = pareto(pts)
            steps = [{"from": a, "to": b, "cost_ratio": (pts[b][0] / pts[a][0]) if pts[a][0] else None,
                      "quality_gain": pts[b][1] - pts[a][1], "significant_holm": sig[qname][(a, b)]}
                     for a, b in zip(front, front[1:])]
            frontiers[f"{axis}|{qname}"] = {"frontier": front, "steps": steps,
                                            "dominated": sorted(set(pts) - set(front))}
    out = {"efficiency": eff, "quality": quality, "frontiers": frontiers,
           "notes": ["GPU = one Tesla T4 (S1 session); CPU = 4 vCPU Xeon 2.0 GHz (same session)",
                     "search = batch-1 exact FAISS IndexFlatIP on CPU over 300,000 passages",
                     "reader cost is batched throughput and is NOT added to per-query latency"]}
    write_json_atomic(cpath(cfg, "results_raw").parent / "statistics" / f"rq34_{args.tag}.json", out)
    tdir = root(cfg) / "tables"
    tdir.mkdir(exist_ok=True)
    cols = ["params", "dim", "gpu_peak_mib", "throughput_gpu_pps", "encode_gpu_s", "throughput_cpu_pps", "q_enc_gpu_ms",
            "q_enc_cpu_ms", "search_ms", "query_ms_gpu", "query_ms_cpu", "index_mb"]
    with open(tdir / f"rq3_efficiency_{args.tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["retriever", "name"] + cols)
        for r in ALL:
            w.writerow([r, NAMES[r]] + [("" if eff[r].get(c) is None else round(eff[r][c], 3)) for c in cols])
    print(f"wrote rq34_{args.tag}.json and tables/rq3_efficiency_{args.tag}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
