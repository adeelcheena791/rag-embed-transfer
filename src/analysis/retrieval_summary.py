"""Small retrieval summary of a session (addendum v0.3, E1), so results can be checked without moving the
large run files: per retriever, Hit@1/5/10, Both@5 (all gold passages in the top 5), nDCG@10 and MRR@10,
on (a) all joined queries and (b) the generation sample (first n_gen of the seeded order), split by type
when the joined file has one. Descriptive only; the confirmatory tests run later in the analysis scripts.

    python -m src.analysis.retrieval_summary --config configs/v03_hp_qwen.yaml --out summary/retrieval_summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.analysis.metrics import load_qrels_tsv, per_query
from src.common.config import load_config, proc_file, qrels_tsv
from src.common.io import read_jsonl, read_lines
from src.common.runs import find_latest_run

RETRIEVERS = ["R0", "R1", "R2", "R3", "R4", "R5", "R6"]


def all_at_k(ranked: list[str], rel: dict, k: int) -> float:
    return float(set(rel) <= set(ranked[:k]))


def summarise(run_file, qrels, qids) -> dict:
    v = per_query(run_file, qrels, qids)
    ranked = {r["qid"]: r["pids"] for r in read_jsonl(run_file)}
    both = np.array([all_at_k(ranked[q], qrels[q], 5) for q in qids])
    out = {m: float(v[m].mean()) for m in ("hit@1", "hit@5", "hit@10", "ndcg@10", "mrr@10")}
    out["both@5"] = float(both.mean())
    out["n"] = len(qids)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus", default="main")
    ap.add_argument("--retrievers", default=",".join(RETRIEVERS), help="e.g. X0,X5 for the reranked runs (E5)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    qrels = load_qrels_tsv(qrels_tsv(cfg))
    joined = {r["qid"]: r for r in read_jsonl(proc_file(cfg, "joined.jsonl"))}
    order = [q for q in read_lines(proc_file(cfg, "query_order.txt")) if q in joined]
    sets = {"judged": sorted(qrels), "joined": sorted(joined), f"gen_first_{cfg['main']['n_gen']}": order[: cfg["main"]["n_gen"]]}
    types = sorted({r.get("type") for r in joined.values() if r.get("type")})
    for t in types:
        sets[f"gen_{t}"] = [q for q in sets[f"gen_first_{cfg['main']['n_gen']}"] if joined[q]["type"] == t]

    res = {"config": args.config, "corpus": args.corpus, "sets": {k: len(v) for k, v in sets.items()}, "retrievers": {}}
    for r in [x for x in args.retrievers.split(",") if x]:
        try:
            run = find_latest_run(cfg, r, args.corpus, want="judged")
        except FileNotFoundError as e:
            res["retrievers"][r] = {"error": str(e)}
            print(f"{r}: MISSING ({e})")
            continue
        res["retrievers"][r] = {"run": run.name, **{k: summarise(run, qrels, v) for k, v in sets.items() if v}}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    first = f"gen_first_{cfg['main']['n_gen']}"
    print(f"{'':4s} {'set':18s} {'hit@5':>7s} {'both@5':>7s} {'ndcg@10':>8s}")
    for r, d in res["retrievers"].items():
        for k in ("judged", "joined", first):
            if k in d:
                print(f"{r:4s} {k:18s} {d[k]['hit@5']:7.3f} {d[k]['both@5']:7.3f} {d[k]['ndcg@10']:8.3f}")
    print(f"summary -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
