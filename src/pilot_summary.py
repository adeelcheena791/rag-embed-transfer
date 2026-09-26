"""One-screen summary of pilot Part A for copy-paste back to the analyst.

Reads ONLY files under results/raw (and data/processed/nq_join_stats.json); computes nothing new.

    python -m src.pilot_summary --config configs/pilot.yaml
"""
from __future__ import annotations

import argparse
import json

from src.common.config import cpath, load_config
from src.common.io import read_jsonl


def _last(path, **match):
    rows = [r for r in read_jsonl(path) if all(r.get(k) == v for k, v in match.items())]
    return rows[-1] if rows else None


def summarize(cfg: dict) -> list[str]:
    raw = cpath(cfg, "results_raw")
    out = ["=== PILOT PART A SUMMARY ==="]
    dl = list(read_jsonl(raw / "download.jsonl"))
    if dl:
        latest = {r["name"]: r for r in dl}
        out.append("P1  sha match: " + ", ".join(f"{k}={'OK' if v.get('sha_match') else 'FAIL'}" for k, v in latest.items()))
    else:
        out.append("P1  not run")
    js = cpath(cfg, "data_processed") / "nq_join_stats.json"
    if js.exists():
        s = json.loads(js.read_text())
        out.append(f"P3  join rate {s['join_rate']:.3f} ({s['n_joined']}/{s['n_beir_judged']}; unmatched {s['n_unmatched']}, "
                   f"ambiguous {s['n_ambiguous']}) -> {'PASS' if s['passed'] else 'FAIL'}")
    else:
        out.append("P3  not run")
    b = _last(raw / "bm25.jsonl", n_docs=None) or (list(read_jsonl(raw / "bm25.jsonl")) or [None])[-1]
    if b:
        out.append(f"P3b BM25 {b['n_docs']} docs, query_set {b['query_set']}, index {b['index_seconds']:.0f}s, "
                   f"search {b['search_seconds_total']:.0f}s")
    sc = list(read_jsonl(raw / "subcorpus.jsonl"))
    if sc:
        s = sc[-1]
        out.append(f"    sub-corpus '{s['name']}': {s['size']} passages (gold {s['n_gold']}, BM25 depth {s['bm25_depth']} "
                   f"-> +{s['n_bm25_added']}, random {s['n_random']})")
    san = list(read_jsonl(raw / "sanity_scifact.jsonl"))
    if san:
        latest = {r["model"]: r for r in san}
        out.append("P2  SciFact nDCG@10: " + "; ".join(
            f"{m} {r.get('ndcg@10', float('nan')):.4f} (ref {r.get('reference')}) {r['verdict']}" for m, r in latest.items()))
    enc = [r for r in read_jsonl(raw / "encode.jsonl")]
    smoke = [r for r in enc if r.get("smoke_test")]
    fails = list(read_jsonl(raw / "failures.jsonl"))
    r6_fail = [f for f in fails if f.get("model") == "R6"]
    if smoke or r6_fail:
        ok = any(r["model"] == "R6" for r in smoke)
        out.append(f"P4c R6 smoke test on GPU: {'OK' if ok else 'FAILED'}"
                   + ("" if ok else f" -> {r6_fail[-1]['error'][:200]}"))
    decisions = sorted(raw.glob("throughput_decision_*.json"))
    for p in decisions:
        d = json.loads(p.read_text())
        if d.get("device") == "cpu":
            out.append(f"F3  R6 CPU fallback: {d.get('r6_cpu')}")
        else:
            per = ", ".join(f"{k} {v:.2f}h" for k, v in d["projected_t4_hours_per_model"].items())
            out.append(f"P4  projected full-corpus encoding {d['projected_t4_hours_total']:.2f} T4-h (cap {d['cap_t4_hours']}) "
                       f"-> {d['decision'].upper()}; per model: {per}; dropped {d['dropped_models']}; "
                       f"rerun_at_common_batch {d['rerun_at_common_batch']}")
    tp = list(read_jsonl(raw / "throughput.jsonl"))
    for r in {x["model"]: x for x in tp if x.get("device") != "cpu"}.values():
        if r.get("dropped"):
            out.append(f"    {r['model']}: DROPPED")
            continue
        out.append(f"    {r['model']}: {r['passages_per_s']:.0f} p/s, bs {r['batch_size_used']}, peak {r.get('peak_mib')} MiB, "
                   f"trunc {r['truncation_rate_passages']:.3f}, q-lat {r['query_latency']['median_ms']:.1f} ms, "
                   f"attn {r.get('attn_implementation')}, params {r.get('n_params')}")
    se = [r for r in read_jsonl(raw / "search.jsonl")]
    for r in se:
        out.append(f"P4b search {r['model']}: {r['query_set']} in {r['search_seconds_total']:.1f}s")
    non_r6 = [f for f in fails if f.get("model") != "R6"]
    out.append(f"failures.jsonl: {len(fails)} records" + (f" (non-R6: {len(non_r6)})" if non_r6 else ""))
    for f in fails[-5:]:
        out.append(f"    {f.get('model')} {f.get('stage')} bs={f.get('batch_size')}: {str(f.get('error'))[:160]}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    print("\n".join(summarize(load_config(args.config))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
