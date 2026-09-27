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
    out = [f"=== {cfg['experiment'].upper()} SUMMARY: data + retrieval ==="]
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
                   f"ambiguous {s['n_ambiguous']}) -> {'PASS' if s['passed'] else 'FAIL'} "
                   f"(gate: n_joined >= {s.get('min_joined')})")
    else:
        out.append("P3  not run")
    jd = cpath(cfg, "data_processed") / "nq_join_diagnostics.json"
    if jd.exists():
        d = json.loads(jd.read_text())
        out.append(f"D1a not joined {d['n_not_joined']}: in NQ-open train {d['unmatched_in_nq_open_train_exact']}; "
                   f"max-Jaccard to val {d['unmatched_max_jaccard_to_val_hist']} (median {d['unmatched_max_jaccard_median']}); "
                   f"q-tokens {d['mean_question_tokens']}; gold/q {d['mean_gold_passages']}")
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
    out += summarize_b(cfg)
    return out


def summarize_b(cfg: dict) -> list[str]:
    """Part B lines (P5-P8). Pilot sanity numbers only; never reported as results."""
    raw, proc = cpath(cfg, "results_raw"), cpath(cfg, "results_processed")
    out = []
    runs = list(read_jsonl(raw / "generation_runs.jsonl"))
    if not runs:
        return out
    out.append(f"=== {cfg['experiment'].upper()}: generation + judge" + (" (pilot sanity only - not results) ===" if cfg["experiment"] == "pilot" else " (raw counts; analysis happens offline) ==="))
    for r in runs:
        out.append(f"P5  reader {r['model_id']} @ {r['model_revision'][:10]} via {r.get('backend')} "
                   f"(vllm {r.get('vllm_version')}, load {r.get('load_seconds', 0):.0f}s, eager {r.get('enforce_eager')}) "
                   f"conds {r['conditions']} n={r['n']}")
    gdir = raw / "generations"
    for f in sorted(gdir.glob(f"{cfg['experiment']}__*.jsonl")):
        rows = list(read_jsonl(f))
        empty = sum(not (x.get("output") or "").strip() for x in rows)
        fr = {}
        for x in rows:
            fr[x["finish_reason"]] = fr.get(x["finish_reason"], 0) + 1
        out.append(f"    {f.stem}: n={len(rows)} empty={empty} ({empty / max(1, len(rows)):.1%}; gate <=1%) "
                   f"finish={fr} mean prompt tokens={sum(x['n_prompt_tokens'] for x in rows) / max(1, len(rows)):.0f}")
    sfile = proc / "answers" / f"{cfg['experiment']}__summary.json"
    if sfile.exists():
        tab = json.loads(sfile.read_text())
        out.append("    answer sanity (EM / F1 / contain / unparsed): " + "; ".join(
            f"{k.split('__')[1]} {v['em']:.3f}/{v['f1']:.3f}/{v['contain']:.3f}/{v['rate_unparsed']:.3f}" for k, v in tab.items()))
    for p in sorted(raw.glob("judge_decision_*.json")):
        d = json.loads(p.read_text())
        per = "; ".join(f"{c} fail {v['n_parse_fail']}/{v['n_judged']} supp {v['rate_supported']}" for c, v in d["per_condition"].items())
        out.append(f"P6  judge {d['variant']}: parse failures {d['n_parse_fail']}/{d['n_judged']} = {d['parse_fail_rate']:.3%} "
                   f"(F2 threshold {d['threshold_f2']:.0%}) -> main variant {d.get('f2_main_variant', '-')}; {per}")
    for p in sorted(raw.glob("pilot_power_*.json")):
        d = json.loads(p.read_text())
        pr = d["projections"]
        out.append(f"P7  p_d={d['p_d']:.3f} (n={d['n_pilot_queries']}) -> N_gen={d['n_gen']}; MDD {d['mdd_at_n_gen_and_pilot_p_d']}; "
                   f"S3 {pr['s3_t4_hours']:.2f} T4-h (F4.2 {pr['f4_2_triggered']}), S4 {pr.get('s4_t4_hours', float('nan')):.2f} T4-h "
                   f"(F4.3 {pr.get('f4_3_triggered')}), S3+S4 {pr.get('s3_plus_s4', float('nan')):.2f} (F4.4 {pr.get('f4_4_triggered')})")
    root = cpath(cfg, "results_raw").parents[1]
    locks = [n for n in ("requirements-retrieval.lock", "requirements-generation.lock") if (root / n).exists()]
    out.append(f"P8  lock files: {locks or 'none yet'}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    print("\n".join(summarize(load_config(args.config))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
