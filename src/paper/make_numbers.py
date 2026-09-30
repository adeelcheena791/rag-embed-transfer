"""Generate paper/latex/numbers.tex: every number the prose uses, as \\NUM{key} / \\CI{key} macros.

Sources (single source of truth): results/statistics/rq12_main.json, rq34_main.json, paper/derived_stats.json,
data/processed/nq_join_stats.json, results/raw/{subcorpus,bm25}.jsonl, configs/main.yaml, annotation sample,
and paper/records/verified_records.json (pilot facts transcribed from the dated addenda, with sources).
An undefined key stops the LaTeX build, so no number can silently fall back to hand-typed text.

    python -m src.paper.make_numbers
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

R = ["R0", "R1", "R2", "R3", "R4", "R5", "R6"]


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def spp(x, d=1):
    """signed percentage points, typographic minus"""
    v = f"{100 * x:+.{d}f}"
    return v.replace("-", "\\ensuremath{-}").replace("+", "\\ensuremath{+}")


def num(x, d=1):
    s = f"{x:.{d}f}"
    return s.replace("-", "\\ensuremath{-}")


def grp(x):
    return f"{x:,.0f}"


def pval(x):
    if x >= 0.995:
        return "1"
    if x >= 0.001:
        return f"{x:.2g}"
    m, e = f"{x:.1e}".split("e")
    return f"\\ensuremath{{{m}{{\\times}}10^{{{int(e)}}}}}"


def main() -> int:
    root = Path(".")
    rq12 = json.load(open(root / "results/statistics/rq12_main.json"))
    rq34 = json.load(open(root / "results/statistics/rq34_main.json"))
    der = json.load(open(root / "paper/derived_stats.json"))
    js = json.load(open(root / "data/processed/nq_join_stats.json"))
    sub = jl(root / "results/raw/subcorpus.jsonl")[-1]
    full_n = max(r.get("n_docs", 0) for r in jl(root / "results/raw/bm25.jsonl"))
    cfg = yaml.safe_load(open(root / "configs/main.yaml"))
    rec = json.load(open(root / "paper/records/verified_records.json"))
    n_human = len(json.load(open(root / "annotation/sample_v1_blind.json")))
    eff, m1, ans, dec = rq34["efficiency"], rq12["rq1_joined"]["means"], rq12["rq2_answers"], rq12["rq2_decomposition"]
    N, C = {}, {}

    def put(k, v):
        assert k not in N, k
        N[k] = v

    def putci(k, lo, hi, f=pct):
        C[k] = f"[{f(lo)}, {f(hi)}]"

    # ---- data and design
    put("n.judged", grp(js["n_beir_judged"])); put("n.joined", grp(js["n_joined"]))
    put("n.unjoined", grp(js["n_unmatched"])); put("join.rate", pct(js["join_rate"]))
    put("join.rate.frac", f"{js['join_rate']:.3f}"); put("join.gate.old", "0.70")
    put("join.gate.new", grp(cfg["join"]["min_joined"]))
    put("sub.size", grp(sub["size"])); put("sub.gold", grp(sub["n_gold"])); put("sub.bm25", grp(sub["n_bm25_added"]))
    put("sub.depth", str(sub["bm25_depth"])); put("sub.random", grp(sub["n_random"]))
    put("corpus.full", grp(full_n)); put("sub.frac", f"{100 * sub['size'] / full_n:.0f}")
    fa = der["faithfulness_UNVALIDATED"]
    put("n.judge", str(fa["n"])); put("n.judge.gold", str(fa["n_gold"])); put("n.judgements", grp(fa["n_judgements"]))
    put("judge.missing", fa["gold_missing"][0]); put("n.parsefail", str(fa["n_unparsed"]))
    put("n.idk", str(der["abstention_A3"]["R0"]["n"])); put("n.det", "100"); put("n.human", str(n_human))
    put("maxlen", str(cfg["encode"]["max_seq_length"])); put("reader.maxtok", str(cfg["generation"]["max_tokens"]))
    put("judge.maxtok", str(cfg["judge"]["max_tokens"])); put("judge.fewshot.pct", f"{100 * cfg['judge']['few_shot_threshold']:.0f}")
    put("bm25.k1", str(cfg["bm25"]["k1"])); put("bm25.b", str(cfg["bm25"]["b"]))
    put("boot.B", "10,000"); put("boot.seed", "0"); put("pairs", "21"); put("dense.pairs", "15"); put("k.main", "5")
    put("ratio.minhit", "2")
    # pilot / records
    put("pilot.pd", f"{rec['pilot_discordant']['value'] / rec['pilot_discordant']['of']:.3f}")
    put("pilot.disc", str(rec["pilot_discordant"]["value"])); put("pilot.n", str(rec["pilot_discordant"]["of"]))
    import sys
    sys.path.insert(0, "design")
    from power_mcnemar import mdd
    put("mdd", f"{100 * mdd(js['n_joined'], rec['pilot_discordant']['value'] / rec['pilot_discordant']['of'], 0.05 / 21):.1f}")
    put("scifact.maxdiff", f"{rec['scifact_max_abs_diff_model_card']['value']:.4f}")
    put("scifact.qwen.ours", f"{rec['scifact_qwen3_ours']['value']:.3f}")
    put("scifact.qwen.mteb", f"{rec['scifact_qwen3_mteb']['value']:.3f}")
    put("scifact.qwen.diff", f"{rec['scifact_qwen3_ours']['value'] - rec['scifact_qwen3_mteb']['value']:.3f}")
    put("d1a.hi", str(rec["d1a_unjoined_jaccard_ge_0.9"]["value"]))
    put("cpu.est", str(rec["cpu_benchmark_estimate_h"]["value"])); put("cpu.meas", rec["cpu_benchmark_measured"]["value"].replace(" h ", "\\,h~").replace(" min", "\\,min"))
    put("tp.lo", num(rec["gpu_throughput_pilot_vs_main_pct"]["lo"])); put("tp.hi", "\\ensuremath{+}" + num(rec["gpu_throughput_pilot_vs_main_pct"]["hi"]))

    # ---- models / cost
    for r in R[1:]:
        e = eff[r]
        put(f"params.{r}", f"{e['params'] / 1e6:.1f}M"); put(f"dim.{r}", str(e["dim"]))
        put(f"enc.{r}", grp(e["encode_gpu_s"])); put(f"mem.{r}", grp(e["gpu_peak_mib"]))
        put(f"cpupps.{r}", f"{e['throughput_cpu_pps']:.2f}" if e["throughput_cpu_pps"] < 1 else f"{e['throughput_cpu_pps']:.1f}")
        put(f"qcpu.{r}", f"{e['q_enc_cpu_ms']:.1f}"); put(f"qgpu.{r}", f"{e['q_enc_gpu_ms']:.1f}")
        put(f"search.{r}", f"{e['search_ms']:.0f}"); put(f"totcpu.{r}", f"{e['query_ms_cpu']:.1f}")
        put(f"index.{r}", grp(e["index_mb"]))
    put("search.R0", f"{eff['R0']['search_ms']:.1f}")
    put("enc.ratio.R4", f"{eff['R4']['encode_gpu_s'] / eff['R5']['encode_gpu_s']:.1f}")
    put("enc.ratio.R6", f"{eff['R6']['encode_gpu_s'] / eff['R5']['encode_gpu_s']:.1f}")
    put("enc.range", f"{eff['R6']['encode_gpu_s'] / eff['R1']['encode_gpu_s']:.0f}")
    put("enc.gap.R3R5", f"{100 * (eff['R5']['encode_gpu_s'] - eff['R3']['encode_gpu_s']) / eff['R5']['encode_gpu_s']:.1f}")
    put("tie.pct", f"{100 * der['pareto_tie5_EXPLORATORY']['tie']:.0f}")

    # ---- RQ1
    for r in R:
        for m in ("hit@1", "hit@5", "hit@10", "ndcg@10"):
            put(f"{m}.{r}", pct(m1[r][m]["mean"])); putci(f"{m}.{r}", m1[r][m]["ci_lo"], m1[r][m]["ci_hi"])
    for met in ("hit@5", "ndcg@10"):
        for t in rq12["rq1_joined"]["pairwise"][met]:
            for x, y, sg in ((t["a"], t["b"], 1), (t["b"], t["a"], -1)):
                k = f"d.{met}.{x}.{y}"
                lo, hi = sorted((sg * t["ci_lo"], sg * t["ci_hi"]))
                put(k, spp(sg * t["diff"])); putci(k, lo, hi, lambda v: num(100 * v)); put("ph." + k[2:], pval(t["p_holm"]))
        put(f"nsig.{met}", str(sum(t["significant_holm_0.05"] for t in rq12["rq1_joined"]["pairwise"][met])))
    diffs = [100 * v["hit@5"]["diff"] for v in rq12["d1b"]["per_retriever"].values()]
    put("d1b.lo", num(min(diffs))); put("d1b.hi", "\\ensuremath{+}" + num(max(diffs)))

    # ---- RQ2
    for c in list(ans):
        for m in ("em", "f1"):
            put(f"{m}.{c}", pct(ans[c][m]["mean"])); putci(f"{m}.{c}", ans[c][m]["ci_lo"], ans[c][m]["ci_hi"])
    for t in rq12["rq2_pairwise_em"]:
        for x, y, sg in ((t["a"], t["b"], 1), (t["b"], t["a"], -1)):
            k = f"d.em.{x}.{y}"
            lo, hi = sorted((sg * t["ci_lo"], sg * t["ci_hi"]))
            put(k, spp(sg * t["diff"])); putci(k, lo, hi, lambda v: num(100 * v)); put("ph." + k[2:], pval(t["p_holm"]))
    put("ph.em.bm25.max", pval(max(t["p_holm"] for t in rq12["rq2_pairwise_em"] if t["a"] == "R0")))
    dense_em = [t for t in rq12["rq2_pairwise_em"] if t["a"] != "R0"]
    put("nsig.em.dense", str(sum(t["significant_holm_0.05"] for t in dense_em)))
    put("nns.em.dense", str(sum(not t["significant_holm_0.05"] for t in dense_em)))
    put("max.ns.em.dense", num(max(abs(100 * t["diff"]) for t in dense_em if not t["significant_holm_0.05"])))
    bm = [-100 * t["diff"] for t in rq12["rq2_pairwise_em"] if t["a"] == "R0"]
    put("em.gain.bm25.lo", num(min(bm))); put("em.gain.bm25.hi", num(max(bm)))
    put("nsig.f1", str(sum(t["significant_holm_0.05"] for t in rq12["rq2_pairwise_f1"])))
    for r in R:
        dc = dec[r]
        put(f"pem.hit.{r}", pct(dc["p_correct_given_hit"])); putci(f"pem.hit.{r}", *dc["p_hit_ci"])
        put(f"pem.miss.{r}", pct(dc["p_correct_given_miss"])); putci(f"pem.miss.{r}", *dc["p_miss_ci"])
    dense = R[1:]
    ph = [100 * dec[r]["p_correct_given_hit"] for r in dense]; pm = [100 * dec[r]["p_correct_given_miss"] for r in dense]
    gap = [100 * (dec[r]["p_correct_given_hit"] - dec[r]["p_correct_given_miss"]) for r in dense]
    put("pem.hit.lo", num(min(ph))); put("pem.hit.hi", num(max(ph))); put("pem.miss.lo", num(min(pm))); put("pem.miss.hi", num(max(pm)))
    put("pem.gap.lo", num(min(gap))); put("pem.gap.hi", num(max(gap)))
    put("pem.hit.strong.lo", num(min(100 * dec[r]["p_correct_given_hit"] for r in dense if r != "R1")))
    put("pem.hit.strong.hi", num(max(100 * dec[r]["p_correct_given_hit"] for r in dense if r != "R1")))
    put("T.ref", f"{dec['R1']['p_correct_given_hit'] - dec['R1']['p_correct_given_miss']:.2f}")
    for t in rq12["rq2_transfer_ratio_vs_R1"]:
        r = t["retriever"]
        put(f"T.{r}", f"{t['ratio']:.2f}"); putci(f"T.{r}", t["ci_lo"], t["ci_hi"], lambda x: num(x, 2))
        put(f"dhit.{r}", spp(t["d_hit"])); put(f"dem.{r}", spp(t["d_em"]))
    tr = [t["ratio"] for t in rq12["rq2_transfer_ratio_vs_R1"] if t["retriever"] != "R0"]
    put("T.lo", f"{min(tr):.2f}"); put("T.hi", f"{max(tr):.2f}")
    put("T.lo.pct", f"{100 * min(tr):.0f}"); put("T.hi.pct", f"{100 * max(tr):.0f}")
    put("rho", f"{rq12['rq2_spearman_hit5_em']['rho']:.1f}"); put("rho.n", str(rq12["rq2_spearman_hit5_em"]["n_points"]))

    # ---- ablations
    pts = {(p["retriever"], p["k"]): p for p in der["topk_ablation"]["points"]}
    for (r, k), p in pts.items():
        put(f"emk.{r}.{k}", pct(p["em"])); putci(f"emk.{r}.{k}", p["ci_lo"], p["ci_hi"]); put(f"tok.{r}.{k}", grp(p["mean_prompt_tokens"]))
    st = der["topk_ablation"]["steps"]
    put("nsig.steps", str(sum(s["significant_holm_0.05"] for s in st))); put("n.steps", str(len(st)))
    for s in st:
        k = f"step.{s['retriever']}.{s['from_k']}.{s['to_k']}"
        put(k, spp(s["diff"])); putci(k, s["ci_lo"], s["ci_hi"], lambda x: num(100 * x)); put("ph." + k, pval(s["p_holm"]))
    ex = der["topk_ablation"]["exploratory_R4k1_vs_R0k10"]
    put("ex.d", spp(ex["diff"])); putci("ex.d", ex["ci_lo"], ex["ci_hi"], lambda x: num(100 * x)); put("ex.p", pval(ex["p"]))
    for row in der["transfer_by_k_EXPLORATORY"]["rows"]:
        k = f"Tk.{row['retriever']}.{row['k']}"
        put(k, f"{row['ratio']:.2f}"); putci(k, row["ci_lo"], row["ci_hi"], lambda x: num(x, 2))
    idk = der["abstention_A3"]
    for c, v in idk.items():
        put(f"idk.{c}", pct(v["idk_rate"]))
        if c != "GOLD":
            put(f"idk.hit.{c}", pct(v["idk_given_hit"])); put(f"idk.miss.{c}", pct(v["idk_given_miss"]))
    hits = [100 * idk[c]["idk_given_hit"] for c in ("R0", "R1", "R4")]
    put("idk.hit.lo", num(min(hits))); put("idk.hit.hi", num(max(hits)))
    drops = [100 * (idk[c]["em_A1"] - idk[c]["em_A3"]) for c in ("R0", "R1", "R4")]
    put("idk.drop.lo", num(min(drops))); put("idk.drop.hi", num(max(drops)))
    put("idk.pmin", pval(min(idk[c]["em_A3_minus_A1_p"] for c in ("R0", "R1", "R4"))))

    # ---- faithfulness (unvalidated)
    for c, v in fa["means"].items():
        put(f"supp.{c}", pct(v["mean"])); putci(f"supp.{c}", v["ci_lo"], v["ci_hi"])

    lines = ["% GENERATED by src/paper/make_numbers.py -- do not edit by hand",
             "\\makeatletter",
             "\\newcommand{\\NUM}[1]{\\ifcsname num@#1\\endcsname\\csname num@#1\\endcsname\\else\\errmessage{Undefined number key #1}\\fi}",
             "\\newcommand{\\CI}[1]{\\ifcsname ci@#1\\endcsname\\csname ci@#1\\endcsname\\else\\errmessage{Undefined CI key #1}\\fi}"]
    lines += [f"\\expandafter\\def\\csname num@{k}\\endcsname{{{v}}}" for k, v in N.items()]
    lines += [f"\\expandafter\\def\\csname ci@{k}\\endcsname{{{v}}}" for k, v in C.items()]
    lines.append("\\makeatother")
    out = root / "paper/latex/numbers.tex"
    out.write_text("\n".join(lines) + "\n")
    (root / "paper/latex/numbers_index.json").write_text(json.dumps({"NUM": N, "CI": C}, indent=1))
    print(f"wrote {len(N)} numbers and {len(C)} intervals to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
