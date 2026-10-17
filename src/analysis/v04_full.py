"""v0.4 Phase 3 analysis (design/addendum_v0.4.md): readers on full-corpus NQ retrieval (session S5-NQ).

Reads only committed files: the v0.3 inputs under paper_inputs/ (sub-corpus = "sub") and the S5-NQ outputs under
paper_inputs/v04_s5nq/ (full corpus = "full"; the top-100 run files are read from the committed session zip).

    python -m src.analysis.v04_full        # writes paper_outputs/v04/v04_full_analysis.json and CSV tables

Pre-specified (addendum v0.4):
  P7  for each reader, every reportable full-corpus EM ratio of a dense encoder (R2, R5) against MiniLM-L6 (R1) has an
      upper 95% bound below 1;
  P8  for each reader, the point estimate T_full(R5) lies inside the sub-corpus 95% interval of T_sub(R5); the paired
      difference T_full - T_sub is reported with its bootstrap interval.
  Reporting rule on the full corpus: |dHit@5| > 0.02 and Holm-significant Hit@5 difference over the 6 pairs among
  R0, R1, R2, R5. Bootstrap: B = 10,000, seed 0, one index for every quantity, so sub and full are paired.
Secondary (descriptive): EM and Hit@5 full vs sub (exact McNemar, Holm over 4 per reader); full-corpus ratios against
BM25; query-level transitions on the full corpus; answer strings in retrieved passages; O8 sensitivity of P5.
Anything else in the output is labelled "not_prespecified".
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np

from src.analysis import stats
from src.analysis.v03_analysis import METRICS, Data, jl
from src.common.ir_metrics import hit_at_k, ndcg_at_k

RF = ["R0", "R1", "R2", "R5"]
DENSE = ["R2", "R5"]
READERS = {"qwen": "qwen_nq", "llama": "llama_nq"}
S5 = "paper_inputs/v04_s5nq"
ZIP = f"{S5}/v04_s5nq_results.zip"
STRATA = [("miss_to_hit", 0, 1), ("hit_to_hit", 1, 1), ("hit_to_miss", 1, 0), ("miss_to_miss", 0, 0)]
P8_ADDENDUM = {"qwen": (0.147, 0.356), "llama": (0.053, 0.265)}      # rounded values written in the addendum


def boot_ratio(num, den, idx):
    """Bootstrap replicates of sum(num)/sum(den) with the shared index (den > 0 in every replicate is checked)."""
    out = np.empty(len(idx))
    for i in range(0, len(idx), 1000):
        n, d = num[idx[i:i + 1000]].sum(axis=1), den[idx[i:i + 1000]].sum(axis=1)
        out[i:i + 1000] = n / d
    return out


def boot_ratio_of_means(em_m, em_ref, hit_m, hit_ref, idx):
    be = stats.boot_means(em_m - em_ref, idx)
    bh = stats.boot_means(hit_m - hit_ref, idx)
    with np.errstate(divide="ignore", invalid="ignore"):
        return be / bh


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return None, None
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(c - h), float(c + h)


def load_full_runs(root: Path, qids: list[str]) -> dict:
    """Top-100 lists of the four full-corpus runs, read from the committed session zip."""
    meta = json.load(open(root / S5 / "v04/passage_answers_meta.json"))["runs"]
    out = {}
    with zipfile.ZipFile(root / ZIP) as z:
        for r in RF:
            name = f"results/raw/runs/{meta['full:' + r]}"
            with z.open(name) as f:
                rows = {x["qid"]: x["pids"] for x in map(json.loads, io.TextIOWrapper(f, encoding="utf-8")) if x}
            out[r] = (meta["full:" + r], rows)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="paper_outputs/v04")
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()
    out_dir = root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    D = Data(root)
    q = D.qids("qwen_nq")
    n = len(q)
    idx = stats.resample_index(n)
    qrels = D.nq_qrels
    v03 = json.load(open(root / "paper_outputs/v03/statistics/v03_analysis.json"))
    res: dict = {"design": "design/addendum_v0.4.md", "n_queries": n, "bootstrap": {"B": stats.B, "seed": stats.SEED},
                 "checks": {}}

    # ------------------------------------------------------------ inputs and integrity checks
    runs = load_full_runs(root, q)
    full_hit, full_ndcg, gens_pids = {}, {}, {}
    for r in RF:
        _, run = runs[r]
        full_hit[r] = np.array([hit_at_k(run[x], qrels[x], 5) for x in q])
        full_ndcg[r] = np.array([ndcg_at_k(run[x], qrels[x], 10) for x in q])
    full_ans = {}
    for rd, cell in READERS.items():
        full_ans[rd] = {}
        for r in RF:
            tag = f"v03_nq_{rd}__{r}_k5_A1__full"
            g = {x["qid"]: x for x in jl(root / S5 / f"raw/generations/{tag}.jsonl")}
            a = {x["qid"]: x for x in jl(root / S5 / f"processed/answers/{tag}.jsonl")}
            assert set(g) == set(q) and set(a) == set(q), (tag, len(g), len(a))
            assert all(g[x]["pids"] == runs[r][1][x][:5] for x in q), f"{tag}: reader passages differ from run top 5"
            assert all(g[x]["corpus"] == "full" and g[x]["k"] == 5 for x in q)
            gens_pids.setdefault(r, {x: g[x]["pids"] for x in q})
            assert gens_pids[r] == {x: g[x]["pids"] for x in q}, f"{tag}: readers saw different passages"
            full_ans[rd][r] = {m: np.array([a[x][m] for x in q], float) for m in METRICS}
            full_ans[rd][r]["unparsed"] = np.array([a[x].get("category") == "unparsed" for x in q], float)
    res["checks"]["reader_passages_equal_run_top5"] = True
    # E9 reproduction: the full-corpus runs of addendum v0.3.2 (session S4) were summarised; this session re-encoded.
    e9 = json.load(open(root / "paper_inputs/v03_nq_s4/summary/v03_nq_full_summary.json"))["retrievers"]
    res["checks"]["e9_reproduction"] = {
        r: {"hit@5_now": float(full_hit[r].mean()), "hit@5_e9": e9[r]["gen_first_2255"]["hit@5"],
            "ndcg@10_now": float(full_ndcg[r].mean()), "ndcg@10_e9": e9[r]["gen_first_2255"]["ndcg@10"]} for r in RF}
    # passage_answers.jsonl hit flags must equal the hits computed here (main: v0.3 top-10 files; full: run files)
    pa = jl(root / S5 / "v04/passage_answers.jsonl")
    pa_by = {(x["corpus"], x["retriever"], x["qid"]): x for x in pa}
    sub_hit = {r: D.retrieval("qwen_nq", r)["hit@5"] for r in ["R0", "R1", "R2", "R3", "R4", "R5", "R6"]}
    for r in sub_hit:
        assert all(pa_by[("main", r, x)]["hit"] == sub_hit[r][i] for i, x in enumerate(q)), f"main {r} hit mismatch"
    for r in RF:
        assert all(pa_by[("full", r, x)]["hit"] == full_hit[r][i] for i, x in enumerate(q)), f"full {r} hit mismatch"
    res["checks"]["passage_answers_hits_match"] = True

    # ------------------------------------------------------------ retrieval on the full corpus
    res["retrieval_full"] = {r: {"hit@5": stats.mean_ci(full_hit[r], idx), "ndcg@10": stats.mean_ci(full_ndcg[r], idx)}
                             for r in RF}
    res["retrieval_sub"] = {r: {"hit@5": stats.mean_ci(sub_hit[r], idx)} for r in RF}
    res["pairwise_hit5_full"] = stats.pairwise({r: full_hit[r] for r in RF}, RF, idx, binary=True)
    h5 = {(t["a"], t["b"]): t for t in res["pairwise_hit5_full"]}

    def hp(a, b):
        return (h5.get((a, b)) or h5.get((b, a)))["p_holm"]

    # sub-corpus pairwise Hit@5 over the 21 pairs (as v0.3), for ratios against BM25 on the sub-corpus
    sub_pw = {(t["a"], t["b"]): t for t in v03["cells"]["qwen_nq"]["pairwise_hit5"]}

    def hp_sub(a, b):
        return (sub_pw.get((a, b)) or sub_pw.get((b, a)))["p_holm"]

    res["readers"] = {}
    em_rows, ratio_rows, trans_rows = [], [], []
    for rd, cell in READERS.items():
        sub_ans = {r: D.answers(cell, f"{r}_k5_A1") for r in RF}
        fa = full_ans[rd]
        R = {"answers_full": {r: {**{m: stats.mean_ci(fa[r][m], idx) for m in METRICS},
                                  "rate_unparsed": float(fa[r]["unparsed"].mean())} for r in RF},
             "answers_sub": {r: {m: stats.mean_ci(sub_ans[r][m], idx) for m in METRICS} for r in RF},
             "pairwise_em_full": stats.pairwise({r: fa[r]["em"] for r in RF}, RF, idx, binary=True)}
        # transfer ratios vs MiniLM-L6 on the full corpus (EM primary; F1 and containment as in E8)
        R["transfer_full_vs_R1"] = {m: [stats.transfer_ratio(full_hit[r], full_hit["R1"], fa[r][m], fa["R1"][m], idx,
                                                             hp("R1", r), r).__dict__ for r in ("R0", "R2", "R5")]
                                    for m in METRICS}
        rep = [t for t in R["transfer_full_vs_R1"]["em"] if t["estimable"] and t["retriever"] in DENSE]
        R["P7"] = {"n_reportable_dense": len(rep), "upper_bounds": {t["retriever"]: t["ci_hi"] for t in rep},
                   "holds": bool(rep) and all(t["ci_hi"] < 1 for t in rep),
                   "not_reportable": {t["retriever"]: t["reason"] for t in R["transfer_full_vs_R1"]["em"]
                                      if not t["estimable"] and t["retriever"] in DENSE}}
        # P8 and the paired difference T_full - T_sub, for e5-base-v2 (pre-specified) and bge-small (not pre-specified)
        sub_t = {t["retriever"]: t for t in v03["cells"][cell]["transfer_vs_R1"]["em"]}
        R["T_full_minus_T_sub"] = {}
        for r in DENSE:
            tf = next(t for t in R["transfer_full_vs_R1"]["em"] if t["retriever"] == r)
            ts = sub_t[r]
            bf = boot_ratio_of_means(fa[r]["em"], fa["R1"]["em"], full_hit[r], full_hit["R1"], idx)
            bs = boot_ratio_of_means(sub_ans[r]["em"], sub_ans["R1"]["em"], sub_hit[r], sub_hit["R1"], idx)
            d = bf - bs
            ok = np.isfinite(d)
            lo, hi = stats.ci(d[ok])
            R["T_full_minus_T_sub"][r] = {"T_full": tf["ratio"], "T_sub": ts["ratio"],
                                          "diff": (tf["ratio"] - ts["ratio"]) if tf["ratio"] is not None else None,
                                          "ci_lo": lo, "ci_hi": hi, "n_finite_replicates": int(ok.sum()),
                                          "prespecified": r == "R5"}
            # check: the sub-corpus ratio interval recomputed here equals the committed v0.3 one
            slo, shi = stats.ci(bs[np.isfinite(bs)])
            assert abs(slo - ts["ci_lo"]) < 1e-12 and abs(shi - ts["ci_hi"]) < 1e-12, (rd, r, slo, ts["ci_lo"])
        t5 = R["T_full_minus_T_sub"]["R5"]
        s5 = sub_t["R5"]
        R["P8"] = {"T_full_e5": t5["T_full"], "sub_interval_unrounded": [s5["ci_lo"], s5["ci_hi"]],
                   "sub_interval_addendum": list(P8_ADDENDUM[rd]),
                   "holds": t5["T_full"] is not None and s5["ci_lo"] <= t5["T_full"] <= s5["ci_hi"],
                   "holds_with_addendum_rounding": t5["T_full"] is not None
                   and P8_ADDENDUM[rd][0] <= t5["T_full"] <= P8_ADDENDUM[rd][1]}
        # secondary: full vs sub per retriever (EM: McNemar, Holm over 4)
        fs = []
        for r in RF:
            t = stats.paired_diff(fa[r]["em"], sub_ans[r]["em"], idx, binary=True)
            th = stats.paired_diff(full_hit[r], sub_hit[r], idx, binary=True)
            fs.append({"retriever": r, "em": t, "hit5": th,
                       "f1": stats.paired_diff(fa[r]["f1"], sub_ans[r]["f1"], idx, binary=False),
                       "contain": stats.paired_diff(fa[r]["contain"], sub_ans[r]["contain"], idx, binary=True)})
        for key in ("em", "hit5"):
            for t, p in zip(fs, stats.holm([t[key]["p"] for t in fs])):
                t[key]["p_holm4"] = p
        R["full_minus_sub"] = fs
        # secondary: ratios against BM25 (R0) as baseline, full and sub
        R["transfer_vs_R0"] = {"full": [], "sub": []}
        for r in ("R1", "R2", "R5"):
            R["transfer_vs_R0"]["full"].append(stats.transfer_ratio(full_hit[r], full_hit["R0"], fa[r]["em"], fa["R0"]["em"],
                                                                    idx, hp("R0", r), r).__dict__)
            R["transfer_vs_R0"]["sub"].append(stats.transfer_ratio(sub_hit[r], sub_hit["R0"], sub_ans[r]["em"],
                                                                   sub_ans["R0"]["em"], idx, hp_sub("R0", r), r).__dict__)
        # secondary: query-level transitions vs MiniLM-L6 on the full corpus
        R["transitions_full"] = {}
        h1, e1 = full_hit["R1"].astype(int), fa["R1"]["em"].astype(int)
        for r in DENSE:
            hr, er = full_hit[r].astype(int), fa[r]["em"].astype(int)
            d = (er - e1).astype(float)
            total = d.sum() / n
            rec, tr = 0.0, {}
            for s, a, b in STRATA:
                m = (h1 == a) & (hr == b)
                contrib = d[m].sum() / n
                rec += contrib
                bc = stats.boot_means(d * m, idx)
                lo, hi = stats.ci(bc)
                tr[s] = {"n": int(m.sum()), "share": float(m.mean()), "em_minilm": float(e1[m].mean()) if m.any() else None,
                         "em_encoder": float(er[m].mean()) if m.any() else None,
                         "em_diff_within": float(d[m].mean()) if m.any() else None,
                         "wrong_to_right": int(((e1 == 0) & (er == 1) & m).sum()),
                         "right_to_wrong": int(((e1 == 1) & (er == 0) & m).sum()),
                         "contribution": float(contrib), "ci_lo": lo, "ci_hi": hi}
                trans_rows.append([rd, r, s, tr[s]["n"], f"{tr[s]['share']:.4f}", f"{contrib:+.4f}", f"{lo:+.4f}", f"{hi:+.4f}",
                                   tr[s]["wrong_to_right"], tr[s]["right_to_wrong"], f"{total:+.4f}"])
            assert abs(rec - total) < 1e-12
            tr["total"] = float(total)
            R["transitions_full"][r] = tr
        res["readers"][rd] = R
        for r in RF:
            af, asb = R["answers_full"][r], R["answers_sub"][r]
            t = next(x for x in fs if x["retriever"] == r)
            em_rows.append([rd, r, f"{asb['em']['mean']:.4f}", f"{af['em']['mean']:.4f}", f"{t['em']['diff']:+.4f}",
                            f"{t['em']['ci_lo']:+.4f}", f"{t['em']['ci_hi']:+.4f}", f"{t['em']['p_holm4']:.3g}",
                            f"{res['retrieval_sub'][r]['hit@5']['mean']:.4f}", f"{res['retrieval_full'][r]['hit@5']['mean']:.4f}",
                            f"{t['hit5']['p_holm4']:.3g}", f"{af['f1']['mean']:.4f}", f"{af['contain']['mean']:.4f}"])
        for t in R["transfer_full_vs_R1"]["em"]:
            ts = sub_t[t["retriever"]]
            ratio_rows.append([rd, t["retriever"], "R1", f"{t['d_hit']:+.4f}", f"{t['d_em']:+.4f}",
                               "" if t["ratio"] is None else f"{t['ratio']:.4f}",
                               "" if t["ci_lo"] is None else f"{t['ci_lo']:.4f}", "" if t["ci_hi"] is None else f"{t['ci_hi']:.4f}",
                               f"{ts['ratio']:.4f}", f"{ts['ci_lo']:.4f}", f"{ts['ci_hi']:.4f}", t["reason"]])

    # ------------------------------------------------------------ answer strings in retrieved passages
    ans_rows = []
    res["answer_strings"] = {}
    for corpus, names in (("main", ["R0", "R1", "R2", "R3", "R4", "R5", "R6"]), ("full", RF)):
        for r in names:
            hit = np.array([pa_by[(corpus, r, x)]["hit"] for x in q], float)
            ans = np.array([pa_by[(corpus, r, x)]["ans_in_topk"] for x in q], float)
            miss = 1 - hit
            bm = boot_ratio(ans * miss, miss, idx)
            bh = boot_ratio(ans * hit, hit, idx)
            v = {"hit@5": float(hit.mean()), "ans_in_top5": float(ans.mean()),
                 "miss_with_answer": {"k": int((ans * miss).sum()), "n": int(miss.sum()), "share": float((ans * miss).sum() / miss.sum()),
                                      "ci": list(stats.ci(bm))},
                 "hit_with_answer": {"k": int((ans * hit).sum()), "n": int(hit.sum()), "share": float((ans * hit).sum() / hit.sum()),
                                     "ci": list(stats.ci(bh))}}
            res["answer_strings"][f"{corpus}:{r}"] = v
            ans_rows.append([corpus, r, f"{v['hit@5']:.4f}", f"{v['ans_in_top5']:.4f}", v["miss_with_answer"]["n"],
                             f"{v['miss_with_answer']['share']:.4f}", f"{v['miss_with_answer']['ci'][0]:.4f}",
                             f"{v['miss_with_answer']['ci'][1]:.4f}", v["hit_with_answer"]["n"], f"{v['hit_with_answer']['share']:.4f}",
                             f"{v['hit_with_answer']['ci'][0]:.4f}", f"{v['hit_with_answer']['ci'][1]:.4f}"])
    gold = {x["qid"]: x for x in jl(root / S5 / "v04/gold_answers.jsonl")}
    g1 = np.array([gold[x]["n_gold_with_answer"] >= 1 for x in q], float)
    res["answer_strings"]["gold_rule_check"] = {"share_queries_gold_contains_answer": stats.mean_ci(g1, idx),
                                                "n_gold_mean": float(np.mean([gold[x]["n_gold"] for x in q]))}

    # ------------------------------------------------------------ O8: hard negatives with the answer; P5 on the clean subset
    e12 = {x["qid"]: x for x in jl(root / S5 / "v04/e12_hardneg_answers.jsonl")}
    q5 = q[:500]
    assert list(e12) == q5 or set(e12) == set(q5)
    base = root / "paper_inputs/v03_nq_s4/processed/answers"
    mech = {c: {x["qid"]: x for x in jl(base / f"v03_nq_qwen__{c}_k5_A1__main.jsonl")} for c in ("M1", "MR")}
    gen_m1 = {x["qid"]: x["pids"] for x in jl(root / "paper_inputs/v03_nq_s4/raw/generations/v03_nq_qwen__M1_k5_A1__main.jsonl")}
    assert all(set(e12[x]["hard"]) == set(gen_m1[x]) - set(qrels[x]) for x in q5), "rebuilt hard negatives differ from M1"
    clean = [x for x in q5 if e12[x]["n_hard_with_answer"] == 0]
    nh = np.array([e12[x]["n_hard_with_answer"] for x in q5])
    idx5 = stats.resample_index(len(clean))
    m1 = np.array([mech["M1"][x]["em"] for x in clean], float)
    mr = np.array([mech["MR"][x]["em"] for x in clean], float)
    m1_all = np.array([mech["M1"][x]["em"] for x in q5], float)
    mr_all = np.array([mech["MR"][x]["em"] for x in q5], float)
    has = nh >= 1
    res["O8"] = {"n_e12": len(q5), "n_with_answer_hard_negative": int(has.sum()), "share": float(has.mean()),
                 "wilson": list(wilson(int(has.sum()), len(q5))), "distribution": {int(k): int((nh == k).sum()) for k in range(5)},
                 "P5_original": v03["mechanism"]["P5"]["test"],
                 "P5_clean_subset": {"n": len(clean), **stats.paired_diff(m1, mr, idx5, binary=True),
                                     "em_M1": float(m1.mean()), "em_MR": float(mr.mean())},
                 "P5_answer_hard_subset_not_prespecified": {"n": int(has.sum()), "em_M1": float(m1_all[has].mean()),
                                                            "em_MR": float(mr_all[has].mean())}}
    res["O8"]["P5_clean_subset"]["holds"] = bool(res["O8"]["P5_clean_subset"]["p"] < 0.05 and res["O8"]["P5_clean_subset"]["diff"] < 0)

    res["predictions"] = {f"P7_{rd}": res["readers"][rd]["P7"]["holds"] for rd in READERS}
    res["predictions"].update({f"P8_{rd}": res["readers"][rd]["P8"]["holds"] for rd in READERS})
    (out_dir / "v04_full_analysis.json").write_text(json.dumps(res, indent=1, default=float))
    _csv(out_dir / "v04_em_hit_full_vs_sub.csv", ["reader", "retriever", "em_sub", "em_full", "em_full_minus_sub", "ci_lo", "ci_hi",
                                                  "p_holm4_em", "hit5_sub", "hit5_full", "p_holm4_hit5", "f1_full", "contain_full"], em_rows)
    _csv(out_dir / "v04_transfer_full.csv", ["reader", "retriever", "baseline", "d_hit5_full", "d_em_full", "T_full", "ci_lo", "ci_hi",
                                             "T_sub", "sub_ci_lo", "sub_ci_hi", "status_full"], ratio_rows)
    _csv(out_dir / "v04_transitions_full.csv", ["reader", "encoder", "stratum", "n", "share", "contribution_to_dEM", "ci_lo", "ci_hi",
                                                "wrong_to_right", "right_to_wrong", "dEM_total"], trans_rows)
    _csv(out_dir / "v04_answer_strings.csv", ["corpus", "retriever", "hit5", "ans_in_top5", "n_miss", "miss_with_answer", "ci_lo", "ci_hi",
                                              "n_hit", "hit_with_answer", "hit_ci_lo", "hit_ci_hi"], ans_rows)
    for rd in READERS:
        R = res["readers"][rd]
        print(rd, "P7", R["P7"], "\n   P8", R["P8"], "\n   diff", R["T_full_minus_T_sub"])
    print("O8", {k: v for k, v in res["O8"].items() if k != "P5_original"})
    print("E9 check", res["checks"]["e9_reproduction"])
    return 0


def _csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
