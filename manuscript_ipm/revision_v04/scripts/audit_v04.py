"""Revision v0.4 audit: recompute transfer ratios, the conditional decomposition, baseline sensitivity, throughput ranges,
GEE with interactions, the random-passage control, HNSW coverage, judge agreement and the sub-corpus composition, from the
committed per-query files under paper_inputs/ and annotation/.

Run from the repository root:
    python3 manuscript_ipm/revision_v04/scripts/audit_v04.py

Reads only committed inputs (no GPU, no network). Writes to manuscript_ipm/revision_v04/outputs/. It never writes to
paper_inputs/, results/ or paper_outputs/. It reuses the data loader and the bootstrap index of src/analysis so that the
resamples are the same as in the original analysis (B = 10,000, seed 0); every formula below is written out here.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.analysis import stats  # noqa: E402
from src.analysis.v03_analysis import CELLS, R, Data, jl  # noqa: E402

OUT = ROOT / "manuscript_ipm/revision_v04/outputs"
ORIG = json.load(open(ROOT / "paper_outputs/v03/statistics/v03_analysis.json"))
LEDGER = {r["claim_id"]: r for r in csv.DictReader(open(ROOT / "manuscript_ipm/records/evidence-ledger.csv", encoding="utf-8"))}
NAME = {"R0": "BM25", "R1": "MiniLM-L6", "R2": "bge-small", "R3": "bge-base", "R4": "bge-large", "R5": "e5-base-v2",
        "R6": "Qwen3-Emb-0.6B"}
SHORT = {"qwen_nq": "qnq", "llama_nq": "lnq", "qwen_hp": "qhp", "llama_hp": "lhp"}


def wcsv(name, header, rows):
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def pct(lo_hi):
    return [float(np.percentile(lo_hi, 2.5)), float(np.percentile(lo_hi, 97.5))]


def boot(x, idx):
    """Bootstrap means of x over resampled queries, in chunks to bound memory."""
    return np.concatenate([x[idx[i:i + 1000]].mean(axis=1) for i in range(0, len(idx), 1000)])


def holm_p(cell, a, b):
    for t in ORIG["cells"][cell]["pairwise_hit5"]:
        if {t["a"], t["b"]} == {a, b}:
            return t["p_holm"]
    raise KeyError((cell, a, b))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    D = Data(ROOT)
    res = {}
    ratio_rows, dec_rows, base_rows = [], [], []
    per_cell = {}
    for cell in CELLS:
        q = D.qids(cell)
        n = len(q)
        idx = stats.resample_index(n)
        ret = {r: D.retrieval(cell, r) for r in R}
        ans = {r: D.answers(cell, f"{r}_k5_A1") for r in R}
        # duplicate / missing query checks
        assert len(set(q)) == n, f"{cell}: duplicate query ids"
        per_cell[cell] = {"n": n, "ret": ret, "ans": ans}
        h = {r: ret[r]["hit@5"] for r in R}
        e = {r: ans[r]["em"] for r in R}
        bh = {r: boot(h[r], idx) for r in R}
        be = {r: boot(e[r], idx) for r in R}
        bhe = {r: boot(h[r] * e[r], idx) for r in R}
        orig_t = {t["retriever"]: t for t in ORIG["cells"][cell]["transfer_vs_R1"]["em"]}
        # ---- 4.1 transfer ratios vs MiniLM-L6 (and 4.3 alternative baselines)
        for base in ("R1", "R0", "R2"):
            for r in R:
                if r == base:
                    continue
                dh, de = h[r].mean() - h[base].mean(), e[r].mean() - e[base].mean()
                p = holm_p(cell, base, r)
                ok = abs(dh) > 0.02 and p < 0.05
                t = de / dh if dh != 0 else float("nan")
                if ok:
                    bt = (be[r] - be[base]) / (bh[r] - bh[base])
                    lo, hi = pct(bt)
                else:
                    lo = hi = float("nan")
                row = [cell, CELLS[cell]["dataset"], CELLS[cell]["reader"], NAME[r], NAME[base], n,
                       f"{e[base].mean():.4f}", f"{e[r].mean():.4f}", f"{h[base].mean():.4f}", f"{h[r].mean():.4f}",
                       f"{de:.4f}", f"{dh:.4f}", f"{t:.4f}" if ok else "", f"{lo:.4f}" if ok else "", f"{hi:.4f}" if ok else "",
                       f"{p:.3g}", "estimable" if ok else ("|dHit@5| <= 2 pp" if abs(dh) <= 0.02 else "Hit@5 difference not Holm-significant")]
                if base == "R1":
                    o = orig_t[r]
                    rep = o["ratio"] if o["estimable"] else None
                    led = LEDGER.get(f"T.{SHORT[cell]}.{r}", {}).get("value", "")
                    diff = "" if rep is None or not ok else f"{t - rep:.2e}"
                    ratio_rows.append(row + [f"{rep:.4f}" if rep is not None else "n.e.", led, diff,
                                             "paper_outputs/v03/statistics/v03_analysis.json cells." + cell + ".transfer_vs_R1.em"])
                else:
                    base_rows.append(row)
        # ---- 4.2 decomposition vs MiniLM-L6, two formulations, with bootstrap shares
        p0 = h["R1"].mean()
        a0 = (h["R1"] * e["R1"]).sum() / h["R1"].sum()
        b0 = ((1 - h["R1"]) * e["R1"]).sum() / (1 - h["R1"]).sum()
        ba0 = bhe["R1"] / bh["R1"]
        bb0 = (be["R1"] - bhe["R1"]) / (1 - bh["R1"])
        for r in R:
            if r == "R1":
                continue
            p1 = h[r].mean()
            a1 = (h[r] * e[r]).sum() / h[r].sum()
            b1 = ((1 - h[r]) * e[r]).sum() / (1 - h[r]).sum()
            dem = e[r].mean() - e["R1"].mean()
            prev = (p1 - p0) * (a0 - b0)
            ca, cb = p1 * (a1 - a0), (1 - p1) * (b1 - b0)
            # symmetric (average of the two orderings)
            s_prev = (p1 - p0) * ((a0 - b0) + (a1 - b1)) / 2
            s_ca = (a1 - a0) * (p0 + p1) / 2
            s_cb = (b1 - b0) * ((1 - p0) + (1 - p1)) / 2
            ident = prev + ca + cb - dem
            ident_s = s_prev + s_ca + s_cb - dem
            assert abs(ident) < 1e-12 and abs(ident_s) < 1e-12, (cell, r, ident, ident_s)
            estimable = orig_t[r]["estimable"]
            share = (ca + cb) / dem if dem != 0 else float("nan")
            share_s = (s_ca + s_cb) / dem if dem != 0 else float("nan")
            # bootstrap of the share (baseline-conditioned), same resamples
            bp1 = bh[r]
            ba1 = bhe[r] / bh[r]
            bb1 = (be[r] - bhe[r]) / (1 - bh[r])
            bdem = be[r] - be["R1"]
            bcond = bp1 * (ba1 - ba0) + (1 - bp1) * (bb1 - bb0)
            with np.errstate(divide="ignore", invalid="ignore"):
                bshare = bcond / bdem
            frac_small = float(np.mean(np.abs(bdem) < 0.005))
            slo, shi = pct(bshare[np.isfinite(bshare)])
            clo, chi = pct(bcond)
            ds_led = LEDGER.get(f"DS.{SHORT[cell]}.{r}", {}).get("value", "")
            dec_rows.append([cell, CELLS[cell]["dataset"], CELLS[cell]["reader"], NAME[r], "MiniLM-L6", n,
                             f"{p0:.4f}", f"{p1:.4f}", f"{a0:.4f}", f"{a1:.4f}", f"{b0:.4f}", f"{b1:.4f}", f"{dem:.4f}",
                             f"{prev:.4f}", f"{ca:.4f}", f"{cb:.4f}", f"{ident:.1e}", f"{share:.3f}", f"{slo:.3f}", f"{shi:.3f}",
                             f"{clo:.4f}", f"{chi:.4f}", f"{frac_small:.3f}",
                             f"{s_prev:.4f}", f"{s_ca:.4f}", f"{s_cb:.4f}", f"{ident_s:.1e}", f"{share_s:.3f}",
                             "yes" if estimable else "no", ds_led])
        res.setdefault("cells", {})[cell] = {"n": n}

    wcsv("transfer_ratio_audit.csv",
         ["cell", "dataset", "reader", "retriever", "baseline", "n_queries", "em_baseline", "em_system", "hit5_baseline", "hit5_system",
          "d_em", "d_hit5", "recomputed_ratio", "ci_lo", "ci_hi", "hit5_p_holm", "status", "reported_ratio_json", "manuscript_value",
          "difference_recomputed_minus_json", "source"], ratio_rows)
    wcsv("transfer_ratio_alt_baselines.csv",
         ["cell", "dataset", "reader", "retriever", "baseline", "n_queries", "em_baseline", "em_system", "hit5_baseline", "hit5_system",
          "d_em", "d_hit5", "ratio", "ci_lo", "ci_hi", "hit5_p_holm", "status"], base_rows)
    wcsv("transfer_decomposition_audit.csv",
         ["cell", "dataset", "reader", "retriever", "baseline", "n_queries", "p0_hit5_base", "p1_hit5_sys", "a0_em_hit_base",
          "a1_em_hit_sys", "b0_em_miss_base", "b1_em_miss_sys", "d_em", "prevalence_term_dp_a0_b0", "hit_term_p1_da",
          "miss_term_1mp1_db", "identity_residual", "share_conditional", "share_ci_lo", "share_ci_hi", "cond_part_ci_lo",
          "cond_part_ci_hi", "frac_boot_abs_dEM_lt_0.5pp", "sym_prevalence", "sym_hit", "sym_miss", "sym_identity_residual",
          "sym_share_conditional", "ratio_estimable", "manuscript_DS_value"], dec_rows)

    # ---- 5.2 throughput repetitions
    rep = [r for r in jl(ROOT / "paper_inputs/v03_nq_s4/raw/throughput.jsonl") if r.get("run_id", "").startswith("20261015")]
    tp = {}
    for m in ["R1", "R2", "R3", "R5", "R4", "R6"]:
        v = sorted(r["passages_per_s"] for r in rep if r["model"] == m)
        tp[m] = v
    overl = []
    ms = list(tp)
    for i, a in enumerate(ms):
        for b in ms[i + 1:]:
            ov = not (max(tp[a]) < min(tp[b]) or min(tp[a]) > max(tp[b]))
            overl.append([NAME[a], NAME[b], f"{min(tp[a]):.1f}-{max(tp[a]):.1f}", f"{min(tp[b]):.1f}-{max(tp[b]):.1f}", ov])
    wcsv("throughput_ranges.csv", ["encoder_a", "encoder_b", "range_a_pps", "range_b_pps", "ranges_overlap"], overl)
    res["throughput"] = {NAME[m]: v for m, v in tp.items()}
    res["throughput_R3_over_R5_median"] = float(np.median(tp["R3"]) / np.median(tp["R5"]))

    # ---- 6 GEE: original, and with retriever x dataset (+ reader) interactions
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    rows = []
    for cell in CELLS:
        q = D.qids(cell)
        for r in R:
            ee = per_cell[cell]["ans"][r]["em"]
            rows += [{"em": ee[i], "retriever": r, "dataset": CELLS[cell]["dataset"], "reader": CELLS[cell]["reader"],
                      "cluster": f"{CELLS[cell]['dataset']}:{x}"} for i, x in enumerate(q)]
    df = pd.DataFrame(rows)
    df["cluster_id"] = df["cluster"].astype("category").cat.codes
    fam, cov = sm.families.Binomial(), sm.cov_struct.Exchangeable()
    ret_t = "C(retriever, Treatment('R5'))"
    fits = {}
    for name, f in {"A_main_effects": f"em ~ {ret_t} + C(dataset) + C(reader)",
                    "B_retr_x_dataset": f"em ~ {ret_t} * C(dataset) + C(reader)",
                    "C_retr_x_dataset_reader": f"em ~ {ret_t} * C(dataset) + {ret_t} * C(reader)",
                    "D_saturated_cells": f"em ~ {ret_t} * C(dataset) * C(reader)"}.items():
        m = smf.gee(f, groups="cluster_id", data=df, family=fam, cov_struct=sm.cov_struct.Exchangeable())
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fr = m.fit()
        fits[name] = {"formula": f, "fit": fr, "warnings": sorted({str(x.message)[:120] for x in w})}
    gee = {"n_rows": len(df), "n_clusters": int(df["cluster_id"].nunique()), "models": {}}

    def wald(fr, pat):
        names = [n for n in fr.params.index if pat(n)]
        L = np.zeros((len(names), len(fr.params)))
        for i, nm in enumerate(names):
            L[i, list(fr.params.index).index(nm)] = 1
        wt = fr.wald_test(L, scalar=True)
        return {"terms": len(names), "chi2": float(wt.statistic), "df": len(names), "p": float(wt.pvalue)}

    for name, d in fits.items():
        fr = d["fit"]
        info = {"formula": d["formula"], "working_corr": float(fr.cov_struct.dep_params), "warnings": d["warnings"],
                "converged": bool(getattr(fr, "converged", True))}
        if name == "A_main_effects":
            for r in ("R4", "R6"):
                k = f"{ret_t}[T.{r}]"
                ci = fr.conf_int().loc[k]
                info[f"OR_{NAME[r]}_vs_e5"] = [float(np.exp(fr.params[k])), float(np.exp(ci[0])), float(np.exp(ci[1]))]
            orig = ORIG["gee"]["contrasts"]
            info["matches_original"] = all(abs(info[f"OR_{NAME[r]}_vs_e5"][0] - orig[c]["odds_ratio"]) < 1e-9
                                           for r, c in (("R4", "bge-large_vs_e5"), ("R6", "qwen3emb_vs_e5")))
        if name in ("B_retr_x_dataset", "C_retr_x_dataset_reader"):
            info["wald_retr_x_dataset"] = wald(fr, lambda n: n.startswith(ret_t) and ":C(dataset)" in n)
        if name == "C_retr_x_dataset_reader":
            info["wald_retr_x_reader"] = wald(fr, lambda n: n.startswith(ret_t) and ":C(reader)" in n)
        if name == "B_retr_x_dataset":
            cov_m = fr.cov_params()
            for r in ("R4", "R6"):
                k = f"{ret_t}[T.{r}]"
                kint = f"{ret_t}[T.{r}]:C(dataset)[T.NQ]"
                for ds, terms in (("HotpotQA", [k]), ("NQ", [k, kint])):
                    b = float(sum(fr.params[t] for t in terms))
                    var = float(sum(cov_m.loc[t1, t2] for t1 in terms for t2 in terms))
                    se = math.sqrt(var)
                    info[f"OR_{NAME[r]}_vs_e5_{ds}"] = [math.exp(b), math.exp(b - 1.96 * se), math.exp(b + 1.96 * se)]
        gee["models"][name] = info
    # stratified per-cell EM differences vs e5 (descriptive, from the original pairwise tests)
    strat = {}
    for cell in CELLS:
        em_pairs = {(t["a"], t["b"]): t for t in ORIG["cells"][cell]["pairwise_em"]}
        for r in ("R4", "R6"):
            t = em_pairs.get(("R5", r)) or em_pairs.get((r, "R5"))
            sign = 1 if (t["a"], t["b"]) == (r, "R5") else -1
            strat[f"{cell}:{NAME[r]}-e5"] = {"diff": sign * t["diff"],
                                             "ci": sorted([sign * t["ci_lo"], sign * t["ci_hi"]]), "p_holm": t["p_holm"]}
    gee["stratified_mcnemar"] = strat
    res["gee"] = gee

    # ---- 7 HNSW: what was measured
    hn = jl(ROOT / "paper_inputs/v03_nq_s4/raw/hnsw.jsonl")
    keys = sorted(set().union(*[set(x) for x in hn]))
    hnsw_answer_files = sorted(str(p.relative_to(ROOT)) for p in ROOT.glob("paper_inputs/**/*hnsw*") if "answers" in str(p))
    res["hnsw"] = {"record_keys": keys, "metrics_ef128": sorted(hn[0]["hnsw_ef128"].keys()), "reader_answer_files": hnsw_answer_files,
                   "retrievers": [x["model"] for x in hn]}

    # ---- 8 random-passage control
    m = ORIG["mechanism"]
    g = m["gold_only_vs_MR"]
    res["random_passages"] = {"MR_minus_MG": {"diff": -g["diff"], "ci_lo": -g["ci_hi"], "ci_hi": -g["ci_lo"], "p": g["p"],
                                              "n_MR_correct_only": g["n01"], "n_MG_correct_only": g["n10"]},
                              "MG_minus_MR": g, "n": m["n_queries"],
                              "M1_minus_MR": m["P5"]["test"]}

    # ---- 11 judge agreement recomputed from the label files
    lab = {x["item_id"]: x["label"] for x in json.load(open(ROOT / "annotation/judge_validation_labels.json"))["labels"]}
    key = json.load(open(ROOT / "annotation/sample_v1_key.json"))
    hj = [(lab[k["item_id"]], k["judge_verdict"], 1 / k["inclusion_prob"]) for k in key]
    nn = len(hj)
    po = sum(a == b for a, b, _ in hj) / nn
    ph, pj = sum(a == "SUPPORTED" for a, _, _ in hj) / nn, sum(b == "SUPPORTED" for _, b, _ in hj) / nn
    pe = ph * pj + (1 - ph) * (1 - pj)
    kappa = (po - pe) / (1 - pe)
    W = sum(w for *_, w in hj)
    res["judge"] = {"n": nn, "kappa": kappa, "raw_agreement": po, "human_supported_rate": ph, "judge_supported_rate": pj,
                    "weighted_agreement": sum(w for a, b, w in hj if a == b) / W,
                    "matches_original": abs(kappa - ORIG["judge_agreement"]["cohen_kappa"]) < 1e-12 if "judge_agreement" in ORIG else None}

    # ---- 9 sub-corpus composition (from the build records)
    sc_hp = jl(ROOT / "paper_inputs/v03_hp/raw/subcorpus.jsonl")
    res["subcorpus"] = {"hp_records": sc_hp, "E9": ORIG["corpus_hnsw_cost"]["E9"]["drop_full_minus_sub"]}

    (OUT / "audit_results.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k not in ("subcorpus",)}, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
