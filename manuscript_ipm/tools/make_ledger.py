"""Gate 2/6: evidence ledger + LaTeX number macros from one source.

Reads only committed artifacts (v03_analysis.json, join/sub-corpus stats) and writes
  manuscript_ipm/latex/numbers.tex          \\ev{<claim id>} expands to the formatted value
  manuscript_ipm/records/evidence-ledger.csv  one row per claim id (schema of the skill's validate_ledger.py)
  manuscript_ipm/records/evidence-ledger.md   the same rows as a table
Every number in the manuscript text is written as \\ev{ID}; an unknown ID stops LaTeX with an error.
Status: VERIFIED = value recomputed by tools/independent_check.py (standard library, no project code);
PARTIALLY VERIFIED = read from the analysis file produced by the project code, reproduced on a second machine, not
independently recomputed.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
S = json.load(open(ROOT / "paper_outputs/v03/statistics/v03_analysis.json"))
SRC = "paper_outputs/v03/statistics/v03_analysis.json"
CELLS = {"qnq": "qwen_nq", "lnq": "llama_nq", "qhp": "qwen_hp", "lhp": "llama_hp"}
CELLNAME = {"qnq": "Qwen2.5-7B / NQ", "lnq": "Llama-3.1-8B / NQ", "qhp": "Qwen2.5-7B / HotpotQA", "lhp": "Llama-3.1-8B / HotpotQA"}
RNAME = {"R0": "BM25", "R1": "MiniLM-L6", "R2": "bge-small", "R3": "bge-base", "R4": "bge-large", "R5": "e5-base", "R6": "Qwen3-Emb-0.6B",
         "CB": "closed book", "GOLD": "gold passages", "X0": "BM25+reranker", "X5": "e5-base+reranker"}
V, P = "VERIFIED", "PARTIALLY VERIFIED"
ROWS = []
SHARES = {}


def f(x, d=3):
    s = f"{x:.{d}f}"
    return s.replace("-", "\\ensuremath{-}", 1) if s.startswith("-") else s


def sgn(x, d=3):
    return ("+" if x >= 0 else "\\ensuremath{-}") + f"{abs(x):.{d}f}"


def pv(p):
    if p >= 0.001:
        return f"{p:.3f}" if p >= 0.01 else f"{p:.4f}"
    m, e = f"{p:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def add(cid, value, claim, ctype, metric, comparator, population, uncertainty, source, status, check):
    ROWS.append(dict(claim_id=cid, claim=claim, claim_type=ctype, value=value, metric=metric, comparator=comparator,
                     population=population, uncertainty=uncertainty, source=source, status=status, check=check))


# ---------------- per cell: retrieval, answers, transfer, decomposition, P2 pairs
for c, cell in CELLS.items():
    C = S["cells"][cell]
    pop = f"{CELLNAME[c]}, n = {C['n_queries']} generation queries"
    ds = "nq" if "nq" in cell else "hp"
    for r, m in C["retrieval"].items():
        h = m["hit@5"]
        if c in ("qnq", "qhp"):  # retrieval is reader-independent; record once per dataset
            add(f"HIT.{ds}.{r}", f(h["mean"]), f"Hit@5 of {RNAME[r]}", "observation", "Hit@5", "—", pop.split(",")[0].split("/")[1].strip()
                + f", n = {h['n']}", "", f"{SRC} cells.{cell}.retrieval.{r}", V, "independent_check §2")
            add(f"HIT.{ds}.{r}.ci", f"[{f(h['ci_lo'])}, {f(h['ci_hi'])}]", f"95% CI of Hit@5 of {RNAME[r]}", "calculation", "Hit@5",
                "—", f"n = {h['n']}", "percentile bootstrap B=10,000 seed 0", f"{SRC} cells.{cell}.retrieval.{r}", P, "second-machine rerun")
            if "both@5" in m:
                b = m["both@5"]
                add(f"BOTH.hp.{r}", f(b["mean"]), f"Both@5 of {RNAME[r]}", "observation", "Both@5 (both gold passages in top 5)", "—",
                    f"HotpotQA, n = {b['n']}", f"CI [{f(b['ci_lo'])}, {f(b['ci_hi'])}]", f"{SRC} cells.{cell}.retrieval.{r}", P,
                    "summary file equality only")
    for r, m in C["answers"].items():
        e = m["em"]
        add(f"EM.{c}.{r}", f(e["mean"]), f"EM of {RNAME[r]}", "observation", "exact match", "—", pop, "",
            f"{SRC} cells.{cell}.answers.{r}", V, "independent_check §2/§6")
        add(f"EM.{c}.{r}.ci", f"[{f(e['ci_lo'])}, {f(e['ci_hi'])}]", f"95% CI of EM of {RNAME[r]}", "calculation", "exact match", "—",
            pop, "percentile bootstrap B=10,000 seed 0", f"{SRC} cells.{cell}.answers.{r}", P, "second-machine rerun")
        add(f"F1.{c}.{r}", f(m["f1"]["mean"]), f"token F1 of {RNAME[r]}", "observation", "SQuAD token F1", "—", pop, "",
            f"{SRC} cells.{cell}.answers.{r}", P, "not independently re-implemented")
        add(f"AC.{c}.{r}", f(m["contain"]["mean"]), f"answer containment of {RNAME[r]}", "observation", "answer containment", "—", pop,
            "", f"{SRC} cells.{cell}.answers.{r}", P, "not independently re-implemented")
    for t in C["transfer_vs_R1"]["em"]:
        r = t["retriever"]
        add(f"DH.{c}.{r}", sgn(t["d_hit"]), f"Hit@5 gain of {RNAME[r]} over MiniLM-L6", "calculation", "ΔHit@5", "MiniLM-L6", pop, "",
            f"{SRC} cells.{cell}.transfer_vs_R1", V, "derived from verified means")
        add(f"DE.{c}.{r}", sgn(t["d_em"]), f"EM gain of {RNAME[r]} over MiniLM-L6", "calculation", "ΔEM", "MiniLM-L6", pop, "",
            f"{SRC} cells.{cell}.transfer_vs_R1", V, "derived from verified means")
        if t["estimable"]:
            add(f"T.{c}.{r}", f(t["ratio"], 2), f"transfer ratio of {RNAME[r]}", "calculation", "T = ΔEM/ΔHit@5", "MiniLM-L6", pop,
                f"CI [{t['ci_lo']:.2f}, {t['ci_hi']:.2f}]", f"{SRC} cells.{cell}.transfer_vs_R1", V, "independent_check §3 (point estimate)")
            add(f"T.{c}.{r}.ci", f"[{f(t['ci_lo'], 2)}, {f(t['ci_hi'], 2)}]", f"95% CI of the transfer ratio of {RNAME[r]}", "calculation",
                "T", "MiniLM-L6", pop, "paired percentile bootstrap over queries B=10,000 seed 0", f"{SRC} cells.{cell}.transfer_vs_R1", P,
                "second-machine rerun")
            add(f"T.{c}.{r}.hi3", f(t["ci_hi"], 3), f"upper 95% bound (3 decimals) of the transfer ratio of {RNAME[r]}", "calculation", "T",
                "MiniLM-L6", pop, "bootstrap", f"{SRC} cells.{cell}.transfer_vs_R1", P, "second-machine rerun")
            add(f"T.{c}.{r}.hi", f(t["ci_hi"], 2), f"upper 95% bound of the transfer ratio of {RNAME[r]}", "calculation", "T", "MiniLM-L6",
                pop, "bootstrap", f"{SRC} cells.{cell}.transfer_vs_R1", P, "second-machine rerun")
        else:
            add(f"T.{c}.{r}.reason", "not estimable", t["reason"], "calculation", "T reporting rule", "MiniLM-L6", pop, "", f"{SRC}", V,
                "rule applied to verified ΔHit@5")
    for met, tag in (("f1", "TF"), ("contain", "TA")):
        for t in C["transfer_vs_R1"][met]:
            if t["estimable"]:
                r = t["retriever"]
                add(f"{tag}.{c}.{r}", f(t["ratio"], 2), f"transfer ratio of {RNAME[r]} with {met} as answer metric", "calculation",
                    f"T = Δ{met}/ΔHit@5", "MiniLM-L6", pop, f"CI [{t['ci_lo']:.2f}, {t['ci_hi']:.2f}]",
                    f"{SRC} cells.{cell}.transfer_vs_R1.{met}", P, "F1/AC not independently re-implemented")
                add(f"{tag}.{c}.{r}.ci", f"[{f(t['ci_lo'], 2)}, {f(t['ci_hi'], 2)}]", f"95% CI, {met} transfer ratio of {RNAME[r]}",
                    "calculation", "T", "MiniLM-L6", pop, "bootstrap", f"{SRC} cells.{cell}.transfer_vs_R1.{met}", P, "")
    for r, d in C["decomposition"].items():
        add(f"PH.{c}.{r}", f(d["p_em_given_hit"]), f"P(EM | hit@5) for {RNAME[r]}", "calculation", "conditional EM", "—",
            pop + f", n_hit = {d['n_hit']}", "", f"{SRC} cells.{cell}.decomposition", P, "not independently recomputed")
        add(f"PM.{c}.{r}", f(d["p_em_given_miss"]), f"P(EM | miss@5) for {RNAME[r]}", "calculation", "conditional EM", "—",
            pop + f", n_miss = {d['n_miss']}", "", f"{SRC} cells.{cell}.decomposition", P, "not independently recomputed")
    # post hoc algebraic decomposition (exact identity): EM_R = h_R a_R + (1-h_R) b_R with a = P(EM|hit), b = P(EM|miss), so
    # T(R) = (a_1 - b_1) + [h_R (a_R - a_1) + (1 - h_R)(b_R - b_1)] / (h_R - h_1)
    D = C["decomposition"]
    n = C["n_queries"]
    a1, b1 = D["R1"]["p_em_given_hit"], D["R1"]["p_em_given_miss"]
    h1 = D["R1"]["n_hit"] / n
    add(f"DB.{c}", f(a1 - b1, 2), "baseline hit-miss gap of MiniLM-L6, P(EM|hit) - P(EM|miss)", "calculation", "difference of conditional EM",
        "—", pop, "post hoc", f"{SRC} cells.{cell}.decomposition", P, "computed in make_ledger.py")
    for t in C["transfer_vs_R1"]["em"]:
        r = t["retriever"]
        if not t["estimable"]:
            continue
        hR, aR, bR = D[r]["n_hit"] / n, D[r]["p_em_given_hit"], D[r]["p_em_given_miss"]
        resid = (hR * (aR - a1) + (1 - hR) * (bR - b1)) / (hR - h1)
        assert abs((a1 - b1) + resid - t["ratio"]) < 1e-9, (cell, r)
        SHARES.setdefault(ds, []).append(resid / t["ratio"])
        add(f"DS.{c}.{r}", f"{100 * resid / t['ratio']:.0f}\\%", f"share of T from changed conditional EM rates, {RNAME[r]}", "calculation",
            "(T - (a_1 - b_1)) / T", "MiniLM-L6", pop, "post hoc", f"{SRC} cells.{cell}.decomposition", P, "computed in make_ledger.py")
        add(f"DR.{c}.{r}", sgn(resid, 2), f"part of T from changed conditional EM rates, {RNAME[r]}", "calculation", "T - (a_1 - b_1)",
            "MiniLM-L6", pop, "post hoc; exact identity checked", f"{SRC} cells.{cell}.decomposition", P, "computed in make_ledger.py")
    add(f"RHO.{c}", f(C["spearman_hit5_em"]["rho"], 2), "Spearman rho between Hit@5 and EM over 7 retrievers", "calculation", "Spearman",
        "—", pop, "descriptive, 7 points", f"{SRC} cells.{cell}.spearman_hit5_em", P, "")
    if "both@5" in C["retrieval"]["R0"]:  # post hoc (not pre-registered): Spearman between Both@5 and EM over R0-R6
        rs = list(C["retrieval"])
        def ranks(v):
            o = sorted(range(len(v)), key=lambda i: v[i])
            rk = [0] * len(v)
            for pos, i in enumerate(o):
                rk[i] = pos + 1
            assert len(set(v)) == len(v), "ties not handled"
            return rk
        a = ranks([C["retrieval"][r]["both@5"]["mean"] for r in rs])
        b = ranks([C["answers"][r]["em"]["mean"] for r in rs])
        n = len(rs)
        rho = 1 - 6 * sum((x - y) ** 2 for x, y in zip(a, b)) / (n * (n * n - 1))
        add(f"RHOB.{c}", f(rho, 2), "Spearman rho between Both@5 and EM over 7 retrievers (post hoc)", "calculation", "Spearman", "—", pop,
            "descriptive, post hoc, 7 points", f"{SRC} cells.{cell}.retrieval/answers", P, "computed in make_ledger.py")
    for big in ("R4", "R6"):
        e = C["P2"]["per_encoder"][big]
        add(f"P2.{c}.{big}", sgn(e["em_diff_minus_e5"]), f"EM of {RNAME[big]} minus EM of e5-base", "inference", "ΔEM", "e5-base", pop,
            f"Holm p = {e['p_holm']:.3g}", f"{SRC} cells.{cell}.P2", V, "independent_check §7 (Holm p)")
        add(f"P2.{c}.{big}.p", pv(e["p_holm"]), f"Holm-adjusted McNemar p, {RNAME[big]} vs e5-base", "inference", "exact McNemar, Holm over 21",
            "e5-base", pop, "", f"{SRC} cells.{cell}.P2", V, "independent_check §7")
    if "by_type" in C:
        for typ, d in C["by_type"].items():
            add(f"N.{c}.{typ}", f"{d['n']:,}", f"number of {typ} questions", "observation", "count", "—", pop, "", f"{SRC} by_type", P, "")
            for k in ("R0", "R1", "R2", "R3", "R4", "R5", "R6", "CB", "GOLD"):
                add(f"BT.{c}.{typ}.{k}", f(d[k]["em"]), f"EM of {RNAME[k]} on {typ} questions", "observation", "exact match", "—",
                    pop + f", {typ} n = {d['n']}", "", f"{SRC} cells.{cell}.by_type", P, "")
            for k in ("R0", "R1", "R2", "R3", "R4", "R5", "R6"):
                add(f"BTB.{c}.{typ}.{k}", f(d["both5_" + k]), f"Both@5 of {RNAME[k]} on {typ} questions", "observation", "Both@5", "—",
                    pop + f", {typ}", "", f"{SRC} cells.{cell}.by_type", P, "")

for ds_, v in SHARES.items():
    if ds_ == "hp":
        add("DS.hp.min", f"{100 * min(v):.0f}\\%", "smallest share of T from changed conditional EM rates, HotpotQA", "calculation", "share", "",
            "HotpotQA, 2 readers x 5 encoders", "post hoc", f"{SRC}", P, "")
        add("DS.hp.max", f"{100 * max(v):.0f}\\%", "largest share of T from changed conditional EM rates, HotpotQA", "calculation", "share", "",
            "HotpotQA, 2 readers x 5 encoders", "post hoc", f"{SRC}", P, "")

# ---------------- unweighted mean EM over the four cells (as plotted in fig5)
for r in ("R0", "R1", "R2", "R3", "R4", "R5", "R6"):
    m = sum(S["cells"][cell]["answers"][r]["em"]["mean"] for cell in CELLS.values()) / 4
    add(f"EMBAR.{r}", f(m), f"unweighted mean EM over the four cells, {RNAME[r]}", "calculation", "mean of 4 cell EMs", "—",
        "4 reader-dataset cells", "descriptive", f"{SRC} cells.*.answers", V, "mean of verified cell EMs")

# ---------------- top-k (E4) and P4
T = S["topk"]
for r, ks in T["em_by_k"].items():
    for k, v in ks.items():
        add(f"TK.{r}.k{k}", f(v["mean"]), f"EM of {RNAME[r]} with k = {k}", "observation", "exact match", "—",
            f"Qwen2.5-7B / NQ, n = {v['n']} ({v['source']})", f"CI [{f(v['ci_lo'])}, {f(v['ci_hi'])}]", f"{SRC} topk", V, "independent_check §6")
for t in T["k1_vs_k5"]:
    r = t["retriever"]
    add(f"K1.{r}", sgn(t["diff"]), f"EM(k=1) minus EM(k=5) for {RNAME[r]}", "inference", "ΔEM", "k = 5", "Qwen2.5-7B / NQ, n = 2255",
        f"CI [{t['ci_lo']:.3f}, {t['ci_hi']:.3f}]; Holm p = {t['p_holm']:.2g}", f"{SRC} topk.k1_vs_k5", P, "")
    add(f"K1.{r}.p", pv(t["p_holm"]), f"Holm p, k=1 vs k=5, {RNAME[r]}", "inference", "exact McNemar, Holm over 7", "k = 5",
        "Qwen2.5-7B / NQ", "", f"{SRC} topk.k1_vs_k5", P, "")

# ---------------- mechanism (E12): P5, P6
M = S["mechanism"]
for k, v in M["answers"].items():
    add(f"MC.{k}", f(v["em"]["mean"]), f"EM of composition {k}", "observation", "exact match", "—", f"Qwen2.5-7B / NQ, first {M['n_queries']} queries",
        f"CI [{f(v['em']['ci_lo'])}, {f(v['em']['ci_hi'])}]", f"{SRC} mechanism.answers", V, "independent_check §6")
p5 = M["P5"]["test"]
add("P5.diff", sgn(p5["diff"]), "EM of M1 minus EM of MR", "inference", "ΔEM", "MR (random negatives)", "Qwen2.5-7B / NQ, n = 500",
    f"CI [{p5['ci_lo']}, {p5['ci_hi']}]", f"{SRC} mechanism.P5", V, "derived from verified means")
add("P5.abs", f(abs(p5["diff"])), "size of the EM drop, M1 vs MR", "inference", "|ΔEM|", "MR", "Qwen2.5-7B / NQ, n = 500", "", f"{SRC} mechanism.P5", V,
    "derived from verified means")
add("P5.ci", f"[{f(p5['ci_lo'])}, {f(p5['ci_hi'])}]", "95% CI of M1 − MR", "calculation", "ΔEM", "MR", "n = 500", "paired bootstrap", f"{SRC}", P, "")
add("P5.p", pv(p5["p"]), "exact McNemar p, M1 vs MR", "inference", "exact McNemar (single pre-registered test)", "MR", "n = 500", "", f"{SRC}", V,
    "independent_check §4")
for t in M["P6"]["tests"]:
    add(f"P6.{t['a']}{t['b']}", sgn(t["diff"]), f"EM of {t['a']} minus {t['b']}", "inference", "ΔEM", t["b"], "n = 500",
        f"Holm p = {t['p_holm']:.2f}", f"{SRC} mechanism.P6", P, "")
    add(f"P6.{t['a']}{t['b']}.p", f"{t['p_holm']:.2f}", f"Holm p, {t['a']} vs {t['b']}", "inference", "exact McNemar, Holm over 3", t["b"], "n = 500",
        "", f"{SRC} mechanism.P6", P, "")
g = M["gold_only_vs_MR"]
add("MG.diff", sgn(g["diff"]), "EM of gold-only (MG) minus MR", "inference", "ΔEM", "MR", "n = 500", f"p = {g['p']:.2f}", f"{SRC}", P, "")
add("MG.p", f"{g['p']:.2f}", "exact McNemar p, MG vs MR", "inference", "exact McNemar", "MR", "n = 500", "", f"{SRC}", P, "")

# ---------------- reranker (E5)
RR = S["reranker"]
for k, v in RR["retrieval_generation_set"].items():
    add(f"RR.hit.{k}", f(v["hit@5"]["mean"]), f"Hit@5 of {RNAME[k]} (generation set)", "observation", "Hit@5", "—", "NQ, n = 2255",
        f"CI [{f(v['hit@5']['ci_lo'])}, {f(v['hit@5']['ci_hi'])}]", f"{SRC} reranker", P, "")
    add(f"RR.ndcg.{k}", f(v["ndcg@10"]["mean"]), f"nDCG@10 of {RNAME[k]} (generation set)", "observation", "nDCG@10", "—", "NQ, n = 2255",
        "", f"{SRC} reranker", P, "")
for k, v in RR["retrieval_all_judged"].items():
    add(f"RRJ.hit.{k}", f(v["hit@5"]), f"Hit@5 of {RNAME[k]} (all judged)", "observation", "Hit@5", "—", "NQ, n = 3452", "", f"{SRC} reranker", P, "")
for k, v in RR["answers"].items():
    add(f"RR.em.{k}", f(v["em"]["mean"]), f"EM of {RNAME[k]} (reranker comparison)", "observation", "exact match", "—", "Qwen2.5-7B / NQ, n = 2255",
        f"CI [{f(v['em']['ci_lo'])}, {f(v['em']['ci_hi'])}]", f"{SRC} reranker", V, "independent_check §2/§6")
for pair, d in RR["pairs"].items():
    for met in ("hit@5", "em"):
        x = d[met]
        tag = "hit" if met == "hit@5" else "em"
        add(f"RR.{pair}.{tag}", sgn(x["diff"]), f"{met} difference {pair}", "inference", met, pair.split("_vs_")[1], "NQ, n = 2255",
            f"CI [{x['ci_lo']:.3f}, {x['ci_hi']:.3f}]; p = {x['p']:.2g}", f"{SRC} reranker.pairs", P, "")
        add(f"RR.{pair}.{tag}.p", pv(x["p"]), f"exact McNemar p {met} {pair}", "inference", "exact McNemar", "", "NQ, n = 2255", "",
            f"{SRC} reranker.pairs", P, "")
        add(f"RR.{pair}.{tag}.ci", f"[{f(x['ci_lo'])}, {f(x['ci_hi'])}]", f"95% CI {met} {pair}", "calculation", met, "", "NQ, n = 2255",
            "paired bootstrap", f"{SRC} reranker.pairs", P, "")
for k, v in RR["latency_ms"].items():
    add(f"RR.lat.{k}", f"{v['median_ms']:.0f}", f"median reranking latency per query ({v['pairs_per_query']} pairs), base {RNAME[k]}",
        "observation", "ms per query", "—", f"n = {v['n']} queries, T4", f"p95 {v['p95_ms']:.0f} ms", f"{SRC} reranker.latency_ms", P, "")

# ---------------- E9 full corpus, E6 HNSW, E7 cost
E9 = S["corpus_hnsw_cost"]["E9"]
for r in E9["retrievers"]:
    for k, lab in (("subcorpus_same_embeddings", "sub"), ("full_corpus", "full")):
        for met in ("ndcg@10", "hit@5"):
            add(f"E9.{lab}.{r}.{met.split('@')[0]}", f(E9[k][r][met]), f"{met} of {RNAME[r]} on the {lab} corpus", "observation", met, "—",
                f"NQ, all {E9['n_judged']} judged queries", "", f"{SRC} corpus_hnsw_cost.E9", P, "")
    add(f"E9.drop.{r}", sgn(E9["drop_full_minus_sub"][r]["ndcg@10"]), f"nDCG@10 full minus sub-corpus, {RNAME[r]}", "calculation", "nDCG@10",
        "sub-corpus", "NQ judged", "", f"{SRC} E9", P, "")
for r in E9["retrievers"]:
    add(f"E9.dropabs.{r}", f(abs(E9["drop_full_minus_sub"][r]["ndcg@10"])), f"size of the nDCG@10 drop on the full corpus, {RNAME[r]}",
        "calculation", "nDCG@10", "sub-corpus", "NQ judged", "", f"{SRC} E9", P, "")
add("E9.tau", f"{E9['P3']['kendall_tau_primary']:.2f}", "Kendall tau between sub- and full-corpus orderings", "calculation", "Kendall tau",
    "—", "4 retrievers", "", f"{SRC} E9.P3", P, "")
E6 = S["corpus_hnsw_cost"]["E6"]
for r, d in E6.items():
    add(f"E6.ov5.{r}", f"{d['ef128']['overlap@5']:.4f}", f"top-5 overlap HNSW(efSearch 128) vs exact, {RNAME[r]}", "observation", "overlap@5",
        "exact search", "NQ judged", "", f"{SRC} E6", P, "")
    add(f"E6.ov5.ef64.{r}", f"{d['ef64']['overlap@5']:.4f}", f"top-5 overlap HNSW(efSearch 64), {RNAME[r]}", "observation", "overlap@5",
        "exact search", "NQ judged", "", f"{SRC} E6", P, "")
    add(f"E6.lat.exact.{r}", f"{d['exact']['latency']['median_ms']:.0f}", f"exact search median latency, {RNAME[r]}", "observation", "ms/query",
        "", "500 queries, CPU", "", f"{SRC} E6", P, "")
    add(f"E6.lat.ef128.{r}", f"{d['ef128']['latency']['median_ms']:.2f}", f"HNSW efSearch 128 median latency, {RNAME[r]}", "observation",
        "ms/query", "", "500 queries, CPU", "", f"{SRC} E6", P, "")
    add(f"E6.dhit.{r}", sgn(d["ef128"]["hit@5"] - d["exact"]["hit@5"], 4), f"Hit@5 HNSW(128) minus exact, {RNAME[r]}", "calculation", "Hit@5",
        "exact", "NQ judged", "", f"{SRC} E6", P, "")
add("E6.dhitabs.max", f"{max(abs(d['ef128']['hit@5'] - d['exact']['hit@5']) for d in E6.values()):.4f}",
    "largest absolute Hit@5 change, HNSW efSearch 128 vs exact", "calculation", "Hit@5", "exact", "NQ judged, 3 encoders", "", f"{SRC} E6", P, "")
E7 = S["corpus_hnsw_cost"]["E7"]
for r in ("R1", "R2", "R3", "R4", "R5", "R6"):
    d = E7[r]
    add(f"E7.tp.{r}", f"{d['passages_per_s']['median']:,.0f}", f"median encoding throughput, {RNAME[r]}", "observation", "passages/s",
        "", "T4, fp16, batch 128, 20,000 passages", f"range {d['passages_per_s']['min']:.0f}–{d['passages_per_s']['max']:.0f} over {d['n_reps']} reps",
        f"{SRC} E7", P, "")
    add(f"E7.tprange.{r}", f"{d['passages_per_s']['min']:,.0f}--{d['passages_per_s']['max']:,.0f}", f"throughput range, {RNAME[r]}",
        "observation", "passages/s", "", "5 repetitions", "", f"{SRC} E7", P, "")
    add(f"E7.lat.{r}", f"{d['query_latency_ms']['median']:.1f}", f"median query-encoding latency, {RNAME[r]}", "observation", "ms",
        "", "batch 1, 500 queries", "", f"{SRC} E7", P, "")
    add(f"E7.par.{r}", f"{d['n_params'] / 1e6:.0f}", f"parameters (millions), {RNAME[r]}", "observation", "count", "", "", "", f"{SRC} E7", P, "")
    add(f"E7.mem.{r}", f"{d['peak_mib']:,.0f}", f"peak GPU memory while encoding, {RNAME[r]}", "observation", "MiB", "", "", "", f"{SRC} E7", P, "")
add("E7.lat.R0", f"{E7['R0']['query_latency_ms']['median']:.1f}", "median BM25 query latency (CPU)", "observation", "ms", "", "batch 1", "",
    f"{SRC} E7", P, "")
add("E7.ratio.R1R6", f"{E7['R1']['passages_per_s']['median'] / E7['R6']['passages_per_s']['median']:.0f}",
    "throughput ratio MiniLM-L6 / Qwen3-Emb", "calculation", "ratio", "", "", "", f"{SRC} E7", P, "")
add("E7.ratio.R5R4", f"{E7['R5']['passages_per_s']['median'] / E7['R4']['passages_per_s']['median']:.1f}",
    "throughput ratio e5-base / bge-large", "calculation", "ratio", "", "", "", f"{SRC} E7", P, "")

# ---------------- GEE (E10)
G = S["gee"]
for name, d in G["contrasts"].items():
    k = {"bge-large_vs_e5": "R4", "qwen3emb_vs_e5": "R6"}[name]
    add(f"GEE.or.{k}", f"{d['odds_ratio']:.3f}", f"pooled odds ratio of EM, {RNAME[k]} vs e5-base", "inference", "GEE logistic OR", "e5-base",
        f"4 cells, {G['n_rows']} rows, {G['n_clusters']} clusters", f"95% CI [{d['or_ci95'][0]:.3f}, {d['or_ci95'][1]:.3f}]", f"{SRC} gee", P, "")
    add(f"GEE.ci.{k}", f"[{d['or_ci95'][0]:.3f}, {d['or_ci95'][1]:.3f}]", f"95% CI of OR, {RNAME[k]} vs e5-base", "inference", "robust SE", "",
        "", "", f"{SRC} gee", P, "")
    add(f"GEE.p.{k}", pv(d["p"]), f"Wald p of the OR, {RNAME[k]} vs e5-base", "inference", "robust Wald", "", "", "", f"{SRC} gee", P, "")
add("GEE.wc", f"{G['working_correlation']:.2f}", "exchangeable working correlation", "calculation", "", "", "", "", f"{SRC} gee", P, "")
add("GEE.rows", f"{G['n_rows']:,}", "rows in the pooled GEE", "observation", "count", "", "", "", f"{SRC} gee", P, "")
add("GEE.clusters", f"{G['n_clusters']:,}", "query clusters in the pooled GEE", "observation", "count", "", "", "", f"{SRC} gee", P, "")

# ---------------- judge agreement (E3)
J = S["judge_agreement"]
add("JA.kappa", f(J["cohen_kappa"]), "Cohen's kappa, author vs LLM judge", "calculation", "Cohen's kappa", "author labels", "200 stratified items",
    f"CI [{J['kappa_ci95'][0]:.3f}, {J['kappa_ci95'][1]:.3f}]", f"{SRC} judge_agreement", V, "independent_check §5")
add("JA.ci", f"[{f(J['kappa_ci95'][0])}, {f(J['kappa_ci95'][1])}]", "95% CI of kappa", "calculation", "bootstrap", "", "200 items", "", f"{SRC}", P, "")
add("JA.raw", f(J["raw_agreement"]), "raw agreement", "calculation", "proportion", "", "200 items", "", f"{SRC}", P, "")
add("JA.kipw", f(J["population_weighted"]["kappa_ipw"]), "inverse-probability-weighted kappa", "calculation", "kappa", "", "200 items", "",
    f"{SRC}", P, "")
add("JA.jrate", f(J["population_weighted"]["judge_supported_rate"]), "population-weighted share judged SUPPORTED by the LLM judge", "calculation",
    "proportion", "", "200 items, inverse-probability weights", "", f"{SRC}", P, "")
add("JA.hrate", f(J["population_weighted"]["human_supported_rate"]), "population-weighted share labelled SUPPORTED by the author", "calculation",
    "proportion", "", "200 items, inverse-probability weights", "", f"{SRC}", P, "")
add("JA.prec", f(J["population_weighted"]["judge_precision_for_SUPPORTED"]), "weighted judge precision for SUPPORTED", "calculation", "", "",
    "", "", f"{SRC}", P, "")
add("JA.rec", f(J["population_weighted"]["judge_recall_for_SUPPORTED"]), "weighted judge recall for SUPPORTED", "calculation", "", "", "",
    "", f"{SRC}", P, "")
for k, v in J["confusion_human_x_judge"].items():
    add("JA." + k.replace("human_", "h").replace("__judge_", "j").replace("NOT_SUPPORTED", "NS").replace("SUPPORTED", "S"), str(v),
        f"confusion count {k}", "observation", "count", "", "200 items", "", f"{SRC}", V, "independent_check §5 inputs")

# ---------------- design counts (from data-preparation records)
nq = json.load(open(ROOT / "paper_inputs/data/processed/nq_join_stats.json"))
hp = json.load(open(ROOT / "paper_inputs/v03_hp/processed/hp_join_stats.json"))
sc_nq = json.loads(open(ROOT / "paper_inputs/results/raw/subcorpus.jsonl").readline())
sc_hp = json.loads(open(ROOT / "paper_inputs/v03_hp/raw/subcorpus.jsonl").readline())
for k, v, what, src in (("N.nq.judged", nq["n_beir_judged"], "BEIR-NQ judged test queries", "nq_join_stats.json"),
                        ("N.nq.joined", nq["n_joined"], "NQ queries joined to NQ-open answers", "nq_join_stats.json"),
                        ("N.nq.unmatched", nq["n_unmatched"], "NQ queries without an NQ-open match", "nq_join_stats.json"),
                        ("N.hp.judged", hp["n_beir_judged"], "BEIR-HotpotQA judged test queries", "hp_join_stats.json"),
                        ("N.hp.bridge", hp["n_by_type"]["bridge"], "HotpotQA bridge questions (all judged)", "hp_join_stats.json"),
                        ("N.hp.comparison", hp["n_by_type"]["comparison"], "HotpotQA comparison questions (all judged)", "hp_join_stats.json"),
                        ("N.hp.gen", S["cells"]["qwen_hp"]["n_queries"], "HotpotQA generation queries", SRC),
                        ("N.sc", sc_nq["size"], "sub-corpus size", "subcorpus.jsonl"),
                        ("N.sc.nq.gold", sc_nq["n_gold"], "gold passages in the NQ sub-corpus", "results/raw/subcorpus.jsonl"),
                        ("N.sc.nq.bm25", sc_nq["n_bm25_added"], "BM25 negatives in the NQ sub-corpus", "results/raw/subcorpus.jsonl"),
                        ("N.sc.nq.rand", sc_nq["n_random"], "random passages in the NQ sub-corpus", "results/raw/subcorpus.jsonl"),
                        ("N.sc.hp.gold", sc_hp["n_gold"], "gold passages in the HotpotQA sub-corpus", "v03_hp/raw/subcorpus.jsonl"),
                        ("N.sc.hp.bm25", sc_hp["n_bm25_added"], "BM25 negatives in the HotpotQA sub-corpus", "v03_hp/raw/subcorpus.jsonl"),
                        ("N.sc.hp.rand", sc_hp["n_random"], "random passages in the HotpotQA sub-corpus", "v03_hp/raw/subcorpus.jsonl"),
                        ("N.gee", G["n_rows"], "GEE rows", SRC)):
    add(k, f"{v:,}", what, "observation", "count", "", "study design (data preparation)", "", src, V, "read from data-preparation record")
IC = json.load(open(ROOT / "manuscript_ipm/records/independent_check.json"))
add("IC.n", f"{IC['n_answers']:,}", "answers re-scored independently", "calculation", "count", "", "all scored answers", "",
    "manuscript_ipm/records/independent_check.json", V, "independent_check §1")
add("IC.plain", f"{IC['n_plain_squad_diff']:,}", "answers whose EM changes under a plain SQuAD normaliser", "calculation", "count", "",
    "all scored answers", "", "manuscript_ipm/records/independent_check.json", V, "independent_check §1")
add("IC.plainpct", f"{100 * IC['n_plain_squad_diff'] / IC['n_answers']:.2f}\\%", "share of answers whose EM changes under a plain SQuAD normaliser",
    "calculation", "proportion", "", "all scored answers", "", "manuscript_ipm/records/independent_check.json", V, "independent_check §1")
CS = json.load(open(ROOT / "manuscript_ipm/records/cross_session_check.json"))
add("CS.n", str(CS["cross_session"]["n_identical_inputs"]), "reader inputs identical across the two study phases", "observation", "count", "",
    "Qwen2.5-7B / NQ", "", "manuscript_ipm/records/cross_session_check.json", V, "tools/cross_session_check.py")
add("CS.same", str(CS["cross_session"]["n_identical_outputs"]), "of these, outputs identical across phases", "observation", "count", "",
    "Qwen2.5-7B / NQ", "", "manuscript_ipm/records/cross_session_check.json", V, "tools/cross_session_check.py")
add("CS.within", f"{CS['within_session_rerun_A2']['n_identical_outputs']}/{CS['within_session_rerun_A2']['n']}",
    "identical outputs in a within-session rerun", "observation", "count", "", "Qwen2.5-7B / NQ, bge-base k=5 prompt A1, 100 queries rerun in the same session", "",
    "manuscript_ipm/records/cross_session_check.json", V, "tools/cross_session_check.py")
add("N.nq.joinrate", f"{nq['join_rate']:.3f}", "NQ join rate", "calculation", "proportion", "", "BEIR-NQ judged test queries", "",
    "nq_join_stats.json", V, "")

# ---------------- carried over from Phase 1 / exploratory extension (src/analysis/v03_extras.py)
XS = "paper_outputs/v03/statistics/v03_extras.json"
X = json.load(open(ROOT / XS))
for r, m in X["retrieval_nq"]["metrics"].items():
    for met in ("hit@1", "hit@10", "ndcg@10", "mrr@10"):
        add(f"RET.nq.{met.replace('@', '')}.{r}", f(m[met]["mean"]), f"{met} of {RNAME[r]} on the NQ generation queries", "observation",
            met, "—", f"NQ, n = {X['retrieval_nq']['n']}", f"CI [{f(m[met]['ci_lo'])}, {f(m[met]['ci_hi'])}]", f"{XS} retrieval_nq", P,
            "Phase 1 pipeline output (rq12_main.json); Hit@5 checked against top-10 lists")
for r, m in X["retrieval_hp"]["metrics"].items():
    for met in ("hit@1", "hit@10", "ndcg@10", "mrr@10"):
        add(f"RET.hp.{met.replace('@', '')}.{r}", f(m[met]), f"{met} of {RNAME[r]} on the HotpotQA generation queries", "observation",
            met, "—", f"HotpotQA, n = {X['retrieval_hp']['n']}", "", f"{XS} retrieval_hp", P, "retrieval summary file")
for r, v in X["deployment_cost"]["efficiency"].items():
    for key, lab, fmt in (("encode_gpu_s", "T4 time to encode 300,000 passages (s)", "{:,.0f}"), ("q_enc_cpu_ms", "CPU query encoding (ms)", "{:.1f}"),
                          ("search_ms", "CPU exact search (ms)", "{:.1f}"), ("query_ms_cpu", "CPU query total (ms)", "{:.1f}"),
                          ("index_mb", "flat index size (MB)", "{:,.0f}"), ("gpu_peak_mib", "peak GPU memory (MiB)", "{:,.0f}")):
        if v.get(key) is None:
            continue
        add(f"DC.{key.split('_')[0] if key != 'query_ms_cpu' else 'qcpu'}.{key}.{r}".replace(f"{key.split('_')[0]}.{key}", key),
            fmt.format(v[key]), f"{lab}, {RNAME[r]}", "observation", lab, "", "Phase 1, single measurement", "",
            f"{XS} deployment_cost", P, "")
for c, v in X["idk"].items():
    add(f"IDK.rate.{c}", f(v["idk_rate"]), f"abstention rate under prompt A3, {RNAME[c]}", "observation", "share", "", f"Qwen / NQ, n = {v['n']}",
        "", f"{XS} idk", P, "")
    if c != "GOLD":
        add(f"IDK.hit.{c}", f(v["idk_given_hit"]), f"abstention on hits, {RNAME[c]}", "observation", "share", "", f"n_hit = {v['n_hit']}", "",
            f"{XS} idk", P, "")
        add(f"IDK.miss.{c}", f(v["idk_given_miss"]), f"abstention on misses, {RNAME[c]}", "observation", "share", "", f"n_miss = {v['n_miss']}",
            "", f"{XS} idk", P, "")
        add(f"IDK.dem.{c}", sgn(v["em_A3"] - v["em_A1"]), f"EM under A3 minus A1, {RNAME[c]}", "inference", "ΔEM", "prompt A1", "n = 200",
            f"p = {v['p_A3_vs_A1']:.2f}", f"{XS} idk", P, "")
        add(f"IDK.p.{c}", f"{v['p_A3_vs_A1']:.2f}", f"exact McNemar p, A3 vs A1, {RNAME[c]}", "inference", "exact McNemar", "", "n = 200", "",
            f"{XS} idk", P, "")
for t in X["transfer_by_k"]["rows"]:
    add(f"TKT.{t['retriever']}.k{t['k']}", f(t["ratio"], 2), f"exploratory transfer ratio at k = {t['k']}, {RNAME[t['retriever']]}",
        "calculation", "dEM@k / dHit@k", "MiniLM-L6", "Qwen / NQ, n = 2255", f"CI [{t['ci_lo']:.2f}, {t['ci_hi']:.2f}]",
        f"{XS} transfer_by_k", P, "post hoc, exploratory")
    add(f"TKT.{t['retriever']}.k{t['k']}.ci", f"[{f(t['ci_lo'], 2)}, {f(t['ci_hi'], 2)}]", f"95% CI, exploratory ratio at k = {t['k']}",
        "calculation", "bootstrap", "MiniLM-L6", "Qwen / NQ", "", f"{XS} transfer_by_k", P, "post hoc, exploratory")

RQ12 = json.load(open(ROOT / "paper_inputs/results/statistics/rq12_main.json"))
D1 = RQ12["d1b"]["per_retriever"]
dd = [v["hit@5"]["diff"] for v in D1.values()]
all0 = all(v["hit@5"]["ci"][0] <= 0 <= v["hit@5"]["ci"][1] for v in D1.values())
assert all0
add("D1B.min", sgn(min(dd)), "smallest Hit@5 difference joined minus unjoined NQ queries", "calculation", "ΔHit@5", "unjoined (1,197)",
    "NQ judged, 7 retrievers", "every 95% CI includes 0", "paper_inputs/results/statistics/rq12_main.json d1b", P, "Phase 1 pipeline output")
add("D1B.max", sgn(max(dd)), "largest Hit@5 difference joined minus unjoined NQ queries", "calculation", "ΔHit@5", "unjoined (1,197)",
    "NQ judged, 7 retrievers", "every 95% CI includes 0", "paper_inputs/results/statistics/rq12_main.json d1b", P, "Phase 1 pipeline output")

for c, v in X["judge_rates"]["rates"].items():
    add(f"JR.{c}", f(v["mean"]), f"share judged SUPPORTED by the excluded LLM judge, {RNAME[c]}", "observation", "share", "",
        f"Qwen / NQ, n = {v['n']}", f"CI [{f(v['ci_lo'])}, {f(v['ci_hi'])}]", f"{XS} judge_rates", P, "judge excluded from conclusions")
for key, v in X["prompt_tokens"].items():
    add(f"TOK.{key}", f"{v:,.0f}", f"mean reader prompt length (tokens), {key}", "observation", "tokens", "", "Qwen / NQ, 2,255 queries", "",
        f"{XS} prompt_tokens", P, "")
x = X["r4k1_vs_r0k10"]
add("X41.diff", sgn(x["diff"]), "EM of bge-large at k = 1 minus BM25 at k = 10 (exploratory)", "inference", "ΔEM", "BM25, k = 10",
    "Qwen / NQ, n = 2255", f"CI [{x['ci_lo']:.3f}, {x['ci_hi']:.3f}]; p = {x['p']:.3f}", f"{XS} r4k1_vs_r0k10", P, "exploratory, uncorrected")
add("X41.ci", f"[{f(x['ci_lo'])}, {f(x['ci_hi'])}]", "95% CI, bge-large k = 1 minus BM25 k = 10", "calculation", "bootstrap", "", "n = 2255", "",
    f"{XS} r4k1_vs_r0k10", P, "")
add("X41.p", f"{x['p']:.3f}", "exact McNemar p, uncorrected, bge-large k = 1 vs BM25 k = 10", "inference", "exact McNemar", "", "n = 2255", "",
    f"{XS} r4k1_vs_r0k10", P, "")


def main():
    ids = [r["claim_id"] for r in ROWS]
    dup = {i for i in ids if ids.count(i) > 1}
    assert not dup, dup
    tex = ["% Generated by manuscript_ipm/tools/make_ledger.py from the committed analysis file. Do not edit by hand.",
           "\\makeatletter",
           "\\newcommand{\\ev}[1]{\\ifcsname ev@#1\\endcsname\\csname ev@#1\\endcsname\\else\\errmessage{Unknown evidence id #1}\\fi}"]
    for r in ROWS:
        tex.append(f"\\expandafter\\def\\csname ev@{r['claim_id']}\\endcsname{{{r['value']}}}")
    tex.append("\\makeatother")
    (ROOT / "manuscript_ipm/latex/numbers.tex").write_text("\n".join(tex) + "\n")
    cols = ["claim_id", "claim", "claim_type", "value", "metric", "comparator", "population", "uncertainty", "source", "status", "check"]
    with open(ROOT / "manuscript_ipm/records/evidence-ledger.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(ROWS)
    md = ["# Evidence ledger (generated) — source of truth for manuscript numbers", "",
          "Generated by manuscript_ipm/tools/make_ledger.py. In the LaTeX source every number is written as `\\ev{ID}`; the ID is the",
          "claim ID below. VERIFIED = recomputed by independent_check.py; PARTIALLY VERIFIED = project code output reproduced on a second",
          "machine, not independently recomputed. Literature claims: literature-matrix.md. Claim strength and conflicts: claim-strength.md.", "",
          f"Rows: {len(ROWS)}; VERIFIED {sum(r['status'] == V for r in ROWS)}; PARTIALLY VERIFIED {sum(r['status'] == P for r in ROWS)}.", "",
          "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in ROWS:
        md.append("| " + " | ".join(str(r[c]).replace("|", "/") for c in cols) + " |")
    (ROOT / "manuscript_ipm/records/evidence-ledger.md").write_text("\n".join(md) + "\n")
    print(f"{len(ROWS)} claims; VERIFIED {sum(r['status'] == V for r in ROWS)}")


if __name__ == "__main__":
    main()
