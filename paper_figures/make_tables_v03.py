"""LaTeX tables (booktabs) for the journal version, generated from the analysis file only:

    paper_outputs/v03/statistics/v03_analysis.json   ->   paper_figures/out/tables/*.tex

Run from the repository root:  python paper_figures/make_tables_v03.py
Needs only the standard library. In the manuscript: \\usepackage{booktabs} and \\input{tables/<name>.tex}.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = json.load(open(ROOT / "paper_outputs/v03/statistics/v03_analysis.json"))
OUT = ROOT / "paper_figures/out/tables"
R = ["R0", "R1", "R2", "R3", "R4", "R5", "R6"]
NAME = {"R0": "BM25", "R1": "MiniLM-L6", "R2": "bge-small", "R3": "bge-base", "R4": "bge-large", "R5": "e5-base-v2",
        "R6": "Qwen3-Emb-0.6B", "CB": "Closed book", "GOLD": "Gold passages", "X0": "BM25 + reranker",
        "X5": "e5-base-v2 + reranker"}
CELLS = [("qwen_nq", "NQ", "Qwen"), ("llama_nq", "NQ", "Llama"), ("qwen_hp", "HotpotQA", "Qwen"), ("llama_hp", "HotpotQA", "Llama")]


def f(x, d=3):
    return "--" if x is None else f"{x:.{d}f}"


def pval(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def table(name, spec, header, rows, caption, label, notes=None, wide=False):
    env = "table*" if wide else "table"
    body = [f"\\begin{{{env}}}[t]", "\\centering", "\\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
            f"\\begin{{tabular}}{{{spec}}}", "\\toprule", header + r" \\", "\\midrule"]
    for r in rows:
        body.append(r if r.startswith("\\") else r + r" \\")
    body += ["\\bottomrule", "\\end{tabular}"]
    if notes:
        body.append(f"\\par\\smallskip\\parbox{{\\linewidth}}{{\\footnotesize {notes}}}")
    body.append(f"\\end{{{env}}}")
    OUT.mkdir(parents=True, exist_ok=True)
    txt = "\n".join(body) + "\n"
    import re
    txt = re.sub(r"(?<![\w$-])-(\d)", r"$-$\1", txt)          # typographic minus for negative numbers
    (OUT / f"{name}.tex").write_text(txt)
    print("wrote", OUT / f"{name}.tex")


def t_main():
    head = "Retriever & " + " & ".join(f"Hit@5 & EM" for _ in CELLS)
    top = (r" & \multicolumn{4}{c}{NQ (2,255 queries)} & \multicolumn{4}{c}{HotpotQA (2,000 queries)} \\"
           r" \cmidrule(lr){2-5}\cmidrule(lr){6-9}"
           r" & \multicolumn{2}{c}{Qwen} & \multicolumn{2}{c}{Llama} & \multicolumn{2}{c}{Qwen} & \multicolumn{2}{c}{Llama} \\"
           r" \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}")
    rows = [top + "\n" + head + r" \\", "\\midrule"]
    for r in R + ["CB", "GOLD"]:
        cols = []
        for c, _, _ in CELLS:
            v = S["cells"][c]
            h = v["retrieval"].get(r, {}).get("hit@5", {}).get("mean")
            cols += [f(h), f(v["answers"][r]["em"]["mean"])]
        rows.append(f"{NAME[r]} & " + " & ".join(cols))
    body = [f"\\begin{{table*}}[t]", "\\centering", "\\small",
            "\\caption{Retrieval (Hit@5) and answer quality (exact match) for each reader and dataset. "
            "Hit@5 is computed on the generation queries; 95\\% bootstrap intervals are in the supplementary tables.}",
            "\\label{tab:main}", "\\begin{tabular}{l" + "rr" * 4 + "}", "\\toprule"] + rows[:1] + rows[1:2] + rows[2:] + \
        ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    body = [b if (b.startswith("\\") or b.endswith(r"\\") or "\n" in b) else b + r" \\" for b in body]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tab_main.tex").write_text("\n".join(body) + "\n")
    print("wrote", OUT / "tab_main.tex")


def t_transfer():
    rows = []
    for r in ["R0", "R2", "R3", "R4", "R5", "R6"]:
        cols = []
        for c, _, _ in CELLS:
            t = next(x for x in S["cells"][c]["transfer_vs_R1"]["em"] if x["retriever"] == r)
            cols.append(f"{t['ratio']:.2f} [{t['ci_lo']:.2f}, {t['ci_hi']:.2f}]" if t["estimable"] else "n.e.")
        rows.append(f"{NAME[r]} & " + " & ".join(cols))
    table("tab_transfer", "lcccc", "Retriever & NQ Qwen & NQ Llama & HotpotQA Qwen & HotpotQA Llama", rows,
          "Transfer ratio $T(R)=\\Delta\\mathrm{EM}/\\Delta\\mathrm{Hit@5}$ relative to MiniLM-L6, with 95\\% bootstrap intervals.",
          "tab:transfer", "n.e.: not estimable under the pre-registered rule ($|\\Delta\\mathrm{Hit@5}|\\le 2$ points or not "
          "significant after Holm). BM25 on NQ retrieves worse than MiniLM-L6, so its ratio describes a loss.", wide=True)


def t_predictions():
    cellname = {"qwen_nq": "NQ, Qwen", "llama_nq": "NQ, Llama", "qwen_hp": "HotpotQA, Qwen", "llama_hp": "HotpotQA, Llama"}
    ok = lambda b: "holds" if b else "\\textbf{fails}"                     # noqa: E731
    rows = []
    for c, lab in cellname.items():
        v = S["cells"][c]["P1"]
        worst = max(v["upper_bounds"].items(), key=lambda kv: kv[1])
        rows.append(f"P1 & {lab} & {ok(v['holds'])} & {v['n_reportable']} ratios; largest upper bound {worst[1]:.3f} ({NAME[worst[0]]})")
    for c, lab in cellname.items():
        v = S["cells"][c]["P2"]["per_encoder"]
        rows.append(f"P2 & {lab} & {ok(S['cells'][c]['P2']['holds'])} & "
                    + "; ".join(f"{NAME[k]} $-$ e5: {x['em_diff_minus_e5']:+.3f} ($p_\\text{{Holm}}$ {pval(x['p_holm'])})" for k, x in v.items()))
    e9 = S["corpus_hnsw_cost"]["E9"]["P3"]
    rows.append(f"P3 & NQ, 4 retrievers & {ok(e9['holds'])} & Kendall's $\\tau$ = {e9['kendall_tau_primary']:.2f}")
    t = S["topk"]["k1_vs_k5"]
    rows.append(f"P4 & NQ, Qwen & {ok(S['topk']['P4']['holds'])} & $k{{=}}1$ below $k{{=}}5$ for all 7 retrievers; "
                f"$\\Delta$EM {min(x['diff'] for x in t):+.3f} to {max(x['diff'] for x in t):+.3f}; all $p_\\text{{Holm}}$ $<$0.001")
    m = S["mechanism"]
    rows.append(f"P5 & NQ, Qwen, 500 q. & {ok(m['P5']['holds'])} & hard vs random negatives: $\\Delta$EM {m['P5']['test']['diff']:+.3f}, $p$ {pval(m['P5']['test']['p'])}")
    rows.append(f"P6 & NQ, Qwen, 500 q. & {ok(m['P6']['holds'])} & gold at position 1/3/5: largest $|\\Delta$EM$|$ "
                f"{max(abs(x['diff']) for x in m['P6']['tests']):.3f}; smallest $p_\\text{{Holm}}$ {min(x['p_holm'] for x in m['P6']['tests']):.2f}")
    table("tab_predictions", "lllp{0.55\\linewidth}", "ID & Scope & Result & Evidence", rows,
          "Pre-registered predictions and their outcomes.", "tab:predictions",
          "P1: every reportable EM transfer ratio has an upper 95\\% bound below 1. P2: no Holm-significant EM gain of bge-large or "
          "Qwen3-Emb-0.6B over e5-base-v2. P3: the full corpus keeps the nDCG@10 ordering. P4: EM at $k{=}1$ below $k{=}5$. "
          "P5: hard negatives lower EM. P6: gold position changes EM.", wide=True)


def t_topk():
    t = S["topk"]
    rows = []
    for r, x in zip(R, t["k1_vs_k5"]):
        e = t["em_by_k"][r]
        rows.append(f"{NAME[r]} & " + " & ".join(f(e[str(k)]["mean"]) for k in (1, 3, 5, 10)) +
                    f" & {x['diff']:+.3f} & {pval(x['p_holm'])}")
    table("tab_topk", "lrrrrrr", "Retriever & $k{=}1$ & $k{=}3$ & $k{=}5$ & $k{=}10$ & $\\Delta_{1-5}$ & $p_\\text{Holm}$", rows,
          "Exact match by number of passages $k$ (Qwen reader, NQ, 2,255 queries).", "tab:topk",
          "Exact McNemar test of $k{=}1$ vs $k{=}5$, Holm over seven retrievers.")


def t_mechanism():
    m = S["mechanism"]
    lab = {"MG": "Gold only", "MR": "Gold@1 + 4 random", "M1": "Gold@1 + 4 hard", "M3": "Gold@3 + 4 hard", "M5": "Gold@5 + 4 hard"}
    rows = [f"{lab[c]} & {f(m['answers'][c]['em']['mean'])} [{f(m['answers'][c]['em']['ci_lo'])}, {f(m['answers'][c]['em']['ci_hi'])}]"
            f" & {f(m['answers'][c]['f1']['mean'])} & {f(m['answers'][c]['contain']['mean'])}" for c in ("MG", "MR", "M1", "M3", "M5")]
    table("tab_mechanism", "lccc", "Context (5 passages) & EM [95\\% CI] & F1 & AC", rows,
          "Distraction and position tests (Qwen reader, first 500 NQ queries).", "tab:mechanism",
          f"Hard vs random negatives: $\\Delta$EM = {m['P5']['test']['diff']:+.3f}, exact McNemar $p$ = {m['P5']['test']['p']:.1e}. "
          "Position: no pair differs after Holm.")


def t_reranker():
    rr = S["reranker"]
    rows = []
    for r in ("R0", "X0", "R5", "X5"):
        lat = rr["latency_ms"].get({"X0": "R0", "X5": "R5"}.get(r), {}).get("median_ms") if r.startswith("X") else None
        rows.append(f"{NAME[r]} & {f(rr['retrieval_all_judged'][r]['hit@5'])} & {f(rr['retrieval_all_judged'][r]['ndcg@10'])} & "
                    f"{f(rr['answers'][r]['em']['mean'])} & {f(lat, 0) if lat else '--'}")
    table("tab_reranker", "lcccc", "System & Hit@5 & nDCG@10 & EM & ms/query", rows,
          "Cross-encoder reranking (bge-reranker-base, top 50) on NQ.", "tab:reranker",
          "Retrieval on all 3,452 judged queries; EM on the 2,255 generation queries (Qwen). Latency: median per query, 50 pairs, one T4.")


def t_cost():
    c = S["corpus_hnsw_cost"]["E7"]
    rows = []
    for r in ["R1", "R2", "R3", "R5", "R4", "R6"]:
        v = c[r]
        rows.append(f"{NAME[r]} & {v['n_params'] / 1e6:.0f} & {v['dim']} & {v['passages_per_s']['median']:.0f} "
                    f"({v['passages_per_s']['min']:.0f}--{v['passages_per_s']['max']:.0f}) & "
                    f"{v['query_latency_ms']['median']:.1f} ({v['query_latency_ms']['min']:.1f}--{v['query_latency_ms']['max']:.1f})")
    table("tab_cost", "lrrcc", "Encoder & Params (M) & Dim & Passages/s & Query ms", rows,
          "Encoding cost on one Tesla T4, median (range) over five repetitions.", "tab:cost",
          f"BM25 batch-1 query latency on CPU: {c['R0']['query_latency_ms']['median']:.1f} ms "
          f"({c['R0']['query_latency_ms']['min']:.1f}--{c['R0']['query_latency_ms']['max']:.1f}).")


def t_corpus_hnsw():
    e9, e6 = S["corpus_hnsw_cost"]["E9"], S["corpus_hnsw_cost"]["E6"]
    rows = [f"{NAME[r]} & {f(e9['subcorpus_same_embeddings'][r]['ndcg@10'])} & {f(e9['full_corpus'][r]['ndcg@10'])} & "
            f"{f(e9['subcorpus_same_embeddings'][r]['hit@5'])} & {f(e9['full_corpus'][r]['hit@5'])}" for r in e9["retrievers"]]
    table("tab_full_corpus", "lcccc", "Retriever & nDCG@10 sub & nDCG@10 full & Hit@5 sub & Hit@5 full", rows,
          "Sub-corpus (300k) vs full NQ corpus (2.68M), all 3,452 judged queries.", "tab:fullcorpus",
          f"Kendall's $\\tau$ between the two orderings = {e9['P3']['kendall_tau_primary']:.2f}.")
    rows = [f"{NAME[r]} & {f(v['exact']['hit@5'])} & {v['exact']['latency']['median_ms']:.1f} & "
            f"{f(v['ef64']['overlap@100'])} & {f(v['ef64']['hit@5'])} & {v['ef64']['latency']['median_ms']:.2f} & "
            f"{f(v['ef128']['overlap@100'])} & {f(v['ef128']['hit@5'])} & {v['ef128']['latency']['median_ms']:.2f}" for r, v in e6.items()]
    table("tab_hnsw", "lcccccccc", "Encoder & \\multicolumn{2}{c}{Exact} & \\multicolumn{3}{c}{HNSW ef=64} & \\multicolumn{3}{c}{HNSW ef=128} \\\\ "
          "& Hit@5 & ms & Overlap & Hit@5 & ms & Overlap & Hit@5 & ms", rows,
          "Approximate search (HNSW, $M{=}32$) vs exact search on the NQ sub-corpus, CPU.", "tab:hnsw",
          "Overlap: mean share of the exact top 100 recovered. ms: median batch-1 query latency on 4 vCPUs.", wide=True)


def t_gee():
    g = S["gee"]
    rows = []
    for k, lab in (("bge-large_vs_e5", "bge-large vs e5-base-v2"), ("qwen3emb_vs_e5", "Qwen3-Emb-0.6B vs e5-base-v2")):
        v = g["contrasts"][k]
        rows.append(f"{lab} & {v['odds_ratio']:.3f} [{v['or_ci95'][0]:.3f}, {v['or_ci95'][1]:.3f}] & {pval(v['p'])}")
    table("tab_gee", "lcc", "Contrast & Odds ratio [95\\% CI] & $p$", rows,
          "Pooled GEE logistic model of EM over the four reader--dataset cells.", "tab:gee",
          f"EM $\\sim$ retriever + dataset + reader; exchangeable working correlation ({g['working_correlation']:.2f}) within query; "
          f"{g['n_rows']:,} answers in {g['n_clusters']:,} query clusters; robust standard errors.")


def t_judge():
    j = S.get("judge_agreement")
    if not j:
        return
    c = j["confusion_human_x_judge"]
    rows = [f"Human SUPPORTED & {c['human_SUPPORTED__judge_SUPPORTED']} & {c['human_SUPPORTED__judge_NOT_SUPPORTED']}",
            f"Human NOT\\_SUPPORTED & {c['human_NOT_SUPPORTED__judge_SUPPORTED']} & {c['human_NOT_SUPPORTED__judge_NOT_SUPPORTED']}"]
    table("tab_judge", "lcc", "& Judge SUPPORTED & Judge NOT\\_SUPPORTED", rows,
          "Agreement between the author's labels and the LLM judge on 200 stratified items.", "tab:judge",
          f"Cohen's $\\kappa$ = {j['cohen_kappa']:.3f} [{j['kappa_ci95'][0]:.3f}, {j['kappa_ci95'][1]:.3f}]; raw agreement "
          f"{j['raw_agreement']:.3f}. Under the pre-registered rule ($\\kappa<0.2$) the judge results are excluded from the conclusions.")


if __name__ == "__main__":
    t_main(); t_transfer(); t_predictions(); t_topk(); t_mechanism(); t_reranker(); t_cost(); t_corpus_hnsw(); t_gee(); t_judge()
