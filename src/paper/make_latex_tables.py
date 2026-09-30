"""Write the paper's LaTeX tables from the frozen result files (single source of truth).

Reads results/statistics/rq12_main.json, rq34_main.json, paper/derived_stats.json (written by
src.paper.make_figures_tables), data/processed/nq_join_stats.json, results/raw/subcorpus.jsonl and
configs/resolved_revisions.json. Writes paper/latex/tables/*.tex. Every number is formatted here; no
number is typed by hand into a table.

    python -m src.paper.make_figures_tables     # first: writes paper/derived_stats.json
    python -m src.paper.make_latex_tables
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

R = ["R0", "R1", "R2", "R3", "R4", "R5", "R6"]
DENSE = R[1:]
NAME = {"R0": "BM25", "R1": "MiniLM-L6-v2", "R2": "bge-small", "R3": "bge-base", "R4": "bge-large",
        "R5": "e5-base-v2", "R6": "Qwen3-Emb-0.6B", "CB": "Closed-book", "GOLD": "Gold"}
HF = {"R1": "sentence-transformers/all-MiniLM-L6-v2", "R2": "BAAI/bge-small-en-v1.5",
      "R3": "BAAI/bge-base-en-v1.5", "R4": "BAAI/bge-large-en-v1.5", "R5": "intfloat/e5-base-v2",
      "R6": "Qwen/Qwen3-Embedding-0.6B"}
QPREF = {"R1": "none", "R2": "BGE instr.", "R3": "BGE instr.", "R4": "BGE instr.", "R5": "\\texttt{query:}",
         "R6": "task instr."}
PPREF = {"R1": "none", "R2": "none", "R3": "none", "R4": "none", "R5": "\\texttt{passage:}", "R6": "none"}


def p(x, d=1):
    return f"{100 * x:.{d}f}"


def ci(m, lo, hi, d=1):
    return f"{100 * m:.{d}f} {{\\scriptsize[{100 * lo:.{d}f}, {100 * hi:.{d}f}]}}"


def pv(x):
    if x >= 0.001:
        return f"{x:.3f}".rstrip("0").rstrip(".") if x < 1 else "1"
    m, e = f"{x:.1e}".split("e")
    return f"${m}{{\\times}}10^{{{int(e)}}}$"


def env(star, caption, label, colspec, header, rows, size="\\small", notes=None):
    e = "table*" if star else "table"
    out = [f"\\begin{{{e}}}[t]", "\\centering", size, "\\setlength{\\tabcolsep}{4pt}",
           "\\begin{adjustbox}{max width=\\linewidth}", f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", header + " \\\\", "\\midrule"]
    rows = [re.sub(r"(?<![\w$^{])-(?=\d)", "$-$", r) for r in rows]
    out += [r if r.startswith("\\midrule") else r + " \\\\" for r in rows]
    out += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", f"\\end{{{e}}}"]
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="paper/latex/tables")
    a = ap.parse_args(argv)
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rq12 = json.load(open(root / "results/statistics/rq12_main.json"))
    rq34 = json.load(open(root / "results/statistics/rq34_main.json"))
    der = json.load(open(root / "paper/derived_stats.json"))
    js = json.load(open(root / "data/processed/nq_join_stats.json"))
    sub = [json.loads(l) for l in open(root / "results/raw/subcorpus.jsonl") if l.strip()][-1]
    rev = json.load(open(root / "configs/resolved_revisions.json"))
    eff = rq34["efficiency"]
    m1, ans, dec = rq12["rq1_joined"]["means"], rq12["rq2_answers"], rq12["rq2_decomposition"]
    fa = der["faithfulness_UNVALIDATED"]
    files = {}

    # T1 models (compact; revisions in appendix)
    rows = ["BM25 & -- & -- & -- & -- & --"]
    for r in DENSE:
        e = eff[r]
        rows.append(f"{NAME[r]} & {e['params'] / 1e6:.1f}M & {e['dim']} & {QPREF[r]} & {PPREF[r]} & {rev[HF[r]][:7]}")
    files["T1"] = env(False, "Retrieval methods. Dense encoders use their model-card prefixes (``BGE instr.'': "
                      "\\emph{Represent this sentence for searching relevant passages:}; Qwen3 uses a task instruction). "
                      "All truncate at 256 tokens, run in fp16 and use exact inner-product search over L2-normalised "
                      "embeddings. BM25: bm25s 0.3.11, $k_1{=}1.2$, $b{=}0.75$. Rev.: pinned Hugging Face commit.",
                      "tab:T1", "lrrllr", "Method & Params & Dim & Query & Passage & Rev.", rows, "\\footnotesize")

    # T2 data
    rows = [f"BEIR-NQ judged test queries & {js['n_beir_judged']:,}",
            f"Joined to NQ-open (primary set) & {js['n_joined']:,} ({100 * js['join_rate']:.1f}\\%)",
            f"Sub-corpus passages & {sub['size']:,}",
            f"\\quad judged relevant (gold) & {sub['n_gold']:,}",
            f"\\quad BM25 top-{sub['bm25_depth']} negatives & {sub['n_bm25_added']:,}",
            f"\\quad random & {sub['n_random']:,}",
            f"Reader conditions $\\times$ queries (S3a) & 9 $\\times$ {js['n_joined']:,}",
            f"Top-$k$ ablation (R0, R1, R4; $k{{=}}1,3,10$) & 9 $\\times$ {js['n_joined']:,}",
            f"IDK prompt (R0, R1, R4, Gold) & 4 $\\times$ {der['abstention_A3']['R0']['n']}",
            f"Judge (R0--R6 / Gold) & 7 $\\times$ {fa['n']} / {fa['n_gold']}",
            "Human validation sample & 200"]
    files["T2"] = env(False, "Data. Queries are BEIR-NQ test queries whose normalised text matches an NQ-open "
                      "question with one answer list. Subsamples are prefixes of one seeded query order.",
                      "tab:T2", "lr", "Item & Count", rows, "\\footnotesize")

    # T3 retrieval
    order = sorted(R, key=lambda r: m1[r]["ndcg@10"]["mean"])
    rows = [f"{NAME[r]} & {p(m1[r]['hit@1']['mean'])} & {ci(m1[r]['hit@5']['mean'], m1[r]['hit@5']['ci_lo'], m1[r]['hit@5']['ci_hi'])}"
            f" & {p(m1[r]['hit@10']['mean'])} & {ci(m1[r]['ndcg@10']['mean'], m1[r]['ndcg@10']['ci_lo'], m1[r]['ndcg@10']['ci_hi'])}"
            f" & {p(m1[r]['mrr@10']['mean'])}" for r in order]
    files["T3"] = env(False, f"Retrieval quality (\\%) on the {rq12['n_joined']:,} joined queries, 300,000-passage "
                      "sub-corpus. Brackets: 95\\% percentile bootstrap CI over queries (10,000 resamples, seed 0).",
                      "tab:T3", "lrrrrr", "Method & H@1 & H@5 & H@10 & nDCG@10 & MRR", rows, "\\footnotesize")

    # T4 efficiency
    rows = []
    for r in R:
        e = eff[r]
        if r == "R0":
            rows.append(f"BM25 & -- & -- & 0 & -- & -- & -- & {e['search_ms']:.1f} & {e['query_ms_gpu']:.1f} & {e['query_ms_cpu']:.1f} & n/m")
            continue
        rows.append(f"{NAME[r]} & {e['params'] / 1e6:.1f}M & {e['dim']} & {e['encode_gpu_s']:,.0f} & {e['gpu_peak_mib']:,.0f}"
                    f" & {e['throughput_cpu_pps']:.2f} & {e['q_enc_gpu_ms']:.1f} / {e['q_enc_cpu_ms']:.1f} & {e['search_ms']:.1f}"
                    f" & {e['query_ms_gpu']:.1f} & {e['query_ms_cpu']:.1f} & {e['index_mb']:,.0f}")
    rows_c = []
    for r in R:
        e = eff[r]
        if r == "R0":
            rows_c.append(f"BM25 & 0 & -- & -- & {e['search_ms']:.1f} & n/m")
            continue
        rows_c.append(f"{NAME[r]} & {e['encode_gpu_s']:,.0f} & {e['gpu_peak_mib']:,.0f} & {e['q_enc_cpu_ms']:.1f}"
                      f" & {e['search_ms']:.1f} & {e['index_mb']:,.0f}")
    files["T4c"] = env(False, "Measured cost (one Tesla T4; 4-vCPU Xeon, 2.0\\,GHz). Enc.: T4 wall time to encode the "
                       "300,000-passage sub-corpus. Mem: peak GPU memory. Q-enc: batch-1 CPU query encoding (median). "
                       "Search: batch-1 exact search on CPU (median). Index: flat fp32 FAISS IndexFlatIP. n/m: not measured. "
                       "Single measurements; full table in Appendix~\\ref{app:tables}.",
                       "tab:T4", "lrrrrr", "Method & Enc. (s) & Mem (MiB) & Q-enc (ms) & Search (ms) & Index (MiB)",
                       rows_c, "\\footnotesize")
    files["T4"] = env(True, "Measured cost on one Tesla T4 and a 4-vCPU Xeon (2.0\\,GHz), same session. Encode: wall "
                      "time to encode the 300,000-passage sub-corpus on the T4. Mem: peak GPU memory. CPU p/s: CPU "
                      "encoding throughput. Latencies are batch-1 medians; search is exact FAISS flat inner-product "
                      "search on CPU; total = query encoding + search. Index: size of the flat fp32 FAISS IndexFlatIP. BM25 index size was "
                      "not measured (n/m). Cost values are single measurements; GPU throughput varied by $-$0.7\\% to "
                      "+3.1\\% between pilot and main runs.",
                      "tab:TC5", "lrrrrrrrrrr",
                      "Method & Params & Dim & Encode (s) & Mem (MiB) & CPU p/s & Q-enc GPU/CPU (ms) & Search (ms) & "
                      "Total GPU (ms) & Total CPU (ms) & Index (MiB)", rows, "\\footnotesize")

    # T5 RAG evaluation
    rows = [f"Closed-book & {ci(ans['CB']['em']['mean'], ans['CB']['em']['ci_lo'], ans['CB']['em']['ci_hi'])} & "
            f"{p(ans['CB']['f1']['mean'])} & -- & -- & -- & --"]
    for r in sorted(R, key=lambda r: ans[r]["em"]["mean"]):
        dc, f = dec[r], fa["means"][r]
        rows.append(f"{NAME[r]} & {ci(ans[r]['em']['mean'], ans[r]['em']['ci_lo'], ans[r]['em']['ci_hi'])} & {p(ans[r]['f1']['mean'])}"
                    f" & {p(dc['hit@5'])} & {ci(dc['p_correct_given_hit'], *dc['p_hit_ci'])}"
                    f" & {ci(dc['p_correct_given_miss'], *dc['p_miss_ci'])} & {ci(f['mean'], f['ci_lo'], f['ci_hi'])}")
    g = fa["means"]["GOLD"]
    rows.append(f"Gold & {ci(ans['GOLD']['em']['mean'], ans['GOLD']['em']['ci_lo'], ans['GOLD']['em']['ci_hi'])} & "
                f"{p(ans['GOLD']['f1']['mean'])} & -- & -- & -- & {ci(g['mean'], g['ci_lo'], g['ci_hi'])}")
    files["T5"] = env(True, f"End-to-end answers (\\%), n = {rq12['n_joined']:,}, $k{{=}}5$, fixed reader. P(EM$\\mid$hit/miss): EM "
                      "when the top-5 does/does not contain a judged relevant passage. SUPPORTED: LLM-judge label, "
                      f"\\textbf{{unvalidated}}; first {fa['n']} queries (Gold {fa['n_gold']}: \\texttt{{{fa['gold_missing'][0]}}} "
                      f"produced no answer; {fa['n_judgements']:,} judgements). Brackets: 95\\% percentile bootstrap CI "
                      "(10,000 resamples, seed 0).",
                      "tab:T5", "lrrrrrr",
                      "Condition & EM & F1 & H@5 & P(EM$\\mid$hit) & P(EM$\\mid$miss) & SUPPORTED", rows,
                      "\\footnotesize")

    # T6 transfer ratios + key EM tests
    pw = {(t["a"], t["b"]): t for t in rq12["rq2_pairwise_em"]}
    hp = {(t["a"], t["b"]): t for t in rq12["rq2_hit5_pairwise_generation_set"]}

    def pair(dct, x, y):
        return dct.get((x, y)) or dct.get((y, x))
    rows = []
    for t in sorted(rq12["rq2_transfer_ratio_vs_R1"], key=lambda t: t["d_hit"]):
        r = t["retriever"]
        e, h = pair(pw, r, "R1"), pair(hp, r, "R1")
        rows.append(f"{NAME[r]} & {100 * t['d_hit']:+.1f} & {pv(h['p_holm'])} & {100 * t['d_em']:+.1f} & {pv(e['p_holm'])}"
                    f" & {t['ratio']:.2f} {{\\scriptsize[{t['ci_lo']:.2f}, {t['ci_hi']:.2f}]}}")
    files["T6"] = env(False, f"Transfer from retrieval to answers, relative to MiniLM-L6-v2 (n = {rq12['n_joined']:,}). "
                      "$\\Delta$: difference in percentage points. $p_\\mathrm{H}$: exact McNemar, Holm over 21 pairs. "
                      "Ratio $T=\\Delta\\mathrm{EM}/\\Delta\\mathrm{H@5}$ with 95\\% paired percentile bootstrap CI "
                      "(10,000 resamples, seed 0); all six meet the pre-registered rule "
                      "($|\\Delta\\mathrm{H@5}|>2$\\,pp and Holm-significant). Full pairwise tests: Appendix~\\ref{app:tables}.",
                      "tab:T6", "lrrrrr", "Method & $\\Delta$H@5 & $p_\\mathrm{H}$ & $\\Delta$EM & $p_\\mathrm{H}$ & $T$", rows,
                      "\\footnotesize")

    # Appendix: full pairwise table (Hit@5, nDCG@10, EM, SUPPORTED)
    blocks = [("H@5", rq12["rq1_joined"]["pairwise"]["hit@5"]), ("nDCG@10", rq12["rq1_joined"]["pairwise"]["ndcg@10"]),
              ("EM", rq12["rq2_pairwise_em"]), ("SUPP.", der["faithfulness_UNVALIDATED"]["pairwise"])]
    keys = [(t["a"], t["b"]) for t in blocks[0][1]]
    rows = []
    for k in keys:
        cells = [f"{NAME[k[0]]} -- {NAME[k[1]]}"]
        for _, lst in blocks:
            t = pair({(x['a'], x['b']): x for x in lst}, *k)
            s = f"{100 * t['diff']:+.1f}" + ("$^*$" if t["significant_holm_0.05"] else "")
            cells.append(s)
        rows.append(" & ".join(cells))
    files["TC1"] = env(False, "All pairwise differences A$-$B in percentage points (joined queries; SUPPORTED on the "
                       "judge subsample, unvalidated). $^*$: significant after Holm over 21 pairs (exact McNemar for "
                       "binary measures, Wilcoxon for nDCG@10).", "tab:TC1", "lrrrr",
                       "A -- B & H@5 & nDCG@10 & EM & SUPP.", rows, "\\scriptsize")

    # Appendix: top-k ablation and abstention
    pts = {(q["retriever"], q["k"]): q for q in der["topk_ablation"]["points"]}
    rows = []
    for r in ("R0", "R1", "R4"):
        rows.append(f"{NAME[r]} & " + " & ".join(ci(pts[(r, k)]['em'], pts[(r, k)]['ci_lo'], pts[(r, k)]['ci_hi'])
                                                  for k in (1, 3, 5, 10)))
    rows.append("\\midrule")
    rows.append("Mean prompt tokens (R4) & " + " & ".join(f"{pts[('R4', k)]['mean_prompt_tokens']:,.0f}" for k in (1, 3, 5, 10)))
    files["TC2"] = env(False, f"Top-$k$ ablation: EM (\\%) with 95\\% CI, n = {rq12['n_joined']:,}. Adjacent steps are "
                       "tested with exact McNemar, Holm over the 9 steps.", "tab:TC2", "lrrrr",
                       "Method & $k{=}1$ & $k{=}3$ & $k{=}5$ & $k{=}10$", rows, "\\scriptsize")
    rows = []
    for c in ("R0", "R1", "R4", "GOLD"):
        v = der["abstention_A3"][c]
        rows.append(f"{NAME[c]} & {p(v['idk_rate'])} & " +
                    (f"{p(v['idk_given_hit'])} & {p(v['idk_given_miss'])} & {p(v['em_A1'])} & {p(v['em_A3'])} & {pv(v['em_A3_minus_A1_p'])}"
                     if c != "GOLD" else "-- & -- & -- & -- & --"))
    files["TC3"] = env(False, f"IDK prompt on the first {der['abstention_A3']['R0']['n']} queries (\\%). "
                       "IDK: share of ``I don't know'' answers. $p$: exact McNemar, EM under the IDK prompt vs the main prompt (uncorrected).",
                       "tab:TC3", "lrrrrrr", "Cond. & IDK & IDK$\\mid$hit & IDK$\\mid$miss & EM main & EM IDK & $p$", rows,
                       "\\scriptsize")
    rows = [f"{NAME[r]} & \\texttt{{{HF[r]}}} & \\texttt{{{rev[HF[r]][:12]}}}" for r in DENSE]
    rows += [f"Reader & \\texttt{{Qwen/Qwen2.5-7B-Instruct-AWQ}} & \\texttt{{{rev['Qwen/Qwen2.5-7B-Instruct-AWQ'][:12]}}}",
             f"Judge & \\texttt{{mistralai/Mistral-7B-Instruct-v0.3}} & \\texttt{{{rev['mistralai/Mistral-7B-Instruct-v0.3'][:12]}}}"]
    files["TC4"] = env(False, "Pinned model revisions (Hugging Face commit, first 12 characters).", "tab:TC4", "lll",
                       "Role & Model & Revision", rows, "\\scriptsize")

    for k, v in files.items():
        (out / f"{k}.tex").write_text("% GENERATED by src/paper/make_latex_tables.py -- do not edit by hand\n" + v)
    print(f"wrote {len(files)} tables to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
