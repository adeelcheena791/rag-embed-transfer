"""v0.5: macros and tables for the revised manuscript, generated from committed analysis outputs only.

    python3 manuscript_ipm/revision_v04/scripts/make_v05_tex.py

Inputs: paper_outputs/v03/statistics/{v03_analysis,v03_extras}.json, the v0.4/v0.5 audit outputs in
manuscript_ipm/revision_v04/outputs/, data-preparation records and environment logs under paper_inputs/.
Writes latex/numbers_v05.tex, latex/tables/*.tex (listed in TABLES) and outputs/evidence_ledger_v05_additions.csv.
Replaces three v0.3 tables (tab_predictions, tab_hnsw, tab_data) with corrected versions; the numbers in them are unchanged.
"""
from __future__ import annotations

import csv
import glob
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "manuscript_ipm/revision_v04/outputs"
TEX = ROOT / "manuscript_ipm/revision_v04/latex"
S = json.load(open(ROOT / "paper_outputs/v03/statistics/v03_analysis.json"))
X = json.load(open(ROOT / "paper_outputs/v03/statistics/v03_extras.json"))
NAME = {"R0": "BM25", "R1": "MiniLM-L6", "R2": "bge-small", "R3": "bge-base", "R4": "bge-large", "R5": "e5-base-v2",
        "R6": "Qwen3-Emb-0.6B"}
DENSE = ["R2", "R3", "R4", "R5", "R6"]
DN = [NAME[r] for r in DENSE]
CELLS = ["qwen_nq", "llama_nq", "qwen_hp", "llama_hp"]
MAC, LED = [], []


def neg(s):
    return re.sub(r"(?<![\w$])-(\d)", r"\\ensuremath{-}\1", s)


def add(cid, value, desc, source):
    MAC.append(f"\\expandafter\\def\\csname ev@{cid}\\endcsname{{{value}}}")
    LED.append([cid, desc, value, source, "make_v05_tex.py", "computed from committed outputs"])


def fx(x, d=2, sign=False):
    return neg(f"{x:+.{d}f}" if sign else f"{x:.{d}f}")


def rng(v, d=2, sign=False):
    lo, hi = fx(min(v), d, sign), fx(max(v), d, sign)
    if lo == hi:
        return lo
    return f"{lo} to {hi}" if (sign or min(v) < 0) else f"{lo}--{hi}"


def table(name, spec, header, rows, caption, label, notes=None, wide=False, size="\\small"):
    env = "table*" if wide else "table"
    body = [f"\\begin{{{env}}}[t]", "\\centering", size, f"\\caption{{{caption}}}", f"\\label{{{label}}}",
            f"\\begin{{tabular}}{{{spec}}}", "\\toprule", header + r" \\", "\\midrule"]
    body += [r if r.startswith(("\\midrule", "\\cmidrule", "\\multicolumn")) and r.rstrip().endswith("\\\\") or
             r.startswith(("\\midrule", "\\cmidrule")) else r + r" \\" for r in rows]
    body += ["\\bottomrule", "\\end{tabular}"]
    if notes:
        body.append(f"\\par\\smallskip\\parbox{{\\linewidth}}{{\\footnotesize {notes}}}")
    body.append(f"\\end{{{env}}}")
    txt = re.sub(r"(?<![\w$\\{-])-(\d)", r"$-$\1", "\n".join(body) + "\n")
    (TEX / "tables" / f"{name}.tex").write_text(txt)


def pval(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


# ------------------------------------------------------------------ A. bounds (concern A)
def bounds():
    ub = {c: {t["retriever"]: t["ci_hi"] for t in S["cells"][c]["transfer_vs_R1"]["em"] if t["estimable"] and t["retriever"] in DENSE}
          for c in CELLS}
    nq = max(max(ub[c].values()) for c in ("qwen_nq", "llama_nq"))
    add("V5.nq.ubmax", f"{nq:.3f}", "largest upper 95% bound of a dense-encoder NQ ratio (unrounded 0.4545)", "v03_analysis.json")
    add("V5.nq.ubmax2", f"{nq:.2f}" if round(nq, 2) >= nq else f"{nq + 0.005:.2f}", "NQ upper bounds stay below this (2 dp)",
        "v03_analysis.json")


# ------------------------------------------------------------------ B. decomposition components
def components():
    rows = list(csv.DictReader(open(OUT / "decomposition_components.csv")))
    for ds, tag in (("NQ", "nq"), ("HotpotQA", "hp")):
        v = [r for r in rows if r["dataset"] == ds]
        add(f"V5.cmp.{tag}.prev", rng([float(r["prevalence"]) for r in v], 3, True), f"hit-prevalence term, {ds}", "decomposition_components.csv")
        add(f"V5.cmp.{tag}.hits", rng([float(r["hits"]) for r in v], 3, True), f"EM-among-hits term, {ds}", "decomposition_components.csv")
        add(f"V5.cmp.{tag}.miss", rng([float(r["misses"]) for r in v], 3, True), f"EM-among-misses term, {ds}", "decomposition_components.csv")
        add(f"V5.cmp.{tag}.hits.pos", str(sum(float(r["hits_lo"]) > 0 for r in v)), f"hits-term intervals above 0, {ds}", "decomposition_components.csv")
        add(f"V5.cmp.{tag}.hits.neg", str(sum(float(r["hits_hi"]) < 0 for r in v)), f"hits-term intervals below 0, {ds}", "decomposition_components.csv")
        add(f"V5.cmp.{tag}.miss.zero", str(sum(float(r["miss_lo"]) <= 0 <= float(r["miss_hi"]) for r in v)),
            f"misses-term intervals containing 0, {ds}", "decomposition_components.csv")
    body = []
    for c in CELLS:
        cl = {"qwen_nq": "NQ Qwen", "llama_nq": "NQ Llama", "qwen_hp": "HotpotQA Qwen", "llama_hp": "HotpotQA Llama"}[c]
        body.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{cl}}}}} \\\\")
        for r in [x for x in rows if x["cell"] == c]:
            body.append(f"{r['encoder']} & {fx(float(r['dEM']), 3, True)} & {fx(float(r['prevalence']), 3, True)} [{fx(float(r['prev_lo']), 3)}, {fx(float(r['prev_hi']), 3)}] & "
                        f"{fx(float(r['hits']), 3, True)} [{fx(float(r['hits_lo']), 3)}, {fx(float(r['hits_hi']), 3)}] & "
                        f"{fx(float(r['misses']), 3, True)} [{fx(float(r['miss_lo']), 3)}, {fx(float(r['miss_hi']), 3)}]")
    table("tab_components", "lcccc", "Encoder & $\\Delta$EM & Hit prevalence & EM among hits & EM among misses", body,
          "Components of the EM gain over MiniLM-L6 (Eq.~\\ref{eq:decomp} multiplied by $\\Delta\\bar h$), with 95\\% bootstrap intervals. Post hoc, descriptive.",
          "tab:components", "Hit prevalence: $\\Delta\\bar h\\,(a_{R_1}-b_{R_1})$; EM among hits: $\\bar h_R\\,\\Delta a$; EM among misses: "
          "$(1-\\bar h_R)\\,\\Delta b$. The three terms add up to $\\Delta$EM exactly. Intervals recompute every term in each resample.",
          wide=True, size="\\footnotesize")


# ------------------------------------------------------------------ Step 4A transitions
def transitions():
    rows = list(csv.DictReader(open(OUT / "transitions_hit5.csv")))
    lab = {"miss_to_hit": "miss $\\to$ hit", "hit_to_hit": "hit $\\to$ hit", "hit_to_miss": "hit $\\to$ miss", "miss_to_miss": "miss $\\to$ miss"}
    body = []
    for s, l in lab.items():
        cells = []
        for ds in ("NQ", "HotpotQA"):
            v = [r for r in rows if r["stratum"] == s and r["dataset"] == ds]
            cells.append(f"{rng([100 * float(r['share']) for r in v], 0)} & {rng([float(r['em_diff_within']) for r in v], 2, True)} & "
                         f"{rng([float(r['contribution_to_dEM']) for r in v], 3, True)}")
        body.append(f"{l} & " + " & ".join(cells))
    tot = {ds: [float(r["dEM_total"]) for r in rows if r["dataset"] == ds and r["stratum"] == "miss_to_hit"] for ds in ("NQ", "HotpotQA")}
    body.append("\\midrule")
    body.append(f"all queries & 100 & & {rng(tot['NQ'], 3, True)} & 100 & & {rng(tot['HotpotQA'], 3, True)}")
    table("tab_transitions", "lcccccc",
          " & \\multicolumn{3}{c}{NQ} & \\multicolumn{3}{c}{HotpotQA} \\\\ \\cmidrule(lr){2-4}\\cmidrule(lr){5-7} "
          "Hit@5 (MiniLM-L6 $\\to$ encoder) & \\% queries & $\\Delta$EM within & Contribution & \\% queries & $\\Delta$EM within & Contribution",
          body, "Where the EM gain over MiniLM-L6 sits: queries grouped by their Hit@5 under MiniLM-L6 and under the dense encoder. "
          "Ranges over the five dense encoders and two readers per dataset. Post hoc, descriptive.", "tab:transitions",
          "$\\Delta$EM within: EM of the encoder minus EM of MiniLM-L6 among the queries of the group. Contribution: the group's "
          "share of the overall EM difference, (sum of paired EM differences in the group) / all queries; the four contributions "
          "add up to the overall $\\Delta$EM exactly. Groups are defined by retrieval outcomes and describe where the EM "
          "difference sits, not why. Per-encoder values with 95\\% bootstrap intervals: Table~\\ref{tab:transdetail}.",
          wide=True, size="\\footnotesize")
    for s in lab:
        for ds, tag in (("NQ", "nq"), ("HotpotQA", "hp")):
            v = [r for r in rows if r["stratum"] == s and r["dataset"] == ds]
            k = s.replace("_to_", "")
            add(f"V5.tr.{tag}.{k}.share", rng([100 * float(r["share"]) for r in v], 0) + "\\%", f"share of queries, {s}, {ds}", "transitions_hit5.csv")
            add(f"V5.tr.{tag}.{k}.within", rng([float(r["em_diff_within"]) for r in v], 2, True), f"EM difference within {s}, {ds}", "transitions_hit5.csv")
            add(f"V5.tr.{tag}.{k}.contrib", rng([float(r["contribution_to_dEM"]) for r in v], 3, True), f"contribution to dEM, {s}, {ds}", "transitions_hit5.csv")
            add(f"V5.tr.{tag}.{k}.embase", rng([float(r["em_minilm"]) for r in v], 2), f"MiniLM-L6 EM within {s}, {ds}", "transitions_hit5.csv")
            add(f"V5.tr.{tag}.{k}.neg", str(sum(float(r["contrib_ci_hi"]) < 0 for r in v)), f"contribution intervals below 0, {s}, {ds}", "transitions_hit5.csv")
            add(f"V5.tr.{tag}.{k}.zero", str(sum(float(r["contrib_ci_lo"]) <= 0 <= float(r["contrib_ci_hi"]) for r in v)),
                f"contribution intervals containing 0, {s}, {ds}", "transitions_hit5.csv")
    # NQ: gross gain from new hits vs net
    nq = [r for r in rows if r["dataset"] == "NQ"]
    pairs = {(r["cell"], r["encoder"]) for r in nq}
    kept = [float(next(r for r in nq if (r["cell"], r["encoder"]) == p and r["stratum"] == "miss_to_hit")["dEM_total"]) /
            float(next(r for r in nq if (r["cell"], r["encoder"]) == p and r["stratum"] == "miss_to_hit")["contribution_to_dEM"]) for p in pairs]
    add("V5.tr.nq.netshare", rng([100 * k for k in kept], 0) + "\\%", "net dEM as a share of the miss->hit contribution, NQ", "transitions_hit5.csv")
    # appendix detail: e5-base-v2 and bge-large
    body = []
    for c in CELLS:
        cl = {"qwen_nq": "NQ Qwen", "llama_nq": "NQ Llama", "qwen_hp": "HotpotQA Qwen", "llama_hp": "HotpotQA Llama"}[c]
        for enc in ("bge-large", "e5-base-v2"):
            v = [r for r in rows if r["cell"] == c and r["encoder"] == enc]
            body.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{cl}, {enc}}}}} \\\\")
            for r in v:
                body.append(f"\\quad {lab[r['stratum']]} & {int(r['n']):,} & {fx(float(r['em_minilm']), 3)} & {fx(float(r['em_encoder']), 3)} & "
                            f"{r['wrong_to_right']} / {r['right_to_wrong']} & {fx(float(r['contribution_to_dEM']), 4, True)} & "
                            f"[{fx(float(r['contrib_ci_lo']), 4)}, {fx(float(r['contrib_ci_hi']), 4)}]")
    table("tab_transdetail", "lcccccc", "Group & $n$ & EM MiniLM-L6 & EM encoder & wrong$\\to$right / right$\\to$wrong & Contribution & 95\\% CI",
          body, "Query-level transitions for bge-large and e5-base-v2 against MiniLM-L6. All 20 encoder--reader--dataset "
          "combinations are in the repository (transitions\\_hit5.csv). Post hoc, descriptive.", "tab:transdetail",
          "Intervals: percentile bootstrap over queries ($B=10{,}000$, seed 0); each query keeps its group, so group sizes vary across "
          "resamples and the interval is for the group's contribution to the population EM difference.", wide=True, size="\\footnotesize")


def hp_evidence():
    rows = list(csv.DictReader(open(OUT / "transitions_hotpotqa_evidence.csv")))
    g = {"evidence_gained": "more supporting passages", "evidence_same": "same number", "evidence_lost": "fewer supporting passages"}
    body = []
    for k, l in g.items():
        v = [r for r in rows if r["group"] == k]
        body.append(f"{l} & {rng([100 * float(r['share']) for r in v], 0)} & {rng([float(r['contribution_to_dEM']) for r in v], 3, True)} & "
                    f"{rng([100 * float(r['share_of_dEM']) for r in v], 0)}")
        tag = k.split("_")[1]
        add(f"V5.ev.{tag}.share", rng([100 * float(r["share"]) for r in v], 0) + "\\%", f"share of HotpotQA queries, {k}", "transitions_hotpotqa_evidence.csv")
        add(f"V5.ev.{tag}.contrib", rng([float(r["contribution_to_dEM"]) for r in v], 3, True), f"contribution to dEM, {k}", "transitions_hotpotqa_evidence.csv")
        add(f"V5.ev.{tag}.pct", rng([100 * float(r["share_of_dEM"]) for r in v], 0) + "\\%", f"share of dEM, {k}", "transitions_hotpotqa_evidence.csv")
        add(f"V5.ev.{tag}.pos", str(sum(float(r["contrib_ci_lo"]) > 0 for r in v)), f"contribution intervals above 0, {k}", "transitions_hotpotqa_evidence.csv")
    c12 = [r for r in rows if r["group"] == "count_1_to_2"]
    add("V5.ev.c12.contrib", rng([float(r["contribution_to_dEM"]) for r in c12], 3, True), "contribution of 1->2 supporting passages", "transitions_hotpotqa_evidence.csv")
    add("V5.ev.c12.pct", rng([100 * float(r["share_of_dEM"]) for r in c12], 0) + "\\%", "share of dEM from 1->2 supporting passages", "transitions_hotpotqa_evidence.csv")
    table("tab_hpevidence", "lccc", "Supporting passages in top 5 (encoder vs MiniLM-L6) & \\% queries & Contribution to $\\Delta$EM & \\% of $\\Delta$EM",
          body, "HotpotQA: EM gain over MiniLM-L6 by change in the number of supporting passages (0, 1 or 2) in the top five. "
                "Ranges over five dense encoders and two readers. Post hoc, descriptive.", "tab:hpevidence",
          "The three contributions add up to $\\Delta$EM exactly. Shares above 100\\% occur because the other two groups contribute small losses.")


# ------------------------------------------------------------------ Step 4C sensitivity summary
def sensitivity():
    def tr(metric, ds):
        v = [t for c in CELLS if S["cells"][c]["dataset"] == ds for t in S["cells"][c]["transfer_vs_R1"][metric]
             if t["estimable"] and t["retriever"] in DENSE]
        return [t["ratio"] for t in v], max(t["ci_hi"] for t in v)
    ab = list(csv.DictReader(open(OUT / "transfer_ratio_alt_baselines.csv")))
    bo = [r for r in csv.DictReader(open(OUT / "transfer_ratio_both5_hotpotqa.csv")) if r["retriever"] != "BM25"]
    rows = []
    for lab, m in (("EM (primary)", "em"), ("token F1", "f1"), ("answer containment", "contain")):
        a, ah = tr(m, "NQ")
        b, bh = tr(m, "HotpotQA")
        rows.append(f"{lab}, Hit@5, MiniLM-L6 & {rng(a)} & {fx(ah)} & {rng(b)} & {fx(bh)}")
    def abr(ds):
        v = [r for r in ab if r["dataset"] == ds and r["baseline"] == "BM25" and r["retriever"] in DN]
        return [float(r["ratio"]) for r in v], max(float(r["ci_hi"]) for r in v)
    a, ah = abr("NQ"); b, bh = abr("HotpotQA")
    rows.append(f"EM, Hit@5, BM25 & {rng(a)} & {fx(ah)} & {rng(b)} & {fx(bh)}")
    rows.append(f"EM, Both@5, MiniLM-L6 & -- & -- & {rng([float(r['ratio_em_per_both5']) for r in bo])} & {fx(max(float(r['ci_hi']) for r in bo))}")
    for k in (1, 3, 10):
        v = [r for r in X["transfer_by_k"]["rows"] if r["k"] == k and r["retriever"] in DENSE]
        rows.append(f"EM, Hit@$k$ at $k={k}$, MiniLM-L6 (Qwen) & {rng([r['ratio'] for r in v])} & {fx(max(r['ci_hi'] for r in v))} & -- & --")
        add(f"V5.k{k}.range", rng([r["ratio"] for r in v]), f"NQ dense ratios at k={k} (Qwen, exploratory)", "v03_extras.json transfer_by_k")
        add(f"V5.k{k}.ubmax", fx(max(r["ci_hi"] for r in v)), f"largest upper bound at k={k}", "v03_extras.json transfer_by_k")
    table("tab_sensitivity", "lcccc", "Answer metric, retrieval metric, baseline & NQ ratios & NQ max. upper & HotpotQA ratios & HotpotQA max. upper",
          rows, "Sensitivity of the dense-encoder transfer ratios (range over five encoders and both readers unless stated). "
                "All rows except the top one are secondary or post hoc.", "tab:sensitivity",
          "Max. upper: largest upper 95\\% bootstrap bound in the row. The $k$ rows match Hit@$k$ to the $k$ passages read "
          "(Qwen reader, NQ only; exploratory, without the reporting rule).", wide=True)
    f1n, _ = tr("f1", "NQ"); acn, _ = tr("contain", "NQ"); f1h, f1hh = tr("f1", "HotpotQA")
    add("V5.f1.nq", rng(f1n), "NQ dense F1 ratios", "v03_analysis.json")
    add("V5.ac.nq", rng(acn), "NQ dense containment ratios", "v03_analysis.json")
    allnq = [h for m in ("em", "f1", "contain") for h in [tr(m, "NQ")[1]]] + [abr("NQ")[1]]
    add("V5.nq.allub", fx(max(allnq)), "largest NQ upper bound over EM/F1/AC and both baselines (k = 5)", "v03_analysis.json; transfer_ratio_alt_baselines.csv")


# ------------------------------------------------------------------ corrected v0.3 tables (C, E, predictions wording)
def predictions():
    cellname = {"qwen_nq": "NQ, Qwen$^\\dagger$", "llama_nq": "NQ, Llama", "qwen_hp": "HotpotQA, Qwen", "llama_hp": "HotpotQA, Llama"}
    met = lambda b: "met" if b else "\\textbf{not met}"  # noqa: E731
    rows = []
    for c, lab in cellname.items():
        ub = {t["retriever"]: t["ci_hi"] for t in S["cells"][c]["transfer_vs_R1"]["em"] if t["estimable"] and t["retriever"] in DENSE}
        worst = max(ub.items(), key=lambda kv: kv[1])
        ok = all(v < 1 for v in ub.values())
        assert ok == S["cells"][c]["P1"]["holds"]
        rows.append(f"P1 & {lab} & {met(ok)} & {len(ub)} dense-encoder ratios; largest upper bound {worst[1]:.3f} ({NAME[worst[0]]})")
    for c, lab in cellname.items():
        v = S["cells"][c]["P2"]["per_encoder"]
        rows.append(f"P2 & {lab} & {met(S['cells'][c]['P2']['holds'])} & "
                    + "; ".join(f"{NAME[k]} $-$ e5: {x['em_diff_minus_e5']:+.3f} ($p_\\text{{Holm}}$ {pval(x['p_holm'])})" for k, x in v.items()))
    e9 = S["corpus_hnsw_cost"]["E9"]["P3"]
    rows.append(f"P3 & NQ, 4 retrievers & {met(e9['holds'])} & Kendall's $\\tau$ = {e9['kendall_tau_primary']:.2f} (retrieval only)")
    t = S["topk"]["k1_vs_k5"]
    rows.append(f"P4 & NQ, Qwen & {met(S['topk']['P4']['holds'])} & $k{{=}}1$ below $k{{=}}5$ for all 7 retrievers$^\\ddagger$; "
                f"$\\Delta$EM {min(x['diff'] for x in t):+.3f} to {max(x['diff'] for x in t):+.3f}; all $p_\\text{{Holm}}$ $<$0.001")
    m = S["mechanism"]
    rows.append(f"P5 & NQ, Qwen, 500 q. & {met(m['P5']['holds'])} & hard vs random negatives: $\\Delta$EM {m['P5']['test']['diff']:+.3f}, $p$ {pval(m['P5']['test']['p'])}")
    rows.append(f"P6 & NQ, Qwen, 500 q. & {met(m['P6']['holds'])} & gold at position 1/3/5: largest $|\\Delta$EM$|$ "
                f"{max(abs(x['diff']) for x in m['P6']['tests']):.3f}; smallest $p_\\text{{Holm}}$ {min(x['p_holm'] for x in m['P6']['tests']):.2f}")
    fp = ROOT / "paper_outputs/v04/v04_full_analysis.json"
    if fp.exists():                                     # Phase 3 (addendum v0.4): P7, P8 on the full NQ corpus
        F = json.load(open(fp))
        for rd, lab in (("qwen", "Qwen"), ("llama", "Llama")):
            p7 = F["readers"][rd]["P7"]
            worst = max(p7["upper_bounds"].items(), key=lambda kv: kv[1])
            rows.append(f"P7 & NQ full corpus, {lab} & {met(p7['holds'])} & {p7['n_reportable_dense']} dense-encoder ratios; "
                        f"largest upper bound {worst[1]:.3f} ({NAME[worst[0]]})")
        for rd, lab in (("qwen", "Qwen"), ("llama", "Llama")):
            p8 = F["readers"][rd]["P8"]
            d = F["readers"][rd]["T_full_minus_T_sub"]["R5"]
            lo, hi = p8["sub_interval_unrounded"]
            rows.append(f"P8 & NQ full corpus, {lab} & {met(p8['holds'])} & e5-base-v2: $T_\\text{{full}}$ {p8['T_full_e5']:.3f} in "
                        f"[{lo:.3f}, {hi:.3f}]; $T_\\text{{full}}-T_\\text{{sub}}$ {d['diff']:+.3f} [{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}]")
    table("tab_predictions", "lllp{0.55\\linewidth}", "ID & Scope & Criterion & Evidence", rows,
          "Registered predictions: whether each stated criterion was met.", "tab:predictions",
          "P1: every reportable EM transfer ratio of a dense encoder has an upper 95\\% bound below 1 (unrounded; a bound of exactly "
          "1.000 does not meet it). P2: no Holm-significant EM gain of bge-large or Qwen3-Emb-0.6B over e5-base-v2. P3: the full "
          "corpus keeps the nDCG@10 ordering. P4: EM at $k{=}1$ below $k{=}5$. P5: hard negatives lower EM. P6: gold position "
          "changes EM. P7 (registered before Phase 3): every reportable full-corpus EM ratio of a dense encoder has an upper "
          "95\\% bound below 1. P8: the full-corpus ratio of e5-base-v2 lies inside its sub-corpus 95\\% interval. "
          "$^\\dagger$Not confirmatory: observed in Phase 1, before registration. $^\\ddagger$Confirmatory for "
          "bge-small, bge-base, e5-base-v2 and Qwen3-Emb-0.6B only. The v0.3 analysis code also counted BM25 in the NQ P1 checks "
          "(six ratios); the registered criterion covers dense encoders, and the verdict is the same either way.", wide=True)


def hnsw():
    e6 = S["corpus_hnsw_cost"]["E6"]
    rows = []
    for r, v in e6.items():
        rows.append(f"{NAME[r]} & {fx(v['exact']['hit@5'], 3)} & {v['exact']['latency']['median_ms']:.1f} & "
                    + " & ".join(f"{fx(v[f'ef{ef}']['overlap@5'], 3)} & {fx(v[f'ef{ef}']['overlap@100'], 3)} & {fx(v[f'ef{ef}']['hit@5'], 3)} & "
                                 f"{v[f'ef{ef}']['latency']['median_ms']:.2f}" for ef in (64, 128)))
    add("V5.hnsw.ov5.min128", fx(min(v["ef128"]["overlap@5"] for v in e6.values()), 3), "smallest top-5 overlap at ef 128", "hnsw.jsonl")
    add("V5.hnsw.ov100.min128", fx(min(v["ef128"]["overlap@100"] for v in e6.values()), 3), "smallest top-100 overlap at ef 128", "hnsw.jsonl")
    add("V5.hnsw.n", f"{next(iter(e6.values()))['n_rows']:,}", "rows in the HNSW index", "hnsw.jsonl")
    table("tab_hnsw", "lcccccccccc",
          "& \\multicolumn{2}{c}{Exact} & \\multicolumn{4}{c}{HNSW efSearch 64} & \\multicolumn{4}{c}{HNSW efSearch 128} \\\\ "
          "\\cmidrule(lr){2-3}\\cmidrule(lr){4-7}\\cmidrule(lr){8-11} Encoder & Hit@5 & ms & Ov@5 & Ov@100 & Hit@5 & ms & Ov@5 & Ov@100 & Hit@5 & ms",
          rows, "Approximate search (HNSW, $M{=}32$, efConstruction 200) against exact search on the NQ sub-corpus, CPU, all 3,452 judged NQ queries.",
          "tab:hnsw", "Ov@5 / Ov@100: mean share of the exact top 5 / top 100 that HNSW also returns. Hit@5 here is over all judged queries, "
          "not the 2,255 generation queries. ms: median batch-1 query latency on 4 vCPUs. No reader was run on HNSW results.",
          wide=True, size="\\footnotesize")


def data_table():
    nq = json.load(open(ROOT / "paper_inputs/data/processed/nq_join_stats.json"))
    hp = json.load(open(ROOT / "paper_inputs/v03_hp/processed/hp_join_stats.json"))
    sn = json.loads(open(ROOT / "paper_inputs/results/raw/subcorpus.jsonl").readline())
    sh = json.loads(open(ROOT / "paper_inputs/v03_hp/raw/subcorpus.jsonl").readline())
    qr = {}
    with open(ROOT / "paper_inputs/data/raw/nq_qrels/test.tsv") as f:
        next(f)
        for line in f:
            q, d, g = line.rstrip("\n").split("\t")[:3]
            if int(g) > 0:
                qr.setdefault(q, set()).add(d)
    hpj = [json.loads(l) for l in open(ROOT / "paper_inputs/v03_hp/data/hp_joined.jsonl")]
    nq_per = [len(v) for v in qr.values()]
    hp_per = sorted({len(set(r["gold_pids"])) for r in hpj})
    assert hp_per == [2]
    rows = [f"Full corpus (passages) & 2,681,468 & 5,233,329",
            f"Judged test queries & {nq['n_beir_judged']:,} & {hp['n_beir_judged']:,}",
            f"Queries joined to reference answers & {nq['n_joined']:,} & {hp['n_joined']:,}",
            f"Generation queries & {S['cells']['qwen_nq']['n_queries']:,} & {S['cells']['qwen_hp']['n_queries']:,} "
            f"({S['cells']['qwen_hp']['by_type']['bridge']['n']:,} bridge, {S['cells']['qwen_hp']['by_type']['comparison']['n']:,} comparison)",
            f"Gold passages per judged query & {sum(nq_per) / len(nq_per):.2f} (range {min(nq_per)}--{max(nq_per)}) & 2 (every query)",
            f"Distinct gold passages in the corpus & {sn['n_gold']:,} & {sh['n_gold']:,} (shared across queries)",
            f"Sub-corpus size & {sn['size']:,} & {sh['size']:,}",
            f"\\quad gold passages & {sn['n_gold']:,} & {sh['n_gold']:,}",
            f"\\quad top-$d$ BM25 passages, non-gold (depth $d$) & {sn['n_bm25_added']:,} ($d$ = {sn['bm25_depth']}) & {sh['n_bm25_added']:,} ($d$ = {sh['bm25_depth']})",
            f"\\quad random passages & {sn['n_random']:,} & {sh['n_random']:,}",
            f"Reference answers & NQ-open (validation) & HotpotQA fullwiki (validation)"]
    table("tab_data", "lll", "Item & NQ & HotpotQA", rows, "Data and sub-corpora.", "tab:data",
          "Corpus sizes as distributed by BEIR. Joins: NQ by normalised question text, HotpotQA by question id. Each HotpotQA query has "
          "exactly two distinct gold passages; 14,810 gold references point to 13,783 distinct passages because some passages "
          "support several questions. Both@5 requires both gold passages of the query in the top five.")


def env_table():
    rows = []
    lab = {"env_main_S1": "Phase 1, NQ retrieval and encoding", "env_main_S3a": "Phase 1, Qwen reader (NQ)",
           "env_main_S4": "Phase 1, judge and CPU cost", "env_v03_S1hp": "Phase 2, HotpotQA retrieval",
           "env_v03_S2hp": "Phase 2, HotpotQA readers", "env_v03_S3nq": "Phase 2, Llama reader (NQ), top-$k$",
           "env_v03_S4nq": "Phase 2, E5, E6, E7, E9, E12", "env_v04_S5nq": "Phase 3, full-corpus NQ readers (S5)"}
    for f in sorted(glob.glob(str(ROOT / "paper_inputs/**/env*.json"), recursive=True)):
        k = Path(f).stem
        if k not in lab:
            continue
        d = json.load(open(f))
        p = d["packages"]
        rows.append(f"{lab[k]} & {d['python']} & {p['torch'].split('+')[0]} & {d['torch_cuda']['cuda']} & {p['transformers']} & "
                    f"{p.get('vllm') or '--'} & {p['numpy']}")
    table("tab_env", "lllllll", "Session & Python & PyTorch & CUDA & Transformers & vLLM & NumPy", rows,
          "Software in each Kaggle session, read from the environment log that each session wrote.", "tab:env",
          "All sessions: bm25s 0.3.11, faiss 1.15.1, sentence-transformers 5.4.1, 4 vCPUs, 31.3 GiB memory, 2 Tesla T4 (14,911 MiB each).",
          wide=True, size="\\footnotesize")


def prompt_counts():
    import sys
    sys.path.insert(0, str(ROOT))
    from src.generation.prompts import PROMPT_HASH
    c, bad = {}, 0
    for f in glob.glob(str(ROOT / "paper_inputs/**/raw/generations/*.jsonl"), recursive=True):
        if "v04_s5nq" in f:                             # Phase 3 records are counted by make_v06_tex.py
            continue
        for line in open(f):
            r = json.loads(line)
            c[r["prompt"]] = c.get(r["prompt"], 0) + 1
            bad += r["prompt_hash"] != PROMPT_HASH[r["prompt"]]
    assert bad == 0, f"{bad} generation records carry a prompt hash that does not match the code"
    for k in ("A1", "A2", "A3"):
        add(f"N.gen.{k}", f"{c[k]:,}", f"generation records with prompt {k} (hash verified)", "paper_inputs/**/raw/generations")


def main():
    (TEX / "tables").mkdir(parents=True, exist_ok=True)
    prompt_counts(); bounds(); components(); transitions(); hp_evidence(); sensitivity(); predictions(); hnsw(); data_table(); env_table()
    (TEX / "numbers_v05.tex").write_text("% Generated by revision_v04/scripts/make_v05_tex.py. Do not edit.\n" + "\n".join(MAC) + "\n")
    with open(OUT / "evidence_ledger_v05_additions.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["claim_id", "claim", "value", "source", "produced_by", "status"])
        w.writerows(LED)
    print(len(MAC), "macros")


if __name__ == "__main__":
    main()
