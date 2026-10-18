"""v0.6: macros and tables for the Phase 3 results (design/addendum_v0.4.md, session S5-NQ).

    python3 manuscript_ipm/revision_v04/scripts/make_v06_tex.py

Inputs: paper_outputs/v04/v04_full_analysis.json (python -m src.analysis.v04_full) and the S5-NQ session records under
paper_inputs/v04_s5nq/. Writes latex/numbers_v06.tex, latex/tables/tab_fullreader.tex, latex/tables/tab_answerstr.tex and
outputs/evidence_ledger_v06_additions.csv.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from make_v05_tex import NAME, ROOT, TEX, OUT, fx, neg, rng, table  # noqa: F401

F = json.load(open(ROOT / "paper_outputs/v04/v04_full_analysis.json"))
S = json.load(open(ROOT / "paper_outputs/v03/statistics/v03_analysis.json"))
MAC, LED = [], []
RD = {"qwen": ("q", "Qwen", "qwen_nq"), "llama": ("l", "Llama", "llama_nq")}
DENSE = ["R2", "R5"]


def add(cid, value, desc, source="v04_full_analysis.json"):
    MAC.append(f"\\expandafter\\def\\csname ev@{cid}\\endcsname{{{value}}}")
    LED.append([cid, desc, value, source, "make_v06_tex.py", "computed from committed outputs"])


def ci(lo, hi, d=2):
    return f"[{fx(lo, d)}, {fx(hi, d)}]"


def pv(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def main():
    R = F["readers"]
    # ---------------------------------------------------------------- transfer on the full corpus
    for rd, (s, lab, cell) in RD.items():
        for r in ("R0", "R2", "R5"):
            t = next(x for x in R[rd]["transfer_full_vs_R1"]["em"] if x["retriever"] == r)
            assert t["estimable"]
            add(f"V6.T.{s}.{r}", fx(t["ratio"]), f"full-corpus EM transfer ratio, {lab}, {NAME[r]} vs MiniLM-L6")
            add(f"V6.T.{s}.{r}.ci", ci(t["ci_lo"], t["ci_hi"]), f"its 95% bootstrap interval")
            add(f"V6.DH.{s}.{r}", fx(t["d_hit"], 3, True), f"full-corpus Hit@5 difference vs MiniLM-L6, {NAME[r]}")
            add(f"V6.DE.{s}.{r}", fx(t["d_em"], 3, True), f"full-corpus EM difference vs MiniLM-L6, {lab}, {NAME[r]}")
        d = R[rd]["T_full_minus_T_sub"]["R5"]
        add(f"V6.dT.{s}", fx(d["diff"], 2, True), f"T_full - T_sub, e5-base-v2, {lab}")
        add(f"V6.dT.{s}.ci", f"[{fx(d['ci_lo'], 2, True)}, {fx(d['ci_hi'], 2, True)}]", "its paired bootstrap interval")
        p8 = R[rd]["P8"]
        add(f"V6.P8.{s}.sub", ci(*p8["sub_interval_unrounded"], d=3), f"sub-corpus interval of T(e5), {lab} (P8 target)")
        add(f"V6.T3.{s}.R5", fx(p8["T_full_e5"], 3), f"T_full(e5), {lab}, 3 dp")
    dense = [t for rd in R for t in R[rd]["transfer_full_vs_R1"]["em"] if t["retriever"] in DENSE]
    add("V6.T.dense.range", rng([t["ratio"] for t in dense]), "range of full-corpus dense ratios (both readers)")
    ub = max(t["ci_hi"] for t in dense)
    add("V6.T.ubmax", fx(ub, 3), "largest upper bound of a full-corpus dense ratio")
    assert ub < 1
    add("V6.T.ubmax.e5", fx(max(R[rd]["P7"]["upper_bounds"]["R5"] for rd in R), 3), "largest upper bound, e5-base-v2, full corpus")
    # ---------------------------------------------------------------- full vs sub
    emd = {rd: {x["retriever"]: x for x in R[rd]["full_minus_sub"]} for rd in R}
    dd = [emd[rd][r]["em"]["diff"] for rd in R for r in ("R1", "R2", "R5")]
    add("V6.dEM.dense", rng(dd, 3, True), "EM full minus sub, dense retrievers (MiniLM, bge-small, e5), both readers")
    add("V6.dEM.dense.abs", rng([-x for x in dd], 3), "EM loss full vs sub, dense retrievers, absolute")
    add("V6.dEM.dense.pmax", pv(max(emd[rd][r]["em"]["p_holm4"] for rd in R for r in ("R1", "R2", "R5"))),
        "largest Holm-adjusted (4) McNemar p, dense, EM full vs sub")
    add("V6.dEM.R0", rng([emd[rd]["R0"]["em"]["diff"] for rd in R], 3, True), "EM full minus sub, BM25")
    add("V6.dEM.R0.pmin", f"{min(emd[rd]['R0']['em']['p_holm4'] for rd in R):.2f}", "smallest Holm p, BM25 EM full vs sub")
    dh = [emd["qwen"][r]["hit5"]["diff"] for r in ("R1", "R2", "R5")]
    add("V6.dHit.dense", rng(dh, 3, True), "Hit@5 full minus sub, dense retrievers")
    add("V6.dHit.dense.abs", rng([-x for x in dh], 3), "Hit@5 loss full vs sub, dense, absolute")
    add("V6.dHit.R0", fx(emd["qwen"]["R0"]["hit5"]["diff"], 3, True), "Hit@5 full minus sub, BM25")
    lossratio = [emd[rd][r]["em"]["diff"] / emd[rd][r]["hit5"]["diff"] for rd in R for r in ("R1", "R2", "R5")]
    add("V6.lossratio", rng(lossratio), "EM loss per Hit@5 loss, full vs sub, dense (not pre-specified)")
    for rd, (s, lab, _) in RD.items():
        for r in ("R0", "R1", "R2", "R5"):
            add(f"V6.EM.{s}.{r}", fx(R[rd]["answers_full"][r]["em"]["mean"], 3), f"full-corpus EM, {lab}, {NAME[r]}")
    for r in ("R0", "R1", "R2", "R5"):
        add(f"V6.HIT.{r}", fx(F["retrieval_full"][r]["hit@5"]["mean"], 3), f"full-corpus Hit@5 on the generation queries, {NAME[r]}")
    # ratios against BM25
    for b in ("full", "sub"):
        v = [t["ratio"] for rd in R for t in R[rd]["transfer_vs_R0"][b] if t["retriever"] in DENSE]
        add(f"V6.b0.{b}", rng(v), f"dense ratios against BM25, {b} corpus, both readers")
    # unparsed answers
    add("V6.unparsed.max", f"{100 * max(R[rd]['answers_full'][r]['rate_unparsed'] for rd in R for r in R[rd]['answers_full']):.1f}\\%",
        "largest share of unparsed reader outputs, full corpus")
    # ---------------------------------------------------------------- transitions on the full corpus
    tr = [R[rd]["transitions_full"][r] for rd in R for r in DENSE]
    add("V6.tr.misshit.contrib", rng([t["miss_to_hit"]["contribution"] for t in tr], 3, True), "miss->hit contribution, full")
    add("V6.tr.hithit.contrib", rng([t["hit_to_hit"]["contribution"] for t in tr], 3, True), "hit->hit contribution, full")
    add("V6.tr.hitmiss.contrib", rng([t["hit_to_miss"]["contribution"] for t in tr], 3, True), "hit->miss contribution, full")
    add("V6.tr.missmiss.contrib", rng([t["miss_to_miss"]["contribution"] for t in tr], 3, True), "miss->miss contribution, full")
    add("V6.tr.misshit.within", rng([t["miss_to_hit"]["em_diff_within"] for t in tr], 2, True), "EM gain within miss->hit, full")
    add("V6.tr.hithit.neg", str(sum(t["hit_to_hit"]["ci_hi"] < 0 for t in tr)), "hit->hit intervals below zero (of 4)")
    add("V6.tr.hithit.zero", str(sum(t["hit_to_hit"]["ci_lo"] <= 0 <= t["hit_to_hit"]["ci_hi"] for t in tr)),
        "hit->hit intervals containing zero (of 4)")
    # ---------------------------------------------------------------- answer strings
    A = F["answer_strings"]
    sub_d = [A[f"main:{r}"]["miss_with_answer"]["share"] for r in ("R1", "R2", "R3", "R4", "R5", "R6")]
    full_d = [A[f"full:{r}"]["miss_with_answer"]["share"] for r in ("R1", "R2", "R5")]
    add("V6.as.miss.sub.dense", rng(sub_d), "share of Hit@5 misses with an answer string in the top 5, sub, dense")
    add("V6.as.miss.full.dense", rng(full_d), "same, full corpus, dense")
    add("V6.as.miss.sub.R0", fx(A["main:R0"]["miss_with_answer"]["share"]), "same, sub, BM25")
    add("V6.as.miss.full.R0", fx(A["full:R0"]["miss_with_answer"]["share"]), "same, full, BM25")
    add("V6.as.miss.sub.R5", fx(A["main:R5"]["miss_with_answer"]["share"]), "same, sub, e5")
    add("V6.as.miss.sub.R5.ci", ci(*A["main:R5"]["miss_with_answer"]["ci"]), "its interval")
    add("V6.as.miss.sub.R5.n", f"{A['main:R5']['miss_with_answer']['n']:,}", "e5 sub misses")
    add("V6.as.hit.all", rng([v["hit_with_answer"]["share"] for k, v in A.items() if ":" in k]), "share of hits with an answer string")
    g = A["gold_rule_check"]["share_queries_gold_contains_answer"]
    add("V6.as.gold", fx(g["mean"]), "share of queries whose gold passage(s) contain an answer string (rule check)")
    add("V6.as.gold.ci", ci(g["ci_lo"], g["ci_hi"]), "its interval")
    # ---------------------------------------------------------------- O8
    O = F["O8"]
    add("V6.o8.n", str(O["n_with_answer_hard_negative"]), "E12 queries with >=1 hard negative containing an answer string")
    add("V6.o8.share", fx(O["share"]), "their share of 500")
    add("V6.o8.wilson", ci(*O["wilson"]), "Wilson 95% interval")
    c = O["P5_clean_subset"]
    add("V6.o8.clean.n", str(c["n"]), "E12 queries with no such hard negative")
    add("V6.o8.clean.M1", fx(c["em_M1"], 3), "EM gold@1 + 4 hard, clean subset")
    add("V6.o8.clean.MR", fx(c["em_MR"], 3), "EM gold@1 + 4 random, clean subset")
    add("V6.o8.clean.diff", fx(c["diff"], 3, True), "P5 difference on the clean subset")
    add("V6.o8.clean.ci", f"[{fx(c['ci_lo'], 3, True)}, {fx(c['ci_hi'], 3, True)}]", "its interval")
    add("V6.o8.clean.p", pv(c["p"]), "exact McNemar p, clean subset")
    a = O["P5_answer_hard_subset_not_prespecified"]
    add("V6.o8.ans.M1", fx(a["em_M1"], 3), "EM M1 on queries with an answer-bearing hard negative (not pre-specified)")
    add("V6.o8.ans.MR", fx(a["em_MR"], 3), "EM MR on the same queries")
    # ---------------------------------------------------------------- session facts
    s5 = ROOT / "paper_inputs/v04_s5nq/raw"
    enc = {json.loads(l)["model"]: json.loads(l) for l in open(s5 / "encode.jsonl") if json.loads(l).get("corpus") == "full"
           and json.loads(l)["run_id"].startswith("20261017")}
    for r in ("R1", "R2", "R5"):
        add(f"V6.enc.{r}", f"{enc[r]['corpus_seconds'] / 60:.0f}", f"minutes to encode the full NQ corpus, {NAME[r]}", "encode.jsonl")
    add("V6.enc.rows", f"{enc['R5']['n_rows']:,}", "passages encoded", "encode.jsonl")
    tim = [json.loads(l) for l in open(s5 / "generation_timing.jsonl")]
    for rd, (s, lab, _) in RD.items():
        runs = [json.loads(l) for l in open(s5 / "generation_runs.jsonl")]
        rid = next(x["run_id"] for x in runs if lab.lower() in x["model_id"].lower())
        add(f"V6.read.{s}", f"{sum(t['wall_seconds'] for t in tim if t['run_id'] == rid) / 60:.0f}",
            f"reader minutes, {lab}, 4 conditions x 2,255 queries", "generation_timing.jsonl")
    e9 = F["checks"]["e9_reproduction"]
    assert all(v["hit@5_now"] == v["hit@5_e9"] and abs(v["ndcg@10_now"] - v["ndcg@10_e9"]) < 1e-12 for v in e9.values())
    add("V6.e9.repro", "identical", "re-encoded full-corpus Hit@5 and nDCG@10 equal the E9 summaries")
    import sys
    sys.path.insert(0, str(ROOT))
    from src.generation.prompts import PROMPT_HASH
    recs = [json.loads(l) for f in sorted((ROOT / "paper_inputs/v04_s5nq/raw/generations").glob("*.jsonl")) for l in open(f)]
    assert len(recs) == 4 * 2 * F["n_queries"] and all(r["prompt"] == "A1" and r["prompt_hash"] == PROMPT_HASH["A1"] for r in recs)
    add("V6.n.gen", f"{len(recs):,}", "reader outputs in Phase 3 (all prompt A1, hash verified)", "paper_inputs/v04_s5nq/raw/generations")

    # ---------------------------------------------------------------- tables
    rows = []
    for rd, (s, lab, cell) in RD.items():
        sub_t = {t["retriever"]: t for t in S["cells"][cell]["transfer_vs_R1"]["em"]}
        full_t = {t["retriever"]: t for t in R[rd]["transfer_full_vs_R1"]["em"]}
        if rd != "qwen":
            rows.append("\\midrule")
        rows.append(f"\\multicolumn{{9}}{{l}}{{\\textit{{{'Qwen2.5-7B' if rd == 'qwen' else 'Llama-3.1-8B'}}}}} \\\\")
        for r in ("R0", "R1", "R2", "R5"):
            e = emd[rd][r]["em"]
            ts = "--" if r == "R1" else f"{fx(sub_t[r]['ratio'])} {ci(sub_t[r]['ci_lo'], sub_t[r]['ci_hi'])}"
            tf = "--" if r == "R1" else f"{fx(full_t[r]['ratio'])} {ci(full_t[r]['ci_lo'], full_t[r]['ci_hi'])}"
            rows.append(f"{NAME[r]} & {fx(F['retrieval_sub'][r]['hit@5']['mean'], 3)} & "
                        f"{fx(F['retrieval_full'][r]['hit@5']['mean'], 3)} & {fx(R[rd]['answers_sub'][r]['em']['mean'], 3)} & "
                        f"{fx(R[rd]['answers_full'][r]['em']['mean'], 3)} & {fx(e['diff'], 3, True)} & "
                        f"{pv(e['p_holm4'])} & {ts} & {tf}")
    table("tab_fullreader", "lccccccll",
          "& \\multicolumn{2}{c}{Hit@5} & \\multicolumn{2}{c}{EM} & & & \\multicolumn{2}{c}{$T$ vs MiniLM-L6 [95\\% CI]} \\\\ "
          "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){8-9} Retriever & sub & full & sub & full & "
          "$\\Delta$EM full$-$sub & $p_\\text{Holm}$ & sub & full", rows,
          "NQ readers on the 300,000-passage sub-corpus and on the full 2,681,468-passage corpus (Phase 3), 2,255 generation queries, "
          "$k=5$. Intervals are 95\\% bootstrap intervals.", "tab:fullreader",
          "Sub-corpus values are those of Table~\\ref{tab:transfer}. $\\Delta$EM: paired difference full $-$ sub (intervals in the repository), exact McNemar test, Holm over the "
          "four retrievers per reader. Full-corpus ratios use the reporting rule with Holm over the six pairs among the four "
          "retrievers. BM25's ratio relates an EM loss to a Hit@5 loss.", wide=True, size="\\footnotesize")
    rows = []
    for r in ("R0", "R1", "R2", "R3", "R4", "R5", "R6"):
        m = A[f"main:{r}"]
        f = A.get(f"full:{r}")
        fm = f"{fx(f['miss_with_answer']['share'])} {ci(*f['miss_with_answer']['ci'])} & {f['miss_with_answer']['n']:,}" if f else "-- & --"
        rows.append(f"{NAME[r]} & {fx(m['miss_with_answer']['share'])} {ci(*m['miss_with_answer']['ci'])} & {m['miss_with_answer']['n']:,} & "
                    f"{fm} & {fx(m['hit_with_answer']['share'])}")
    table("tab_answerstr", "lccccc",
          "& \\multicolumn{2}{c}{Misses, sub-corpus} & \\multicolumn{2}{c}{Misses, full corpus} & Hits, sub \\\\ "
          "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5} Retriever & with answer & $n$ & with answer & $n$ & with answer",
          rows, "Share of Hit@5 misses and hits whose top five passages contain a reference answer string (NQ, 2,255 generation queries).",
          "tab:answerstr",
          "Automatic, lexical check: a normalised reference answer occurs as a whole-token span of a normalised passage "
          f"(title and text), the rule of answer containment. Under this rule the gold passages contain an answer for {fx(g['mean'])} "
          "of the queries. A match does not show that the passage supports the answer.", size="\\footnotesize")
    audit()
    (TEX / "numbers_v06.tex").write_text("% Generated by revision_v04/scripts/make_v06_tex.py. Do not edit.\n" + "\n".join(MAC) + "\n")
    with open(OUT / "evidence_ledger_v06_additions.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["claim_id", "claim", "value", "source", "produced_by", "status"])
        w.writerows(LED)
    for m in MAC:
        print(m.split("ev@")[1].replace("\\endcsname", " "))


def np_and(U):
    t = U["not_prespecified"]["A_by_auto_flag"]
    k = U["not_prespecified"]["kappa_A_vs_auto_yes_no_items"]
    return f"{t['yes|auto=1'] + t['no|auto=0']} of {k['n']} yes/no items ($\\kappa$ = {k['kappa']:.2f})"


def audit():
    """Human audit of misses (addendum v0.4): outputs/audit_v04_results.json from scripts/audit_v04_results.py."""
    f = OUT / "audit_v04_results.json"
    if not f.exists():
        return
    U = json.load(open(f))
    src = "audit_v04_results.json"
    w = lambda x: f"[{x['wilson'][0]:.2f}, {x['wilson'][1]:.2f}]"  # noqa: E731
    for k in ("yes", "no", "unsure"):
        add(f"V6.au.A.{k}", str(U["A"][k]["k"]), f"audit: A = {k} (of 100)", src)
    add("V6.au.A.yes.ci", w(U["A"]["yes"]), "Wilson interval, A = yes", src)
    add("V6.au.B.yes", str(U["B_yes_given_A_yes"]["k"]), "audit: B = yes among A = yes", src)
    add("V6.au.C.yes", str(U["C"]["yes"]["k"]), "audit: reader answer judged correct (of 100)", src)
    add("V6.au.C.yes.ci", w(U["C"]["yes"]), "Wilson interval, C = yes", src)
    add("V6.au.C.no", str(U["C"]["no"]["k"]), "audit: C = no", src)
    add("V6.au.C.unsure", str(U["C"]["unsure"]["k"]), "audit: C = unsure", src)
    add("V6.au.EM", str(U["EM"]["k"]), "EM = 1 among the 100 audited misses", src)
    add("V6.au.CyEM0", str(U["C_yes_EM0"]["k"]), "correct answers with EM = 0", src)
    add("V6.au.D.yes", str(U["D_yes_given_C_yes_EM0"]["k"]), "of these, alias or formatting difference", src)
    add("V6.au.D.no", str(U["C_yes_EM0"]["k"] - U["D_yes_given_C_yes_EM0"]["k"]), "of these, other (no reference matches)", src)
    np_ = U["not_prespecified"]
    add("V6.au.auto", str(np_["auto_flag_share"]["k"]), "automatic answer-string flag among the 100 (not pre-specified)", src)
    agree = np_["A_by_auto_flag"]["yes|auto=1"] + np_["A_by_auto_flag"]["no|auto=0"]
    add("V6.au.agree", f"{agree} of {np_['kappa_A_vs_auto_yes_no_items']['n']}", "human yes/no labels agreeing with the flag", src)
    add("V6.au.kappa", f"{np_['kappa_A_vs_auto_yes_no_items']['kappa']:.2f}", "Cohen's kappa, A vs flag, yes/no items", src)
    add("V6.au.CbyA.yes", f"{np_['C_yes_by_A']['yes']['k']} of {np_['C_yes_by_A']['yes']['n']}", "correct answers when A = yes", src)
    add("V6.au.CbyA.no", f"{np_['C_yes_by_A']['no']['k']} of {np_['C_yes_by_A']['no']['n']}", "correct answers when A = no", src)

    rows = [f"A: a passage contains a correct answer & yes / no / unsure & {U['A']['yes']['k']} / {U['A']['no']['k']} / {U['A']['unsure']['k']} & {w(U['A']['yes'])}",
            f"B: that passage is about the question (of A = yes) & yes & {U['B_yes_given_A_yes']['k']} of {U['B_yes_given_A_yes']['n']} & {w(U['B_yes_given_A_yes'])}",
            f"C: reader answer factually correct & yes / no / unsure & {U['C']['yes']['k']} / {U['C']['no']['k']} / {U['C']['unsure']['k']} & {w(U['C']['yes'])}",
            f"\\quad EM = 1 (automatic) & & {U['EM']['k']} & {w(U['EM'])}",
            f"D: alias or formatting only (of C = yes, EM = 0) & yes & {U['D_yes_given_C_yes_EM0']['k']} of {U['D_yes_given_C_yes_EM0']['n']} & {w(U['D_yes_given_C_yes_EM0'])}"]
    table("tab_humanaudit", "p{0.42\\linewidth}lll", "Label & Value & Count & 95\\% CI", rows,
          "Blinded human audit of 100 seeded Hit@5 misses of e5-base-v2 (NQ sub-corpus, Qwen reader; one annotator, the author).",
          "tab:humanaudit", "Wilson intervals for the first value of each row. Screen 2 (C, D) was opened only after all Screen 1 labels (A, B) "
          "were complete. Not pre-specified: the automatic answer-string flag agreed with A on " + np_and(U) + ".", size="\\footnotesize")


if __name__ == "__main__":
    main()
