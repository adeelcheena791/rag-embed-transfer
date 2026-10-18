"""Independent check, part 3 (v0.6): the Phase 3 (S5-NQ, full corpus) numbers, without project imports.

Reads the raw files directly (the ranked lists from the committed session zip, the raw reader outputs, the v0.3 inputs)
and recomputes with its own code:
  1. EM of every Phase 3 answer from the raw reader output (own normaliser) vs the scored files;
  2. per-query Hit@5 on the full corpus from the ranked lists, and that each reader saw exactly the top five;
  3. Hit@5 and EM means, EM full - sub with own exact McNemar and Holm over 4;
  4. the 6 pairwise full-corpus Hit@5 tests with Holm (reporting rule), the transfer ratios with the percentile bootstrap
     (resampling index by definition numpy default_rng(0).integers(0, n, (10000, n)); own ratio and quantile code), P7, P8
     and T_full - T_sub for e5-base-v2;
  5. the query-level transition contributions on the full corpus;
  6. the answer-string share among Hit@5 misses for MiniLM-L6 and e5-base-v2 (sub and full), from the stored passage
     texts with an own whole-token containment rule;
  7. P5 on the E12 questions without an answer-bearing hard negative (own McNemar).
Writes outputs/independent_check_v06.json and .md and compares with paper_outputs/v04/v04_full_analysis.json.

    python3 -I manuscript_ipm/revision_v04/scripts/independent_check_v06.py
"""
import io
import json
import math
import re
import string
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "manuscript_ipm/revision_v04/outputs"
S5 = ROOT / "paper_inputs/v04_s5nq"
A = json.load(open(ROOT / "paper_outputs/v04/v04_full_analysis.json"))
S = json.load(open(ROOT / "paper_outputs/v03/statistics/v03_analysis.json"))
RF = ["R0", "R1", "R2", "R5"]
REP, RES, BAD = [], {}, []


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def log(s):
    REP.append(s)
    print(s)


def check(name, mine, theirs, tol=1e-9):
    ok = abs(mine - theirs) <= tol
    if not ok:
        BAD.append(f"{name}: mine {mine} vs analysis {theirs}")
    return ok


def norm(s):
    s = s.lower()
    drop = set(string.punctuation) | {"‘", "’", "“", "”", "–", "—", "…"}
    s = "".join(ch for ch in s if ch not in drop)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def mcnemar(a, b):
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n01 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n10 + n01
    if n == 0:
        return 1.0
    k = min(n10, n01)
    logp = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2) for i in range(n + 1)]
    return min(1.0, sum(math.exp(v) for v in logp if v <= logp[k] + 1e-7))


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def quantile(x, q):
    s = np.sort(x)
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return float(s[lo] + (s[hi] - s[lo]) * (pos - lo))


def mean(x):
    return sum(x) / len(x)


# ------------------------------------------------------------------ data
qrels = {}
with open(ROOT / "paper_inputs/data/raw/nq_qrels/test.tsv") as f:
    next(f)
    for line in f:
        q, d, g = line.rstrip("\n").split("\t")[:3]
        if int(g) > 0:
            qrels.setdefault(q, set()).add(d)
joined = {r["qid"]: r for r in jl(ROOT / "paper_inputs/data/processed/nq_joined.jsonl")}
Q = [l.strip() for l in open(ROOT / "paper_inputs/data/processed/nq_query_order.txt") if l.strip() in joined][:2255]
n = len(Q)
meta = json.load(open(S5 / "v04/passage_answers_meta.json"))["runs"]
full_top = {}
with zipfile.ZipFile(S5 / "v04_s5nq_results.zip") as z:
    for r in RF:
        with z.open("results/raw/runs/" + meta["full:" + r]) as f:
            rows = {}
            for line in io.TextIOWrapper(f, encoding="utf-8"):
                if line.strip():
                    x = json.loads(line)
                    rows[x["qid"]] = x["pids"]
        full_top[r] = rows
sub_top = {r: {x["qid"]: x["pids"] for x in jl(ROOT / f"paper_inputs/results/raw/top10/{r}__main__top10.jsonl")} for r in RF}
hit = {(c, r): [int(bool(set(top[r][q][:5]) & qrels[q])) for q in Q] for c, top in (("full", full_top), ("sub", sub_top)) for r in RF}

# ------------------------------------------------------------------ 1-2. EM re-scored; reader passages
log("## 1. Phase 3 EM re-scored from raw reader outputs; reader passages vs ranked lists")
em_full, tot, dis = {}, 0, 0
for rd in ("qwen", "llama"):
    for r in RF:
        tag = f"v03_nq_{rd}__{r}_k5_A1__full.jsonl"
        gens = {x["qid"]: x for x in jl(S5 / "raw/generations" / tag)}
        scored = {x["qid"]: x["em"] for x in jl(S5 / "processed/answers" / tag)}
        assert all(gens[q]["pids"] == full_top[r][q][:5] for q in Q), tag
        mine = []
        for q in Q:
            out = gens[q]["output"].strip() if gens[q]["output"] else None
            e = float(out is not None and any(norm(out) == norm(g) for g in joined[q]["answers"]))
            mine.append(e)
            tot += 1
            dis += e != scored[q]
        em_full[(rd, r)] = [scored[q] for q in Q]
log(f"- {tot} answers re-scored, {dis} disagreements; every reader saw exactly the top five of the stored ranked list")
RES["rescore"] = {"n": tot, "disagree": dis}
sub_dirs = {"qwen": ("paper_inputs/results/processed/answers", "main"), "llama": ("paper_inputs/v03_nq/processed/answers", "v03_nq_llama")}
em_sub = {}
for rd, (d, pre) in sub_dirs.items():
    for r in RF:
        sc = {x["qid"]: x["em"] for x in jl(ROOT / d / f"{pre}__{r}_k5_A1__main.jsonl")}
        em_sub[(rd, r)] = [sc[q] for q in Q]

# ------------------------------------------------------------------ 3. means and full - sub
log("## 2. Means and full - sub")
for r in RF:
    check(f"hit full {r}", mean(hit[("full", r)]), A["retrieval_full"][r]["hit@5"]["mean"])
    check(f"hit sub {r}", mean(hit[("sub", r)]), A["retrieval_sub"][r]["hit@5"]["mean"])
    RES[f"hit.full.{r}"] = mean(hit[("full", r)])
for rd in ("qwen", "llama"):
    ps = [mcnemar(em_full[(rd, r)], em_sub[(rd, r)]) for r in RF]
    adj = holm(ps)
    fs = {x["retriever"]: x for x in A["readers"][rd]["full_minus_sub"]}
    for r, p in zip(RF, adj):
        d = mean(em_full[(rd, r)]) - mean(em_sub[(rd, r)])
        check(f"EM full {rd} {r}", mean(em_full[(rd, r)]), A["readers"][rd]["answers_full"][r]["em"]["mean"])
        check(f"dEM {rd} {r}", d, fs[r]["em"]["diff"])
        check(f"p_holm4 {rd} {r}", p, fs[r]["em"]["p_holm4"], 1e-9 + 1e-6 * p)
        RES[f"dEM.{rd}.{r}"], RES[f"pEM.{rd}.{r}"] = d, p
        RES[f"EM.full.{rd}.{r}"] = mean(em_full[(rd, r)])
        dh = mean(hit[("full", r)]) - mean(hit[("sub", r)])
        RES[f"dHit.{r}"] = dh
log(f"- checked; mismatches so far: {len(BAD)}")

# ------------------------------------------------------------------ 4. ratios
log("## 3. Reporting rule, transfer ratios, P7, P8")
pairs = [(a, b) for i, a in enumerate(RF) for b in RF[i + 1:]]
padj = dict(zip(pairs, holm([mcnemar(hit[("full", a)], hit[("full", b)]) for a, b in pairs])))
idx = np.random.default_rng(0).integers(0, n, (10000, n))


def ratio(hm, hr, em_m, em_r):
    dh, de = mean(hm) - mean(hr), mean(em_m) - mean(em_r)
    bh = (np.array(hm, float) - np.array(hr, float))[idx].mean(axis=1)
    be = (np.array(em_m, float) - np.array(em_r, float))[idx].mean(axis=1)
    return de / dh, be / bh


for rd in ("qwen", "llama"):
    an = {t["retriever"]: t for t in A["readers"][rd]["transfer_full_vs_R1"]["em"]}
    sub = {t["retriever"]: t for t in S["cells"][f"{rd}_nq"]["transfer_vs_R1"]["em"]}
    for r in ("R0", "R2", "R5"):
        p = padj.get(("R0", "R1") if r == "R0" else ("R1", r))
        ok = abs(mean(hit[("full", r)]) - mean(hit[("full", "R1")])) > 0.02 and p < 0.05
        assert ok == an[r]["estimable"]
        t, b = ratio(hit[("full", r)], hit[("full", "R1")], em_full[(rd, r)], em_full[(rd, "R1")])
        lo, hi = quantile(b, 0.025), quantile(b, 0.975)
        check(f"T {rd} {r}", t, an[r]["ratio"]), check(f"lo {rd} {r}", lo, an[r]["ci_lo"]), check(f"hi {rd} {r}", hi, an[r]["ci_hi"])
        RES[f"T.{rd}.{r}"], RES[f"T.{rd}.{r}.lo"], RES[f"T.{rd}.{r}.hi"] = t, lo, hi
        if r == "R5":
            ts, bs = ratio(hit[("sub", r)], hit[("sub", "R1")], em_sub[(rd, r)], em_sub[(rd, "R1")])
            d = b - bs
            RES[f"dT.{rd}"], RES[f"dT.{rd}.lo"], RES[f"dT.{rd}.hi"] = t - ts, quantile(d, 0.025), quantile(d, 0.975)
            check(f"dT {rd}", t - ts, A["readers"][rd]["T_full_minus_T_sub"]["R5"]["diff"])
            check(f"dT lo {rd}", RES[f"dT.{rd}.lo"], A["readers"][rd]["T_full_minus_T_sub"]["R5"]["ci_lo"])
            p8 = sub[r]["ci_lo"] <= t <= sub[r]["ci_hi"]
            assert p8 == A["readers"][rd]["P8"]["holds"]
            RES[f"P8.{rd}"] = p8
    p7 = all(RES[f"T.{rd}.{r}.hi"] < 1 for r in ("R2", "R5"))
    assert p7 == A["readers"][rd]["P7"]["holds"]
    RES[f"P7.{rd}"] = p7
    log(f"- {rd}: P7 {p7}, P8 {RES[f'P8.{rd}']}; T(e5) {RES[f'T.{rd}.R5']:.4f} [{RES[f'T.{rd}.R5.lo']:.4f}, {RES[f'T.{rd}.R5.hi']:.4f}]")

# ------------------------------------------------------------------ 5. transitions
log("## 4. Transition contributions (full corpus)")
for rd in ("qwen", "llama"):
    for r in ("R2", "R5"):
        h1, hr, e1, er = hit[("full", "R1")], hit[("full", r)], em_full[(rd, "R1")], em_full[(rd, r)]
        tr = A["readers"][rd]["transitions_full"][r]
        for s, a, b in (("miss_to_hit", 0, 1), ("hit_to_hit", 1, 1), ("hit_to_miss", 1, 0), ("miss_to_miss", 0, 0)):
            c = sum(y - x for x, y, u, v in zip(e1, er, h1, hr) if u == a and v == b) / n
            check(f"tr {rd} {r} {s}", c, tr[s]["contribution"])
            RES[f"tr.{rd}.{r}.{s}"] = c
log(f"- checked; mismatches so far: {len(BAD)}")

# ------------------------------------------------------------------ 6. answer strings (R1, R5; texts stored for their misses)
log("## 5. Answer strings among Hit@5 misses (own containment rule, stored passage texts)")
texts = {x["pid"]: f"{x['title'] or ''} {x['text'] or ''}".strip() for x in jl(S5 / "v04/audit_texts.jsonl")}


def contains(p, answers):
    p = f" {norm(p)} "
    return any(norm(a) and f" {norm(a)} " in p for a in answers)


for c, top in (("main", sub_top), ("full", full_top)):
    for r in ("R1", "R5"):
        miss = [q for q in Q if not set(top[r][q][:5]) & qrels[q]]
        k = sum(any(contains(texts[p], joined[q]["answers"]) for p in top[r][q][:5]) for q in miss)
        sh = k / len(miss)
        check(f"answer strings {c} {r}", sh, A["answer_strings"][f"{c}:{r}"]["miss_with_answer"]["share"])
        RES[f"as.{c}.{r}"] = sh
        log(f"- {c} {r}: {k} of {len(miss)} misses ({sh:.4f})")

# ------------------------------------------------------------------ 7. P5 on the clean E12 subset
log("## 6. P5 without answer-bearing hard negatives")
e12 = {x["qid"]: x["n_hard_with_answer"] for x in jl(S5 / "v04/e12_hardneg_answers.jsonl")}
base = ROOT / "paper_inputs/v03_nq_s4/processed/answers"
m1 = {x["qid"]: x["em"] for x in jl(base / "v03_nq_qwen__M1_k5_A1__main.jsonl")}
mr = {x["qid"]: x["em"] for x in jl(base / "v03_nq_qwen__MR_k5_A1__main.jsonl")}
clean = [q for q in Q[:500] if e12[q] == 0]
a, b = [m1[q] for q in clean], [mr[q] for q in clean]
RES["o8.n"], RES["o8.clean.diff"], RES["o8.clean.p"] = sum(1 for q in Q[:500] if e12[q] > 0), mean(a) - mean(b), mcnemar(a, b)
check("o8 diff", RES["o8.clean.diff"], A["O8"]["P5_clean_subset"]["diff"])
check("o8 p", RES["o8.clean.p"], A["O8"]["P5_clean_subset"]["p"], 1e-12 + 1e-6 * RES["o8.clean.p"])
log(f"- {len(clean)} clean questions; M1 - MR = {RES['o8.clean.diff']:+.4f}, p = {RES['o8.clean.p']:.3g}")

# ------------------------------------------------------------------ 8. human audit counts (own Wilson)
log("## 7. Human audit of misses")
import csv as _csv
lab = {int(r["item"]): r for r in _csv.DictReader(open(ROOT / "annotation/v04_miss_audit/labels_author.csv"))}
key = {int(r["item"]): r for r in _csv.DictReader(open(ROOT / "annotation/v04_miss_audit/key.csv"))}
sample = json.load(open(ROOT / "annotation/v04_miss_audit/sample.json"))["qids"]
miss5 = [q for q in Q if not set(sub_top["R5"][q][:5]) & qrels[q]]
assert sample == [miss5[i] for i in np.random.default_rng(20261017).choice(len(miss5), 100, replace=False)]
assert [key[i]["qid"] for i in range(1, 101)] == sample


def wil(k, m, z=1.959963984540054):
    p = k / m
    d = 1 + z * z / m
    c = (p + z * z / (2 * m)) / d
    h = z * math.sqrt(p * (1 - p) / m + z * z / (4 * m * m)) / d
    return c - h, c + h


RES["au.A.yes"] = sum(r["A"] == "yes" for r in lab.values())
RES["au.A.yes.lo"], RES["au.A.yes.hi"] = wil(RES["au.A.yes"], 100)
RES["au.C.yes"] = sum(r["C"] == "yes" for r in lab.values())
RES["au.C.yes.lo"], RES["au.C.yes.hi"] = wil(RES["au.C.yes"], 100)
RES["au.EM"] = sum(int(key[i]["em"]) for i in lab)
cy0 = [i for i in lab if lab[i]["C"] == "yes" and key[i]["em"] == "0"]
RES["au.CyEM0"], RES["au.D.yes"] = len(cy0), sum(lab[i]["D"] == "yes" for i in cy0)
U = json.load(open(OUT / "audit_v04_results.json"))
for a_, b_ in ((RES["au.A.yes"], U["A"]["yes"]["k"]), (RES["au.C.yes"], U["C"]["yes"]["k"]), (RES["au.EM"], U["EM"]["k"]),
               (RES["au.D.yes"], U["D_yes_given_C_yes_EM0"]["k"]), (RES["au.A.yes.lo"], U["A"]["yes"]["wilson"][0])):
    check("audit", a_, b_)
log(f"- sample re-drawn from the seed and equal to the audited items; A yes {RES['au.A.yes']}, C yes {RES['au.C.yes']}, "
    f"EM {RES['au.EM']}, D yes {RES['au.D.yes']} of {RES['au.CyEM0']}")

log(f"\n**Result: {len(BAD)} mismatches.**")
for b in BAD:
    log(f"- {b}")
(OUT / "independent_check_v06.json").write_text(json.dumps(RES, indent=1))
(OUT / "independent_check_v06.md").write_text("# Independent check v0.6 (Phase 3)\n\n" + "\n".join(REP) + "\n")
raise SystemExit(1 if BAD else 0)
