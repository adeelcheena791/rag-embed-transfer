"""Gate 2: independent recomputation of the key numbers, written from scratch with the standard library only.

It does NOT import any project module. It re-derives, from raw files:
  1. exact match (EM) from the raw reader outputs with an own SQuAD-style normaliser, per query, and compares it with
     the scored answer files used by the analysis;
  2. Hit@5 (NQ from ranked lists + qrels; HotpotQA from the passages given to the reader + gold ids);
  3. mean EM / Hit@5 per cell and condition, transfer-ratio point estimates vs MiniLM-L6;
  4. exact McNemar p-values (two-sided binomial, own implementation) for P2 (HotpotQA) and P5;
  5. Cohen's kappa of the human-judge agreement.
and compares each with paper_outputs/v03/statistics/v03_analysis.json. Output: manuscript_ipm/records/independent_check.md
"""
import csv
import json
import math
import re
import string
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
S = json.load(open(ROOT / "paper_outputs/v03/statistics/v03_analysis.json"))
OUT = []


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def norm(s, uni=True):
    s = s.lower()
    # the documented rule (src/evaluation/answer.py docstring and design v0.2): ASCII punctuation plus typographic
    # quotes, en/em dash and ellipsis; re-implemented here, not imported. uni=False gives the plain SQuAD rule.
    drop = set(string.punctuation) | ({"\u2018", "\u2019", "\u201c", "\u201d", "\u2013", "\u2014", "\u2026"} if uni else set())
    s = "".join(ch for ch in s if ch not in drop)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def em(pred, golds, uni=True):
    if pred is None:
        return 0.0
    return float(any(norm(pred, uni) == norm(g, uni) for g in golds))


def mcnemar_p(a, b):
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n01 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n10 + n01
    if n == 0:
        return 1.0
    k = min(n10, n01)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def log(line):
    OUT.append(line)
    print(line)


R = ["R0", "R1", "R2", "R3", "R4", "R5", "R6"]
# ---------------- data
nq_joined = {r["qid"]: r for r in jl(ROOT / "paper_inputs/data/processed/nq_joined.jsonl")}
nq_order = [l.strip() for l in open(ROOT / "paper_inputs/data/processed/nq_query_order.txt") if l.strip() in nq_joined][:2255]
qrels = {}
with open(ROOT / "paper_inputs/data/raw/nq_qrels/test.tsv") as f:
    next(f)
    for row in csv.reader(f, delimiter="\t"):
        if int(row[2]) > 0:
            qrels.setdefault(row[0], set()).add(row[1])
hp_joined = {r["qid"]: r for r in jl(ROOT / "paper_inputs/v03_hp/data/hp_joined.jsonl")}
hp_order = [l.strip() for l in open(ROOT / "paper_inputs/v03_hp/data/hp_query_order.txt") if l.strip() in hp_joined][:2000]

# ---------------- 1. EM re-scored from raw outputs vs scored files
log("## 1. EM re-scored from raw reader outputs (own normaliser) vs scored answer files")
raw_sets = [("HotpotQA Qwen", "paper_inputs/v03_hp/raw/generations", "paper_inputs/v03_hp/processed/answers", "v03_hp_qwen", hp_joined),
            ("HotpotQA Llama", "paper_inputs/v03_hp/raw/generations", "paper_inputs/v03_hp/processed/answers", "v03_hp_llama", hp_joined),
            ("NQ Llama", "paper_inputs/v03_nq/raw/generations", "paper_inputs/v03_nq/processed/answers", "v03_nq_llama", nq_joined),
            ("NQ Qwen top-k (v0.3)", "paper_inputs/v03_nq/raw/generations", "paper_inputs/v03_nq/processed/answers", "v03_nq_qwen", nq_joined),
            ("NQ Qwen S4", "paper_inputs/v03_nq_s4/raw/generations", "paper_inputs/v03_nq_s4/processed/answers", "v03_nq_qwen", nq_joined),
            ("NQ Qwen v0.2 (all conditions, k ablation, A3, A2rep)", "paper_inputs/results/raw/generations", "paper_inputs/results/processed/answers", "main", nq_joined)]
total, disagree, examples, plain_diff = 0, 0, [], 0
for name, gdir, adir, prefix, joined in raw_sets:
    n_s, d_s = 0, 0
    for gf in sorted((ROOT / gdir).glob(f"{prefix}__*.jsonl")):
        scored = {r["qid"]: r["em"] for r in jl(ROOT / adir / gf.name)}
        for r in jl(gf):
            out = r["output"].strip() if r["output"] else None
            mine = em(out, joined[r["qid"]]["answers"])
            plain_diff += mine != em(out, joined[r["qid"]]["answers"], uni=False)
            n_s += 1
            if mine != scored[r["qid"]]:
                d_s += 1
                if len(examples) < 8:
                    examples.append((gf.name, r["qid"], r["output"], joined[r["qid"]]["answers"][:3], scored[r["qid"]], mine))
    total += n_s; disagree += d_s
    log(f"- {name}: {n_s} answers re-scored, {d_s} disagreements")
log(f"- TOTAL: {total} answers, {disagree} disagreements ({100 * disagree / total:.3f}%)")
log(f"- sensitivity: a plain SQuAD normaliser (ASCII punctuation only) changes EM for {plain_diff} of {total} answers "
    f"({100 * plain_diff / total:.2f}%)")
SUMMARY = {"n_answers": total, "n_disagree": disagree, "n_plain_squad_diff": plain_diff}
for e in examples:
    log(f"  - example: {e[0]} {e[1]} output={e[2]!r} golds={e[3]} scored={e[4]} mine={e[5]}")

# ---------------- 2-3. Hit@5, EM means, transfer ratios
log("\n## 2-3. Hit@5, EM means and transfer-ratio point estimates per cell (own computation vs analysis file)")
cells = {"qwen_nq": ("paper_inputs/results/processed/answers", "main", "NQ"),
         "llama_nq": ("paper_inputs/v03_nq/processed/answers", "v03_nq_llama", "NQ"),
         "qwen_hp": ("paper_inputs/v03_hp/processed/answers", "v03_hp_qwen", "HP"),
         "llama_hp": ("paper_inputs/v03_hp/processed/answers", "v03_hp_llama", "HP")}
top10 = {r: {x["qid"]: x["pids"] for x in jl(ROOT / f"paper_inputs/results/raw/top10/{r}__main__top10.jsonl")} for r in R}
maxdiff = 0.0
vec = {}
for cell, (adir, prefix, ds) in cells.items():
    q = nq_order if ds == "NQ" else hp_order
    hits, ems = {}, {}
    for r in R:
        rows = {x["qid"]: x["em"] for x in jl(ROOT / adir / f"{prefix}__{r}_k5_A1__main.jsonl")}
        ems[r] = [rows[x] for x in q]
        if ds == "NQ":
            hits[r] = [float(bool(set(top10[r][x][:5]) & qrels[x])) for x in q]
        else:
            g = {x["qid"]: x["pids"] for x in jl(ROOT / f"paper_inputs/v03_hp/raw/generations/{prefix}__{r}_k5_A1__main.jsonl")}
            hits[r] = [float(bool(set(g[x][:5]) & set(hp_joined[x]["gold_pids"]))) for x in q]
    vec[cell] = (hits, ems)
    for r in R:
        mh, me = sum(hits[r]) / len(q), sum(ems[r]) / len(q)
        ah, ae = S["cells"][cell]["retrieval"][r]["hit@5"]["mean"], S["cells"][cell]["answers"][r]["em"]["mean"]
        maxdiff = max(maxdiff, abs(mh - ah), abs(me - ae))
    tr = {t["retriever"]: t for t in S["cells"][cell]["transfer_vs_R1"]["em"]}
    line = []
    for r in ["R0", "R2", "R3", "R4", "R5", "R6"]:
        dh = sum(hits[r]) / len(q) - sum(hits["R1"]) / len(q)
        de = sum(ems[r]) / len(q) - sum(ems["R1"]) / len(q)
        mine = de / dh if dh else float("nan")
        a = tr[r]["ratio"]
        maxdiff = max(maxdiff, abs(mine - a) if a is not None else 0)
        line.append(f"{r} {mine:.3f}/{'n.e.' if a is None else f'{a:.3f}'}")
    log(f"- {cell}: T(own/analysis) " + ", ".join(line))
log(f"- largest absolute difference over all Hit@5, EM means and ratios: {maxdiff:.2e}")

# ---------------- 4. McNemar
log("\n## 4. Exact McNemar p-values (own binomial) vs analysis (unadjusted p)")
for cell in ("qwen_hp", "llama_hp"):
    hits, ems = vec[cell]
    pw = {(t["a"], t["b"]): t for t in S["cells"][cell]["pairwise_em"]}
    for big in ("R4", "R6"):
        mine = mcnemar_p(ems["R5"], ems[big])
        a = (pw.get(("R5", big)) or pw[(big, "R5")])["p"]
        log(f"- {cell} e5 vs {big}: own p = {mine:.6g}, analysis p = {a:.6g}")
m = {c: {x["qid"]: x["em"] for x in jl(ROOT / f"paper_inputs/v03_nq_s4/processed/answers/v03_nq_qwen__{c}_k5_A1__main.jsonl")} for c in ("M1", "MR")}
q5 = nq_order[:500]
mine = mcnemar_p([m["M1"][x] for x in q5], [m["MR"][x] for x in q5])
log(f"- P5 M1 vs MR: own p = {mine:.6g}, analysis p = {S['mechanism']['P5']['test']['p']:.6g}")

# ---------------- 5. kappa
log("\n## 5. Cohen's kappa (own) vs analysis")
key = {k["item_id"]: k["judge_verdict"] for k in json.load(open(ROOT / "annotation/sample_v1_key.json"))}
lab = {x["item_id"]: x["label"] for x in json.load(open(ROOT / "annotation/judge_validation_labels.json"))["labels"]}
h = [lab[i] == "SUPPORTED" for i in key]
j = [key[i] == "SUPPORTED" for i in key]
n = len(h)
po = sum(a == b for a, b in zip(h, j)) / n
ph, pj = sum(h) / n, sum(j) / n
pe = ph * pj + (1 - ph) * (1 - pj)
k = (po - pe) / (1 - pe)
log(f"- kappa own = {k:.6f}, analysis = {S['judge_agreement']['cohen_kappa']:.6f}; n = {n}")

# ---------------- 6. all other EM means in the analysis file (CB, GOLD, top-k, mechanism, reranker)
log("\n## 6. Other EM means (own mean over the documented query set) vs analysis")


def mean_em(path, qs):
    rows = {x["qid"]: x["em"] for x in jl(ROOT / path)}
    return sum(rows[x] for x in qs) / len(qs)


checks = []
for cell, (adir, prefix, ds) in cells.items():
    q = nq_order if ds == "NQ" else hp_order
    for c, f in (("CB", "CB_A2"), ("GOLD", "GOLD_A1")):
        checks.append((f"{cell} {c}", mean_em(f"{adir}/{prefix}__{f}__main.jsonl", q), S["cells"][cell]["answers"][c]["em"]["mean"]))
for r, ks in S["topk"]["em_by_k"].items():
    for k, v in ks.items():
        if v["source"] == "v0.3":
            path = f"paper_inputs/v03_nq/processed/answers/v03_nq_qwen__{r}_k{k}_A1__main.jsonl"
        else:
            path = f"paper_inputs/results/processed/answers/main__{r}_k{k}_A1__main.jsonl"
        checks.append((f"top-k {r} k={k}", mean_em(path, nq_order), v["mean"]))
for c in ("M1", "M3", "M5", "MR", "MG"):
    checks.append((f"mechanism {c}", mean_em(f"paper_inputs/v03_nq_s4/processed/answers/v03_nq_qwen__{c}_k5_A1__main.jsonl", nq_order[:500]),
                   S["mechanism"]["answers"][c]["em"]["mean"]))
for c in ("X0", "X5"):
    checks.append((f"reranker {c}", mean_em(f"paper_inputs/v03_nq_s4/processed/answers/v03_nq_qwen__{c}_k5_A1__main.jsonl", nq_order),
                   S["reranker"]["answers"][c]["em"]["mean"]))
worst = max(abs(a - b) for _, a, b in checks)
log(f"- {len(checks)} EM means compared; largest absolute difference {worst:.2e}")
for name, a, b in checks:
    if abs(a - b) > 1e-12:
        log(f"  - MISMATCH {name}: own {a:.6f} vs analysis {b:.6f}")

# ---------------- 7. all 21 pairwise EM McNemar tests per cell with own Holm adjustment
log("\n## 7. Pairwise EM tests: own exact McNemar + own Holm (21 pairs per cell) vs analysis")


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


worst_p, worst_h, n_sig_diff = 0.0, 0.0, 0
for cell in cells:
    hits, ems = vec[cell]
    pw = S["cells"][cell]["pairwise_em"]
    own = [mcnemar_p(ems[t["a"]], ems[t["b"]]) for t in pw]
    own_h = holm(own)
    for t, p, ph in zip(pw, own, own_h):
        worst_p = max(worst_p, abs(p - t["p"]))
        worst_h = max(worst_h, abs(ph - t["p_holm"]))
        n_sig_diff += (ph < 0.05) != (t["p_holm"] < 0.05)
log(f"- 84 pairs: largest |p difference| {worst_p:.2e}; largest |Holm p difference| {worst_h:.2e}; "
    f"pairs whose Holm significance at 0.05 differs: {n_sig_diff}")

SUMMARY.update(n_em_means=len(checks), n_pairs=84, holm_sig_diff=n_sig_diff)
json.dump(SUMMARY, open(ROOT / "manuscript_ipm/records/independent_check.json", "w"), indent=1)
(ROOT / "manuscript_ipm/records/independent_check.md").write_text(
    "# Independent recomputation (Gate 2)\n\nScript: manuscript_ipm/tools/independent_check.py (standard library only, no project imports).\n\n" + "\n".join(OUT) + "\n")
