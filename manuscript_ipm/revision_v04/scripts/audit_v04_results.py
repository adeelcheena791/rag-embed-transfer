"""Human audit of NQ retrieval misses (design/addendum_v0.4.md): counts with Wilson 95% intervals.

    python3 manuscript_ipm/revision_v04/scripts/audit_v04_results.py

Input: annotation/v04_miss_audit/labels_author.csv (the author's labels, exported from the labeling page) and key.csv
(automatic answer-string flag, EM, gold ids; not shown during labelling). Output: outputs/audit_v04_results.json.
Pre-specified: counts of A, B, C, D with Wilson intervals. Not pre-specified (labelled so): agreement of A with the automatic
answer-string flag; C by A.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
AUD = ROOT / "annotation/v04_miss_audit"
OUT = ROOT / "manuscript_ipm/revision_v04/outputs"


def wilson(k, n, z=1.959963984540054):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [c - h, c + h]


def share(k, n):
    return {"k": k, "n": n, "share": k / n, "wilson": wilson(k, n)}


def main():
    L = {int(r["item"]): r for r in csv.DictReader(open(AUD / "labels_author.csv"))}
    K = {int(r["item"]): r for r in csv.DictReader(open(AUD / "key.csv"))}
    assert sorted(L) == sorted(K) == list(range(1, 101))
    for i, r in L.items():                                       # label consistency (the page enforced it)
        em = int(K[i]["em"])
        assert (r["A"] == "yes") == bool(r["A_passage"]) == bool(r["B"])
        assert bool(r["D"]) == (r["C"] == "yes" and em == 0)
    n = 100
    A = Counter(r["A"] for r in L.values())
    C = Counter(r["C"] for r in L.values())
    ay = [i for i in L if L[i]["A"] == "yes"]
    cy0 = [i for i in L if L[i]["C"] == "yes" and int(K[i]["em"]) == 0]
    res = {
        "n": n, "seed": 20261017,
        "A": {v: share(A[v], n) for v in ("yes", "no", "unsure")},
        "A_passage_rank": dict(sorted(Counter(int(L[i]["A_passage"]) for i in ay).items())),
        "B_yes_given_A_yes": share(sum(L[i]["B"] == "yes" for i in ay), len(ay)),
        "C": {v: share(C[v], n) for v in ("yes", "no", "unsure")},
        "EM": share(sum(int(K[i]["em"]) for i in L), n),
        "C_yes_EM0": share(len(cy0), n),
        "D_yes_given_C_yes_EM0": share(sum(L[i]["D"] == "yes" for i in cy0), len(cy0)),
    }
    # not pre-specified: agreement with the automatic answer-string flag; correctness by A
    auto = {i: int(K[i]["auto_answer_string_in_top5"]) for i in L}
    tab = Counter((L[i]["A"], auto[i]) for i in L)
    res["not_prespecified"] = {
        "A_by_auto_flag": {f"{a}|auto={f}": tab[(a, f)] for a in ("yes", "no", "unsure") for f in (1, 0)},
        "auto_flag_share": share(sum(auto.values()), n),
        "C_yes_by_A": {a: share(sum(L[i]["C"] == "yes" for i in L if L[i]["A"] == a), A[a]) for a in ("yes", "no", "unsure")},
    }
    yy, yn = tab[("yes", 1)], tab[("yes", 0)]
    ny, nn = tab[("no", 1)], tab[("no", 0)]
    m = yy + yn + ny + nn                                          # Cohen's kappa on yes/no items
    po = (yy + nn) / m
    pe = ((yy + yn) * (yy + ny) + (ny + nn) * (yn + nn)) / m ** 2
    res["not_prespecified"]["kappa_A_vs_auto_yes_no_items"] = {"n": m, "kappa": (po - pe) / (1 - pe), "raw_agreement": po}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit_v04_results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
