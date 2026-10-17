"""Targeted checks for the v0.4/v0.5 analyses. Run from the repository root:

    python3 -m pytest -q manuscript_ipm/revision_v04/tests
"""
import csv
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if not (ROOT / "paper_outputs/v03/statistics/v03_analysis.json").exists():
    pytest.skip("paper_outputs not present (e.g. inside a Kaggle notebook)", allow_module_level=True)
OUT = ROOT / "manuscript_ipm/revision_v04/outputs"
TEX = ROOT / "manuscript_ipm/revision_v04/latex"
S = json.load(open(ROOT / "paper_outputs/v03/statistics/v03_analysis.json"))


def rows(name):
    return list(csv.DictReader(open(OUT / name)))


def test_transitions_reconstruct_em_difference():
    by = {}
    for r in rows("transitions_hit5.csv"):
        by.setdefault((r["cell"], r["encoder"]), []).append(r)
    assert len(by) == 20
    for k, v in by.items():
        assert {r["stratum"] for r in v} == {"miss_to_hit", "hit_to_hit", "hit_to_miss", "miss_to_miss"}
        assert sum(int(r["n"]) for r in v) == (2255 if "nq" in k[0] else 2000)
        total = float(v[0]["dEM_total"])
        assert abs(sum(float(r["contribution_to_dEM"]) for r in v) - total) < 5e-4      # rounded to 4 dp in the CSV


def test_transitions_match_analysis_em_difference():
    names = {"bge-small": "R2", "bge-base": "R3", "bge-large": "R4", "e5-base-v2": "R5", "Qwen3-Emb-0.6B": "R6"}
    for r in rows("transitions_hit5.csv"):
        a = S["cells"][r["cell"]]["answers"]
        assert abs(float(r["dEM_total"]) - (a[names[r["encoder"]]]["em"]["mean"] - a["R1"]["em"]["mean"])) < 5e-5


def test_hotpotqa_evidence_groups_partition():
    by = {}
    for r in rows("transitions_hotpotqa_evidence.csv"):
        if r["group"].startswith("evidence_"):
            by.setdefault((r["cell"], r["encoder"]), []).append(r)
    assert len(by) == 10
    for v in by.values():
        assert sum(int(r["n"]) for r in v) == 2000


def test_decomposition_components_add_up():
    for r in rows("decomposition_components.csv"):
        assert abs(float(r["prevalence"]) + float(r["hits"]) + float(r["misses"]) - float(r["dEM"])) < 2e-4
        for t, lo, hi in (("prevalence", "prev_lo", "prev_hi"), ("hits", "hits_lo", "hits_hi"), ("misses", "miss_lo", "miss_hi")):
            assert float(r[lo]) <= float(r[t]) + 1e-4 <= float(r[hi]) + 2e-4


def test_ratios_match_committed_analysis():
    for r in rows("transfer_ratio_audit.csv"):
        if r["status"] == "estimable":
            assert r["difference_recomputed_minus_json"] in ("0.00e+00", "-0.00e+00")


def test_predictions_table_matches_analysis_verdicts():
    t = (TEX / "tables/tab_predictions.tex").read_text()
    for cell, lab in (("llama_hp", "HotpotQA, Llama"), ("qwen_hp", "HotpotQA, Qwen")):
        line = next(l for l in t.splitlines() if l.startswith(f"P1 & {lab}"))
        assert ("not met" in line) == (not S["cells"][cell]["P1"]["holds"])


def test_nq_upper_bound_statement():
    ub = max(t["ci_hi"] for c in ("qwen_nq", "llama_nq") for t in S["cells"][c]["transfer_vs_R1"]["em"]
             if t["estimable"] and t["retriever"] != "R0")
    mac = re.search(r"ev@V5\.nq\.ubmax\\endcsname\{([0-9.]+)\}", (TEX / "numbers_v05.tex").read_text()).group(1)
    assert abs(float(mac) - ub) < 5e-4 and ub > 0.45


def test_no_v03_phrases_that_the_audit_corrected():
    text = " ".join(p.read_text() for p in (TEX / "sections").glob("*.tex"))
    for bad in ("harmless", "no measurable cost in answers", "same picture", "easier for BM25", "hurt e5-base-v2",
                "No upper bound on NQ exceeded", "gold bound"):
        assert bad not in text, bad


def test_every_macro_defined():
    defined = set()
    for f in ("numbers.tex", "numbers_v04.tex", "numbers_v05.tex", "numbers_v06.tex"):
        defined |= set(re.findall(r"ev@([^\\]+)\\endcsname", (TEX / f).read_text()))
    used = set()
    for p in list((TEX / "sections").glob("*.tex")) + list((TEX / "tables").glob("*.tex")):
        used |= set(re.findall(r"\\ev\{([^}]+)\}", p.read_text()))
    assert not (used - defined), sorted(used - defined)


# ---------------------------------------------------------------- v0.6 (Phase 3, addendum v0.4)
F = json.load(open(ROOT / "paper_outputs/v04/v04_full_analysis.json"))


def test_phase3_predictions_in_table():
    t = (TEX / "tables/tab_predictions.tex").read_text()
    for p in ("P7", "P8"):
        for rd, lab in (("qwen", "Qwen"), ("llama", "Llama")):
            line = next(l for l in t.splitlines() if l.startswith(f"{p} & NQ full corpus, {lab}"))
            assert ("not met" in line) == (not F["readers"][rd][p]["holds"])


def test_phase3_p8_uses_unrounded_interval_and_matches_v03():
    for rd in ("qwen", "llama"):
        t = next(x for x in S["cells"][f"{rd}_nq"]["transfer_vs_R1"]["em"] if x["retriever"] == "R5")
        assert F["readers"][rd]["P8"]["sub_interval_unrounded"] == [t["ci_lo"], t["ci_hi"]]
        assert F["readers"][rd]["T_full_minus_T_sub"]["R5"]["T_sub"] == t["ratio"]


def test_phase3_reproduces_e9_and_independent_check():
    for v in F["checks"]["e9_reproduction"].values():
        assert v["hit@5_now"] == v["hit@5_e9"] and abs(v["ndcg@10_now"] - v["ndcg@10_e9"]) < 1e-12
    ic = json.load(open(OUT / "independent_check_v06.json"))
    assert ic["rescore"]["disagree"] == 0
    for rd in ("qwen", "llama"):
        assert abs(ic[f"T.{rd}.R5"] - F["readers"][rd]["P8"]["T_full_e5"]) < 1e-12
        assert ic[f"P7.{rd}"] == F["readers"][rd]["P7"]["holds"] and ic[f"P8.{rd}"] == F["readers"][rd]["P8"]["holds"]


def test_phase3_transitions_add_up():
    for rd in ("qwen", "llama"):
        for r, tr in F["readers"][rd]["transitions_full"].items():
            assert abs(sum(v["contribution"] for k, v in tr.items() if k != "total") - tr["total"]) < 1e-12


def test_no_stale_full_corpus_claims():
    text = " ".join(p.read_text() for p in (TEX / "sections").glob("*.tex"))
    for bad in ("We did not run the readers on the full corpus", "Whether the ratios themselves hold on the full corpus was not tested",
                "whether the ratios hold on the full corpus", "the passage texts are not stored"):
        assert bad not in text, bad
