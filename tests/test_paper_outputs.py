from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT / "results" / "statistics" / "rq12_main.json").exists(), reason="main results not restored")
def test_make_figures_tables_reproduces_verified_numbers(tmp_path):
    import csv
    import json

    from src.paper.make_figures_tables import main
    assert main(["--root", str(ROOT), "--out", str(tmp_path)]) == 0
    assert len(list((tmp_path / "figures").glob("F*.pdf"))) == 9
    t5 = {r["condition"]: r for r in csv.DictReader(open(tmp_path / "tables" / "T5_rag_evaluation.csv"))}
    assert t5["e5-base-v2"]["EM [95% CI]"].startswith("51.0") and t5["BM25"]["SUPPORTED (judge, UNVALIDATED) [95% CI]"] == "79.0 [75.4, 82.6]"
    d = json.load(open(tmp_path / "derived_stats.json"))
    step = [s for s in d["topk_ablation"]["steps"] if s["retriever"] == "R4" and s["from_k"] == 5][0]
    assert round(step["p_holm"], 3) == 0.044
    assert d["abstention_A3"]["R0"]["idk_rate"] == 0.215 and d["faithfulness_UNVALIDATED"]["n"] == 500
    fa = d["faithfulness_UNVALIDATED"]
    assert fa["n_gold"] == 499 and fa["gold_missing"] == ["test478"] and fa["n_judgements"] == 3999
    ex = d["topk_ablation"]["exploratory_R4k1_vs_R0k10"]
    assert (round(100 * ex["ci_lo"], 1), round(100 * ex["ci_hi"], 1)) == (0.1, 4.2)      # same query order as rq12
    tk = {(r["retriever"], r["k"]): r["ratio"] for r in d["transfer_by_k_EXPLORATORY"]["rows"]}
    assert round(tk[("R4", 1)], 2) == 0.61 and round(tk[("R4", 10)], 2) == 0.19 and round(tk[("R4", 5)], 2) == 0.24
