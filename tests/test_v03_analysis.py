"""The v0.3 analysis reproduces every v0.2 number exactly and agrees with the Kaggle-side summaries."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/statistics/v03/v03_analysis.json"


@pytest.fixture(scope="module")
def res(tmp_path_factory):
    from src.analysis import v03_analysis
    out = tmp_path_factory.mktemp("v03")
    assert v03_analysis.main(["--root", str(ROOT), "--out", str(out / "s"), "--tables", str(out / "t")]) == 0
    return json.loads((out / "s" / "v03_analysis.json").read_text())


def test_reproduces_v02(res):
    v2 = json.loads((ROOT / "paper_inputs/results/statistics/rq12_main.json").read_text())
    a = {t["retriever"]: t for t in v2["rq2_transfer_ratio_vs_R1"]}
    b = {t["retriever"]: t for t in res["cells"]["qwen_nq"]["transfer_vs_R1"]["em"]}
    for k in a:
        assert a[k]["ratio"] == pytest.approx(b[k]["ratio"]) and a[k]["ci_hi"] == pytest.approx(b[k]["ci_hi"])
    for x, y in zip(v2["rq2_pairwise_em"], res["cells"]["qwen_nq"]["pairwise_em"]):
        assert x["p_holm"] == pytest.approx(y["p_holm"])


def test_hotpot_matches_kaggle_summary(res):
    s = json.loads((ROOT / "paper_inputs/v03_hp/summary/v03_hp_retrieval_summary.json").read_text())["retrievers"]
    for cell in ("qwen_hp", "llama_hp"):
        for r, d in s.items():
            assert res["cells"][cell]["retrieval"][r]["hit@5"]["mean"] == pytest.approx(d["gen_first_2000"]["hit@5"])
            assert res["cells"][cell]["retrieval"][r]["both@5"]["mean"] == pytest.approx(d["gen_first_2000"]["both@5"])


def test_gee_shape(res):
    assert res["gee"]["n_rows"] == 7 * (2 * 2255 + 2 * 2000) and res["gee"]["n_clusters"] == 2255 + 2000
