import numpy as np
import pytest

from src.analysis import stats
from src.analysis.metrics import METRICS, per_query


def test_metrics_match_ranx(tmp_path):
    ranx = pytest.importorskip("ranx")
    rng = np.random.default_rng(1)
    docs = [f"d{i}" for i in range(60)]
    qrels, run_rows = {}, []
    for q in range(40):
        rel = rng.choice(docs, size=rng.integers(1, 3), replace=False)
        qrels[f"q{q}"] = {d: 1 for d in rel}
        ranked = list(rng.permutation(docs)[:20])
        run_rows.append({"qid": f"q{q}", "pids": ranked, "scores": list(np.linspace(1, 0, 20))})
    import json
    rf = tmp_path / "run.jsonl"
    rf.write_text("\n".join(json.dumps(r) for r in run_rows))
    ours = per_query(rf, qrels, sorted(qrels))
    R = ranx.Run({r["qid"]: dict(zip(r["pids"], r["scores"])) for r in run_rows})
    Q = ranx.Qrels(qrels)
    ref = ranx.evaluate(Q, R, ["hit_rate@1", "hit_rate@3", "hit_rate@5", "hit_rate@10", "ndcg@10", "mrr@10"])
    for mine, theirs in zip(METRICS, ["hit_rate@1", "hit_rate@3", "hit_rate@5", "hit_rate@10", "ndcg@10", "mrr@10"]):
        assert ours[mine].mean() == pytest.approx(ref[theirs], abs=1e-9), mine


def test_mcnemar_exact_hand_computed():
    a = np.array([1, 1, 1, 0, 0, 1, 1, 0])
    b = np.array([0, 0, 1, 0, 1, 1, 0, 0])
    m = stats.mcnemar_exact(a, b)
    assert (m["n10"], m["n01"]) == (3, 1)
    assert m["p"] == pytest.approx(0.625)                        # 2 * P(X <= 1 | n=4, p=.5) = 2 * 5/16
    assert stats.mcnemar_exact(a, a)["p"] == 1.0


def test_holm_hand_computed():
    assert stats.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert stats.holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])


def test_bootstrap_ci_and_pairwise():
    rng = np.random.default_rng(3)
    x = rng.integers(0, 2, 2000).astype(float)
    y = np.where(rng.random(2000) < 0.1, 1 - x, x)             # 10% discordant
    idx = stats.resample_index(2000, b=2000)
    mc = stats.mean_ci(x, idx)
    assert mc["ci_lo"] < mc["mean"] < mc["ci_hi"] and mc["ci_hi"] - mc["ci_lo"] < 0.06
    rows = stats.pairwise({"A": x, "B": y, "C": x}, ["A", "B", "C"], idx, binary=True)
    ac = [r for r in rows if (r["a"], r["b"]) == ("A", "C")][0]
    assert ac["diff"] == 0 and ac["p"] == 1.0 and ac["p_holm"] == 1.0
    assert all(r["p_holm"] >= r["p"] for r in rows)


def test_transfer_ratio_reporting_rule():
    rng = np.random.default_rng(5)
    n = 3000
    idx = stats.resample_index(n, b=1000)
    hit_ref = (rng.random(n) < 0.6).astype(float)
    hit_m = np.maximum(hit_ref, (rng.random(n) < 0.15).astype(float))   # +~6 pp
    em_ref = hit_ref * (rng.random(n) < 0.7)
    em_m = hit_m * (rng.random(n) < 0.7)
    t = stats.transfer_ratio(hit_m, hit_ref, em_m, em_ref, idx, hit_p_holm=0.001, name="Rx")
    assert t.estimable and t.ci_lo < t.ratio < t.ci_hi
    small = stats.transfer_ratio(hit_ref, hit_ref, em_ref, em_ref, idx, hit_p_holm=0.001, name="Ry")
    assert not small.estimable and "<= 2 pp" in small.reason
    ns = stats.transfer_ratio(hit_m, hit_ref, em_m, em_ref, idx, hit_p_holm=0.2, name="Rz")
    assert not ns.estimable and "not significant" in ns.reason


def test_pareto_frontier():
    from src.analysis.rq34 import pareto
    pts = {"cheap_bad": (1, 0.3), "mid": (2, 0.5), "dominated": (3, 0.45), "best": (4, 0.6), "tie_cost_worse": (2, 0.4)}
    assert pareto(pts) == ["cheap_bad", "mid", "best"]
    assert pareto({"a": (1, 0.5), "b": (1, 0.5)}) == ["a", "b"]          # exact ties: neither dominates
