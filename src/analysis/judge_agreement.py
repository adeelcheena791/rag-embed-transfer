"""E3 (addendum v0.3; design v0.2 §5.4): agreement between the author's labels and the LLM judge.

Inputs: the key (annotation/sample_v1_key.json, committed only after labelling;, rebuilt and checked against the seeded sampler),
the blind sample shown to the annotator, and the annotator's export (judge_validation_labels.json).
Reported, as pre-registered:
  * Cohen's kappa (human vs judge) on the 200 items, 95% percentile-bootstrap CI over items (10,000 resamples, seed 0);
  * the decision band (>= 0.4 use as secondary evidence; 0.2-0.4 limited, with sensitivity analysis; < 0.2 exclude);
  * raw agreement, confusion matrix, per-retriever agreement;
  * population-weighted agreement and judge precision/recall vs the human label using inverse inclusion probabilities
    (the sample over-represents NOT_SUPPORTED by design);
  * the sensitivity analysis: retriever SUPPORTED rates (a) by human labels on the sample (weighted), (b) by the judge
    on the items where human and judge agree, and whether the retriever ranking changes.
Single annotator (the author): no inter-annotator kappa (limitation, addendum v0.2.4).

    python -m src.analysis.judge_agreement --labels annotation/judge_validation_labels.json --out paper_outputs/v03/statistics/judge_agreement.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

B, SEED = 10_000, 0
LABELS = ("SUPPORTED", "NOT_SUPPORTED")


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    po = float(np.mean(a == b))
    pe = float(np.mean(a) * np.mean(b) + (1 - np.mean(a)) * (1 - np.mean(b)))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def weighted_kappa_ipw(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    w = w / w.sum()
    po = float(np.sum(w * (a == b)))
    pa, pb = float(np.sum(w * a)), float(np.sum(w * b))
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe)


def band(k: float) -> str:
    return "moderate or better: use as secondary evidence" if k >= 0.4 else \
        "fair: limited reliability, conclusions only with the sensitivity analysis" if k >= 0.2 else \
        "slight: exclude from conclusions"


def analyse(key: list[dict], labels: dict[str, str]) -> dict:
    items = [k for k in key if k["item_id"] in labels]
    if len(items) != len(key):
        raise ValueError(f"{len(key) - len(items)} sample items have no human label")
    bad = {labels[k["item_id"]] for k in items} - set(LABELS)
    if bad:
        raise ValueError(f"unknown labels {bad}")
    h = np.array([labels[k["item_id"]] == "SUPPORTED" for k in items], dtype=float)
    j = np.array([k["judge_verdict"] == "SUPPORTED" for k in items], dtype=float)
    w = np.array([1.0 / k["inclusion_prob"] for k in items])
    n = len(items)
    kappa = cohen_kappa(h, j)
    idx = np.random.default_rng(SEED).integers(0, n, size=(B, n))
    boots = np.array([cohen_kappa(h[i], j[i]) for i in idx])
    boots = boots[~np.isnan(boots)]
    lo, hi = np.quantile(boots, [0.025, 0.975])
    conf = Counter((labels[k["item_id"]], k["judge_verdict"]) for k in items)
    per_r = {}
    for r in sorted({k["retriever"] for k in items}):
        m = np.array([k["retriever"] == r for k in items])
        per_r[r] = {"n": int(m.sum()), "agreement": float(np.mean(h[m] == j[m])),
                    "human_supported_rate_weighted": float(np.sum(w[m] * h[m]) / np.sum(w[m])),
                    "judge_supported_rate_weighted": float(np.sum(w[m] * j[m]) / np.sum(w[m])),
                    "judge_supported_rate_on_agreeing_items_weighted":
                        float(np.sum((w * j)[m & (h == j)]) / np.sum(w[m & (h == j)])) if np.any(m & (h == j)) else None}
    wn = w / w.sum()
    tp, fp = float(np.sum(wn * (j == 1) * (h == 1))), float(np.sum(wn * (j == 1) * (h == 0)))
    fn = float(np.sum(wn * (j == 0) * (h == 1)))
    rank = lambda key_: sorted(per_r, key=lambda r: -per_r[r][key_])    # noqa: E731
    return {
        "n_items": n, "annotators": 1,
        "cohen_kappa": kappa, "kappa_ci95": [float(lo), float(hi)], "bootstrap": {"B": B, "seed": SEED},
        "decision_band": band(kappa), "decision_band_ci_low": band(float(lo)),
        "raw_agreement": float(np.mean(h == j)),
        "confusion_human_x_judge": {f"human_{a}__judge_{b}": conf.get((a, b), 0) for a in LABELS for b in LABELS},
        "human_supported_rate_sample": float(h.mean()), "judge_supported_rate_sample": float(j.mean()),
        "population_weighted": {"agreement": float(np.sum(wn * (h == j))), "kappa_ipw": weighted_kappa_ipw(h, j, w),
                                "judge_precision_for_SUPPORTED": tp / (tp + fp) if tp + fp else None,
                                "judge_recall_for_SUPPORTED": tp / (tp + fn) if tp + fn else None,
                                "human_supported_rate": float(np.sum(wn * h)), "judge_supported_rate": float(np.sum(wn * j))},
        "per_retriever": per_r,
        "sensitivity": {"ranking_judge_weighted": rank("judge_supported_rate_weighted"),
                        "ranking_human_weighted": rank("human_supported_rate_weighted"),
                        "ranking_judge_on_agreeing_items": rank("judge_supported_rate_on_agreeing_items_weighted"),
                        "note": "Rankings from ~28 items per retriever are noisy; they are reported, not tested."},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--key", default="annotation/sample_v1_key.json")    # unblinded after the labels were final (2026-10-16)
    ap.add_argument("--blind", default="annotation/sample_v1_blind.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    lab = json.load(open(args.labels))
    key = json.load(open(args.key))
    blind_ids = [b["item_id"] for b in json.load(open(args.blind))]
    if sorted(blind_ids) != sorted(k["item_id"] for k in key):
        raise ValueError("key and blind sample do not contain the same items")
    labels = {x["item_id"]: x["label"] for x in lab["labels"]}
    res = analyse(key, labels)
    res.update(instructions_hash=lab.get("instructions_hash"), annotator=lab.get("annotator"),
               labelled_from=min(x["time"] for x in lab["labels"]), labelled_to=max(x["time"] for x in lab["labels"]))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(f"kappa = {res['cohen_kappa']:.3f} [{res['kappa_ci95'][0]:.3f}, {res['kappa_ci95'][1]:.3f}] -> {res['decision_band']}")
    print(f"raw agreement {res['raw_agreement']:.3f}; weighted agreement {res['population_weighted']['agreement']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
