"""Pre-registered statistics (design v0.2 §7; addenda v0.2.1 C4, v0.2.2).

Unit of analysis = the query; every comparison is paired on the same queries.
  binary outcomes (hit@k, EM)   -> exact two-sided McNemar test on discordant pairs (binomial, p0 = .5)
  graded outcomes (nDCG, F1)    -> Wilcoxon signed-rank p-value (zero differences dropped, 'wilcox')
  every difference              -> 95% paired percentile-bootstrap CI (10,000 resamples, seed 0)
  every mean                    -> 95% percentile-bootstrap CI (same resamples)
  multiple comparisons          -> Holm-Bonferroni within each metric family (21 retriever pairs)
  transfer ratio vs R1          -> dEM / dHit@5 with a joint bootstrap CI; reported only if
                                   |dHit@5| > 2 pp AND the Hit@5 difference is significant after Holm
A single resample-index matrix is shared by all quantities, so CIs are mutually consistent.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import binomtest, spearmanr, wilcoxon

B = 10_000
SEED = 0


def resample_index(n: int, b: int = B, seed: int = SEED) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(b, n), dtype=np.int32)


def boot_means(x: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return x[idx].mean(axis=1)


def ci(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    a = (1 - level) / 2
    lo, hi = np.quantile(samples, [a, 1 - a])
    return float(lo), float(hi)


def mean_ci(x: np.ndarray, idx: np.ndarray) -> dict:
    lo, hi = ci(boot_means(x, idx))
    return {"mean": float(x.mean()), "ci_lo": lo, "ci_hi": hi, "n": int(len(x))}


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> dict:
    a, b = a.astype(bool), b.astype(bool)
    n10 = int((a & ~b).sum())       # a correct, b wrong
    n01 = int((~a & b).sum())
    nd = n10 + n01
    p = 1.0 if nd == 0 else float(binomtest(n10, nd, 0.5).pvalue)
    return {"n10": n10, "n01": n01, "p": p, "odds_ratio": (n10 / n01) if n01 else float("inf") if n10 else float("nan")}


def wilcoxon_p(x: np.ndarray, y: np.ndarray) -> float:
    d = x - y
    if np.all(d == 0):
        return 1.0
    return float(wilcoxon(x, y, zero_method="wilcox", alternative="two-sided").pvalue)


def paired_diff(x: np.ndarray, y: np.ndarray, idx: np.ndarray, binary: bool) -> dict:
    d = x.astype(float) - y.astype(float)
    lo, hi = ci(boot_means(d, idx))
    out = {"diff": float(d.mean()), "ci_lo": lo, "ci_hi": hi}
    if binary:
        m = mcnemar_exact(x, y)
        out.update(test="exact McNemar", p=m["p"], n10=m["n10"], n01=m["n01"], odds_ratio=m["odds_ratio"])
    else:
        out.update(test="Wilcoxon signed-rank", p=wilcoxon_p(x, y))
    return out


def holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (monotone, capped at 1), in the input order."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj.tolist()


def pairwise(values: dict[str, np.ndarray], names: list[str], idx: np.ndarray, binary: bool) -> list[dict]:
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rows.append({"a": a, "b": b, **paired_diff(values[a], values[b], idx, binary)})
    for r, p in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = p
        r["significant_holm_0.05"] = p < 0.05
    return rows


@dataclass
class TransferRatio:
    retriever: str
    d_hit: float
    d_em: float
    ratio: float | None
    ci_lo: float | None
    ci_hi: float | None
    estimable: bool
    reason: str


def transfer_ratio(hit_m, hit_ref, em_m, em_ref, idx, hit_p_holm: float, name: str, min_abs_dhit: float = 0.02) -> TransferRatio:
    dh, de = float(hit_m.mean() - hit_ref.mean()), float(em_m.mean() - em_ref.mean())
    if abs(dh) <= min_abs_dhit or hit_p_holm >= 0.05:
        why = f"|dHit@5| = {abs(dh) * 100:.2f} pp" + (" <= 2 pp" if abs(dh) <= min_abs_dhit else "") + \
              ("" if hit_p_holm < 0.05 else f"; Hit@5 difference not significant after Holm (p = {hit_p_holm:.3g})")
        return TransferRatio(name, dh, de, None, None, None, False, "not estimable: " + why)
    bh = boot_means(hit_m.astype(float) - hit_ref, idx)
    be = boot_means(em_m.astype(float) - em_ref, idx)
    with np.errstate(divide="ignore", invalid="ignore"):
        br = be / bh
    br = br[np.isfinite(br)]
    lo, hi = ci(br)
    return TransferRatio(name, dh, de, de / dh, lo, hi, True, "estimable")


def spearman(x: list[float], y: list[float]) -> dict:
    r = spearmanr(x, y)
    return {"rho": float(r.statistic), "n_points": len(x), "note": "descriptive only (pre-registered)"}
