"""Minimum detectable difference (MDD) for the exact two-sided McNemar test.
Analytic calculation, NOT an experiment.

Model: n_d = round(n * p_d) discordant pairs; under H1 a discordant pair favours A with
probability q = (p_d + delta) / (2 p_d). The exact McNemar test is a two-sided binomial test of
the n_d discordant pairs against 0.5, which rejects iff min(k, n_d - k) <= c, where c is the
largest integer with 2 * BinomCDF(c; n_d, 0.5) <= alpha (symmetric null => this equals
scipy's binomtest p-value <= alpha). Power = P(K <= c) + P(K >= n_d - c) under q.
"""
import numpy as np
from scipy.stats import binom


def critical_c(nd: int, alpha: float) -> int:
    """Largest c with 2*P(K<=c | p=.5) <= alpha; -1 if no rejection region exists."""
    cdf = binom.cdf(np.arange(nd + 1), nd, 0.5)
    ok = np.nonzero(np.minimum(2 * cdf, 1.0) <= alpha)[0]
    return int(ok[-1]) if len(ok) else -1


def power(n: int, p_d: float, delta: float, alpha: float) -> float:
    nd = int(round(n * p_d))
    q = (p_d + delta) / (2 * p_d)
    if q >= 1:
        return 1.0
    c = critical_c(nd, alpha)
    if c < 0:
        return 0.0
    return float(binom.cdf(c, nd, q) + binom.sf(nd - c - 1, nd, q))


def mdd(n: int, p_d: float, alpha: float, target: float = 0.8) -> float:
    for d in np.arange(0.001, p_d, 0.001):
        if power(n, p_d, d, alpha) >= target:
            return float(d)
    return float("nan")


if __name__ == "__main__":
    print("n     p_disc  MDD(alpha=.05)  MDD(alpha=.05/21, Holm worst case)")
    for n in (1000, 2255, 3000):
        for p_d in (0.05, 0.10, 0.15, 0.20, 0.30):
            print(f"{n:<5} {p_d:<7.2f} {mdd(n, p_d, 0.05)*100:>6.1f} pp        {mdd(n, p_d, 0.05/21)*100:>6.1f} pp")
