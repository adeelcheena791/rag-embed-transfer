"""Minimum detectable difference (MDD) for the exact two-sided McNemar test.
Analytic calculation, NOT an experiment. Assumes the number of discordant pairs
n_d = round(n * p_d) and, under H1, P(discordant pair favours A) = (p_d + delta) / (2 p_d).
"""
import numpy as np
from scipy.stats import binom, binomtest

def power(n, p_d, delta, alpha):
    nd = int(round(n * p_d))
    q = (p_d + delta) / (2 * p_d)
    if q > 1: return 1.0
    # rejection region of exact two-sided binomial test with p0 = 0.5
    rej = np.array([binomtest(k, nd, 0.5).pvalue <= alpha for k in range(nd + 1)])
    return binom.pmf(np.arange(nd + 1), nd, q)[rej].sum()

def mdd(n, p_d, alpha, target=0.8):
    for d in np.arange(0.001, p_d, 0.001):
        if power(n, p_d, d, alpha) >= target:
            return d
    return float("nan")

rows = []
for n in (1000, 3000):
    for p_d in (0.05, 0.10, 0.15, 0.20, 0.30):
        rows.append((n, p_d, mdd(n, p_d, 0.05), mdd(n, p_d, 0.05 / 21)))
print("n     p_disc  MDD(alpha=.05)  MDD(alpha=.05/21, Holm worst case)")
for n, p, a, b in rows:
    print(f"{n:<5} {p:<7.2f} {a*100:>6.1f} pp        {b*100:>6.1f} pp")
