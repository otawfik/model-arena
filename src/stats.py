"""Significance and uncertainty on the held-out set.

McNemar's exact test asks whether two classifiers' error rates differ, using
only the documents where they disagree; that's the right paired test when both
models predict on the same test set. Holm's step-down correction keeps the
family-wise error rate at alpha across all pairs.
"""
import math

from scipy.stats import binomtest


def mcnemar(correct_a, correct_b):
    """Exact McNemar test. Returns (b, c, p): b = only A right, c = only B right."""
    b = int((correct_a & ~correct_b).sum())
    c = int((correct_b & ~correct_a).sum())
    p = 1.0 if b + c == 0 else binomtest(b, b + c, 0.5).pvalue
    return b, c, float(p)


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values, same order as the input."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adj[i] = running
    return adj


def wilson(k, n, z=1.96):
    """Wilson score interval for a proportion k/n."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half
