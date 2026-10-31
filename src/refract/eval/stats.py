"""Statistics for the evaluation report, standard library only: bootstrap CIs, agreement, paired tests."""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence


def mean(values: Sequence[float | None]) -> float:
    xs = [v for v in values if v is not None]
    return statistics.fmean(xs) if xs else float("nan")


def percentile(values: Sequence[float], q: float) -> float:
    s = sorted(values)
    if not s:
        return float("nan")
    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def bootstrap_ci(
    values: Sequence[float], n: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean."""
    xs = [v for v in values if v is not None]
    if len(xs) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(xs, k=len(xs))) for _ in range(n))
    return means[int(n * alpha / 2)], means[int(n * (1 - alpha / 2)) - 1]


def paired_bootstrap_diff(
    a: Sequence[float], b: Sequence[float], n: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of a − b over paired items with a 95% bootstrap CI."""
    d = [x - y for x, y in zip(a, b, strict=True)]
    lo, hi = bootstrap_ci(d, n=n, seed=seed)
    return mean(d), lo, hi


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact sign test (ties dropped)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def weighted_kappa(
    a: Sequence[int], b: Sequence[int], categories: Sequence[int], weights: str = "quadratic"
) -> float:
    """Cohen's kappa with linear or quadratic weights, for ordinal ratings (e.g. 0–2 scores, low/medium/high)."""
    pairs = [(x, y) for x, y in zip(a, b, strict=True) if x is not None and y is not None]
    if not pairs:
        return float("nan")
    idx = {c: i for i, c in enumerate(categories)}
    k, n = len(categories), len(pairs)
    obs = [[0.0] * k for _ in range(k)]
    for x, y in pairs:
        obs[idx[x]][idx[y]] += 1
    row = [sum(r) for r in obs]
    col = [sum(obs[i][j] for i in range(k)) for j in range(k)]

    def w(i: int, j: int) -> float:
        if k == 1:
            return 0.0
        d = abs(i - j) / (k - 1)
        return d * d if weights == "quadratic" else d

    num = sum(w(i, j) * obs[i][j] for i in range(k) for j in range(k)) / n
    den = sum(w(i, j) * row[i] * col[j] for i in range(k) for j in range(k)) / (n * n)
    return 1.0 - num / den if den else 1.0


def _ranks(xs: Sequence[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for m in range(i, j + 1):
            ranks[order[m]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    pairs = [(a, b) for a, b in zip(x, y, strict=True) if a is not None and b is not None]
    if len(pairs) < 3:
        return float("nan")
    rx, ry = _ranks([a for a, _ in pairs]), _ranks([b for _, b in pairs])
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return cov / (sx * sy) if sx and sy else float("nan")
