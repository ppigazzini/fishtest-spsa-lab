"""Small-sample statistics for comparing optimizer arms.

A cross-optimizer claim needs a confidence interval and a paired difference, not
two point estimates. The between-seed standard deviation of the final Elo is the
same size as the between-optimizer spread, so a single run per arm resolves
nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

# --- Student-t distribution ---------------------------------------------------
#
# Computed rather than tabulated. The previous lookup table fell back to the
# normal quantile 1.96 for every df >= 31, which made intervals about 4% too
# NARROW exactly where a longer seed sweep would put them (true t at df = 31 is
# 2.0395). Below 30 the table rounded df down and was therefore conservative, so
# the defect only bit on the side the repository is moving toward.
#
# The regularized incomplete beta function is the whole of what is needed, and
# the continued-fraction form is short, standard and self-contained; it keeps the
# package free of a SciPy dependency for four numbers.

_BETACF_MAX_ITER: int = 300
_BETACF_EPS: float = 3.0e-16
_BETACF_TINY: float = 1.0e-300


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _BETACF_TINY:
        d = _BETACF_TINY
    d = 1.0 / d
    h = d

    for m in range(1, _BETACF_MAX_ITER + 1):
        m2 = 2 * m
        # even step
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _BETACF_TINY:
            d = _BETACF_TINY
        c = 1.0 + aa / c
        if abs(c) < _BETACF_TINY:
            c = _BETACF_TINY
        d = 1.0 / d
        h *= d * c
        # odd step
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _BETACF_TINY:
            d = _BETACF_TINY
        c = 1.0 + aa / c
        if abs(c) < _BETACF_TINY:
            c = _BETACF_TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _BETACF_EPS:
            break

    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x),
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return (
        1.0
        - math.exp(
            math.lgamma(a + b)
            - math.lgamma(a)
            - math.lgamma(b)
            + b * math.log1p(-x)
            + a * math.log(x),
        )
        * _betacf(b, a, 1.0 - x)
        / b
    )


def t_two_sided_p(t: float, df: int) -> float:
    """Two-sided p-value of a Student-t statistic."""
    if df <= 0:
        return math.nan
    if not math.isfinite(t):
        return 0.0 if math.isinf(t) else math.nan
    return _betainc(0.5 * df, 0.5, df / (df + t * t))


def t_quantile(alpha: float, df: int) -> float:
    """Two-sided ``1 - alpha`` Student-t quantile, by bisection on the p-value.

    ``alpha = 0.05`` gives the familiar 95% multiplier. Other levels are needed
    once a family-wise correction enters, which is why this is not hard-coded to
    0.05 the way the old table was.
    """
    if df <= 0 or not (0.0 < alpha < 1.0):
        return math.nan
    lo, hi = 0.0, 1.0
    while t_two_sided_p(hi, df) > alpha:
        hi *= 2.0
        if hi > 1e6:  # noqa: PLR2004
            return hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_two_sided_p(mid, df) > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def t95(df: int) -> float:
    """Return the two-sided 95% t quantile for ``df`` degrees of freedom."""
    return t_quantile(0.05, df)


def holm_adjusted(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values, in the input order.

    Eleven optimizers are compared against one baseline. At 95% per comparison
    the family-wise error rate under the null is 1 - 0.95**11 = 43%, not 5%, so
    marking each interval independently overstates the evidence by a lot. Holm
    controls the family-wise rate, needs no independence assumption, and is
    uniformly more powerful than plain Bonferroni.

    Compare the returned values against the same 0.05 as before.
    """
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        scaled = (m - rank) * p_values[idx]
        running = max(running, min(1.0, scaled))
        adjusted[idx] = running
    return adjusted


@dataclass(frozen=True, slots=True)
class Estimate:
    """A sample mean with a 95% confidence interval."""

    mean: float
    half_width: float
    n: int

    @property
    def low(self) -> float:
        """Lower bound of the 95% interval."""
        return self.mean - self.half_width

    @property
    def high(self) -> float:
        """Upper bound of the 95% interval."""
        return self.mean + self.half_width

    @property
    def separated_from_zero(self) -> bool:
        """Whether the interval excludes zero."""
        return math.isfinite(self.half_width) and abs(self.mean) > self.half_width


def mean_ci(values: list[float] | np.ndarray) -> Estimate:
    """Return the mean of ``values`` with a 95% t confidence interval."""
    arr = np.asarray(values, dtype=float)
    n = int(arr.size)
    if n == 0:
        return Estimate(mean=math.nan, half_width=math.nan, n=0)
    if n == 1:
        return Estimate(mean=float(arr[0]), half_width=math.inf, n=1)
    sem = float(arr.std(ddof=1)) / math.sqrt(n)
    return Estimate(mean=float(arr.mean()), half_width=t95(n - 1) * sem, n=n)


def paired_diff(
    treatment: list[float] | np.ndarray,
    baseline: list[float] | np.ndarray,
) -> Estimate:
    """Return the paired difference ``treatment - baseline`` with a 95% interval.

    Both sequences must be aligned seed by seed. Pairing removes the
    between-seed variance, which is what makes a difference of this size
    measurable at all.
    """
    a = np.asarray(treatment, dtype=float)
    b = np.asarray(baseline, dtype=float)
    if a.shape != b.shape:
        msg = f"paired samples must align: {a.shape} != {b.shape}"
        raise ValueError(msg)
    return mean_ci(a - b)
