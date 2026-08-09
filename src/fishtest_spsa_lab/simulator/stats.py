"""Small-sample statistics for comparing optimizer arms.

A cross-optimizer claim needs a confidence interval and a paired difference, not
two point estimates. The between-seed standard deviation of the final Elo is the
same size as the between-optimizer spread, so a single run per arm resolves
nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Two-sided 95% Student-t quantiles, indexed by degrees of freedom.
# Small samples are the norm here (8 to 16 seeds), where the normal
# approximation is noticeably optimistic.
_T95: dict[int, float] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    20: 2.086,
    25: 2.060,
    30: 2.042,
}
_T95_LARGE: float = 1.96


def t95(df: int) -> float:
    """Return the two-sided 95% t quantile for ``df`` degrees of freedom."""
    if df <= 0:
        return math.nan
    if df in _T95:
        return _T95[df]
    keys = sorted(k for k in _T95 if k <= df)
    return _T95[keys[-1]] if keys and df < 30 else _T95_LARGE  # noqa: PLR2004


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
