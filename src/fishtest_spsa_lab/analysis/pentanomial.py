"""Pentanomial primitives: moments, sampling, and the online report estimator.

Extracted from ``validate_variance.py`` to break a dependency inversion.
``common.py`` -- the module every analysis script imports -- was importing from
``validate_variance``, a leaf console-script module whose ``main()`` runs a
simulation. Three more edges pointed the same way (``validate_sf_adam`` and
``validate_spsa_u2`` both imported from it too), so importing the shared helpers
pulled in a script.

Nothing here runs anything: these are pure functions and one accumulator.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


def compute_pentanomial_moments(
    p5: tuple[float, float, float, float, float],
) -> tuple[float, float, float]:
    """Compute mean, second moment, and variance for a pentanomial distribution."""
    # Values correspond to [-2, -1, 0, +1, +2]
    vals = (-2.0, -1.0, 0.0, 1.0, 2.0)
    mu = sum(p * v for p, v in zip(p5, vals, strict=True))
    mu2 = sum(p * (v * v) for p, v in zip(p5, vals, strict=True))
    var = mu2 - mu * mu
    return mu, mu2, var


def gen_pentanomial_outcomes(
    seed: int,
    n: int,
    p5: tuple[float, float, float, float, float],
) -> list[int]:
    """Generate n outcomes from a pentanomial distribution."""
    rng = random.Random(seed)  # noqa: S311
    vals = [-2, -1, 0, +1, +2]
    outs = rng.choices(vals, weights=p5, k=n)
    rng.shuffle(outs)
    return outs


# ----- init aggregates (same math as in Adam script, but no p5 leaks) -----


@dataclass(slots=True)
class InitStats:
    """Initial statistics for warm-starting the estimator."""

    reports: float = 0.0
    sum_n: float = 0.0
    sum_s: float = 0.0
    sum_s2_over_n: float = 0.0


def compute_init_stats_from_prior(
    p5: tuple[float, float, float, float, float],
    reports: float,
    mean_n: float,
) -> InitStats:
    """Compute initial statistics from a prior distribution."""
    if reports <= 0.0 or mean_n <= 0.0:
        return InitStats()
    mu_p, _mu2_p, var_p = compute_pentanomial_moments(p5)
    return InitStats(
        reports=reports,
        sum_n=reports * mean_n,
        sum_s=reports * mean_n * mu_p,
        sum_s2_over_n=reports * (var_p + mean_n * (mu_p * mu_p)),
    )


# ----- online estimator using only (s, N) -----


class OnlineReportStats:
    """Online estimator using only block-level summaries (s, N) per report.

    Maintains exact block-averaged aggregates (no EMA).
    """

    def __init__(self) -> None:
        """Initialize the online estimator."""
        self.reports: float = 0.0
        self.sum_n: float = 0.0
        self.sum_s: float = 0.0
        self.sum_s2_over_n: float = 0.0

    def apply_init_stats(self, init: InitStats) -> None:
        """Apply initial statistics to the estimator."""
        # Warm-start by adding externally computed aggregates.
        if init.reports <= 0.0:
            return
        self.reports += float(init.reports)
        self.sum_n += float(init.sum_n)
        self.sum_s += float(init.sum_s)
        self.sum_s2_over_n += float(init.sum_s2_over_n)

    def update(self, s: float, n: int) -> None:
        """Update the estimator with a new report."""
        if n <= 0:
            return
        self.reports += 1.0
        self.sum_n += float(n)
        self.sum_s += float(s)
        self.sum_s2_over_n += (float(s) * float(s)) / float(n)

    # Exact block-averaged estimates

    def mean(self) -> float:
        """Compute the mean estimate."""
        return (self.sum_s / self.sum_n) if self.sum_n > 0.0 else float("nan")

    def variance_block_avg(self) -> float:
        """Compute the block-averaged variance estimate."""
        if self.reports == 0.0 or self.sum_n == 0.0:
            return float("nan")
        e_s2_over_n = self.sum_s2_over_n / self.reports
        e_n = self.sum_n / self.reports
        mu = self.mean()
        sigma2 = e_s2_over_n - (mu * mu) * e_n
        return max(sigma2, 0.0)

    def second_moment_block_avg(self) -> float:
        """Compute the block-averaged second moment estimate."""
        mu = self.mean()
        sigma2 = self.variance_block_avg()
        if math.isnan(mu) or math.isnan(sigma2):
            return float("nan")
        return mu * mu + sigma2
