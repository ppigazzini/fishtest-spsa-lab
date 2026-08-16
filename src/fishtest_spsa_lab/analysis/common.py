"""Shared utilities for analysis scripts."""

from __future__ import annotations

import random
from dataclasses import dataclass
from math import isclose
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from .validate_variance import (
    compute_pentanomial_moments,
    gen_pentanomial_outcomes,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import matplotlib.pyplot as plt


# ----- protocols for duck-typed state -----


@runtime_checkable
class HasSfWeightSum(Protocol):
    """Object exposing a schedule-free weight accumulator."""

    sf_weight_sum: float


@runtime_checkable
class HasMu2Stats(Protocol):
    """Object exposing online mu2 estimator aggregates."""

    reports: float
    sum_n: float
    sum_s: float
    sum_s2_over_n: float
    mu2_init: float


# ----- plotting -----


@dataclass(slots=True)
class Line:
    """Data class representing a line to be plotted."""

    t: Sequence[int]
    y: Sequence[float]
    label: str
    linestyle: str = "-"
    linewidth: float = 2.0
    alpha: float = 1.0


def plot_many(
    ax: plt.Axes,
    *lines: Line,
    y_label: str | None = None,
    legend_ncol: int = 2,
) -> None:
    """Plot multiple lines on a single axes."""
    for ln in lines:
        ax.plot(
            ln.t,
            ln.y,
            label=ln.label,
            linestyle=ln.linestyle,
            linewidth=ln.linewidth,
            alpha=ln.alpha,
        )
    if y_label:
        ax.set_ylabel(y_label)
    ax.grid(visible=True, alpha=0.3)
    ax.legend(ncol=legend_ncol)


# ----- schedules + sequences -----


def make_schedule(  # noqa: PLR0913
    num_reports: int,
    n_min: int,
    n_max: int,
    p5: tuple[float, float, float, float, float],
    base_seed: int,
    *,
    outcome_fn: Callable[
        [int, int, tuple[float, float, float, float, float]],
        list[int],
    ] = gen_pentanomial_outcomes,
) -> tuple[list[int], list[list[int]]]:
    """Create a list of N per report and the corresponding outcomes with a local RNG.

    Per-report seeds are spawned from ``base_seed`` rather than derived as
    ``base_seed + r``. Under the old arithmetic form, report ``r`` of the arm
    seeded ``s`` and report ``r - 1`` of the arm seeded ``s + 1`` received the
    same seed and therefore the same outcomes: arms meant to be independent
    shared all but one of their reports, shifted by one position. Measured
    identical for every report tested. ``SeedSequence`` spawning gives streams
    with no such relationship, which is what ``validate_spsa_u2`` already did.
    """
    rng = random.Random(base_seed)  # noqa: S311
    ns = [rng.randint(n_min, n_max) for _ in range(num_reports)]
    children = np.random.SeedSequence(base_seed).spawn(num_reports)
    outcomes_by_report = [
        outcome_fn(
            int(children[r].generate_state(1, dtype=np.uint32)[0]),
            ns[r],
            p5,
        )
        for r in range(num_reports)
    ]
    return ns, outcomes_by_report


def end_adjacent_shuffle(order: list[int], p: float, rng: random.Random) -> list[int]:
    """Single backward sweep: for pos from end->1, swap (pos,pos-1) with prob p."""
    idx = order.copy()
    for pos in range(len(idx) - 1, 0, -1):
        if rng.random() < p:
            idx[pos], idx[pos - 1] = idx[pos - 1], idx[pos]
    return idx


def build_sequence(outcomes: Sequence[int], kind: str) -> list[float]:
    """Build generic sequence for SPSA/SGD.

    - 'outcomes': per-outcome values
    - 'const_mean': N copies of the block mean
    """
    n = len(outcomes)
    if n == 0:
        return []
    s = float(sum(outcomes))
    mean = s / n
    if kind == "outcomes":
        return [float(o) for o in outcomes]
    if kind == "const_mean":
        return [mean] * n
    msg = "kind must be 'outcomes' or 'const_mean'"
    raise ValueError(msg)


# ----- small utilities -----


def series_allclose(
    a: Sequence[float],
    b: Sequence[float],
    rel: float = 1e-12,
    abs_tol: float = 1e-12,
) -> bool:
    """Check if two series are element-wise equal within a tolerance."""
    return all(
        isclose(x, y, rel_tol=rel, abs_tol=abs_tol) for x, y in zip(a, b, strict=True)
    )


def max_abs_gap(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the largest element-wise absolute difference between two series.

    The measured quantity behind ``series_allclose``. A gate reports this rather
    than a boolean so the margin against the tolerance is visible: a gap of
    2.5e-16 against a tolerance of 1e-12 says the tolerance has four orders of
    headroom, where "True" says nothing.
    """
    if not a and not b:
        return 0.0
    return max(abs(float(x) - float(y)) for x, y in zip(a, b, strict=True))


def series_scale(*series: Sequence[float]) -> float:
    """Return the largest absolute value across the given series.

    A gap is only interpretable next to the scale it sits on.
    """
    values = [abs(float(v)) for s in series for v in s]
    return max(values) if values else 0.0


def compute_a_from_outcomes(
    outcomes_by_report: Sequence[Sequence[int]],
    frac: float = 0.1,
) -> float:
    """SPSA convenience: A = frac * total_pairs based on realized block lengths."""
    total_pairs = float(sum(len(outs) for outs in outcomes_by_report))
    return frac * total_pairs


# ----- schedule-free shared math -----


def reconstruct_x_prev(theta_prev: float, z_prev: float, beta: float) -> float:
    """Reconstruct x_prev from theta_prev and z_prev.

    Caller must ensure beta != 0 (when beta == 0, x == z directly).
    """
    return (theta_prev - (1.0 - beta) * z_prev) / beta


def sf_weighting_update(glob: HasSfWeightSum, n: int, lr: float) -> float:
    """Advance the schedule-free weight accumulator by *n* pairs at rate *lr*.

    Returns the interpolation coefficient for the current report.
    """
    report_weight = lr * n
    glob.sf_weight_sum += report_weight
    return report_weight / glob.sf_weight_sum if glob.sf_weight_sum > 0 else 1.0


# ----- online mu2 estimator -----


def mu2_hat(state: HasMu2Stats) -> float:
    """Block-averaged E[g^2] estimator from report-level (N, s) aggregates.

    Before any reports arrive, returns ``state.mu2_init``.
    """
    if state.reports <= 0.0:
        return state.mu2_init
    mu = (state.sum_s / state.sum_n) if state.sum_n > 0.0 else 0.0
    e_s2_over_n = state.sum_s2_over_n / state.reports
    e_n = state.sum_n / state.reports
    sigma2 = e_s2_over_n - (mu * mu) * e_n
    sigma2 = max(sigma2, 0.0)
    mu2 = mu * mu + sigma2
    return min(max(mu2, 1e-12), 4.0)


def update_mu2_stats(state: HasMu2Stats, n: int, s: float) -> None:
    """Update mu2 aggregates AFTER using the current estimate for this report."""
    if n <= 0:
        return
    state.reports += 1.0
    state.sum_n += float(n)
    state.sum_s += float(s)
    state.sum_s2_over_n += (float(s) * float(s)) / float(n)


__all__ = [
    "HasMu2Stats",
    "HasSfWeightSum",
    "Line",
    "build_sequence",
    "compute_a_from_outcomes",
    "compute_pentanomial_moments",
    "end_adjacent_shuffle",
    "gen_pentanomial_outcomes",
    "make_schedule",
    "max_abs_gap",
    "mu2_hat",
    "plot_many",
    "reconstruct_x_prev",
    "series_allclose",
    "series_scale",
    "sf_weighting_update",
    "update_mu2_stats",
]
