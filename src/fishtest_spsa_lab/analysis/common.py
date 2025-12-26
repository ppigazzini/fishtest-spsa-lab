"""Shared utilities for analysis scripts."""

from __future__ import annotations

import random
from dataclasses import dataclass
from math import isclose
from typing import TYPE_CHECKING

from .validate_variance import (
    compute_pentanomial_moments,
    gen_pentanomial_outcomes,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import matplotlib.pyplot as plt


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
    """Create a list of N per report and the corresponding outcomes with a local RNG."""
    rng = random.Random(base_seed)  # noqa: S311
    ns = [rng.randint(n_min, n_max) for _ in range(num_reports)]
    outcomes_by_report = [
        outcome_fn(base_seed + r, ns[r], p5) for r in range(num_reports)
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
        isclose(x, y, rel_tol=rel, abs_tol=abs_tol) for x, y in zip(a, b, strict=False)
    )


def compute_a_from_outcomes(
    outcomes_by_report: Sequence[Sequence[int]],
    frac: float = 0.1,
) -> float:
    """SPSA convenience: A = frac * total_pairs based on realized block lengths."""
    total_pairs = float(sum(len(outs) for outs in outcomes_by_report))
    return frac * total_pairs


__all__ = [
    "Line",
    "build_sequence",
    "compute_a_from_outcomes",
    "compute_pentanomial_moments",
    "end_adjacent_shuffle",
    "gen_pentanomial_outcomes",
    "make_schedule",
    "plot_many",
    "series_allclose",
]
