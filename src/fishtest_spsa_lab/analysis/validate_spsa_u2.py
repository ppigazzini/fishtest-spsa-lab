"""Compare naive vs μ2-normalized SPSA under a changing p5.

- p5 moves from a biased distribution to a balanced one over reports
  (to mimic starting far from balance and ending near a draw-heavy regime).
- We run:
    - macro_plain: corrected SPSA macro (mean gain over block)
    - macro_mu2  : same macro, but scalar signal scaled by 1/sqrt(μ2_hat)

This is NOT a macro-vs-micro correctness test; it's a toy experiment
for how μ2-normalization behaves when the noise distribution changes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

import matplotlib.pyplot as plt

from .common import (
    Line,
    compute_a_from_outcomes,
    mu2_hat,
    plot_many,
    update_mu2_stats,
)
from .gate import Gate, show
from .validate_spsa import SpsaSchedule, mean_gain_over_block
from .pentanomial import (
    InitStats,
    compute_init_stats_from_prior,
    compute_pentanomial_moments,
    gen_pentanomial_outcomes,
)

#: How close the final mu2 estimate must sit to the realized per-pair mean
#: square. Measured 0.0755 on a realized level of 0.886, i.e. 8.5%; the bound is
#: set at 0.15 so a broken estimator, or one pinned at a clamp, fails.
MU2_TOLERANCE: float = 0.15

#: The estimator's clamp in common.py :: mu2_hat. Sitting on either end means the
#: statistic is not measuring anything.
MU2_CLAMP_LOW: float = 1e-12
MU2_CLAMP_HIGH: float = 4.0

# ----- data models -----


@dataclass(slots=True)
class GlobalState:
    """Tracks the global state of the simulation."""

    iter_pairs: int = 0  # cumulative pairs processed


@dataclass(slots=True)
class Mu2State:
    """Online mu2 estimator using only (N, s) per report."""

    reports: float = 0.0
    sum_n: float = 0.0
    sum_s: float = 0.0
    sum_s2_over_n: float = 0.0
    mu2_init: float = 1.0


@dataclass(slots=True)
class Series:
    """Holds the time series data for plotting."""

    t_pairs: list[int]
    theta: list[float]
    mu2: list[float] | None = None


# ----- macro updates -----


def macro_update_plain(
    glob: GlobalState,
    theta: float,
    *,
    outcomes: Sequence[int],
    sched: SpsaSchedule,
) -> float:
    """Corrected macro update (mean gain, no μ2)."""
    n = len(outcomes)
    if n == 0:
        return theta
    k0 = glob.iter_pairs + 1
    g_bar = mean_gain_over_block(sched, k0, n)
    result = float(sum(outcomes))
    theta = theta + g_bar * result
    glob.iter_pairs += n
    return theta


def macro_update_mu2(  # noqa: PLR0913
    glob: GlobalState,
    theta: float,
    *,
    outcomes: Sequence[int],
    sched: SpsaSchedule,
    mu2_state: Mu2State,
    mu2_ref: float,
) -> tuple[float, float]:
    """Corrected macro update with μ2-normalized scalar signal.

    Returns
    -------
    theta : float
        Updated parameter value.
    mu2_used : float
        The μ2 estimate used *before* incorporating this block.

    """
    n = len(outcomes)
    if n == 0:
        # No update; return current theta and the current μ2 estimate.
        return theta, mu2_hat(mu2_state)
    k0 = glob.iter_pairs + 1
    g_bar = mean_gain_over_block(sched, k0, n)
    result = float(sum(outcomes))

    mu2 = mu2_hat(mu2_state)
    raw_scale = 1.0 if mu2 <= 0.0 else (mu2_ref / mu2) ** 0.5

    # Conservative clipping of the μ2-based rescaling factor to
    # avoid large, noisy swings in the effective learning rate.
    min_scale = 0.5
    max_scale = 2.0
    scale = max(min_scale, min(max_scale, raw_scale))

    theta = theta + g_bar * scale * result
    glob.iter_pairs += n
    update_mu2_stats(mu2_state, n, result)
    return theta, mu2


def run_macro_plain(
    outcomes_by_report: list[list[int]],
    *,
    sched: SpsaSchedule,
) -> Series:
    """Run the corrected macro update without μ2 normalization."""
    glob = GlobalState()
    theta = 0.0
    t: list[int] = [0]
    th: list[float] = [theta]
    for outs in outcomes_by_report:
        theta = macro_update_plain(glob, theta, outcomes=outs, sched=sched)
        t.append(glob.iter_pairs)
        th.append(theta)
    return Series(t_pairs=t, theta=th)


def run_macro_mu2(
    outcomes_by_report: list[list[int]],
    *,
    sched: SpsaSchedule,
    mu2_init: float,
    init_stats: InitStats | None = None,
) -> Series:
    """Run the corrected macro update with μ2-normalized scalar signal."""
    glob = GlobalState()
    mu_state = Mu2State(mu2_init=mu2_init)
    # Warm-start the μ2 estimator with externally computed aggregates,
    # mirroring OnlineReportStats.apply_init_stats.
    if init_stats is not None and init_stats.reports > 0.0:
        mu_state.reports = float(init_stats.reports)
        mu_state.sum_n = float(init_stats.sum_n)
        mu_state.sum_s = float(init_stats.sum_s)
        mu_state.sum_s2_over_n = float(init_stats.sum_s2_over_n)
    theta = 0.0
    t: list[int] = [0]
    th: list[float] = [theta]
    mu2_values: list[float] = [mu2_init]
    for outs in outcomes_by_report:
        theta, mu2_val = macro_update_mu2(
            glob,
            theta,
            outcomes=outs,
            sched=sched,
            mu2_state=mu_state,
            mu2_ref=mu2_init,
        )
        t.append(glob.iter_pairs)
        th.append(theta)
        mu2_values.append(mu2_val)
    return Series(t_pairs=t, theta=th, mu2=mu2_values)


# ----- changing p5 schedule -----


def interpolate_p5(
    p_start: tuple[float, float, float, float, float],
    p_end: tuple[float, float, float, float, float],
    alpha: float,
) -> tuple[float, float, float, float, float]:
    """Linear interpolation between two pentanomials."""
    p0, p1, p2, p3, p4 = (
        (1.0 - alpha) * ps + alpha * pe for ps, pe in zip(p_start, p_end, strict=True)
    )
    return (p0, p1, p2, p3, p4)


def make_changing_p5_schedule(  # noqa: PLR0913
    num_reports: int,
    n_min: int,
    n_max: int,
    p5_start: tuple[float, float, float, float, float],
    p5_end: tuple[float, float, float, float, float],
    base_seed: int,
) -> tuple[list[tuple[float, float, float, float, float]], list[list[int]]]:
    """Build a schedule where p5 drifts from p5_start to p5_end over reports."""
    rng = random.Random(base_seed)  # noqa: S311
    p5s: list[tuple[float, float, float, float, float]] = []
    outcomes_by_report: list[list[int]] = []
    for r in range(num_reports):
        alpha = r / (num_reports - 1) if num_reports > 1 else 1.0
        p5_r = interpolate_p5(p5_start, p5_end, alpha)
        n = rng.randint(n_min, n_max)
        seed_r = rng.randint(0, 10**9)
        outs = gen_pentanomial_outcomes(seed_r, n, p5_r)
        p5s.append(p5_r)
        outcomes_by_report.append(outs)
    return p5s, outcomes_by_report


# ----- main -----


def main() -> int:
    """Run naive vs μ2 SPSA comparison with changing p5, and return an exit code."""
    gate = Gate(
        "validate-spsa-u2",
        "online mu2 tracks a changing outcome distribution",
    )

    base_seed: int = 424242
    num_reports: int = 120
    n_min, n_max = 1, 32

    # Start with a biased p5 (positive mean), end with a balanced one (mean ≈ 0).
    p5_start: tuple[float, float, float, float, float] = (
        0.10,  # LL
        0.20,  # LD+DL
        0.40,  # DD+WL+LW
        0.20,  # WD+DW
        0.10,  # WW
    )
    p5_end: tuple[float, float, float, float, float] = (
        0.025,
        0.20,
        0.55,
        0.20,
        0.025,
    )

    _p5s, outcomes_by_report = make_changing_p5_schedule(
        num_reports,
        n_min,
        n_max,
        p5_start,
        p5_end,
        base_seed,
    )

    # Use realized block lengths to set A (stability offset) like validate_spsa.
    a_val = compute_a_from_outcomes(outcomes_by_report)
    sched = SpsaSchedule(
        a=0.1,
        a_stability=a_val,
        alpha=0.602,
        c=1.0,
        gamma=0.101,
    )

    # μ2 init from the starting p5 (second moment of [-2..2] with p5_start).
    _mu_start, mu2_start, _var_start = compute_pentanomial_moments(p5_start)

    # Warm-start aggregates using the same math as validate_variance.
    prior_reports: float = 5.0
    prior_mean_n: float = (n_min + n_max) / 2.0
    init_stats = compute_init_stats_from_prior(
        p5_start,
        prior_reports,
        prior_mean_n,
    )

    macro_plain = run_macro_plain(outcomes_by_report, sched=sched)
    macro_mu2 = run_macro_mu2(
        outcomes_by_report,
        sched=sched,
        mu2_init=mu2_start,
        init_stats=init_stats,
    )

    # This script is a behavioural experiment, not a macro-vs-micro identity, so
    # what it can honestly gate is the estimator: mu2 must track the realized
    # outcome mix as p5 moves from biased to balanced, and must not be pinned at
    # a clamp. Without these checks the run asserted nothing and exited 0.
    outcomes_flat = [o for report in outcomes_by_report for o in report]
    realized_mu2 = sum(float(o) * float(o) for o in outcomes_flat) / len(outcomes_flat)
    _mu_end, mu2_end, _var_end = compute_pentanomial_moments(p5_end)
    mu2_series = macro_mu2.mu2 or []

    gate.note("reports", num_reports)
    gate.note("total pairs", macro_plain.t_pairs[-1])
    gate.note("base seed", base_seed)
    gate.note("mu2 of p5_start", mu2_start)
    gate.note("mu2 of p5_end", mu2_end)
    gate.note("realized per-pair E[g^2]", realized_mu2)
    gate.note("final mu2 estimate", mu2_series[-1] if mu2_series else float("nan"))

    gate.check_le(
        "time axes agree",
        0.0 if macro_plain.t_pairs == macro_mu2.t_pairs else 1.0,
        0.0,
    )
    gate.check_le(
        "mu2 series is populated",
        0.0 if len(mu2_series) == len(macro_mu2.t_pairs) else 1.0,
        0.0,
    )
    gate.check_close(
        "final mu2 tracks the realized second moment",
        mu2_series[-1] if mu2_series else float("nan"),
        realized_mu2,
        MU2_TOLERANCE,
    )
    gate.check_le(
        "mu2 never reaches the low clamp",
        MU2_CLAMP_LOW,
        min(mu2_series) if mu2_series else 0.0,
        f"min observed {min(mu2_series):.6g}" if mu2_series else "no data",
    )
    gate.check_le(
        "mu2 never reaches the high clamp",
        max(mu2_series) if mu2_series else float("inf"),
        MU2_CLAMP_HIGH,
    )
    # The estimate must actually move: starting at the biased p5 second moment
    # and ending near the balanced one is the whole point of the experiment.
    gate.check_le(
        "mu2 fell from the biased level toward the balanced one",
        mu2_series[-1] if mu2_series else float("inf"),
        mu2_start,
        f"start {mu2_start:.6g} -> end {mu2_series[-1]:.6g}" if mu2_series else "",
    )

    fig, (ax_theta, ax_mu2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Theta trajectories (plain vs μ2-normalized)
    plot_many(
        ax_theta,
        Line(macro_plain.t_pairs, macro_plain.theta, "theta — macro (plain)"),
        Line(
            macro_mu2.t_pairs,
            macro_mu2.theta,
            "theta — macro μ2-normalized",
            linestyle="--",
        ),
        y_label="theta",
    )

    # μ2 trajectory used by the μ2-normalized macro run
    if macro_mu2.mu2 is not None:
        ax_mu2.plot(macro_mu2.t_pairs, macro_mu2.mu2, label="μ2 estimate")
        ax_mu2.set_ylabel("μ2")
        ax_mu2.legend(loc="best")

    ax_mu2.set_xlabel("pairs")
    fig.suptitle(
        "SPSA μ2 experiment — changing p5 (biased → balanced)",
        y=0.98,
    )
    plt.tight_layout()
    show(fig)

    return gate.report()


if __name__ == "__main__":
    raise SystemExit(main())
