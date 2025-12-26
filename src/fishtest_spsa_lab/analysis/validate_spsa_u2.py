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

from .common import Line, compute_a_from_outcomes, plot_many
from .validate_variance import (
    InitStats,
    compute_init_stats_from_prior,
    compute_pentanomial_moments,
    gen_pentanomial_outcomes,
)

# ----- data models -----


@dataclass(slots=True)
class GlobalState:
    """Tracks the global state of the simulation."""

    iter_pairs: int = 0  # cumulative pairs processed


@dataclass(slots=True)
class Mu2State:
    """Online μ2 estimator using only (N, s) per report."""

    reports: float = 0.0
    sum_n: float = 0.0
    sum_s: float = 0.0
    sum_s2_over_n: float = 0.0
    mu2_init: float = 1.0


@dataclass(slots=True)
class SpsaSchedule:
    """Defines the SPSA schedule parameters."""

    a: float
    a_stability: float
    alpha: float
    c: float
    gamma: float


@dataclass(slots=True)
class Series:
    """Holds the time series data for plotting."""

    t_pairs: list[int]
    theta: list[float]
    mu2: list[float] | None = None


# ----- μ2 estimator (report-level, same math as SFAdamBlock) -----


def mu2_hat(state: Mu2State) -> float:
    """Estimate E[g^2] from report-level aggregates."""
    if state.reports <= 0.0:
        return state.mu2_init
    mu = (state.sum_s / state.sum_n) if state.sum_n > 0.0 else 0.0
    e_s2_over_n = state.sum_s2_over_n / state.reports
    e_n = state.sum_n / state.reports
    sigma2 = e_s2_over_n - (mu * mu) * e_n
    sigma2 = max(sigma2, 0.0)
    mu2 = mu * mu + sigma2
    return min(max(mu2, 1e-12), 4.0)


def update_mu2_stats(state: Mu2State, n: int, s: float) -> None:
    """Update μ2 aggregates AFTER using the current estimate."""
    if n <= 0:
        return
    state.reports += 1.0
    state.sum_n += float(n)
    state.sum_s += float(s)
    state.sum_s2_over_n += (float(s) * float(s)) / float(n)


# ----- SPSA gain math -----


def a_k(schedule: SpsaSchedule, k: int) -> float:
    """Return the SPSA $a_k$ term for pair index $k$."""
    return schedule.a / ((schedule.a_stability + k) ** schedule.alpha)


def c_k(schedule: SpsaSchedule, k: int) -> float:
    """Return the SPSA $c_k$ term for pair index $k$."""
    return schedule.c / (k**schedule.gamma)


def gain(schedule: SpsaSchedule, k: int) -> float:
    """Return the SPSA gain $a_k / c_k$ for pair index $k$."""
    ak = a_k(schedule, k)
    ck = c_k(schedule, k)
    return ak / ck if ck != 0.0 else 0.0


def mean_gain_over_block(schedule: SpsaSchedule, k0: int, n: int) -> float:
    """Return mean gain over a block starting at $k_0$ with length $n$."""
    if n <= 0:
        return 0.0
    return sum(gain(schedule, k0 + j) for j in range(n)) / n


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
    return tuple(
        (1.0 - alpha) * ps + alpha * pe for ps, pe in zip(p_start, p_end, strict=True)
    )  # type: ignore[return-value]


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


def main() -> None:
    """Run naive vs μ2 SPSA comparison with changing p5."""
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
    plt.show()


if __name__ == "__main__":
    main()
