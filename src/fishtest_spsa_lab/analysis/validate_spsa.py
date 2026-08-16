"""SPSA simulation refactored to mimic the schedule-free SGD script structure.

We compare:
- Macro (uncorrected): one shot per report using g(k0) at the block start
- Macro (corrected): one shot using mean per-pair gain Ḡ = (1/n) Σ a_k/c_k
- Micro (const-mean): sequential, distributing result/n across n steps
- Micro (real): sequential, using the actual outcomes sequence

Plot overlay: original order vs end-adjacent shuffled order (same as SGD).
Gate: corrected macro == micro const-mean (for both orders), at 1e-12.
"""
# ruff: noqa: I001

import random
from dataclasses import dataclass
from collections.abc import Sequence

import matplotlib.pyplot as plt

from .common import (
    Line,
    plot_many,
    make_schedule,
    end_adjacent_shuffle,
    build_sequence,
    max_abs_gap,
    series_scale,
    compute_a_from_outcomes,
)
from .gate import Gate, show

#: Tolerance for the macro-vs-micro anchor. The measured gap is of order 1e-16
#: on a theta scale of 0.16, so this leaves about four orders of headroom.
TOLERANCE: float = 1e-12

# ----- data models -----


@dataclass(slots=True)
class GlobalState:
    """Tracks the global state of the simulation."""

    iter_pairs: int = 0  # cumulative pairs processed


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


# ----- core math -----


def a_k(schedule: SpsaSchedule, k: int) -> float:
    """Compute the step size a_k at step k."""
    return schedule.a / ((schedule.a_stability + k) ** schedule.alpha)


def c_k(schedule: SpsaSchedule, k: int) -> float:
    """Compute the perturbation size c_k at step k.

    Precondition: k >= 1 (k = 0 causes ZeroDivisionError when gamma > 0).
    """
    return schedule.c / (k**schedule.gamma)


def gain(schedule: SpsaSchedule, k: int) -> float:
    """Compute the gain g(k) = a_k / c_k at step k.

    Precondition: k >= 1 (c_k is undefined at k = 0).
    """
    # g(k) = a_k / c_k
    ak = a_k(schedule, k)
    ck = c_k(schedule, k)
    return ak / ck if ck != 0.0 else 0.0


def mean_gain_over_block(schedule: SpsaSchedule, k0: int, n: int) -> float:
    """Compute the mean gain over a block of size n starting at k0."""
    if n <= 0:
        return 0.0
    return sum(gain(schedule, k0 + j) for j in range(n)) / n


def macro_update_uncorrected(
    glob: GlobalState,
    theta: float,
    *,
    outcomes: Sequence[int],
    sched: SpsaSchedule,
) -> float:
    """Perform an uncorrected macro update."""
    # Uncorrected: use g(k0) for the whole block
    n = len(outcomes)
    if n == 0:
        return theta
    k0 = glob.iter_pairs + 1
    g0 = gain(sched, k0)
    result = float(sum(outcomes))
    theta = theta + g0 * result
    glob.iter_pairs += n
    return theta


def macro_update_corrected(
    glob: GlobalState,
    theta: float,
    *,
    outcomes: Sequence[int],
    sched: SpsaSchedule,
) -> float:
    """Perform a corrected macro update using mean gain."""
    # Corrected: use mean per-pair gain Ḡ across the block
    n = len(outcomes)
    if n == 0:
        return theta
    k0 = glob.iter_pairs + 1
    g_bar = mean_gain_over_block(sched, k0, n)
    result = float(sum(outcomes))
    theta = theta + g_bar * result
    glob.iter_pairs += n
    return theta


def micro_apply_sequence(
    glob0: GlobalState,
    theta0: float,
    *,
    seq_num: Sequence[float],
    sched: SpsaSchedule,
) -> float:
    """Apply a sequence of micro updates."""
    # True per-pair sequential updates (local copy of glob for per-step k)
    glob = GlobalState(glob0.iter_pairs)
    theta = theta0
    for num in seq_num:
        k = glob.iter_pairs + 1
        theta = theta + gain(sched, k) * float(num)
        glob.iter_pairs += 1
    return theta


# ----- runners -----


def run_macro_uncorrected(
    outcomes_by_report: list[list[int]],
    *,
    sched: SpsaSchedule,
) -> Series:
    """Run the simulation using uncorrected macro updates."""
    glob = GlobalState()
    theta = 0.0
    t: list[int] = [0]
    th: list[float] = [theta]
    for outs in outcomes_by_report:
        theta = macro_update_uncorrected(glob, theta, outcomes=outs, sched=sched)
        t.append(glob.iter_pairs)
        th.append(theta)
    return Series(t_pairs=t, theta=th)


def run_macro_corrected(
    outcomes_by_report: list[list[int]],
    *,
    sched: SpsaSchedule,
) -> Series:
    """Run the simulation using corrected macro updates."""
    glob = GlobalState()
    theta = 0.0
    t: list[int] = [0]
    th: list[float] = [theta]
    for outs in outcomes_by_report:
        theta = macro_update_corrected(glob, theta, outcomes=outs, sched=sched)
        t.append(glob.iter_pairs)
        th.append(theta)
    return Series(t_pairs=t, theta=th)


def run_micro(
    seqs_by_report: list[list[float]],
    *,
    sched: SpsaSchedule,
) -> Series:
    """Run the simulation using micro updates."""
    glob = GlobalState()
    theta = 0.0
    # Start at t=0 for parity with SGD/Adam
    t: list[int] = [0]
    th: list[float] = [theta]
    for seq_num in seqs_by_report:
        theta = micro_apply_sequence(glob, theta, seq_num=seq_num, sched=sched)
        # advance outer time to the end of the report (derive n from sequence)
        n_block = len(seq_num)
        glob.iter_pairs += n_block
        t.append(glob.iter_pairs)
        th.append(theta)
    return Series(t_pairs=t, theta=th)


# ----- main -----


def _axis_gap(*series: Series) -> float:
    """Return 0 when every series shares one time axis, 1 otherwise.

    Reported as a check rather than asserted so a mismatch names itself in the
    result table instead of raising a traceback.
    """
    first = series[0].t_pairs
    return 0.0 if all(s.t_pairs == first for s in series[1:]) else 1.0


def main() -> int:
    """Run the SPSA macro-vs-micro validation and return an exit code."""
    gate = Gate(
        "validate-spsa",
        "classic SPSA: mean-gain macro == const-mean micro",
    )

    # schedule (mirror SGD)
    base_seed: int = 424242
    num_reports: int = 100
    n_min, n_max = 1, 32
    p5: tuple[float, float, float, float, float] = (0.025, 0.20, 0.55, 0.20, 0.025)

    # Build schedule; discard Ns to stay airtight (derive n from sequences)
    _, outcomes_by_report = make_schedule(num_reports, n_min, n_max, p5, base_seed)

    # For A we need total pairs; convenience helper from common
    a_val = compute_a_from_outcomes(outcomes_by_report)
    # Textbook SPSA params
    sched = SpsaSchedule(
        a=0.1,
        a_stability=a_val,
        alpha=0.602,
        c=1.0,
        gamma=0.101,
    )

    # original order
    macro_cor = run_macro_corrected(outcomes_by_report, sched=sched)
    macro_unc = run_macro_uncorrected(outcomes_by_report, sched=sched)
    seqs_mean = [build_sequence(outs, "const_mean") for outs in outcomes_by_report]
    seqs_real = [build_sequence(outs, "outcomes") for outs in outcomes_by_report]
    micro_mean = run_micro(seqs_mean, sched=sched)
    micro_real = run_micro(seqs_real, sched=sched)

    gate.note("reports", num_reports)
    gate.note("pairs per report", f"{n_min}..{n_max}")
    gate.note("total pairs", macro_cor.t_pairs[-1])
    gate.note("base seed", base_seed)
    gate.note("theta scale", series_scale(micro_mean.theta))

    # The anchor: one batched update equals N sequential micro-updates wherever
    # the dynamics are linear. Asserted at 1e-12 against a measured gap of order
    # 1e-16, so the tolerance carries four orders of headroom.
    gate.check_le(
        "time axes agree (original)",
        _axis_gap(macro_cor, micro_mean, micro_real, macro_unc),
        0.0,
    )
    gate.check_le(
        "mean-gain macro == const-mean micro",
        max_abs_gap(macro_cor.theta, micro_mean.theta),
        TOLERANCE,
    )

    # Figure 1: only the original schedule
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6), sharex=True)
    plot_many(
        ax1,
        # Ground-truth sequential path first
        Line(
            micro_real.t_pairs,
            micro_real.theta,
            "theta — micro real",
            linestyle="-.",
        ),
        # Theoretical bridge second
        Line(
            micro_mean.t_pairs,
            micro_mean.theta,
            "theta — micro mean",
            linestyle="--",
        ),
        # Production macro path
        Line(macro_cor.t_pairs, macro_cor.theta, "theta — macro"),
        # Incorrect macro baseline last
        Line(
            macro_unc.t_pairs,
            macro_unc.theta,
            "theta — macro (uncorrected)",
            linestyle=":",
            linewidth=2,
        ),
        y_label="theta",
    )
    ax1.set_xlabel("pairs")
    fig1.suptitle("SPSA — single schedule (theta)", y=0.98)
    plt.tight_layout()
    show(fig1)

    # custom shuffled order (same end-adjacent scheme as SGD)
    p_swap = 4.0 / 5.0
    idx = end_adjacent_shuffle(
        list(range(num_reports)),
        p=p_swap,
        rng=random.Random(base_seed + 1337),  # noqa: S311
    )
    outcomes_by_report_shuf = [outcomes_by_report[i] for i in idx]

    macro_cor2 = run_macro_corrected(outcomes_by_report_shuf, sched=sched)
    macro_unc2 = run_macro_uncorrected(outcomes_by_report_shuf, sched=sched)
    seqs_mean_shuf = [
        build_sequence(outs, "const_mean") for outs in outcomes_by_report_shuf
    ]
    seqs_real_shuf = [
        build_sequence(outs, "outcomes") for outs in outcomes_by_report_shuf
    ]
    micro_mean2 = run_micro(seqs_mean_shuf, sched=sched)
    micro_real2 = run_micro(seqs_real_shuf, sched=sched)

    gate.check_le(
        "time axes agree (shuffled)",
        _axis_gap(macro_cor2, micro_mean2, micro_real2, macro_unc2),
        0.0,
    )
    gate.check_le(
        "mean-gain macro == const-mean micro (shuffled)",
        max_abs_gap(macro_cor2.theta, micro_mean2.theta),
        TOLERANCE,
    )

    # The tolerance is only meaningful if it can reject something. The
    # uncorrected macro -- which reuses the first micro-step's gain for the whole
    # block -- is the defect the mean-gain correction removes, and it must land
    # far outside the tolerance the corrected path is asserted at.
    gate.check_le(
        "uncorrected macro is rejected by that tolerance",
        TOLERANCE,
        max_abs_gap(macro_unc.theta, micro_mean.theta),
        "the check above can fail; it is not a tautology",
    )

    # Figure 2: original vs shuffled overlay
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6), sharex=True)
    plot_many(
        ax2,
        # 1) Micro real (ground truth): original vs shuffled
        Line(
            micro_real.t_pairs,
            micro_real.theta,
            "theta — micro real (orig)",
            linestyle="-.",
        ),
        Line(
            micro_real2.t_pairs,
            micro_real2.theta,
            "theta — micro real (shuf)",
            linestyle="-.",
            linewidth=1.5,
            alpha=0.6,
        ),
        # 2) Micro mean (theoretical bridge): original vs shuffled
        Line(
            micro_mean.t_pairs,
            micro_mean.theta,
            "theta — micro mean (orig)",
            linestyle="--",
        ),
        Line(
            micro_mean2.t_pairs,
            micro_mean2.theta,
            "theta — micro mean (shuf)",
            linestyle="--",
            linewidth=1.5,
            alpha=0.6,
        ),
        # 3) Correct macro: original vs shuffled
        Line(macro_cor.t_pairs, macro_cor.theta, "theta — macro (orig)"),
        Line(
            macro_cor2.t_pairs,
            macro_cor2.theta,
            "theta — macro (shuf)",
            linewidth=1.5,
            alpha=0.6,
        ),
        # 4) Incorrect macro baseline: original vs shuffled
        Line(
            macro_unc.t_pairs,
            macro_unc.theta,
            "theta — macro unc. (orig)",
            linestyle=":",
            linewidth=2,
        ),
        Line(
            macro_unc2.t_pairs,
            macro_unc2.theta,
            "theta — macro unc. (shuf)",
            linestyle=":",
            linewidth=1.5,
            alpha=0.6,
        ),
        y_label="theta",
    )
    ax2.set_xlabel("pairs")
    fig2.suptitle("SPSA — original vs end-adjacent shuffled (theta)", y=0.98)
    plt.tight_layout()
    show(fig2)

    return gate.report()


if __name__ == "__main__":
    raise SystemExit(main())
