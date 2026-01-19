"""Clean-room simulation for schedule-free SGD vs micro loop.

States:
- z: fast iterate (unclamped update state in θ-space)
- x: Polyak surrogate (slow moving average of z via schedule-free mass)
- theta: blended state, theta = (1 - beta) * z + beta * x  (the exported value)

Three paths over a shared schedule (Ns and outcomes per report):
- Fishtest (macro): z += lr*c*result; x uses triangular closed form inside report
- Micro loop (const): N equal micro-steps s = (lr*c*result)/N; exact per-step mass
  blend for x
- Real micro step (outcomes): N micro-steps s_j = lr*c*out_j; exact per-step mass
  blend for x

Start from z=x=theta=0, iter_pairs=0, sf_weight_sum=0. Plot x vs cumulative pairs.
"""

import random
from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt

from .common import (
    Line,
    build_sequence,
    end_adjacent_shuffle,
    make_schedule,
    plot_many,
    series_allclose,
)

# ----- data models -----


@dataclass(slots=True)
class GlobalState:
    """Global state of the simulation."""

    iter_pairs: int = 0
    sf_weight_sum: float = 0.0


@dataclass(slots=True)
class ParamState:
    """Parameter state for SGD."""

    theta: float = 0.0
    z: float = 0.0
    c: float = 0.5
    beta: float = 0.9


@dataclass(slots=True)
class Update:
    """Result of an update step."""

    x: float
    z: float
    theta: float


@dataclass(slots=True)
class Series:
    """Time series data for plotting."""

    t_pairs: list[int]
    x: list[float]
    z: list[float]
    theta: list[float]


# ----- core math -----


def reconstruct_x_prev(theta_prev: float, z_prev: float, beta: float) -> float:
    """Reconstruct x_prev from theta_prev and z_prev."""
    # If beta == 0, we never call this; x=z is used directly.
    return (theta_prev - (1.0 - beta) * z_prev) / beta


def sf_weighting_update(glob: GlobalState, n: int, lr: float) -> float:
    """Update schedule-free weighting."""
    # schedule-free mass increment
    report_weight = lr * n
    glob.sf_weight_sum += report_weight
    return report_weight / glob.sf_weight_sum if glob.sf_weight_sum > 0 else 1.0


def macro_update_sgd(
    glob: GlobalState,
    param: ParamState,
    *,
    outcomes: Sequence[int],
    lr: float,
) -> Update:
    """Single-report (macro) update for SGD."""
    n = len(outcomes)
    result = float(sum(outcomes))

    # advance time/mass
    glob.iter_pairs += n
    weight_sum_prev = glob.sf_weight_sum
    report_weight = lr * n
    glob.sf_weight_sum += report_weight
    weight_sum_curr = glob.sf_weight_sum

    # fast iterate
    z_prev = param.z
    delta_total = lr * param.c * result
    z_new = z_prev + delta_total

    # surrogate
    if param.beta == 0.0:
        x_new = z_new
    else:
        x_prev = reconstruct_x_prev(param.theta, z_prev, param.beta)
        tri_factor = (n + 1) / 2.0
        x_num = (
            weight_sum_prev * x_prev
            + report_weight * z_prev
            + lr * delta_total * tri_factor
        )
        x_new = x_num / weight_sum_curr

    theta_new = (1.0 - param.beta) * z_new + param.beta * x_new
    return Update(x=x_new, z=z_new, theta=theta_new)


def micro_apply_sequence(
    glob0: GlobalState,
    param0: ParamState,
    *,
    seq_num: Sequence[float],
    lr: float,
) -> Update:
    """Apply a sequence of micro-steps."""
    glob = GlobalState(glob0.iter_pairs, glob0.sf_weight_sum)
    z = param0.z
    x = z if param0.beta == 0.0 else reconstruct_x_prev(param0.theta, z, param0.beta)

    for num in seq_num:
        glob.iter_pairs += 1
        a_k = sf_weighting_update(glob, 1, lr)
        step = lr * param0.c * num
        z = z + step
        if param0.beta != 0.0:
            x = (1.0 - a_k) * x + a_k * z

    theta = (1.0 - param0.beta) * z + param0.beta * x
    return Update(x=x, z=z, theta=theta)


# ----- runners -----


def run_macro(
    outcomes_by_report: list[list[int]],
    *,
    lr: float,
    beta: float,
    c: float,
) -> Series:
    """Run the macro simulation."""
    glob = GlobalState()
    param = ParamState(beta=beta, c=c)

    # Start at t=0 for parity with Adam
    t: list[int] = [0]
    if param.beta == 0.0:
        x0 = param.z
    else:
        x0 = reconstruct_x_prev(param.theta, param.z, param.beta)
    xs: list[float] = [x0]
    zs: list[float] = [param.z]
    ths: list[float] = [param.theta]

    for outs in outcomes_by_report:
        upd = macro_update_sgd(glob, param, outcomes=outs, lr=lr)
        param.z, param.theta = upd.z, upd.theta
        t.append(glob.iter_pairs)
        xs.append(upd.x)
        zs.append(upd.z)
        ths.append(upd.theta)
    return Series(t_pairs=t, x=xs, z=zs, theta=ths)


def run_micro(
    seqs_by_report: list[list[float]],
    *,
    lr: float,
    beta: float,
    c: float,
) -> Series:
    """Run the micro simulation."""
    glob = GlobalState()
    param = ParamState(beta=beta, c=c)

    # Start at t=0 for parity with Adam
    t: list[int] = [0]
    if param.beta == 0.0:
        x0 = param.z
    else:
        x0 = reconstruct_x_prev(param.theta, param.z, param.beta)
    xs: list[float] = [x0]
    zs: list[float] = [param.z]
    ths: list[float] = [param.theta]

    for seq_num in seqs_by_report:
        upd = micro_apply_sequence(glob, param, seq_num=seq_num, lr=lr)
        param.z, param.theta = upd.z, upd.theta
        n_block = len(seq_num)
        glob.iter_pairs += n_block
        glob.sf_weight_sum += lr * n_block
        t.append(glob.iter_pairs)
        xs.append(upd.x)
        zs.append(upd.z)
        ths.append(upd.theta)
    return Series(t_pairs=t, x=xs, z=zs, theta=ths)


# ----- main -----


def main() -> None:
    """Run the main simulation."""
    # hyper
    lr: float = 0.1
    beta: float = 0.9
    c: float = 0.5

    # schedule
    base_seed: int = 424242
    num_reports: int = 100
    n_min, n_max = 1, 32
    p5: tuple[float, float, float, float, float] = (
        0.025,
        0.20,
        0.55,
        0.20,
        0.025,
    )

    # Discard Ns — derive N from the sequences (outcomes only)
    _, outcomes_by_report = make_schedule(
        num_reports,
        n_min,
        n_max,
        p5,
        base_seed,
    )

    # original order
    macro = run_macro(outcomes_by_report, lr=lr, beta=beta, c=c)
    seqs_mean = [build_sequence(outs, "const_mean") for outs in outcomes_by_report]
    seqs_real = [build_sequence(outs, "outcomes") for outs in outcomes_by_report]
    micro_mean = run_micro(seqs_mean, lr=lr, beta=beta, c=c)
    micro_real = run_micro(seqs_real, lr=lr, beta=beta, c=c)

    # sanity: macro == micro_mean exactly (by construction)
    assert macro.t_pairs == micro_mean.t_pairs == micro_real.t_pairs, (  # noqa: S101
        "time axes differ"
    )
    assert series_allclose(macro.x, micro_mean.x), (  # noqa: S101
        "macro x != micro const-mean x"
    )
    assert series_allclose(macro.z, micro_mean.z), (  # noqa: S101
        "macro z != micro const-mean z"
    )
    assert series_allclose(macro.theta, micro_mean.theta), (  # noqa: S101
        "macro theta != micro const-mean theta"
    )

    # Figure 1: only the original schedule (parity with Adam)
    fig1, axs1 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    plot_many(
        axs1[0],
        # micro real (ground truth)
        Line(
            micro_real.t_pairs,
            micro_real.x,
            "x — micro real",
            linestyle="-.",
        ),
        # micro mean (theoretical bridge)
        Line(
            micro_mean.t_pairs,
            micro_mean.x,
            "x — micro mean",
            linestyle="--",
        ),
        # macro (production path)
        Line(macro.t_pairs, macro.x, "x — macro"),
        y_label="x",
    )
    plot_many(
        axs1[1],
        Line(
            micro_real.t_pairs,
            micro_real.z,
            "z — micro real",
            linestyle="-.",
        ),
        Line(
            micro_mean.t_pairs,
            micro_mean.z,
            "z — micro mean",
            linestyle="--",
        ),
        Line(macro.t_pairs, macro.z, "z — macro"),
        y_label="z",
    )
    plot_many(
        axs1[2],
        Line(
            micro_real.t_pairs,
            micro_real.theta,
            "theta — micro real",
            linestyle="-.",
        ),
        Line(
            micro_mean.t_pairs,
            micro_mean.theta,
            "theta — micro mean",
            linestyle="--",
        ),
        Line(macro.t_pairs, macro.theta, "theta — macro"),
        y_label="theta",
    )
    axs1[-1].set_xlabel("pairs")
    fig1.suptitle("Schedule-free SGD — single schedule (x, z, theta)", y=0.98)
    plt.tight_layout()
    plt.show()

    # custom shuffled order: single backward sweep with p per adjacent swap
    p_swap = 4.0 / 5.0
    idx = end_adjacent_shuffle(
        list(range(num_reports)),
        p=p_swap,
        rng=random.Random(base_seed + 1337),  # noqa: S311
    )
    outcomes_by_report_shuf = [outcomes_by_report[i] for i in idx]

    macro2 = run_macro(outcomes_by_report_shuf, lr=lr, beta=beta, c=c)
    seqs_mean_shuf = [
        build_sequence(outs, "const_mean") for outs in outcomes_by_report_shuf
    ]
    seqs_real_shuf = [
        build_sequence(outs, "outcomes") for outs in outcomes_by_report_shuf
    ]
    micro_mean2 = run_micro(seqs_mean_shuf, lr=lr, beta=beta, c=c)
    micro_real2 = run_micro(seqs_real_shuf, lr=lr, beta=beta, c=c)

    assert macro2.t_pairs == micro_mean2.t_pairs == micro_real2.t_pairs, (  # noqa: S101
        "time axes differ (shuffled)"
    )

    # Figure 2: original vs shuffled overlay (parity with Adam)
    fig2, axs2 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    plot_many(
        axs2[0],
        # x: micro real, micro mean, macro — orig vs shuf
        Line(
            micro_real.t_pairs,
            micro_real.x,
            "x — micro real (orig)",
            linestyle="-.",
        ),
        Line(
            micro_real2.t_pairs,
            micro_real2.x,
            "x — micro real (shuf)",
            linestyle="-.",
            linewidth=1.5,
            alpha=0.6,
        ),
        Line(
            micro_mean.t_pairs,
            micro_mean.x,
            "x — micro mean (orig)",
            linestyle="--",
        ),
        Line(
            micro_mean2.t_pairs,
            micro_mean2.x,
            "x — micro mean (shuf)",
            linestyle="--",
            linewidth=1.5,
            alpha=0.6,
        ),
        Line(macro.t_pairs, macro.x, "x — macro (orig)"),
        Line(
            macro2.t_pairs,
            macro2.x,
            "x — macro (shuf)",
            linewidth=1.5,
            alpha=0.6,
        ),
        y_label="x",
    )
    plot_many(
        axs2[1],
        Line(
            micro_real.t_pairs,
            micro_real.z,
            "z — micro real (orig)",
            linestyle="-.",
        ),
        Line(
            micro_real2.t_pairs,
            micro_real2.z,
            "z — micro real (shuf)",
            linestyle="-.",
            linewidth=1.5,
            alpha=0.6,
        ),
        Line(
            micro_mean.t_pairs,
            micro_mean.z,
            "z — micro mean (orig)",
            linestyle="--",
        ),
        Line(
            micro_mean2.t_pairs,
            micro_mean2.z,
            "z — micro mean (shuf)",
            linestyle="--",
            linewidth=1.5,
            alpha=0.6,
        ),
        Line(macro.t_pairs, macro.z, "z — macro (orig)"),
        Line(
            macro2.t_pairs,
            macro2.z,
            "z — macro (shuf)",
            linewidth=1.5,
            alpha=0.6,
        ),
        y_label="z",
    )
    plot_many(
        axs2[2],
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
        Line(macro.t_pairs, macro.theta, "theta — macro (orig)"),
        Line(
            macro2.t_pairs,
            macro2.theta,
            "theta — macro (shuf)",
            linewidth=1.5,
            alpha=0.6,
        ),
        y_label="theta",
    )
    axs2[-1].set_xlabel("pairs")
    fig2.suptitle(
        "Schedule-free SGD — original vs end-adjacent shuffled (x, z, theta)",
        y=0.98,
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
