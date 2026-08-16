"""Clean-room simulation for Adam vs micro loop.

States (Adam, θ-space):
- theta: parameter (1D) updated by Adam.
- m: first-moment EMA of gradients.
- v: second-moment EMA of squared gradients.

Three paths over a shared schedule (Ns and outcomes per report):
- Micro real: one textbook Adam step per outcome (g_j = outcome_j).
- Micro const-mean: N steps per report with g = s/N inside the block.
- Macro const-mean: one block-level function that sees only (N, s) but
  internally replays those N classic-Adam steps with g = s/N, so it
  matches the micro const-mean path exactly.

Start from theta=m=v=0, iterate over the same synthetic sequence of
pentanomial outcomes as the other validators, and plot theta vs cumulative
pairs for original and shuffled schedules.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import matplotlib.pyplot as plt

from .common import (
    Line,
    build_sequence,
    end_adjacent_shuffle,
    make_schedule,
    max_abs_gap,
    plot_many,
    series_scale,
)
from .gate import Gate, show

#: The macro const-mean path replays the same N textbook steps the micro path
#: takes, so that equality is exact and is asserted at float tolerance.
TOLERANCE: float = 1e-12

#: The block path is an approximation: it freezes the denominator at v_N for all
#: N micro-steps. Measured gap 8.12e-02 on a theta scale of 3.1415. The bound
#: rejects the uncorrected closed form this file shipped until now, whose gap was
#: 1.74e+00 -- 21x worse, and 4.29x on the very first block.
BLOCK_GAP_BOUND: float = 1.5e-1

# ----- data models -----


@dataclass(slots=True)
class AdamState:
    """Parameter state for Adam (1D)."""

    theta: float = 0.0
    m: float = 0.0
    v: float = 0.0


@dataclass(slots=True)
class Series:
    """Time series data for plotting."""

    t_pairs: list[int]
    theta: list[float]


# ----- core math -----


def adam_step(  # noqa: PLR0913
    state: AdamState,
    *,
    grad: float,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    t: int,
) -> AdamState:
    """Single Adam update for a scalar parameter.

    This is the textbook Adam update (no weight decay):

    m_{t+1} = β1 * m_t + (1 - β1) * g_t
    v_{t+1} = β2 * v_t + (1 - β2) * g_t^2
    m̂_{t+1} = m_{t+1} / (1 - β1^{t+1})
    v̂_{t+1} = v_{t+1} / (1 - β2^{t+1})
    θ_{t+1} = θ_t - η * m̂_{t+1} / (sqrt(v̂_{t+1}) + ε)
    """
    m_new = beta1 * state.m + (1.0 - beta1) * grad
    v_new = beta2 * state.v + (1.0 - beta2) * (grad * grad)

    # Bias correction uses step index t+1
    t1 = t + 1
    m_hat = m_new / (1.0 - beta1**t1) if beta1 < 1.0 else m_new
    v_hat = v_new / (1.0 - beta2**t1) if beta2 < 1.0 else v_new

    denom = math.sqrt(v_hat) + eps
    step = m_hat / denom if denom > 0.0 else 0.0
    theta_new = state.theta - lr * step
    return AdamState(theta=theta_new, m=m_new, v=v_new)


def adam_multi_step_const_grad(  # noqa: PLR0913
    state: AdamState,
    *,
    grad: float,
    n: int,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    t_start: int,
) -> tuple[AdamState, int]:
    """Apply N classic-Adam steps with constant gradient g, return (state, t)."""
    s = state
    t = t_start
    for _ in range(n):
        s = adam_step(
            s,
            grad=grad,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            t=t,
        )
        t += 1
    return s, t


def adam_block_update(  # noqa: PLR0913
    state: AdamState,
    *,
    grad: float,
    n: int,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    t_start: int,
) -> tuple[AdamState, int]:
    """Approximate N-step classic Adam with constant grad using a block update.

    Mirrors ``simulator/optimizer.py :: AdamBlock.step``. m_N and v_N are updated
    in closed form for constant g; the theta step sums the exact m_i ladder and
    freezes the denominator at v_N, and both moments are bias-corrected at the
    end-of-block step index.

    Returns the new state and the advanced step index.
    """
    if n <= 0:
        return state, t_start

    theta0 = state.theta
    m0 = state.m
    v0 = state.v
    g = grad

    beta1_n = beta1**n
    beta2_n = beta2**n

    m_n = beta1_n * m0 + (1.0 - beta1_n) * g
    v_n = beta2_n * v0 + (1.0 - beta2_n) * (g * g)

    s_beta1 = beta1 * (1.0 - beta1_n) / (1.0 - beta1) if beta1 != 1.0 else float(n)

    sum_m = m0 * s_beta1 + g * (n - s_beta1)

    # Bias correction at the end-of-block index. Its absence, not the frozen
    # denominator, was the dominant error: 1bea8b1 measured the uncorrected form
    # overshooting N textbook micro-steps by 4.0x on the first block. That commit
    # fixed simulator/optimizer.py and left this file on the old rule, so the
    # gate for AdamBlock validated an update the lab no longer ships.
    t_end = t_start + n
    t1 = float(t_end)
    bc1 = (1.0 - beta1**t1) if 0.0 <= beta1 < 1.0 else 1.0
    bc2 = (1.0 - beta2**t1) if 0.0 <= beta2 < 1.0 else 1.0

    denom = math.sqrt(v_n / bc2) + eps
    step = (sum_m / bc1) / denom if denom > 0.0 else 0.0
    theta_n = theta0 - lr * step

    return AdamState(theta=theta_n, m=m_n, v=v_n), t_end


# ----- runners -----


def run_micro_real_adam(
    outcomes_by_report: list[list[int]],
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
) -> Series:
    """Per-outcome Adam: one step per actual outcome (micro real)."""
    state = AdamState()
    t_pairs: list[int] = [0]
    thetas: list[float] = [state.theta]

    step_idx = 0
    total_pairs = 0
    for outs in outcomes_by_report:
        seq = build_sequence(outs, "outcomes")
        for g in seq:
            state = adam_step(
                state,
                grad=g,
                lr=lr,
                beta1=beta1,
                beta2=beta2,
                eps=eps,
                t=step_idx,
            )
            step_idx += 1
        total_pairs += len(seq)
        t_pairs.append(total_pairs)
        thetas.append(state.theta)

    return Series(t_pairs=t_pairs, theta=thetas)


def run_micro_const_mean_adam(
    outcomes_by_report: list[list[int]],
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
) -> Series:
    """Micro path with constant-mean gradients inside each report.

    For report i with N_i outcomes and sum s_i, we take N_i steps, each
    with grad = s_i / N_i. This is the “micro const-mean” reference path.
    """
    state = AdamState()
    t_pairs: list[int] = [0]
    thetas: list[float] = [state.theta]

    step_idx = 0
    total_pairs = 0
    for outs in outcomes_by_report:
        n = len(outs)
        if n == 0:
            continue
        s = float(sum(outs))
        g = s / n
        for _ in range(n):
            state = adam_step(
                state,
                grad=g,
                lr=lr,
                beta1=beta1,
                beta2=beta2,
                eps=eps,
                t=step_idx,
            )
            step_idx += 1
        total_pairs += n
        t_pairs.append(total_pairs)
        thetas.append(state.theta)

    return Series(t_pairs=t_pairs, theta=thetas)


def run_macro_const_mean_adam(
    outcomes_by_report: list[list[int]],
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
) -> Series:
    """Macro path that exactly matches the micro const-mean path.

    For each report: N_i, s_i -> grad = s_i / N_i, then internally take
    N_i classic-Adam steps with that constant grad, using the same global
    step index t as run_micro_const_mean_adam.
    """
    state = AdamState()
    t_pairs: list[int] = [0]
    thetas: list[float] = [state.theta]

    step_idx = 0
    total_pairs = 0
    for outs in outcomes_by_report:
        n = len(outs)
        if n == 0:
            continue
        s = float(sum(outs))
        g = s / n

        state, step_idx = adam_multi_step_const_grad(
            state,
            grad=g,
            n=n,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            t_start=step_idx,
        )
        total_pairs += n
        t_pairs.append(total_pairs)
        thetas.append(state.theta)

    return Series(t_pairs=t_pairs, theta=thetas)


def run_macro_block_adam(
    outcomes_by_report: list[list[int]],
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
) -> Series:
    """Macro path using the block-Adam approximation (new optimizer).

    One update per report, using only (N_i, s_i) inside each block.
    """
    state = AdamState()
    t_pairs: list[int] = [0]
    thetas: list[float] = [state.theta]

    step_idx = 0
    total_pairs = 0
    for outs in outcomes_by_report:
        n = len(outs)
        if n == 0:
            continue
        s = float(sum(outs))
        g = s / n

        state, step_idx = adam_block_update(
            state,
            grad=g,
            n=n,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            t_start=step_idx,
        )
        total_pairs += n
        t_pairs.append(total_pairs)
        thetas.append(state.theta)

    return Series(t_pairs=t_pairs, theta=thetas)


# ----- main -----


def main() -> int:
    """Run the Adam macro-vs-micro validation and return an exit code."""
    gate = Gate(
        "validate-adam",
        "Adam: macro const-mean == micro const-mean; block macro bounded",
    )

    # hyperparameters (textbook defaults, tweak as desired)
    lr: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    # schedule (mirror other analysis scripts)
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

    # Derive schedule; discard Ns since we take len(outcomes) per report
    _, outcomes_by_report = make_schedule(
        num_reports,
        n_min,
        n_max,
        p5,
        base_seed,
    )

    # Original order
    micro_real = run_micro_real_adam(
        outcomes_by_report,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
    )
    micro_mean = run_micro_const_mean_adam(
        outcomes_by_report,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
    )
    macro_const = run_macro_const_mean_adam(
        outcomes_by_report,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
    )
    macro_block = run_macro_block_adam(
        outcomes_by_report,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
    )

    gate.note("reports", num_reports)
    gate.note("pairs per report", f"{n_min}..{n_max}")
    gate.note("total pairs", micro_mean.t_pairs[-1])
    gate.note("base seed", base_seed)
    gate.note("lr / beta1 / beta2", f"{lr:g} / {beta1:g} / {beta2:g}")
    gate.note("theta scale", series_scale(micro_mean.theta))
    gate.note(
        "micro real vs micro const-mean",
        max_abs_gap(micro_real.theta, micro_mean.theta),
    )

    gate.check_le(
        "time axes agree (original)",
        0.0
        if micro_mean.t_pairs == macro_const.t_pairs == macro_block.t_pairs
        else 1.0,
        0.0,
    )
    gate.check_le(
        "macro const-mean == micro const-mean",
        max_abs_gap(macro_const.theta, micro_mean.theta),
        TOLERANCE,
    )
    gate.check_le(
        "block macro within bound of micro const-mean",
        max_abs_gap(macro_block.theta, micro_mean.theta),
        BLOCK_GAP_BOUND,
    )

    # Figure 1: original schedule
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6), sharex=True)
    plot_many(
        ax1,
        Line(micro_real.t_pairs, micro_real.theta, "theta — micro real"),
        Line(
            micro_mean.t_pairs,
            micro_mean.theta,
            "theta — micro const-mean",
            linestyle="--",
        ),
        Line(
            macro_const.t_pairs,
            macro_const.theta,
            "theta — macro const-mean",
            linestyle=":",
        ),
        Line(
            macro_block.t_pairs,
            macro_block.theta,
            "theta — macro block-Adam",
            linestyle="-.",
        ),
        y_label="theta",
    )
    ax1.set_xlabel("pairs")
    fig1.suptitle("Adam — single schedule (theta)", y=0.98)
    plt.tight_layout()
    show(fig1)

    # Shuffled order (same end-adjacent scheme as other scripts)
    p_swap = 4.0 / 5.0
    idx = end_adjacent_shuffle(
        list(range(num_reports)),
        p=p_swap,
        rng=random.Random(base_seed + 1337),  # noqa: S311
    )
    outcomes_by_report_shuf = [outcomes_by_report[i] for i in idx]

    micro_real2 = run_micro_real_adam(
        outcomes_by_report_shuf,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
    )
    micro_mean2 = run_micro_const_mean_adam(
        outcomes_by_report_shuf,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
    )
    macro_const2 = run_macro_const_mean_adam(
        outcomes_by_report_shuf,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
    )
    macro_block2 = run_macro_block_adam(
        outcomes_by_report_shuf,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
    )

    gate.check_le(
        "time axes agree (shuffled)",
        0.0
        if micro_mean2.t_pairs == macro_const2.t_pairs == macro_block2.t_pairs
        else 1.0,
        0.0,
    )
    gate.check_le(
        "macro const-mean == micro const-mean (shuffled)",
        max_abs_gap(macro_const2.theta, micro_mean2.theta),
        TOLERANCE,
    )
    gate.check_le(
        "block macro within bound of micro const-mean (shuffled)",
        max_abs_gap(macro_block2.theta, micro_mean2.theta),
        BLOCK_GAP_BOUND,
    )

    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6), sharex=True)
    plot_many(
        ax2,
        # original
        Line(micro_real.t_pairs, micro_real.theta, "theta — micro real (orig)"),
        Line(
            micro_mean.t_pairs,
            micro_mean.theta,
            "theta — micro const-mean (orig)",
            linestyle="--",
        ),
        Line(
            macro_const.t_pairs,
            macro_const.theta,
            "theta — macro const-mean (orig)",
            linestyle=":",
        ),
        Line(
            macro_block.t_pairs,
            macro_block.theta,
            "theta — macro block-Adam (orig)",
            linestyle="-.",
        ),
        # shuffled
        Line(
            micro_real2.t_pairs,
            micro_real2.theta,
            "theta — micro real (shuf)",
            linewidth=1.5,
            alpha=0.6,
        ),
        Line(
            micro_mean2.t_pairs,
            micro_mean2.theta,
            "theta — micro const-mean (shuf)",
            linestyle="--",
            linewidth=1.5,
            alpha=0.6,
        ),
        Line(
            macro_const2.t_pairs,
            macro_const2.theta,
            "theta — macro const-mean (shuf)",
            linestyle=":",
            linewidth=1.5,
            alpha=0.6,
        ),
        Line(
            macro_block2.t_pairs,
            macro_block2.theta,
            "theta — macro block-Adam (shuf)",
            linestyle="-.",
            linewidth=1.5,
            alpha=0.6,
        ),
        y_label="theta",
    )
    ax2.set_xlabel("pairs")
    fig2.suptitle(
        "Adam — original vs end-adjacent shuffled (theta)",
        y=0.98,
    )
    plt.tight_layout()
    show(fig2)

    return gate.report()


if __name__ == "__main__":
    raise SystemExit(main())
