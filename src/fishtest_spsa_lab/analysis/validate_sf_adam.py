"""Clean-room simulation for schedule-free Adam vs micro loop, mirroring SGD structure.

States:
- z: fast iterate (unclamped update state in θ-space)
- x: Polyak surrogate (slow moving average of z via schedule-free mass)
- theta: blended state, theta = (1 - beta1) * z + beta1 * x  (the exported value)

Three paths over a shared schedule (Ns and outcomes per report):
- Fishtest (macro): per-report closed-form v with online μ2; mass blend x with a_k
- Micro loop (const_mean_online): N equal micro-steps; per-step v uses online μ2;
  per-step mass blend
- Real micro step (outcomes): N per-outcome steps; per-step v, per-step mass blend

Start from z=x=theta=0, v=0, iter_pairs=0, sf_weight_sum=0.
Plot x, z, theta vs cumulative pairs for original vs end-adjacent shuffled order.
"""

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt

from .common import (
    Line,
    compute_pentanomial_moments,
    end_adjacent_shuffle,
    make_schedule,
    plot_many,
)

# ----- data models -----


@dataclass(slots=True)
class GlobalState:
    """Global state of the simulation."""

    iter_pairs: int = 0
    sf_weight_sum: float = 0.0
    # Online μ2 estimator state (from report-level summaries only)
    # Use exact block-averaged aggregates like
    # src/fishtest_spsa_lab/analysis/validate_variance.py
    # (OnlineReportStats) to match the macro logic.
    reports: float = 0.0
    sum_n: float = 0.0
    sum_s: float = 0.0
    sum_s2_over_n: float = 0.0
    mu2_init: float = 1.0  # used only before the first report


@dataclass(slots=True)
class ParamState:
    """Parameter state for Adam."""

    theta: float = 0.0
    z: float = 0.0
    v: float = 0.0
    c: float = 0.5
    beta1: float = 0.9


@dataclass(slots=True)
class Update:
    """Result of an update step."""

    x: float
    z: float
    theta: float
    v: float


@dataclass(slots=True)
class Series:
    """Time series data for plotting."""

    t_pairs: list[int]
    x: list[float]
    z: list[float]
    theta: list[float]


@dataclass(slots=True)
class InitStats:
    """Precomputed initialization aggregates (virtual prior), passed into functions."""

    reports: float = 0.0
    sum_n: float = 0.0
    sum_s: float = 0.0
    sum_s2_over_n: float = 0.0


# ----- core math -----


def reconstruct_x_prev(theta_prev: float, z_prev: float, beta1: float) -> float:
    """Reconstruct x_prev from theta_prev and z_prev."""
    # If beta1 == 0, we won't call this; x=z is used directly.
    return (theta_prev - (1.0 - beta1) * z_prev) / beta1


def sf_weighting_update(glob: GlobalState, n: int, lr: float) -> float:
    """Update schedule-free weighting."""
    # schedule-free mass increment
    report_weight = lr * n
    glob.sf_weight_sum += report_weight
    return report_weight / glob.sf_weight_sum if glob.sf_weight_sum > 0 else 1.0


def adam_k(n: int, beta2: float) -> float:
    """Intra-block geometric mean adjustment for Adam's denominator."""
    if not (n > 1 and 0.0 < beta2 < 1.0):
        return 1.0
    q = math.sqrt(beta2)
    tiny = 1e-12
    if abs(1.0 - q) > tiny:
        k = (1.0 - (beta2 ** (0.5 * n))) / (n * (1.0 - q))
    else:
        k = 1.0 - ((n - 1) * 0.25) * (1.0 - beta2)
    return max(min(k, 1.0), 1e-12)


def adam_v_closed_form(  # noqa: PLR0913
    v_prev: float,
    beta2: float,
    n: int,
    g_sq_mean: float,
    micro_steps_after: int,
    eps: float,
) -> tuple[float, float]:
    """Compute closed form v update over n steps with constant mean g^2."""
    if beta2 < 1.0:
        v_new = (beta2**n) * v_prev + (1.0 - beta2**n) * g_sq_mean
        bc = 1.0 - (beta2**micro_steps_after)
        v_hat = v_new / bc if bc > 1e-16 else v_new  # noqa: PLR2004
    else:
        v_new = v_prev
        v_hat = v_new
    denom = math.sqrt(v_hat) + eps
    return v_new, denom


# ----- online μ2 estimation (report-level only: uses N and sum s) -----


def _mu_hat(glob: GlobalState) -> float:
    # Block-average mean per pair: μ̂ = (Σ s_i) / (Σ N_i)
    return (glob.sum_s / glob.sum_n) if glob.sum_n > 0.0 else 0.0


def mu2_hat(glob: GlobalState) -> float:
    """Exact block-averaged estimator, using only (N, s) per report.

    E_blocks[s^2 / N] = σ^2 + μ^2 E_blocks[N]
    ⇒ σ̂^2 = E_s2_over_N - μ̂^2 E_N
    ⇒ μ̂2  = μ̂^2 + σ̂^2
    """  # noqa: RUF002
    # Before any reports, use the configured initial guess
    if glob.reports <= 0.0:
        return glob.mu2_init
    mu = _mu_hat(glob)
    e_s2_over_n = glob.sum_s2_over_n / glob.reports
    e_n = glob.sum_n / glob.reports
    sigma2 = e_s2_over_n - (mu * mu) * e_n
    sigma2 = max(sigma2, 0.0)  # numerical guard
    mu2 = mu * mu + sigma2
    # clamp to plausible range for outcomes in [-2..2]
    return min(max(mu2, 1e-12), 4.0)


def update_mu2_stats(glob: GlobalState, n: int, s: float) -> None:
    """Update estimator AFTER using it for the current report."""
    glob.reports += 1.0
    glob.sum_n += float(n)
    glob.sum_s += float(s)
    glob.sum_s2_over_n += (float(s) * float(s)) / max(float(n), 1.0)


# ----- macro + micro -----


def macro_update(  # noqa: PLR0913
    glob: GlobalState,
    param: ParamState,
    *,
    n: int,
    result: float,
    lr: float,
    beta2: float,
    eps: float,
    use_k: bool = True,
) -> Update:
    """Single-report (macro) update that only depends on the block summary.

    - n: number of pairs in the report
    - result: sum of outcomes over the block
    Uses online μ2 estimated from previous reports (no per-outcome squares).
    """
    # advance time/mass
    glob.iter_pairs += n
    a_k = sf_weighting_update(glob, n, lr)

    # v via closed form: online μ2 with exact block-averaged estimator
    # (prior to this block)
    g_sq_mean = mu2_hat(glob)
    v_new, denom_end = adam_v_closed_form(
        param.v,
        beta2,
        n,
        g_sq_mean,
        glob.iter_pairs,
        eps,
    )

    # fast iterate
    step_phi = (lr * result) / denom_end if denom_end > 0.0 else 0.0
    if use_k:
        step_phi *= adam_k(n, beta2)
    z_new = param.z + step_phi * param.c

    # surrogate
    if param.beta1 == 0.0:
        x_new = z_new
    else:
        x_prev = reconstruct_x_prev(param.theta, param.z, param.beta1)
        x_new = (1.0 - a_k) * x_prev + a_k * z_new

    theta_new = (1.0 - param.beta1) * z_new + param.beta1 * x_new
    return Update(x=x_new, z=z_new, theta=theta_new, v=v_new)


def micro_apply_sequence(  # noqa: PLR0913
    glob0: GlobalState,
    param0: ParamState,
    *,
    seq_num: Sequence[float],
    seq_gsq: Sequence[float],
    lr: float,
    beta2: float,
    eps: float,
) -> Update:
    """Apply a sequence of micro-steps."""
    # local copies for per-step evolution
    glob = GlobalState(glob0.iter_pairs, glob0.sf_weight_sum)
    z = param0.z
    v = param0.v
    x = z if param0.beta1 == 0.0 else reconstruct_x_prev(param0.theta, z, param0.beta1)

    for num, g_sq in zip(seq_num, seq_gsq, strict=True):
        glob.iter_pairs += 1
        a_k = sf_weighting_update(glob, 1, lr)
        if beta2 < 1.0:
            v = beta2 * v + (1.0 - beta2) * g_sq
            bc = 1.0 - (beta2**glob.iter_pairs)
            v_hat = v / bc if bc > 1e-16 else v  # noqa: PLR2004
        else:
            v_hat = v
        denom = math.sqrt(v_hat) + eps
        z = z + ((lr * num) / denom) * param0.c
        if param0.beta1 != 0.0:
            x = (1.0 - a_k) * x + a_k * z

    theta = (1.0 - param0.beta1) * z + param0.beta1 * x
    return Update(x=x, z=z, theta=theta, v=v)


# ----- schedule + sequences -----


def build_sequence(
    outcomes: Sequence[int],
    kind: str,
) -> tuple[list[float], list[float]]:
    """Build a sequence of outcomes and squared outcomes."""
    n = len(outcomes)
    if n == 0:
        return [], []
    if kind == "outcomes":
        # per-outcome numerators and per-outcome squares (for the real micro path only)
        out_sq = [float(o * o) for o in outcomes]
        return [float(o) for o in outcomes], out_sq
    msg = "kind must be 'outcomes'"
    raise ValueError(msg)


def build_const_mean_online_sequences(
    outcomes_by_report: list[list[int]],
    mu2_init: float,
    init_stats: InitStats | None = None,
) -> list[tuple[list[float], list[float]]]:
    """Build per-report constant-mean sequences using exact block-averaged estimator.

    Matches OnlineReportStats in src/fishtest_spsa_lab/analysis/validate_variance.py.
    Uses pre-block μ2_hat and updates after.
    Seeds with externally computed InitStats (virtual prior).
    """
    seqs: list[tuple[list[float], list[float]]] = []
    # Seed local aggregates from InitStats
    reports: float = init_stats.reports if init_stats else 0.0
    sum_n: float = init_stats.sum_n if init_stats else 0.0
    sum_s: float = init_stats.sum_s if init_stats else 0.0
    sum_s2_over_n: float = init_stats.sum_s2_over_n if init_stats else 0.0

    def _mu_hat_local() -> float:
        return (sum_s / sum_n) if sum_n > 0.0 else 0.0

    def _mu2_hat_local() -> float:
        if reports <= 0.0:
            return mu2_init
        mu = _mu_hat_local()
        e_s2_over_n = sum_s2_over_n / reports
        e_n = sum_n / reports
        sigma2 = e_s2_over_n - (mu * mu) * e_n
        sigma2 = max(sigma2, 0.0)
        mu2 = mu * mu + sigma2
        return min(max(mu2, 1e-12), 4.0)

    for outs in outcomes_by_report:
        n = len(outs)
        s = float(sum(outs))
        mean = s / n if n > 0 else 0.0
        g2 = _mu2_hat_local()
        seqs.append(([mean] * n, [g2] * n))
        # update stats after using them for this block
        reports += 1.0
        sum_n += float(n)
        sum_s += float(s)
        sum_s2_over_n += (float(s) * float(s)) / max(float(n), 1.0)
    return seqs


# ----- runners -----


def run_macro(  # noqa: PLR0913
    outcomes_by_report: list[list[int]],
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    c: float,
    mu2_init: float,
    init_stats: InitStats | None = None,
) -> Series:
    """Run the macro simulation."""
    # Set up global state and seed with externally computed InitStats
    glob = GlobalState(mu2_init=mu2_init)
    if init_stats:
        glob.reports = float(init_stats.reports)
        glob.sum_n = float(init_stats.sum_n)
        glob.sum_s = float(init_stats.sum_s)
        glob.sum_s2_over_n = float(init_stats.sum_s2_over_n)

    param = ParamState(beta1=beta1, c=c)

    t: list[int] = [0]
    if param.beta1 == 0.0:
        x0 = param.z
    else:
        x0 = reconstruct_x_prev(param.theta, param.z, param.beta1)
    xs: list[float] = [x0]
    zs: list[float] = [param.z]
    ths: list[float] = [param.theta]

    for outs in outcomes_by_report:
        n_block = len(outs)
        result = float(sum(outs))
        upd = macro_update(
            glob,
            param,
            n=n_block,
            result=result,
            lr=lr,
            beta2=beta2,
            eps=eps,
            use_k=True,
        )
        # After using current online μ2, update stats with this block
        update_mu2_stats(glob, n_block, result)

        param.z, param.theta, param.v = upd.z, upd.theta, upd.v
        t.append(glob.iter_pairs)
        xs.append(upd.x)
        zs.append(upd.z)
        ths.append(upd.theta)
    return Series(t_pairs=t, x=xs, z=zs, theta=ths)


def run_micro(  # noqa: PLR0913
    seqs_by_report: list[tuple[list[float], list[float]]],
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    c: float,
) -> Series:
    """Run the micro simulation."""
    glob = GlobalState()
    param = ParamState(beta1=beta1, c=c)

    t: list[int] = [0]
    if param.beta1 == 0.0:
        x0 = param.z
    else:
        x0 = reconstruct_x_prev(param.theta, param.z, param.beta1)
    xs: list[float] = [x0]
    zs: list[float] = [param.z]
    ths: list[float] = [param.theta]

    for seq_num, seq_gsq in seqs_by_report:
        # guard against accidental mismatch
        if len(seq_num) != len(seq_gsq):
            msg = "seq_num and seq_gsq length mismatch"
            raise ValueError(msg)
        upd = micro_apply_sequence(
            glob,
            param,
            seq_num=seq_num,
            seq_gsq=seq_gsq,
            lr=lr,
            beta2=beta2,
            eps=eps,
        )
        param.z, param.theta, param.v = upd.z, upd.theta, upd.v
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
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    c: float = 0.5

    # schedule (mirror SGD)
    base_seed: int = 424242
    num_reports: int = 100
    n_min, n_max = 1, 32

    # Generator pentanomial (used to draw outcomes)
    p5: tuple[float, float, float, float, float] = (
        0.025,
        0.20,
        0.55,
        0.20,
        0.025,
    )

    # Initial guess for μ2 before any data arrives (only used if no init_stats
    # and no data yet)
    mu2_init: float = 1.0

    # Optional: compute initialization stats ONCE externally (from a prior you choose)
    # Example uses a symmetric draw-heavy prior; tweak or set prior_reports=0
    # to disable.
    prior_p5: tuple[float, float, float, float, float] = (
        0.05,
        0.20,
        0.50,
        0.20,
        0.05,
    )
    prior_reports: float = 5.0  # 0.0 disables warm start
    prior_mean_n: float = (n_min + n_max) / 2.0

    # Compute InitStats externally; only aggregates are passed below.
    def compute_init_stats_from_prior(
        p5_: tuple[float, float, float, float, float],
        reports_: float,
        mean_n_: float,
    ) -> InitStats:
        if reports_ <= 0.0 or mean_n_ <= 0.0:
            return InitStats()
        mu_p, _mu2_p, var_p = compute_pentanomial_moments(p5_)
        return InitStats(
            reports=reports_,
            sum_n=reports_ * mean_n_,
            sum_s=reports_ * mean_n_ * mu_p,
            sum_s2_over_n=reports_ * (var_p + mean_n_ * (mu_p * mu_p)),
        )

    init_stats = compute_init_stats_from_prior(
        prior_p5,
        prior_reports,
        prior_mean_n,
    )

    # Derive schedule
    _, outcomes_by_report = make_schedule(
        num_reports,
        n_min,
        n_max,
        p5,
        base_seed,
    )

    # original order
    macro = run_macro(
        outcomes_by_report,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        c=c,
        mu2_init=mu2_init,
        init_stats=init_stats,
    )
    # Build micro mean sequences with the same online μ2 logic, seeded with the
    # same init_stats
    seqs_mean = build_const_mean_online_sequences(
        outcomes_by_report,
        mu2_init,
        init_stats=init_stats,
    )
    seqs_real = [build_sequence(outs, "outcomes") for outs in outcomes_by_report]
    micro_mean = run_micro(
        seqs_mean,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        c=c,
    )
    micro_real = run_micro(
        seqs_real,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        c=c,
    )

    # Figure 1: only the original schedule
    fig1, axs1 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    plot_many(
        axs1[0],
        # x: micro real, micro mean, macro
        Line(
            micro_real.t_pairs,
            micro_real.x,
            "x — micro real",
            linestyle="-.",
        ),
        Line(
            micro_mean.t_pairs,
            micro_mean.x,
            "x — micro mean",
            linestyle="--",
        ),
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
    fig1.suptitle(
        "Schedule-free Adam — single schedule (x, z, theta)",
        y=0.98,
    )
    plt.tight_layout()
    plt.show()

    # Figure 2: original vs shuffled overlay
    p_swap = 4.0 / 5.0
    idx = end_adjacent_shuffle(
        list(range(num_reports)),
        p=p_swap,
        rng=random.Random(base_seed + 1337),  # noqa: S311
    )
    outcomes_by_report_shuf = [outcomes_by_report[i] for i in idx]

    macro2 = run_macro(
        outcomes_by_report_shuf,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        c=c,
        mu2_init=mu2_init,
        init_stats=init_stats,
    )
    seqs_mean_shuf = build_const_mean_online_sequences(
        outcomes_by_report_shuf,
        mu2_init,
        init_stats=init_stats,
    )
    seqs_real_shuf = [
        build_sequence(outs, "outcomes") for outs in outcomes_by_report_shuf
    ]
    micro_mean2 = run_micro(
        seqs_mean_shuf,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        c=c,
    )
    micro_real2 = run_micro(
        seqs_real_shuf,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        c=c,
    )

    assert macro2.t_pairs == micro_mean2.t_pairs == micro_real2.t_pairs, (  # noqa: S101
        "time axes differ (shuffled)"
    )

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
        "Schedule-free Adam — original vs end-adjacent shuffled (x, z, theta)",
        y=0.98,
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
