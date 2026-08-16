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
    end_adjacent_shuffle,
    make_schedule,
    max_abs_gap,
    mu2_hat,
    plot_many,
    reconstruct_x_prev,
    series_scale,
    sf_weighting_update,
    update_mu2_stats,
)
from .gate import Gate, show
from .pentanomial import (
    InitStats,
    compute_init_stats_from_prior,
)

# Unlike SPSA and SF-SGD, schedule-free Adam is NOT an exact macro-vs-micro
# identity. The macro path applies one denominator, taken at the end of the
# block, to all N micro-steps; the micro path re-derives it each step. Adam's
# bias correction cancels that ramp exactly only while the second-moment level is
# constant, and the online mu2 estimate moves from block to block, so a residual
# remains. SPSA_macro_micro.md once called this equality "exact"; it is not, and
# these bounds are what it actually is.
#
# The bounds are calibrated over 12 independent schedules, not one. A first
# attempt set them from a single realization and they were promptly falsified the
# moment make_schedule's seeding was fixed: the theta gap moved 0.0504 -> 0.1174
# against a 0.10 bound. Measured over seeds 424242..424253:
#
#   beta2 = 0.999   z gap 0.0022 .. 0.0204     relative theta gap up to 0.474
#   beta2 = 0.9     z gap 0.0080 .. 0.0361     relative theta gap up to 0.477
#
# WHICH CHECK CATCHES WHAT. Reinstating the historical k(N, beta2) factor gives,
# over the same 12 schedules:
#
#   beta2 = 0.999   with-k z gap DOWN to 0.0060, i.e. it OVERLAPS the correct
#                   range and can look better than the correct rule. The default
#                   hyperparameters cannot detect this defect at all.
#   beta2 = 0.9     with-k z gap never below 0.3121, against a correct-rule
#                   maximum of 0.0361 -- 8.7x, disjoint.
#
# So BETA2_PROBE is not a nice-to-have; it is the only check here that rejects
# the defect this file used to contain. The beta2 = 0.999 bound is a
# coarse regression guard, and is deliberately not advertised as more.
Z_GAP_BOUND: float = 5.0e-2
Z_GAP_BOUND_LOW_BETA2: float = 1.0e-1

#: Bounded relative to the |theta| scale. The absolute theta gap varies 10x
#: across schedules (0.025 .. 0.219) because it is dominated by the Polyak
#: endpoint approximation, which is data-dependent; the ratio is the stable form.
THETA_REL_GAP_BOUND: float = 0.75

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


# ----- core math -----


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

    # fast iterate. No intra-block damping factor: the bias correction applied in
    # adam_v_closed_form already removes the in-block ramp that a geometric
    # k(N, beta2) term was meant to correct, so the exact factor is 1 for all N
    # and beta2. This mirrors simulator/optimizer.py :: SFAdamBlock.step; the
    # factor that used to sit here carried a dropped minus sign and was clipped
    # to (0, 1], which made the correct direction unreachable.
    step_phi = (lr * result) / denom_end if denom_end > 0.0 else 0.0
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
    # The estimator itself lives in simulator/moments.py; this was the third
    # copy of it, written as a closure over four locals rather than a state
    # object, which is why it drifted out of sight of the other two.
    state = GlobalState(mu2_init=mu2_init)
    if init_stats:
        state.reports = init_stats.reports
        state.sum_n = init_stats.sum_n
        state.sum_s = init_stats.sum_s
        state.sum_s2_over_n = init_stats.sum_s2_over_n

    for outs in outcomes_by_report:
        n = len(outs)
        s = float(sum(outs))
        mean = s / n if n > 0 else 0.0
        g2 = mu2_hat(state)
        seqs.append(([mean] * n, [g2] * n))
        # update stats after using them for this block
        update_mu2_stats(state, n, s)
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

#: A second beta2 at which the intra-block denominator is far more sensitive.
BETA2_PROBE: float = 0.9


def _axis_gap(*series: Series) -> float:
    """Return 0 when every series shares one time axis, 1 otherwise."""
    first = series[0].t_pairs
    return 0.0 if all(sr.t_pairs == first for sr in series[1:]) else 1.0


def main() -> int:
    """Run the SF-Adam macro-vs-micro validation and return an exit code."""
    gate = Gate(
        "validate-sf-adam-block",
        "schedule-free Adam: block macro vs const-mean micro (bounded, not exact)",
    )

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

    gate.note("reports", num_reports)
    gate.note("pairs per report", f"{n_min}..{n_max}")
    gate.note("total pairs", macro.t_pairs[-1])
    gate.note("base seed", base_seed)
    gate.note("lr / beta1 / beta2", f"{lr:g} / {beta1:g} / {beta2:g}")
    gate.note("z scale", series_scale(micro_mean.z))
    gate.note("theta gap (original)", max_abs_gap(macro.theta, micro_mean.theta))

    gate.check_le(
        "time axes agree (original)",
        _axis_gap(macro, micro_mean, micro_real),
        0.0,
    )
    gate.check_le(
        "macro z within bound of const-mean micro z",
        max_abs_gap(macro.z, micro_mean.z),
        Z_GAP_BOUND,
    )
    theta_scale = series_scale(micro_mean.theta)
    gate.check_le(
        "macro theta within relative bound of const-mean micro theta",
        max_abs_gap(macro.theta, micro_mean.theta) / theta_scale
        if theta_scale > 0.0
        else 0.0,
        THETA_REL_GAP_BOUND,
    )

    # The load-bearing check. See the calibration note at the top of this file:
    # at the default beta2 = 0.999 the correct and k-damped rules produce
    # OVERLAPPING z gaps across schedules, so that check cannot reject the
    # defect. At beta2 = 0.9 they are disjoint by 8.7x.
    macro_lb = run_macro(
        outcomes_by_report,
        lr=lr,
        beta1=beta1,
        beta2=BETA2_PROBE,
        eps=eps,
        c=c,
        mu2_init=mu2_init,
        init_stats=init_stats,
    )
    micro_lb = run_micro(
        build_const_mean_online_sequences(
            outcomes_by_report,
            mu2_init,
            init_stats=init_stats,
        ),
        lr=lr,
        beta1=beta1,
        beta2=BETA2_PROBE,
        eps=eps,
        c=c,
    )
    gate.check_le(
        f"macro z within bound of micro z at beta2={BETA2_PROBE:g}",
        max_abs_gap(macro_lb.z, micro_lb.z),
        Z_GAP_BOUND_LOW_BETA2,
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
    show(fig1)

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

    gate.check_le(
        "time axes agree (shuffled)",
        _axis_gap(macro2, micro_mean2, micro_real2),
        0.0,
    )
    gate.check_le(
        "macro z within bound of const-mean micro z (shuffled)",
        max_abs_gap(macro2.z, micro_mean2.z),
        Z_GAP_BOUND,
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
    show(fig2)

    return gate.report()


if __name__ == "__main__":
    raise SystemExit(main())
