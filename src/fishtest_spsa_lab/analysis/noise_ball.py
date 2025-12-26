#!/usr/bin/env python3
"""Elo loss from the optimum under SPSA match-outcome noise (diagnostic).

The quadratic Elo model is initialized to match the toy script conventions from
`optimize_spsa_toy.py`:

- Peak at x_peak = (100, 100, ..., 100)
- Normalized coordinates d = (x - x_peak) / X_SCALE with X_SCALE=100
- Elo_peak = 0
- `--trace-g` defines the total curvature budget trace(G).
    Under this script's coordinate conventions (same as the toy optimizer),
    Elo(0,...,0) = -0.5*trace(G) for any N, so origin_drop = 0.5*trace(G).

In this script we use the simplest "isotropic N-quadric":

    Elo(x) = 0 - 0.5 * d^T G d,
    G = (trace(G)/N) * I.

SPSA approximation:

- delta in {±1}^N.
- We use a constant-gain proxy for the schedule, representing the *end* of the
    run:
    - c_end is calibrated via `--c-diag-elo-drop` (same meaning as the toy script)
    - r_end is taken from `--r-end`
- Match outcome noise uses the vendored pentanomial model (PentaModel).

    Each SPSA batch corresponds to one noisy "plus vs minus" match between
    x_plus=x+c*delta and x_minus=x-c*delta. The match returns a noisy net score
    (net wins) for plus, aggregated over `--batch-size-pairs` pairs.

What gets printed (KPI) for each N:

- N
    Dimension (same meaning as `--num-params` in the toy script).

- elo_drop_stationary_mean
    Predicted steady-state mean Elo loss from the peak (in Elo) under the
    constant-gain proxy (c_end, r_end).

    This answers: "starting SPSA from the peak, how far do we get knocked off
    the peak by match-outcome noise in the long run?" and how it scales with N.

- elo_drop_after_1000, elo_drop_after_10000
    Predicted mean Elo loss from the peak after 1000 / 10000 SPSA batches,
    assuming we start exactly at the peak and then run with the constant-gain
    proxy (c_end, r_end) for that many batches.

Plot mode:
- With `--n-max 1000 --plot`, sweeps N=1..n_max and plots:
    - `elo_drop_stationary_mean`
    - `elo_drop_after_1000`
    - `elo_drop_after_10000`
- Add `--plot-step1` to also overlay the one-update drop (pure-noise step at the peak).
- With `--plot`, a second subplot repeats the same curves under an N-normalized
    learning-rate variant (`--lr-norm`).

How to interpret the numbers (what you should expect):

- With `--trace-g` fixed, for the isotropic N-quadric the per-dimension
    curvature scales like 1/N.

- With `--c-diag-elo-drop` fixed, the diagonal perturbation calibration makes
    c_end independent of N.

Together, in this *specific* toy model (and with r_end and batch_size_pairs held
fixed), `elo_drop_stationary_mean` typically grows roughly linearly with N.

More useful sensitivity checks (directional):

- Increasing `--batch-size-pairs` reduces `elo_drop_stationary_mean` (roughly ~ 1/B).
- Increasing `--r-end` increases `elo_drop_stationary_mean` (approximately linearly).

Tip:
- If you want the noise-ball Elo loss not to blow up with N in this toy,
        try scaling the learning rate down with dimension.

    In this specific isotropic toy, the stationary KPI tends to scale ~ N for
    fixed r_end, so r_end/N is the normalization that approximately flattens it.

    Note: the finite-time curves (after 1k / 10k batches) can scale differently
    with N than the stationary curve.

Sanity / validity checks:

- This is a constant-gain, near-peak approximation; treat it as
    order-of-magnitude guidance.

Run:
  uv run python -m fishtest_spsa_lab.analysis.noise_ball

Notes:
- This is an approximation; treat results as order-of-magnitude guidance.
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from fishtest_spsa_lab.vendor.pentamodel.pentamodel import PentaModel

ELO_CLIP_RANGE: float = 599.0

X_PEAK: float = 100.0
X_SCALE: float = 100.0
ELO_PEAK: float = 0.0
DEFAULT_TRACE_G: float = 10.0

_OUTCOMES = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)


def _penta_mu_var_per_pair(*, elo_diff: float) -> tuple[float, float]:
    """Return (mean, var) of the per-pair outcome in {-2,-1,0,1,2}."""
    input_elo = float(np.clip(-float(elo_diff), -ELO_CLIP_RANGE, ELO_CLIP_RANGE))
    probs = np.asarray(PentaModel(opponentElo=input_elo).pentanomialProbs, dtype=float)
    mu = float(np.dot(probs, _OUTCOMES))
    mu2 = float(np.dot(probs, _OUTCOMES * _OUTCOMES))
    var = max(0.0, mu2 - mu * mu)
    return mu, var


def _lower_bounds_elo_drop_from_trace_g(*, trace_g: float) -> float:
    """Under the toy conventions: elo(0,...,0) = -0.5*trace(G)."""
    tr = float(trace_g)
    if not math.isfinite(tr) or tr <= 0.0:
        raise ValueError("trace_g must be positive")
    return 0.5 * tr


def _trace_g_from_lower_bounds_drop(*, lower_bounds_elo_drop: float) -> float:
    """Map lower-bounds Elo drop to trace(G): trace(G)=2*drop."""
    drop = float(lower_bounds_elo_drop)
    if not math.isfinite(drop) or drop <= 0.0:
        raise ValueError("lower_bounds_elo_drop must be positive")
    return 2.0 * drop


def _c_end_from_c_diag_elo_drop(*, c_diag_elo_drop: float, trace_g: float) -> float:
    """Match toy script calibration for diagonal (SPSA) perturbations at the peak."""
    drop = float(c_diag_elo_drop)
    if not math.isfinite(drop) or drop <= 0.0:
        raise ValueError("c_diag_elo_drop must be positive")

    tr = float(trace_g)
    if not math.isfinite(tr) or tr <= 0.0:
        raise ValueError("trace_g must be positive")

    # See optimize_spsa_toy.py:
    #   E[DeltaE_diag(c)] = 0.5 * (c/X_SCALE)^2 * trace(G)
    return float(X_SCALE) * float(math.sqrt((2.0 * drop) / tr))


def estimate_noise_ball_isotropic_end(
    *,
    num_params: int,
    batch_size_pairs: int,
    r_end: float,
    trace_g: float,
    c_diag_elo_drop: float,
    elo_diff_step: float,
) -> dict[str, float | int | bool] | None:
    """Estimate stationary noise-ball size for isotropic quadratic + SPSA.

    Returns a plain dict suitable for printing.
    """
    n = int(num_params)
    if n <= 0:
        raise ValueError("num_params must be positive")

    n_pairs = int(batch_size_pairs)
    if n_pairs <= 0:
        return None

    r_end_f = float(r_end)
    if not math.isfinite(r_end_f) or r_end_f <= 0.0:
        return None

    h = float(elo_diff_step)
    if not math.isfinite(h) or h <= 0.0:
        raise ValueError("elo_diff_step must be positive")

    trace_g_f = float(trace_g)
    if not math.isfinite(trace_g_f) or trace_g_f <= 0.0:
        raise ValueError("trace_g must be positive")
    lower_bounds_elo_drop = _lower_bounds_elo_drop_from_trace_g(
        trace_g=float(trace_g_f)
    )
    c_end = _c_end_from_c_diag_elo_drop(
        c_diag_elo_drop=float(c_diag_elo_drop),
        trace_g=float(trace_g_f),
    )

    # Quadratic in x-units: Elo(x) = ELO_PEAK - k_x * ||x-x_peak||^2.
    # For isotropic toy objective: Elo(x) = ELO_PEAK - 0.5 * (||e||^2 / X_SCALE^2) * (trace(G)/N).
    # So k_x = 0.5 * (trace(G)/N) / X_SCALE^2.
    k_x = 0.5 * (float(trace_g_f) / float(n)) / (float(X_SCALE) * float(X_SCALE))
    if not math.isfinite(k_x) or k_x <= 0.0:
        return None

    # Match-model slope at elo_diff=0.
    mu_p, _ = _penta_mu_var_per_pair(elo_diff=+h)
    mu_m, _ = _penta_mu_var_per_pair(elo_diff=-h)
    dmu_delo = (mu_p - mu_m) / (2.0 * h)
    slope_net_score_per_elo = float(n_pairs) * float(dmu_delo)

    # Match-model variance at elo_diff=0.
    _mu0, var0 = _penta_mu_var_per_pair(elo_diff=0.0)
    var_net_score = float(n_pairs) * float(var0)

    # grad_signal = net_score / 2 (same convention as the toy script).
    slope_grad_signal_per_elo = slope_net_score_per_elo / 2.0
    var_grad_signal = var_net_score / 4.0

    # Constant gain proxy.
    a_end = r_end_f * (c_end * c_end)

    # For the quadratic, Elo(x+cδ)-Elo(x-cδ) ≈ -4*k_x*c*<e,δ>.
    # E[grad_signal] ≈ slope_grad_signal_per_elo * (elo_plus-elo_minus).
    # => mean update: e_next ≈ e - eta_eff * <e,δ> δ.
    g_factor = 2.0 * k_x * slope_net_score_per_elo
    eta_eff = a_end * g_factor

    denom = c_end * g_factor
    if not math.isfinite(denom) or abs(denom) <= 0.0:
        return None

    # zeta is the additive noise in the canonical scalar: (-<e,δ> + zeta)
    var_zeta = var_grad_signal / (denom * denom)

    stability_denom = 2.0 - float(eta_eff) * float(n)
    stable = (float(eta_eff) > 0.0) and (stability_denom > 0.0)

    # Second-moment recursion (starting at the peak):
    #   s_{t+1} = a*s_t + b,
    # where s_t := E[||e_t||^2], a = 1 - 2*eta + eta^2*N, b = eta^2*N*Var(zeta).
    # Then s_* = b/(1-a) = (eta*N*Var(zeta)) / (2 - eta*N).
    a_rec = 1.0 - 2.0 * float(eta_eff) + (float(eta_eff) * float(eta_eff) * float(n))

    if stable:
        s_star = (float(eta_eff) * float(n) * float(var_zeta)) / float(stability_denom)
        s_star = float(max(0.0, s_star))
        theta_rms = float(math.sqrt(s_star))
        elo_drop_stationary_mean = float(k_x * s_star)
    else:
        s_star = float("inf")
        theta_rms = float("inf")
        elo_drop_stationary_mean = float("inf")

    def _elo_drop_after_batches(num_batches: int) -> float:
        t = int(num_batches)
        if t <= 0:
            return 0.0
        if not (stable and math.isfinite(s_star) and math.isfinite(a_rec)):
            return float("inf")
        # For second-moment stability we need |a_rec| < 1.
        if abs(float(a_rec)) >= 1.0:
            return float("inf")
        s_t = float(s_star) * (1.0 - float(a_rec) ** float(t))
        if not math.isfinite(s_t):
            return float("inf")
        return float(max(0.0, float(k_x) * s_t))

    elo_drop_after_1000 = _elo_drop_after_batches(1000)
    elo_drop_after_10000 = _elo_drop_after_batches(10000)

    # Optional: one-step expected drop from the peak when starting at the peak.
    # This is NOT the main KPI; it is mostly for debugging/intuition.
    elo_drop_step1_mean = (
        float(k_x)
        * (float(r_end_f) ** 2)
        * (float(var_net_score) / 4.0)
        * (float(n) * (float(c_end) ** 2))
    )

    # Parameter-space radius diagnostics (kept for verbose/debug only).
    x_rms_over_c_end = (
        float(theta_rms) / float(c_end) if math.isfinite(theta_rms) else float("inf")
    )
    x_rms_over_xscale = (
        float(theta_rms) / float(X_SCALE) if math.isfinite(theta_rms) else float("inf")
    )

    return {
        "n": int(n),
        "batch_size_pairs": int(n_pairs),
        "r_end": float(r_end_f),
        "lower_bounds_elo_drop": float(lower_bounds_elo_drop),
        "c_diag_elo_drop": float(c_diag_elo_drop),
        "trace_g": float(trace_g_f),
        "c_end": float(c_end),
        "k_x": float(k_x),
        "slope_net_score_per_elo": float(slope_net_score_per_elo),
        "var_net_score": float(var_net_score),
        "eta_eff": float(eta_eff),
        "eta_eff_n": float(eta_eff) * float(n),
        "stable": bool(stable),
        "a_rec": float(a_rec),
        "s_star": float(s_star),
        "theta_rms": float(theta_rms),
        "elo_drop_stationary_mean": float(elo_drop_stationary_mean),
        "elo_drop_after_1000": float(elo_drop_after_1000),
        "elo_drop_after_10000": float(elo_drop_after_10000),
        "elo_drop_step1_mean": float(elo_drop_step1_mean),
        "x_rms_over_c_end": float(x_rms_over_c_end),
        "x_rms_over_xscale": float(x_rms_over_xscale),
    }


def _fit_power_law_exponent(
    *,
    xs: list[int],
    ys: list[float],
    x_min: int | None = None,
    x_max: int | None = None,
) -> float | None:
    """Fit y ~ x^a in log-log space and return a.

    Returns None if insufficient finite, positive points exist.

    Notes:
    - This is a quick diagnostic fit; it is sensitive to the N-range.
    - Use x_min/x_max to avoid small-N transients or unstable tail regions.
    """
    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)

    mask = (x_arr > 0.0) & np.isfinite(x_arr) & (y_arr > 0.0) & np.isfinite(y_arr)
    if x_min is not None:
        mask &= x_arr >= float(int(x_min))
    if x_max is not None:
        mask &= x_arr <= float(int(x_max))
    if int(np.sum(mask)) < 2:
        return None

    lx = np.log(x_arr[mask])
    ly = np.log(y_arr[mask])
    a, _b = np.polyfit(lx, ly, deg=1)
    if not np.isfinite(a):
        return None
    return float(a)


def _estimate_for_n(
    *,
    n: int,
    batch_size_pairs: int,
    r_end: float,
    trace_g: float,
    c_diag_elo_drop: float,
    elo_diff_step: float,
) -> dict[str, float | int | bool] | None:
    return estimate_noise_ball_isotropic_end(
        num_params=int(n),
        batch_size_pairs=int(batch_size_pairs),
        r_end=float(r_end),
        trace_g=float(trace_g),
        c_diag_elo_drop=float(c_diag_elo_drop),
        elo_diff_step=float(elo_diff_step),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the stationary Elo drop from the optimum under SPSA evaluation "
            "noise for an isotropic quadratic toy model (order-of-magnitude diagnostic)."
        )
    )
    parser.add_argument(
        "--num-params",
        type=int,
        default=12,
        help="Dimension N (same meaning as the toy script).",
    )
    parser.add_argument(
        "--batch-size-pairs",
        type=int,
        default=36,
        help="Pairs per SPSA update (same meaning as the toy script).",
    )
    parser.add_argument(
        "--n-max",
        type=int,
        default=None,
        help=("If set, ignore --num-params and sweep N=1..n_max (useful with --plot)."),
    )
    parser.add_argument(
        "--r-end",
        type=float,
        default=None,
        help="End-of-run learning rate r_end (same flag name as the toy script).",
    )
    parser.add_argument(
        "--c-diag-elo-drop",
        type=float,
        default=2.0,
        help=(
            "Calibrate c_end so that the diagonal SPSA perturbation x +/- c_end*delta "
            "at the peak causes about this many Elo drop (same meaning as the toy script)."
        ),
    )
    parser.add_argument(
        "--trace-g",
        type=float,
        default=None,
        help=(
            "Total curvature budget trace(G). Implies Elo(0,...,0) = -0.5*trace(G) "
            "under this script's coordinate conventions."
        ),
    )
    parser.add_argument(
        "--lower-bounds-elo-drop",
        type=float,
        dest="lower_bounds_elo_drop",
        default=None,
        help=(
            "Alternative way to set trace(G) by specifying the Elo drop at the "
            "lower-bounds corner (x=(0,...,0) under the default bounds): "
            "trace(G)=2*drop. Mutually exclusive with --trace-g."
        ),
    )
    parser.add_argument(
        "--elo-diff-step",
        type=float,
        default=1.0,
        help="Finite-difference step (in Elo) used to estimate the local slope.",
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot key diagnostics across N (requires matplotlib).",
    )

    parser.add_argument(
        "--plot-step1",
        action="store_true",
        help="In plot mode, also overlay the one-update drop from the peak.",
    )

    parser.add_argument(
        "--lr-norm",
        choices=("none", "div-sqrt-n", "div-n", "auto"),
        default="auto",
        help=(
            "Learning-rate normalization used for the second plot. "
            "'div-sqrt-n' uses r_end/sqrt(N); 'div-n' uses r_end/N; "
            "'auto' fits exponents a in elo_drop~N^a (separately for the stationary/1k/10k curves) "
            "and uses r_end/N^a per curve; "
            "'none' disables the second plot."
        ),
    )

    parser.add_argument(
        "--lr-norm-fit-min-n",
        type=int,
        default=None,
        help=(
            "When --lr-norm auto, only use N>=min_n for the log-log exponent fit. "
            "(Helps avoid small-N transients.)"
        ),
    )
    parser.add_argument(
        "--lr-norm-fit-max-n",
        type=int,
        default=None,
        help=(
            "When --lr-norm auto, only use N<=max_n for the log-log exponent fit. "
            "(Helps avoid unstable tail regions.)"
        ),
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra intermediate diagnostics (eta/stability/slope/variance).",
    )

    args = parser.parse_args(argv)

    n_default = int(args.num_params)
    if n_default <= 0:
        raise ValueError("--num-params must be positive")

    batch_size_pairs = int(args.batch_size_pairs)
    if batch_size_pairs <= 0:
        raise ValueError("--batch-size-pairs must be positive")

    r_end = float(args.r_end) if args.r_end is not None else 0.002
    if not math.isfinite(r_end) or r_end <= 0.0:
        raise ValueError("--r-end must be positive")

    if args.trace_g is not None and (
        not math.isfinite(float(args.trace_g)) or float(args.trace_g) <= 0.0
    ):
        raise ValueError("--trace-g must be positive")
    if args.lower_bounds_elo_drop is not None and (
        not math.isfinite(float(args.lower_bounds_elo_drop))
        or float(args.lower_bounds_elo_drop) <= 0.0
    ):
        raise ValueError("--lower-bounds-elo-drop must be positive")

    if args.trace_g is not None and args.lower_bounds_elo_drop is not None:
        raise ValueError("Use only one of --trace-g or --lower-bounds-elo-drop")

    if args.trace_g is not None:
        trace_g = float(args.trace_g)
    elif args.lower_bounds_elo_drop is not None:
        trace_g = _trace_g_from_lower_bounds_drop(
            lower_bounds_elo_drop=float(args.lower_bounds_elo_drop)
        )
    else:
        trace_g = float(DEFAULT_TRACE_G)

    c_diag_elo_drop = float(args.c_diag_elo_drop)
    if not math.isfinite(c_diag_elo_drop) or c_diag_elo_drop <= 0.0:
        raise ValueError("--c-diag-elo-drop must be positive")

    if args.n_max is not None:
        n_max = int(args.n_max)
        if n_max <= 0:
            raise ValueError("--n-max must be positive")
        ns = list(range(1, n_max + 1))
    else:
        ns = [n_default]

    print("Noise-ball estimate sweep (isotropic, constant-gain proxy)")
    if args.verbose:
        print("  verbose: prints key=value diagnostics per N")
    else:
        print(
            "  columns: N elo_drop_stationary_mean elo_drop_after_1000 elo_drop_after_10000"
        )

    # Collect baseline estimates first.
    n_list: list[int] = []
    elo_drop_stationary_list: list[float] = []
    elo_drop_1000_list: list[float] = []
    elo_drop_10000_list: list[float] = []
    elo_drop_step1_list: list[float] = []

    for n in ns:
        est = _estimate_for_n(
            n=int(n),
            batch_size_pairs=int(batch_size_pairs),
            r_end=float(r_end),
            trace_g=float(trace_g),
            c_diag_elo_drop=float(c_diag_elo_drop),
            elo_diff_step=float(args.elo_diff_step),
        )
        if est is None:
            print(f"  N={int(n):5d}: unavailable")
            continue

        n_list.append(int(est["n"]))
        elo_drop_stationary_list.append(float(est["elo_drop_stationary_mean"]))
        elo_drop_1000_list.append(float(est["elo_drop_after_1000"]))
        elo_drop_10000_list.append(float(est["elo_drop_after_10000"]))
        elo_drop_step1_list.append(float(est["elo_drop_step1_mean"]))

        if args.verbose:
            print(f"  N={int(est['n'])}")
            print(
                "    "
                f"eta={float(est['eta_eff']):.6g} "
                f"etaN={float(est['eta_eff_n']):.6g} "
                f"stable={bool(est['stable'])} "
                f"a={float(est['a_rec']):.6g}"
            )
            print(
                "    "
                f"theta_rms={float(est['theta_rms']):.6g} "
                f"c_end={float(est['c_end']):.6g}"
            )
            print(
                "    "
                f"x_rms/c={float(est['x_rms_over_c_end']):.6g} "
                f"x_rms/X={float(est['x_rms_over_xscale']):.6g}"
            )
            print(
                "    "
                f"E_inf={float(est['elo_drop_stationary_mean']):.6g} "
                f"E_1k={float(est['elo_drop_after_1000']):.6g} "
                f"E_10k={float(est['elo_drop_after_10000']):.6g}"
            )
            print(f"    E_step1={float(est['elo_drop_step1_mean']):.6g}")
            print(
                "    "
                f"slope={float(est['slope_net_score_per_elo']):.6g} "
                f"var={float(est['var_net_score']):.6g}"
            )
        else:
            print(
                "  "
                f"{int(est['n']):5d} "
                f"{float(est['elo_drop_stationary_mean']):12.6g} "
                f"{float(est['elo_drop_after_1000']):15.6g} "
                f"{float(est['elo_drop_after_10000']):16.6g}"
            )

    # Optional: compute a second series with learning-rate normalization.
    lrnorm_label: str | None = None
    lrnorm_exponent: float | None = None
    lrnorm_exponent_stationary: float | None = None
    lrnorm_exponent_1000: float | None = None
    lrnorm_exponent_10000: float | None = None
    elo_drop_stationary_lrnorm_list: list[float] = []
    elo_drop_1000_lrnorm_list: list[float] = []
    elo_drop_10000_lrnorm_list: list[float] = []

    if args.lr_norm != "none" and n_list:
        if args.lr_norm == "div-sqrt-n":
            lrnorm_label = "r_end/sqrt(N)"
            lrnorm_exponent = 0.5
        elif args.lr_norm == "div-n":
            lrnorm_label = "r_end/N"
            lrnorm_exponent = 1.0
        elif args.lr_norm == "auto":
            fitted_inf = _fit_power_law_exponent(
                xs=n_list,
                ys=elo_drop_stationary_list,
                x_min=args.lr_norm_fit_min_n,
                x_max=args.lr_norm_fit_max_n,
            )
            fitted_1k = _fit_power_law_exponent(
                xs=n_list,
                ys=elo_drop_1000_list,
                x_min=args.lr_norm_fit_min_n,
                x_max=args.lr_norm_fit_max_n,
            )
            fitted_10k = _fit_power_law_exponent(
                xs=n_list,
                ys=elo_drop_10000_list,
                x_min=args.lr_norm_fit_min_n,
                x_max=args.lr_norm_fit_max_n,
            )

            # Fallback: the isotropic toy tends to behave close to linear.
            lrnorm_exponent_stationary = (
                float(fitted_inf) if fitted_inf is not None else 1.0
            )
            lrnorm_exponent_1000 = float(fitted_1k) if fitted_1k is not None else 1.0
            lrnorm_exponent_10000 = float(fitted_10k) if fitted_10k is not None else 1.0
            lrnorm_label = "r_end/N^a (fit per curve)"

            if args.verbose or args.plot:
                print(
                    "  lr-norm auto: fitted exponents "
                    f"a_inf={float(lrnorm_exponent_stationary):.6g} "
                    f"a_1k={float(lrnorm_exponent_1000):.6g} "
                    f"a_10k={float(lrnorm_exponent_10000):.6g}"
                )

        if args.lr_norm != "auto":
            lrnorm_exponent_stationary = lrnorm_exponent
            lrnorm_exponent_1000 = lrnorm_exponent
            lrnorm_exponent_10000 = lrnorm_exponent

        for n in n_list:
            if (
                lrnorm_exponent_stationary is None
                or lrnorm_exponent_1000 is None
                or lrnorm_exponent_10000 is None
            ):
                elo_drop_stationary_lrnorm_list.append(float("nan"))
                elo_drop_1000_lrnorm_list.append(float("nan"))
                elo_drop_10000_lrnorm_list.append(float("nan"))
                continue

            exp_inf = float(lrnorm_exponent_stationary)
            exp_1k = float(lrnorm_exponent_1000)
            exp_10k = float(lrnorm_exponent_10000)

            def _run_lrnorm(*, exponent: float) -> dict[str, float | int | bool] | None:
                r_end_lrnorm = float(r_end) / float(float(n) ** float(exponent))
                return _estimate_for_n(
                    n=int(n),
                    batch_size_pairs=int(batch_size_pairs),
                    r_end=float(r_end_lrnorm),
                    trace_g=float(trace_g),
                    c_diag_elo_drop=float(c_diag_elo_drop),
                    elo_diff_step=float(args.elo_diff_step),
                )

            # If all exponents coincide, run the estimator once.
            if (exp_inf == exp_1k) and (exp_inf == exp_10k):
                est_lrnorm = _run_lrnorm(exponent=exp_inf)
                if est_lrnorm is None:
                    elo_drop_stationary_lrnorm_list.append(float("nan"))
                    elo_drop_1000_lrnorm_list.append(float("nan"))
                    elo_drop_10000_lrnorm_list.append(float("nan"))
                else:
                    elo_drop_stationary_lrnorm_list.append(
                        float(est_lrnorm["elo_drop_stationary_mean"])
                    )
                    elo_drop_1000_lrnorm_list.append(
                        float(est_lrnorm["elo_drop_after_1000"])
                    )
                    elo_drop_10000_lrnorm_list.append(
                        float(est_lrnorm["elo_drop_after_10000"])
                    )
            else:
                est_inf = _run_lrnorm(exponent=exp_inf)
                est_1k = _run_lrnorm(exponent=exp_1k)
                est_10k = _run_lrnorm(exponent=exp_10k)

                elo_drop_stationary_lrnorm_list.append(
                    float(est_inf["elo_drop_stationary_mean"])
                    if est_inf
                    else float("nan")
                )
                elo_drop_1000_lrnorm_list.append(
                    float(est_1k["elo_drop_after_1000"]) if est_1k else float("nan")
                )
                elo_drop_10000_lrnorm_list.append(
                    float(est_10k["elo_drop_after_10000"]) if est_10k else float("nan")
                )

    if args.plot:
        if not n_list:
            print("plot: no data")
            return 0

        try:
            import matplotlib.pyplot as plt
            from matplotlib.ticker import ScalarFormatter
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"matplotlib required for --plot: {exc}")

        ns_arr = np.asarray(n_list, dtype=float)
        stationary_arr = np.asarray(elo_drop_stationary_list, dtype=float)
        after_1000_arr = np.asarray(elo_drop_1000_list, dtype=float)
        after_10000_arr = np.asarray(elo_drop_10000_list, dtype=float)
        step1_arr = np.asarray(elo_drop_step1_list, dtype=float)

        # Two plots:
        #  1) baseline stationary KPI vs N
        #  2) stationary KPI vs N under a learning-rate normalization
        n_rows = 2 if args.lr_norm != "none" else 1
        fig, axes = plt.subplots(n_rows, 1, figsize=(10.5, 7.0), sharex=True)
        if n_rows == 1:
            axes = [axes]

        ax0 = axes[0]
        ax0.set_title("Predicted mean Elo loss from peak (baseline r_end)")
        ax0.plot(ns_arr, stationary_arr, label="elo_drop_stationary_mean")
        ax0.plot(ns_arr, after_1000_arr, label="elo_drop_after_1000")
        ax0.plot(ns_arr, after_10000_arr, label="elo_drop_after_10000")
        if args.plot_step1:
            ax0.plot(ns_arr, step1_arr, label="elo_drop_step1_mean")
        ax0.set_ylabel("Elo")
        ax0.grid(True, alpha=0.25)
        ax0.legend(loc="best")

        for ax in axes:
            ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)

        if args.lr_norm != "none":
            ax1 = axes[1]
            lrnorm_arr = np.asarray(elo_drop_stationary_lrnorm_list, dtype=float)
            lrnorm_1000_arr = np.asarray(elo_drop_1000_lrnorm_list, dtype=float)
            lrnorm_10000_arr = np.asarray(elo_drop_10000_lrnorm_list, dtype=float)
            ax1.set_title(f"Elo loss with learning-rate normalization: {lrnorm_label}")
            if args.lr_norm == "auto":
                ax1.plot(
                    ns_arr,
                    lrnorm_arr,
                    label=(
                        "elo_drop_stationary_mean (a="
                        f"{float(lrnorm_exponent_stationary):.3g})"
                    ),
                )
                ax1.plot(
                    ns_arr,
                    lrnorm_1000_arr,
                    label=(
                        f"elo_drop_after_1000 (a={float(lrnorm_exponent_1000):.3g})"
                    ),
                )
                ax1.plot(
                    ns_arr,
                    lrnorm_10000_arr,
                    label=(
                        f"elo_drop_after_10000 (a={float(lrnorm_exponent_10000):.3g})"
                    ),
                )
            else:
                ax1.plot(ns_arr, lrnorm_arr, label="elo_drop_stationary_mean (lr-norm)")
                ax1.plot(ns_arr, lrnorm_1000_arr, label="elo_drop_after_1000 (lr-norm)")
                ax1.plot(
                    ns_arr, lrnorm_10000_arr, label="elo_drop_after_10000 (lr-norm)"
                )
            ax1.set_xlabel("N")
            ax1.set_ylabel("Elo")
            ax1.grid(True, alpha=0.25)
            ax1.legend(loc="best")

        fig.tight_layout()
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
