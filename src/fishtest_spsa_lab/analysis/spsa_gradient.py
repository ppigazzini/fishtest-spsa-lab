#!/usr/bin/env python3
"""
Compute a 2-point SPSA pseudo-gradient at a given point.

SPSA (Rademacher):
  delta_i ∈ {+1,-1}
  y+ = f(x + c*delta)
  y- = f(x - c*delta)
  ghat_i = ((y+ - y-) / (2*c)) * delta_i      # since 1/delta_i == delta_i for ±1

This script can also generate a sequence of independent SPSA estimates over
multiple trials at a fixed x and optionally plot them against the true
gradient for the example objective.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SPSAResult:
    delta: np.ndarray
    y_plus: float
    y_minus: float
    dy: float
    dir_deriv: float
    ghat: np.ndarray


@dataclass(frozen=True)
class SPSAMeanResult:
    num_trials: int
    mean_ghat: np.ndarray
    se_ghat: np.ndarray


def summarize_trials(ghat_trials: np.ndarray) -> SPSAMeanResult:
    g = np.asarray(ghat_trials, dtype=np.float64)
    if g.ndim != 2:
        raise ValueError("ghat_trials must have shape [trials, n]")
    t = int(g.shape[0])
    if t <= 0:
        raise ValueError("need at least 1 trial")

    mean_g = np.mean(g, axis=0)
    mean_g2 = np.mean(g * g, axis=0)
    var_g = np.maximum(0.0, mean_g2 - mean_g * mean_g)
    se_mean_g = np.sqrt(var_g / float(t))
    return SPSAMeanResult(num_trials=t, mean_ghat=mean_g, se_ghat=se_mean_g)


def plot_series(
    *,
    y_est: np.ndarray,
    y_true: float,
    y_se: np.ndarray | None,
    title: str,
    ylabel: str,
) -> None:
    # Import lazily so non-plot usage doesn't require matplotlib.
    import matplotlib.pyplot as plt

    trials = np.arange(1, int(y_est.shape[0]) + 1)
    trials, y_est = _decimate_for_plot(trials, y_est)
    plt.figure(figsize=(8, 4.5))
    if y_se is None:
        plt.plot(trials, y_est, linewidth=1.5)
    else:
        plt.errorbar(trials, y_est, yerr=y_se, linestyle="-", capsize=3)

    plt.axhline(float(y_true), color="black", linestyle="--", linewidth=1.25)
    plt.title(title)
    plt.xlabel("Trial")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()


def plot_phi_convergence(
    *, dir_hat: np.ndarray, dir_true: np.ndarray, title: str
) -> None:
    import matplotlib.pyplot as plt

    dir_hat = np.asarray(dir_hat, dtype=np.float64)
    dir_true = np.asarray(dir_true, dtype=np.float64)
    if dir_hat.shape != dir_true.shape or dir_hat.ndim != 1:
        raise ValueError("dir_hat and dir_true must be 1D arrays of the same shape")

    t = int(dir_hat.shape[0])
    x = np.arange(1, t + 1, dtype=np.float64)
    err = dir_hat - dir_true
    run_mean_err = np.cumsum(err) / x
    run_mse = np.cumsum(err * err) / x

    x_d, run_mean_err_d = _decimate_for_plot(x, run_mean_err)
    _, run_mse_d = _decimate_for_plot(x, run_mse)

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.0), sharex=True)
    axes[0].plot(x_d, run_mean_err_d, linewidth=1.5)
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    axes[0].set_ylabel("running mean error")
    axes[0].grid(True, alpha=0.2)

    axes[1].plot(x_d, run_mse_d, linewidth=1.5)
    axes[1].set_ylabel("running MSE")
    axes[1].set_xlabel("Trial")
    axes[1].grid(True, alpha=0.2)

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def _decimate_for_plot(
    x: np.ndarray, y: np.ndarray, *, max_points: int = 2000
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x)
    y = np.asarray(y)
    n = int(x.shape[0])
    if n <= max_points:
        return x, y
    step = int(np.ceil(n / float(max_points)))
    x2 = x[::step]
    y2 = y[::step]
    if x2[-1] != x[-1]:
        x2 = np.concatenate([x2, x[-1:]])
        y2 = np.concatenate([y2, y[-1:]])
    return x2, y2


def plot_running_mean_all_coords(
    *,
    g_est: np.ndarray,
    g_true: np.ndarray,
    title: str,
    max_coords: int | None = None,
) -> None:
    """Plot per-coordinate running mean vs trial index.

    Each subplot corresponds to one coordinate i:
      running_mean_t[i] = (1/t) * sum_{s<=t} ghat_s[i]
    with a dashed horizontal line at the true gradient value.
    """

    import matplotlib.pyplot as plt

    g_est = np.asarray(g_est, dtype=np.float64)
    g_true = np.asarray(g_true, dtype=np.float64)
    if g_est.ndim != 2:
        raise ValueError("g_est must be a 2D array [trials, n]")
    if g_true.ndim != 1:
        raise ValueError("g_true must be a 1D vector [n]")
    trials, n = g_est.shape
    if g_true.shape[0] != n:
        raise ValueError("g_true length must match g_est second dimension")

    k = n if max_coords is None else max(0, min(int(max_coords), n))
    if k <= 0:
        return

    x = np.arange(1, int(trials) + 1)
    denom = x[:, None].astype(np.float64)
    running_mean = np.cumsum(g_est[:, :k], axis=0) / denom

    cols = min(5, k)
    rows = int(np.ceil(k / float(cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.4 * rows), sharex=True)
    axes_arr = np.asarray(axes).reshape(-1)

    for i in range(rows * cols):
        ax = axes_arr[i]
        if i >= k:
            ax.axis("off")
            continue
        xi, yi = _decimate_for_plot(x, running_mean[:, i])
        ax.plot(xi, yi, linewidth=1.3)
        ax.axhline(
            float(g_true[i]), color="black", linestyle="--", linewidth=1.0, alpha=0.6
        )
        ax.set_title(f"coord {i}", fontsize=10)
        ax.grid(True, alpha=0.2)

    fig.suptitle(title)
    fig.supxlabel("Trial")
    fig.supylabel("running mean ghat[i]")
    fig.tight_layout()
    plt.show()


def rademacher_delta(
    rng: np.random.Generator, n: int, p_pos: float = 0.5
) -> np.ndarray:
    if not (0.0 < p_pos < 1.0):
        raise ValueError("p_pos must be in (0,1)")
    u = rng.random(n)
    return np.where(u < p_pos, 1.0, -1.0).astype(np.float64, copy=False)


def finite_difference_grad(
    f,
    x: np.ndarray,
    *,
    rel_step: float = 1e-6,
    abs_step: float | None = None,
) -> np.ndarray:
    """Central finite-difference gradient of a scalar function.

    Uses per-coordinate steps:
      h_i = abs_step, if provided
      h_i = rel_step * max(1, |x_i|), otherwise
    """

    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("x must be a 1D vector")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must be finite")
    if abs_step is None:
        rel = float(rel_step)
        if not np.isfinite(rel) or rel <= 0.0:
            raise ValueError("--fd-rel-step must be a positive finite float")
    else:
        h = float(abs_step)
        if not np.isfinite(h) or h <= 0.0:
            raise ValueError("--fd-abs-step must be a positive finite float")

    n = int(x.shape[0])
    grad = np.zeros(n, dtype=np.float64)

    x_plus = x.copy()
    x_minus = x.copy()

    for i in range(n):
        h_i = (
            float(abs_step)
            if abs_step is not None
            else float(rel_step) * max(1.0, abs(float(x[i])))
        )
        x_plus[i] = x[i] + h_i
        x_minus[i] = x[i] - h_i
        y_plus = float(f(x_plus))
        y_minus = float(f(x_minus))
        grad[i] = (y_plus - y_minus) / (2.0 * h_i)
        x_plus[i] = x[i]
        x_minus[i] = x[i]

    return grad


def finite_difference_dir_deriv(
    f,
    x: np.ndarray,
    d: np.ndarray,
    *,
    rel_step: float = 1e-6,
    abs_step: float | None = None,
) -> float:
    """Central finite-difference directional derivative along direction d.

    Computes (f(x+h*d) - f(x-h*d)) / (2h) with a step h.
    The direction d is used as-is (not normalized).
    """

    x = np.asarray(x, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    if x.ndim != 1 or d.ndim != 1 or x.shape != d.shape:
        raise ValueError("x and d must be 1D vectors of the same shape")

    if abs_step is None:
        rel = float(rel_step)
        if not np.isfinite(rel) or rel <= 0.0:
            raise ValueError("--phi-fd-rel-step must be a positive finite float")
        # Scale h relative to x magnitude to keep numerical conditioning reasonable.
        scale = float(max(1.0, np.max(np.abs(x))))
        h = rel * scale
    else:
        h = float(abs_step)
        if not np.isfinite(h) or h <= 0.0:
            raise ValueError("--phi-fd-abs-step must be a positive finite float")

    x_plus = x + h * d
    x_minus = x - h * d
    return (float(f(x_plus)) - float(f(x_minus))) / (2.0 * h)


def spsa_pseudograd(
    f,
    x: np.ndarray,
    *,
    c: float,
    rng: np.random.Generator,
    p_pos: float = 0.5,
) -> SPSAResult:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("x must be a 1D vector")
    if not np.isfinite(c) or c <= 0.0:
        raise ValueError("c must be a positive finite float")

    n = int(x.shape[0])
    delta = rademacher_delta(rng, n, p_pos=p_pos)

    x_plus = x + c * delta
    x_minus = x - c * delta

    y_plus = float(f(x_plus))
    y_minus = float(f(x_minus))

    dy = y_plus - y_minus
    dir_deriv = dy / (2.0 * c)
    ghat = dir_deriv * delta
    return SPSAResult(
        delta=delta,
        y_plus=y_plus,
        y_minus=y_minus,
        dy=dy,
        dir_deriv=dir_deriv,
        ghat=ghat,
    )


def parse_vec(s: str) -> np.ndarray:
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    if not parts:
        raise ValueError("empty --x")
    v = np.asarray([float(p) for p in parts], dtype=np.float64)
    if not np.all(np.isfinite(v)):
        raise ValueError("--x must contain only finite numbers")
    return v


def expand_x_to_n(x: np.ndarray, *, n: int) -> np.ndarray:
    """Expand/broadcast x to length n.

    Rules:
      - If len(x) == n: return x
      - If len(x) == 1: broadcast that value
      - If len(x) > 1 and all entries are equal: broadcast that value
      - Otherwise: raise (non-broadcastable)

    This supports e.g. `--x 1000,1000 --n 10`.
    """

    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("x must be a 1D vector")

    n_i = int(n)
    if n_i <= 0:
        raise ValueError("--n must be >= 1")

    m = int(x.shape[0])
    if m == n_i:
        return x
    if m == 1:
        return np.full((n_i,), float(x[0]), dtype=np.float64)

    # Allow broadcasting only if user provided repeated identical values.
    if np.all(x == x[0]):
        return np.full((n_i,), float(x[0]), dtype=np.float64)

    raise ValueError(
        f"--x has length {m} but --n={n_i}. "
        "Provide exactly N comma-separated values, or a single value, "
        "or repeated identical values (e.g. '1000,1000') to broadcast."
    )


def make_constant_vec(*, n: int, x0: float) -> np.ndarray:
    n_i = int(n)
    if n_i <= 0:
        raise ValueError("--n must be >= 1")
    x0_f = float(x0)
    if not np.isfinite(x0_f):
        raise ValueError("--x0 must be finite")
    return np.full((n_i,), x0_f, dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--x",
        type=str,
        default=None,
        help=(
            "Comma-separated vector, e.g. '1000,1000'. "
            "If --n is provided and differs from len(--x), --x may be broadcast only "
            "when it is a single value or repeated identical values. "
            "If omitted, uses an N-dimensional constant vector from --n and --x0."
        ),
    )
    ap.add_argument(
        "--n",
        type=int,
        default=2,
        help="Target dimension N (used to construct --x when omitted; can also broadcast --x)",
    )
    ap.add_argument(
        "--x0",
        type=float,
        default=1000.0,
        help="Constant coordinate used when --x is omitted",
    )
    ap.add_argument(
        "--c", type=float, default=0.1, help="SPSA perturbation scale c > 0"
    )
    ap.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Number of SPSA trials to generate. If omitted: 10 when --plot, else 1.",
    )
    ap.add_argument("--seed", type=int, default=0, help="RNG seed")
    ap.add_argument("--p-pos", type=float, default=0.5, help="P(delta_i = +1)")
    ap.add_argument(
        "--fd-rel-step",
        type=float,
        default=1e-6,
        help="Relative step for central finite-difference reference gradient (h_i = rel*max(1,|x_i|))",
    )
    ap.add_argument(
        "--fd-abs-step",
        type=float,
        default=None,
        help="Absolute step for central finite-difference reference gradient (overrides --fd-rel-step)",
    )
    ap.add_argument(
        "--phi-true-mode",
        type=str,
        choices=("fd", "proj"),
        default="fd",
        help=(
            "How to compute 'dir_true' in phi diagnostics. "
            "'fd' uses central finite difference along the same delta. "
            "'proj' uses <ref_grad_fd, delta>."
        ),
    )
    ap.add_argument(
        "--phi-fd-rel-step",
        type=float,
        default=1e-6,
        help="Relative step for phi-direction finite difference when --phi-true-mode=fd",
    )
    ap.add_argument(
        "--phi-fd-abs-step",
        type=float,
        default=None,
        help="Absolute step for phi-direction finite difference when --phi-true-mode=fd (overrides rel)",
    )
    ap.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, shows Matplotlib charts (no file output)",
    )
    ap.add_argument(
        "--plot-mode",
        type=str,
        choices=("coords", "phi"),
        default="coords",
        help="Plot mode: 'coords' plots running mean per coordinate; 'phi' plots directional-derivative convergence",
    )
    ap.add_argument(
        "--plot-coord",
        type=int,
        default=-1,
        help="Which gradient coordinate to plot (0-based). Use -1 to plot all coordinates.",
    )
    ap.add_argument(
        "--plot-max-coords",
        type=int,
        default=None,
        help="When plotting all coordinates, optionally plot only the first K",
    )
    args = ap.parse_args()

    target_n = int(args.n)
    if args.x is None:
        x = make_constant_vec(n=target_n, x0=float(args.x0))
    else:
        x_raw = parse_vec(str(args.x))
        x = expand_x_to_n(x_raw, n=target_n)
    rng = np.random.default_rng(np.uint64(args.seed))

    # Example objective: z = sum_i x_i^2 (i.e. x^2 + y^2 in 2D)
    def f(v: np.ndarray) -> float:
        v = np.asarray(v, dtype=np.float64)
        return float(np.sum(v * v * v))

    n = int(x.shape[0])
    print(f"x        = {x}")
    print(f"c        = {float(args.c)}")
    print(f"N        = {n}")
    trials = (
        10
        if bool(args.plot) and args.trials is None
        else (1 if args.trials is None else int(args.trials))
    )
    print(f"trials   = {int(trials)}")
    if trials <= 0:
        raise ValueError("--trials must be > 0")

    coord = int(args.plot_coord)
    if coord != -1 and (coord < 0 or coord >= n):
        raise ValueError(f"--plot-coord must be -1 or in [0,{n - 1}]")

    # Reference gradient computed numerically so the user can change f without updating math here.
    # Note: phi diagnostics can also compute a *directional* finite difference along delta.
    ref_grad = finite_difference_grad(
        f,
        x,
        rel_step=float(args.fd_rel_step),
        abs_step=None if args.fd_abs_step is None else float(args.fd_abs_step),
    )
    print(f"ref_grad_fd (central diff) = {ref_grad}")

    # Collect a per-trial series (each point is one SPSA pseudo-gradient sample).
    series_g = np.zeros((trials, n), dtype=np.float64)
    series_dir = np.zeros((trials,), dtype=np.float64)
    series_dir_true = np.zeros((trials,), dtype=np.float64)
    series_dir_proj = np.zeros((trials,), dtype=np.float64)
    last: SPSAResult | None = None
    for t in range(trials):
        last = spsa_pseudograd(f, x, c=float(args.c), rng=rng, p_pos=float(args.p_pos))
        series_g[t, :] = last.ghat
        series_dir[t] = float(last.dir_deriv)
        # Linearized directional derivative from the reference gradient.
        series_dir_proj[t] = float(np.dot(ref_grad, last.delta))

        if str(args.phi_true_mode) == "fd":
            series_dir_true[t] = finite_difference_dir_deriv(
                f,
                x,
                last.delta,
                rel_step=float(args.phi_fd_rel_step),
                abs_step=None
                if args.phi_fd_abs_step is None
                else float(args.phi_fd_abs_step),
            )
        else:
            series_dir_true[t] = series_dir_proj[t]

    if trials == 1 and last is not None:
        print(f"delta    = {last.delta}")
        print(f"y_plus   = {last.y_plus:.12g}")
        print(f"y_minus  = {last.y_minus:.12g}")
        print(f"dy       = {last.dy:.12g}")
        print(f"dir_hat  = {last.dir_deriv:.12g}   (=(y_plus-y_minus)/(2c))")
        print(
            f"dir_proj = {float(np.dot(ref_grad, last.delta)):.12g}   (=<ref_grad_fd, delta>)"
        )
        print(
            f"dir_true = {float(series_dir_true[0]):.12g}   (mode={str(args.phi_true_mode)})"
        )
        print(f"ghat     = {last.ghat}")
    else:
        summary = summarize_trials(series_g)
        print(f"mean_ghat          = {summary.mean_ghat}")
        print(f"se(mean_ghat)      = {summary.se_ghat}")

        # Scalar directional-derivative diagnostics (phi-domain).
        err = series_dir - series_dir_true
        mean_err = float(np.mean(err))
        se_mean_err = float(np.std(err, ddof=0) / np.sqrt(float(trials)))
        rmse = float(np.sqrt(np.mean(err * err)))
        print(f"dir_hat mean       = {float(np.mean(series_dir)):.12g}")
        print(f"dir_true mean      = {float(np.mean(series_dir_true)):.12g}")
        print(f"dir_proj mean      = {float(np.mean(series_dir_proj)):.12g}")
        print(
            f"dir_hat - dir_true: mean={mean_err:.12g} se(mean)={se_mean_err:.12g} rmse={rmse:.12g}"
        )

    if bool(args.plot):
        if str(args.plot_mode) == "phi":
            plot_phi_convergence(
                dir_hat=series_dir,
                dir_true=series_dir_true,
                title=f"Directional derivative (Rademacher) convergence (N={n}, c={float(args.c)}, trials={trials})",
            )
        else:
            if coord == -1:
                plot_running_mean_all_coords(
                    g_est=series_g,
                    g_true=ref_grad,
                    max_coords=args.plot_max_coords,
                    title=f"Running mean SPSA gradient vs true (N={n}, c={float(args.c)}, trials={trials})",
                )
            else:
                x_trials = np.arange(1, int(trials) + 1, dtype=np.float64)
                y_est = np.cumsum(series_g[:, coord]) / x_trials
                y_true = float(ref_grad[coord])
                plot_series(
                    y_est=y_est,
                    y_true=y_true,
                    y_se=None,
                    title=f"Running mean SPSA gradient vs true (coord={coord}, N={n}, c={float(args.c)}, trials={trials})",
                    ylabel=f"running mean ghat[{coord}]",
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
