"""Toy SPSA optimizer demo using the vendored pentamodel.

This is an engine-Elo maximization problem:
- Peak (best) is 0 Elo at (100, 100, ..., 100).
- The objective is a quadratic bowl in normalized coordinates:
    d := (x - x_peak) / X_SCALE
    elo(x) = ELO_PEAK - 0.5 * d^T G d

In this parameterization, the total curvature scale is controlled by trace(G).
Because the orthonormal rotation is constructed to keep the all-ones direction
fixed, the lower-bound origin (0, ..., 0) satisfies d = (-1, ..., -1) and:
    elo(0, ..., 0) = ELO_PEAK - 0.5 * trace(G).

Supports arbitrary dimensionality; 2D plots always use parameters 0 and 1 and are
skipped if there are fewer than 2 parameters.
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends import BackendFilter, backend_registry
from numpy.typing import NDArray

from fishtest_spsa_lab.vendor.pentamodel.pentamodel import PentaModel

logger = logging.getLogger(__name__)


X_PEAK: float = 100.0
X_SCALE: float = 100.0
ELO_PEAK: float = 0.0
DEFAULT_TRACE_G: float = 10.0
PLOT_BOUNDS_DEFAULT: tuple[float, float] = (0.0, 200.0)

P_DELTA_POS: float = 0.5
MIN_DIMS_FOR_PLOT_2D: int = 2

# PentaModel opponentElo clipping range (copied from pentamodel conventions).
ELO_CLIP_RANGE: float = 599.0

_NET_WINS_WEIGHTS = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)

ELO_CALIBRATION_EPS: float = 1e-9
ELO_PEAK_EPS: float = 1e-12

# SPSA stability constant A: fishtest typically uses a fraction of total pairs.
DEFAULT_STABILITY_A_FRAC: float = 0.1

# Automatic trajectory downsampling: keep at most this many points (including start).
MAX_STORED_POINTS: int = 10_000

# If the eigen-spectrum spans at least this ratio, use a log y-axis.
SPECTRUM_LOG_SCALE_RATIO: float = 1_000.0


type FloatArray = NDArray[np.float64]
type ObjectiveFn = Callable[[FloatArray], float]


@dataclass(frozen=True, slots=True)
class QuadraticObjectiveModel:
    """Quadratic objective in normalized coordinates.

    Model:
        d := (x - X_PEAK) / X_SCALE
        elo(x) = ELO_PEAK - 0.5 * d^T G d

    The eigenvalue sum trace(G) is typically held constant as N changes.
    """

    objective: ObjectiveFn
    g_matrix: FloatArray  # shape: (N, N)
    eigvals: FloatArray  # shape: (N,)
    u_matrix: FloatArray  # shape: (N, N)
    trace_g: float


@dataclass(frozen=True, slots=True)
class SpectrumParams:
    """Parameters controlling the quadratic spectrum + rotation."""

    shape: str
    exponent: float
    active: int | None
    trace: float
    use_identity_u: bool
    u_seed: int


def _make_spectrum_eigenvalues(  # noqa: C901, PLR0913
    *,
    num_params: int,
    shape: str,
    exponent: float,
    active: int | None,
    trace_target: float,
) -> FloatArray:
    n = int(num_params)
    if n <= 0:
        msg = "num_params must be > 0"
        raise ValueError(msg)

    trace_t = float(trace_target)
    if not np.isfinite(trace_t) or trace_t <= 0.0:
        msg = "spectrum_trace must be a finite float > 0"
        raise ValueError(msg)

    shape_s = str(shape)
    if shape_s not in {"power", "linear", "uniform"}:
        msg = "spectrum_shape must be one of: power, linear, uniform"
        raise ValueError(msg)

    is_uniform = shape_s == "uniform"
    is_linear = shape_s == "linear"
    p = float(exponent)
    if (not is_uniform) and (not is_linear) and (not np.isfinite(p) or p < 0.0):
        msg = "spectrum_exponent must be a finite float >= 0"
        raise ValueError(msg)

    active_n = _spectrum_active_dim(n_total=n, active_k=active)

    weights = _spectrum_raw_weights(
        k=active_n,
        shape=shape_s,
        uniform=is_uniform,
        exponent=p,
    )
    if active_n < n:
        weights = np.concatenate(
            [weights, np.zeros((n - active_n,), dtype=np.float64)],
            axis=0,
        )

    s = float(np.sum(weights))
    if not np.isfinite(s) or s <= 0.0:
        msg = "invalid spectrum: sum of eigenvalues must be > 0"
        raise ValueError(msg)

    eigvals = weights * (trace_t / s)
    return np.asarray(eigvals, dtype=np.float64)


def _spectrum_active_dim(*, n_total: int, active_k: int | None) -> int:
    if active_k is None:
        return int(n_total)
    k = int(active_k)
    if k <= 0 or k > int(n_total):
        msg = "spectrum_active must be in [1, num_params]"
        raise ValueError(msg)
    return k


def _spectrum_raw_weights(
    *,
    k: int,
    shape: str,
    uniform: bool,
    exponent: float,
) -> FloatArray:
    kk = int(k)
    if kk <= 0:
        msg = "k must be > 0"
        raise ValueError(msg)

    shape_s = str(shape)
    if shape_s not in {"power", "linear", "uniform"}:
        msg = "spectrum_shape must be one of: power, linear, uniform"
        raise ValueError(msg)

    if uniform:
        return np.ones((kk,), dtype=np.float64)

    # Linear spectrum: single canonical slope (no shape parameter).
    # Weights decrease linearly from 1 down to 1/K on the active subspace.
    if shape_s == "linear":
        if kk == 1:
            return np.ones((1,), dtype=np.float64)

        # weights from 1 down to 1/K (strictly positive)
        return np.linspace(1.0, 1.0 / float(kk), kk, dtype=np.float64)

    # Power spectrum: lambda_i ~ 1 / i^p.
    if exponent == 0.0:
        return np.ones((kk,), dtype=np.float64)

    idx = np.arange(1, kk + 1, dtype=np.float64)
    return 1.0 / np.power(idx, float(exponent))


def _random_orthonormal_fixing_ones(
    *,
    num_params: int,
    rng: np.random.Generator,
) -> FloatArray:
    """Random orthonormal U with U @ 1 = 1 (all-ones direction fixed)."""
    n = int(num_params)
    if n <= 0:
        msg = "num_params must be > 0"
        raise ValueError(msg)

    if n == 1:
        return np.asarray([[1.0]], dtype=np.float64)

    ones = np.ones((n,), dtype=np.float64)
    e0 = ones / float(np.linalg.norm(ones))

    probe = rng.normal(size=(n, n - 1))
    probe = probe - e0[:, None] * (e0 @ probe)[None, :]
    q_perp, _r = np.linalg.qr(probe, mode="reduced")
    q_perp = np.asarray(q_perp, dtype=np.float64)

    a = rng.normal(size=(n - 1, n - 1))
    r_perp, _r2 = np.linalg.qr(a)
    r_perp = np.asarray(r_perp, dtype=np.float64)

    basis = np.column_stack([e0, q_perp])
    block = np.eye(n, dtype=np.float64)
    block[1:, 1:] = r_perp

    u = basis @ block @ basis.T
    return np.asarray(u, dtype=np.float64)


def _make_spectrum_objective(
    *,
    num_params: int,
    params: SpectrumParams,
) -> QuadraticObjectiveModel:
    n = int(num_params)
    eigvals = _make_spectrum_eigenvalues(
        num_params=n,
        shape=str(params.shape),
        exponent=float(params.exponent),
        active=params.active,
        trace_target=float(params.trace),
    )
    trace_g = float(np.sum(eigvals))

    rng = np.random.default_rng(np.uint64(int(params.u_seed)))
    if params.use_identity_u:
        u = np.eye(n, dtype=np.float64)
    else:
        u = _random_orthonormal_fixing_ones(num_params=n, rng=rng)

    g = u.T @ (eigvals[:, None] * u)
    g = np.asarray(0.5 * (g + g.T), dtype=np.float64)

    def _objective(x: FloatArray) -> float:
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.ndim != 1:
            msg = "elo function expects a 1D parameter vector"
            raise ValueError(msg)
        if int(x_arr.shape[0]) != n:
            msg = "elo function expects a vector of length == num_params"
            raise ValueError(msg)

        d = (x_arr - X_PEAK) / X_SCALE
        return float(ELO_PEAK) - 0.5 * float(d.T @ g @ d)

    return QuadraticObjectiveModel(
        objective=_objective,
        g_matrix=g,
        eigvals=eigvals,
        u_matrix=u,
        trace_g=float(trace_g),
    )


def _make_toy_objective(*, num_params: int, seed: int) -> QuadraticObjectiveModel:
    """Create the default objective (spectrum + orthonormal rotation)."""
    return _make_spectrum_objective(
        num_params=int(num_params),
        params=SpectrumParams(
            shape="power",
            exponent=1.0,
            active=None,
            trace=float(DEFAULT_TRACE_G),
            use_identity_u=False,
            u_seed=int(seed),
        ),
    )


def _c_end_from_c_diag_elo_drop(
    *,
    c_diag_elo_drop: float,
    trace_g: float,
) -> float:
    """Compute a per-dimension `c_end` vector calibrated to an Elo gap.

    Calibration matches the actual SPSA (diagonal) perturbation: perturbing all
    coordinates by +/-c (i.e. x +/- c * delta with delta_i in {-1,+1}) at the
    peak corresponds to approximately `c_diag_elo_drop` Elo loss.

    For a quadratic objective elo(x) = ELO_PEAK - 0.5 * d^T G d with
    d=(x-X_PEAK)/X_SCALE, the expected diagonal/SPSA Elo drop at the peak is:

        E[DeltaE_diag(c)] = 0.5 * (c/X_SCALE)^2 * trace(G)

    so c_end = X_SCALE * sqrt(2 * c_diag_elo_drop / trace(G)).
    """
    elo_drop_f = float(c_diag_elo_drop)
    if not np.isfinite(elo_drop_f) or elo_drop_f <= 0.0:
        msg = "c_diag_elo_drop must be a positive finite float"
        raise ValueError(msg)

    trace_f = float(trace_g)
    if not np.isfinite(trace_f) or trace_f <= 0.0:
        msg = "trace_g must be a positive finite float"
        raise ValueError(msg)

    c_target = float(X_SCALE) * float(np.sqrt((2.0 * elo_drop_f) / trace_f))
    return float(c_target)


def _c_end_vec_from_dev_eigenbasis(
    *,
    c_diag_elo_drop: float,
    g_diag_true: FloatArray,
    u_matrix: FloatArray,
    eigvals_dev: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Construct per-axis c_end from a *predicted* eigen-spectrum (dev model).

    True objective (in normalized coordinates d):

        elo(x) = ELO_PEAK - 0.5 * d^T G d
        G = U^T diag(lambda_true) U

    Developer prediction is specified in the eigenbasis as lambda_dev.

    The dev chooses an eigen-coordinate probe magnitude:

        c_z_i ∝ 1 / sqrt(lambda_dev_i)

    and we map that to an axis-aligned per-parameter probe by matching the
    per-axis variance implied by independent Rademacher perturbations in z:

        Var(d_j) = sum_i U_{i,j}^2 * c_z_i^2
        c_x_shape_j := X_SCALE * sqrt(Var(d_j))

    Finally, apply a single global rescale so the achieved diagonal/SPSA Elo drop
    under the *true* diagonal g_diag_true matches c_diag_elo_drop exactly.

    Returns (c_end_vec, c_x_shape_vec).
    """
    drop = float(c_diag_elo_drop)
    if not np.isfinite(drop) or drop <= 0.0:
        msg = "c_diag_elo_drop must be a positive finite float"
        raise ValueError(msg)

    g = np.asarray(g_diag_true, dtype=np.float64).reshape(-1)
    u = np.asarray(u_matrix, dtype=np.float64)
    eig = np.asarray(eigvals_dev, dtype=np.float64).reshape(-1)

    n = int(g.shape[0])
    if n <= 0:
        msg = "g_diag_true must be non-empty"
        raise ValueError(msg)
    if u.shape != (n, n) or eig.shape != (n,):
        msg = "u_matrix and eigvals_dev must match g_diag_true shape"
        raise ValueError(msg)
    if np.any(~np.isfinite(g)) or np.any(~np.isfinite(u)) or np.any(~np.isfinite(eig)):
        msg = "g_diag_true, u_matrix, and eigvals_dev must be finite"
        raise ValueError(msg)

    active = eig > 0.0
    if int(np.count_nonzero(active)) <= 0:
        c_shape = np.zeros((n,), dtype=np.float64)
        return c_shape, c_shape

    c_z = np.zeros((n,), dtype=np.float64)
    c_z[active] = 1.0 / np.sqrt(np.maximum(eig[active], 1e-30))

    u_sq = u * u
    var_d = u_sq.T @ (c_z * c_z)
    var_d = np.maximum(var_d, 0.0)

    c_x_shape = float(X_SCALE) * np.sqrt(var_d)
    c_end = _rescale_c_end_to_hit_diag_drop(
        c=c_x_shape,
        g_diag=g,
        c_diag_elo_drop=float(drop),
    )

    return np.asarray(c_end, dtype=np.float64), np.asarray(c_x_shape, dtype=np.float64)


def _rescale_c_end_to_hit_diag_drop(
    *,
    c: FloatArray,
    g_diag: FloatArray,
    c_diag_elo_drop: float,
) -> FloatArray:
    """Rescale c on active dims so diagonal/SPSA drop matches target.

    Uses the toy quadratic convention:

        DeltaE_diag(c) = 0.5 * (1/X_SCALE^2) * sum_i g_i * c_i^2.

    Rescales only active dims (g_i > 0). Inactive dims remain 0.
    """
    drop = float(c_diag_elo_drop)
    if not np.isfinite(drop) or drop <= 0.0:
        msg = "c_diag_elo_drop must be a positive finite float"
        raise ValueError(msg)

    g = np.asarray(g_diag, dtype=np.float64).reshape(-1)
    c_arr = np.asarray(c, dtype=np.float64).reshape(-1)
    if g.shape != c_arr.shape:
        msg = "g_diag and c must have the same shape"
        raise ValueError(msg)

    active = g > 0.0
    if int(np.count_nonzero(active)) <= 0:
        return np.zeros_like(c_arr)

    achieved = float(
        0.5
        * float(np.sum(g[active] * (c_arr[active] * c_arr[active])))
        / (float(X_SCALE) * float(X_SCALE)),
    )
    if not np.isfinite(achieved) or achieved <= 0.0:
        return np.zeros_like(c_arr)

    scale = math.sqrt(float(drop) / float(achieved))
    c_arr = c_arr.copy()
    c_arr[active] = c_arr[active] * float(scale)
    return np.asarray(c_arr, dtype=np.float64)


def _diag_elo_drop_at_peak_from_c_end(
    *, c_end: FloatArray, g_diag: FloatArray
) -> float:
    """Compute diagonal/SPSA Elo drop at the peak for a given c_end vector.

    Uses the toy quadratic convention:

        DeltaE_diag(c_end) = 0.5 * (1/X_SCALE^2) * sum_i g_i * c_end_i^2

    Only active dims (g_i > 0) contribute.
    """
    g = np.asarray(g_diag, dtype=np.float64).reshape(-1)
    c = np.asarray(c_end, dtype=np.float64).reshape(-1)
    if g.shape != c.shape:
        msg = "g_diag and c_end must have the same shape"
        raise ValueError(msg)

    active = g > 0.0
    if int(np.count_nonzero(active)) <= 0:
        return 0.0

    drop = float(
        0.5
        * float(np.sum(g[active] * (c[active] * c[active])))
        / (float(X_SCALE) * float(X_SCALE))
    )
    return float(drop)


def _format_params(params: Sequence[float], *, max_items: int = 8) -> str:
    """Format a parameter vector compactly for logs."""
    n = len(params)
    if n == 0:
        return "()"

    first = float(params[0])
    if all(float(v) == first for v in params):
        return f"(n={n}, all={first:g})"

    if n <= max_items:
        return str(tuple(float(v) for v in params))

    head_n = max(2, max_items // 2)
    tail_n = max(2, max_items - head_n)
    head = ", ".join(f"{float(v):g}" for v in params[:head_n])
    tail = ", ".join(f"{float(v):g}" for v in params[-tail_n:])
    return f"(n={n}, head=[{head}], tail=[{tail}])"


def _rms_normalized_distance_to_peak(params: Sequence[float]) -> float:
    """RMS normalized distance to the peak (100,...,100), unitless."""
    x = np.asarray([float(v) for v in params], dtype=np.float64)
    deltas = (x - X_PEAK) / X_SCALE
    return float(np.sqrt(np.mean(deltas * deltas)))


def _max_drawdown(values: FloatArray) -> float:
    """Max drawdown (peak-to-trough) in the stored value history."""
    if values.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(values)
    return float(np.max(running_max - values))


def _count_record_highs(values: FloatArray) -> int:
    """Count how many times a new record high occurs (including the first point)."""
    if values.size == 0:
        return 0
    running_max = np.maximum.accumulate(values)
    # New record highs correspond to strict increases in running_max.
    return int(1 + np.count_nonzero(np.diff(running_max) > 0.0))


def _format_minmax(min_val: float, max_val: float) -> str:
    """Format a min..max range, collapsing to a single value if equal."""
    a = float(min_val)
    b = float(max_val)
    if (
        np.isfinite(a)
        and np.isfinite(b)
        and abs(a - b) <= 1e-12 * max(1.0, abs(a), abs(b))
    ):
        return f"{a:.6g}"
    return f"{a:.6g}..{b:.6g}"


def _estimate_first_update_l2_scale(
    *,
    x0: FloatArray,
    games: MatchSimulator,
    optimizer: SPSAOptimizer,
    seed: int,
) -> tuple[float, float, float, float, float]:
    """Estimate the first SPSA step L2 scale at the starting point.

    This is a model-based diagnostic meant for console output.

    We:
    - take the actual schedule at the first update (pair_index=1)
    - sample one representative SPSA direction delta (deterministic RNG seed)
    - compute Elo(x_plus) and Elo(x_minus) deterministically
    - compute expected net score mean/variance from PentaModel
    - turn that into an RMS net score scale, then into an RMS ||Δx||_2 scale

    Returns:
    - c_k_l2: ||c_k||_2 at pair_index=1
    - r_k: r_k at pair_index=1
    - net_score_mean: E[net_score]
    - net_score_rms: sqrt(E[net_score^2])
    - dx_l2_rms: RMS estimate of ||Δx||_2

    """
    x0 = np.asarray(x0, dtype=np.float64)
    c_k, r_k = optimizer._schedule(1.0)  # noqa: SLF001

    np_rng = np.random.default_rng(np.uint64(seed))
    delta = np.where(
        np_rng.random(optimizer.num_params) < P_DELTA_POS,
        1.0,
        -1.0,
    ).astype(
        np.float64,
        copy=False,
    )

    x_plus = x0 + c_k * delta
    x_minus = x0 - c_k * delta

    elo_plus = float(games.objective_function(x_plus))
    elo_minus = float(games.objective_function(x_minus))

    # PentaModel convention: opponentElo = (opponent - player).
    opponent_elo = elo_minus - elo_plus
    model = PentaModel(opponentElo=float(opponent_elo))
    probs = np.asarray(model.pentanomialProbs, dtype=np.float64)
    prob_sum = float(np.sum(probs))
    if not np.isfinite(prob_sum) or prob_sum <= 0.0:
        msg = f"Invalid pentanomial probability sum: {prob_sum}"
        raise RuntimeError(msg)
    if abs(prob_sum - 1.0) > ELO_PEAK_EPS:
        probs = probs / prob_sum

    # Net-wins weights per pair for categories:
    # 0=LL, 1=LD+DL, 2=DD+WL+LW, 3=WD+DW, 4=WW
    score_weights = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    score_mean_per_pair = float(np.dot(score_weights, probs))
    score2_mean_per_pair = float(np.dot(score_weights * score_weights, probs))
    score_var_per_pair = max(
        0.0,
        score2_mean_per_pair - score_mean_per_pair * score_mean_per_pair,
    )

    batch = float(games.batch_size_pairs)
    net_score_mean = batch * score_mean_per_pair
    net_score_var = batch * score_var_per_pair
    net_score_rms = float(np.sqrt(net_score_mean * net_score_mean + net_score_var))

    # Update:
    #   grad_signal = net_score / 2  # noqa: ERA001
    #   Δx = (c_k * (r_k * grad_signal)) * delta  # noqa: ERA001
    # L2 norm given delta in {-1,+1}:
    #   ||Δx||_2 = |r_k * grad_signal| * ||c_k||_2
    c_k_l2 = float(np.linalg.norm(c_k))
    grad_signal_rms = float(net_score_rms) / 2.0
    dx_l2_rms = abs(float(r_k)) * abs(float(grad_signal_rms)) * c_k_l2

    return (
        c_k_l2,
        float(r_k),
        float(net_score_mean),
        float(net_score_rms),
        float(dx_l2_rms),
    )


def _penta_net_wins_mu_var_per_pair(*, elo_diff: float) -> tuple[float, float]:
    """Return (mean, var) for per-pair net-wins in {-2,-1,0,1,2}.

    Here elo_diff means (player - opponent) Elo.
    PentaModel expects opponentElo = (opponent - player) = -elo_diff.
    """
    input_elo = float(np.clip(-float(elo_diff), -ELO_CLIP_RANGE, ELO_CLIP_RANGE))
    model = PentaModel(opponentElo=float(input_elo))
    probs = np.asarray(model.pentanomialProbs, dtype=np.float64)
    prob_sum = float(np.sum(probs))
    if not np.isfinite(prob_sum) or prob_sum <= 0.0:
        msg = f"Invalid pentanomial probability sum: {prob_sum}"
        raise RuntimeError(msg)
    if abs(prob_sum - 1.0) > ELO_PEAK_EPS:
        probs = probs / prob_sum

    mu = float(np.dot(_NET_WINS_WEIGHTS, probs))
    mu2 = float(np.dot(_NET_WINS_WEIGHTS * _NET_WINS_WEIGHTS, probs))
    var = max(0.0, mu2 - mu * mu)
    return mu, var


def _estimate_noise_drop_end_proxy(
    *,
    optimizer: SPSAOptimizer,
    h_phi_trace: float,
    n_eff_phi: float,
    elo_diff_step: float = 1.0,
) -> float:
    """Estimate infinite-time noise drop at fixed end-of-run (c_end, r_end).

    This diagnostic is computed in Fishtest's phi-normalized coordinates
    (see docs/ALGORITHMS.md): theta = c * phi elementwise. In phi-space the
    constant learning rate is r_end and c only enters once via the objective
    curvature (H_phi = C G C / X_SCALE^2).
    """
    n = int(optimizer.num_params)
    if n <= 0:
        msg = "num_params must be > 0"
        raise ValueError(msg)

    n_eff_f = float(n_eff_phi)
    if not np.isfinite(n_eff_f) or n_eff_f <= 0.0:
        msg = "n_eff must be a positive finite float"
        raise ValueError(msg)
    n_eff_f = float(min(float(n), max(1.0, n_eff_f)))

    trace_f = float(h_phi_trace)
    if not np.isfinite(trace_f) or trace_f <= 0.0:
        msg = "h_phi_trace must be a positive finite float"
        raise ValueError(msg)

    # Quadratic in phi-units: Elo(phi) = ELO_PEAK - k_phi * ||phi-phi_peak||^2.
    # Here trace_f is trace(H_phi) = sum_i diag(H_phi)_i.
    k_phi = 0.5 * (trace_f / float(n_eff_f))
    if not np.isfinite(k_phi) or k_phi <= 0.0:
        msg = "invalid k_phi"
        raise RuntimeError(msg)

    # Match-model slope/variance at elo_diff=0.
    h = float(elo_diff_step)
    if not np.isfinite(h) or h <= 0.0:
        msg = "elo_diff_step must be positive"
        raise ValueError(msg)

    mu_p, _var_p = _penta_net_wins_mu_var_per_pair(elo_diff=+h)
    mu_m, _var_m = _penta_net_wins_mu_var_per_pair(elo_diff=-h)
    dmu_delo = (float(mu_p) - float(mu_m)) / (2.0 * float(h))
    slope_net_score_per_elo = float(optimizer.batch_size_pairs) * float(dmu_delo)

    _mu0, var0 = _penta_net_wins_mu_var_per_pair(elo_diff=0.0)
    var_net_score = float(optimizer.batch_size_pairs) * float(var0)

    # The actual update uses grad_signal = net_score / 2.
    var_grad_signal = float(var_net_score) / 4.0

    # Mean dynamics coefficient (noise_ball.py convention), in phi-space.
    g_factor = 2.0 * float(k_phi) * float(slope_net_score_per_elo)

    # End-of-run stationary proxy (constant gain at (phi, r_end)).
    denom = float(g_factor) * float(g_factor)
    if not np.isfinite(denom) or denom <= 0.0:
        var_zeta_end = float("inf")
    else:
        # In phi-space there is no extra c factor in the step; c is already
        # absorbed into H_phi. So var_zeta scales as var_g / g_factor^2.
        var_zeta_end = float(var_grad_signal) / denom

    eta_end = float(optimizer.r_end) * float(g_factor)
    eta_end_n = float(eta_end) * float(n_eff_f)
    stable_end = (float(eta_end) > 0.0) and (2.0 - float(eta_end_n) > 0.0)
    if stable_end and np.isfinite(var_zeta_end):
        s_star_end = (
            float(eta_end)
            * float(n_eff_f)
            * float(var_zeta_end)
            / (2.0 - float(eta_end_n))
        )
        s_star_end = float(max(0.0, s_star_end))
        elo_drop_end_inf = float(k_phi) * float(s_star_end)
    else:
        elo_drop_end_inf = float("inf")

    return float(elo_drop_end_inf)


class MatchSimulator:
    """Generates SPSA match outcomes from an engine-Elo objective."""

    __slots__ = ("batch_size_pairs", "objective_function", "rng")

    def __init__(
        self,
        objective_function: ObjectiveFn,
        *,
        batch_size_pairs: int = 1000,
        rng: random.Random | None = None,
    ) -> None:
        """Create a match outcome generator for an objective function."""
        self.objective_function = objective_function
        self.batch_size_pairs = int(batch_size_pairs)
        if self.batch_size_pairs <= 0:
            msg = "batch_size_pairs must be > 0"
            raise ValueError(msg)
        self.rng = rng if rng is not None else random.Random()  # noqa: S311

    def match_net_wins(
        self,
        x_plus: FloatArray,
        x_minus: FloatArray,
        *,
        base_seed: int | None = None,
    ) -> float:
        """Simulate a plus-vs-minus match; return net wins for plus."""
        if base_seed is None:
            base_seed = self.rng.randrange(1 << 63)

        elo_plus = float(self.objective_function(x_plus))
        elo_minus = float(self.objective_function(x_minus))

        # PentaModel convention: opponentElo = (opponent - player).
        # Here we treat `x_plus` as the player and `x_minus` as the opponent.
        opponent_elo = elo_minus - elo_plus

        # Fast sampling: the pentanomial match result is multinomial with the
        # precomputed category probabilities. This avoids Python-per-round loops.
        model = PentaModel(opponentElo=opponent_elo)
        probs = np.asarray(model.pentanomialProbs, dtype=np.float64)
        prob_sum = float(np.sum(probs))
        if not np.isfinite(prob_sum) or prob_sum <= 0.0:
            msg = f"Invalid pentanomial probability sum: {prob_sum}"
            raise RuntimeError(msg)
        if abs(prob_sum - 1.0) > ELO_PEAK_EPS:
            probs = probs / prob_sum

        np_rng = np.random.default_rng(np.uint64(base_seed))
        counts = np_rng.multinomial(self.batch_size_pairs, probs)

        # Net wins: (2*WW + (WD+DW)) - (2*LL + (LD+DL))
        # counts indices: 0=LL, 1=LD+DL, 2=DD+WL+LW, 3=WD+DW, 4=WW
        net_wins = (2 * counts[4] + counts[3]) - (2 * counts[0] + counts[1])
        return float(net_wins)


@dataclass(frozen=True, slots=True)
class SPSAResult:
    """Output container for a single SPSA run."""

    best_params: tuple[float, ...]
    best_value: float
    best_pairs: float
    trajectory: FloatArray  # shape: (num_stored_steps, num_params)
    pairs_history: FloatArray  # shape: (num_stored_steps,)
    value_history: FloatArray  # shape: (num_stored_steps,)


class SPSAOptimizer:
    """Simple SPSA optimizer (pair-indexed schedule, unbounded parameters)."""

    def __init__(  # noqa: PLR0913
        self,
        games: MatchSimulator,
        *,
        num_params: int,
        num_batches: int,
        c_diag_elo_drop: float = 0.1,
        r_end: float = 0.01,
        c_end: Sequence[float] | None = None,
        a_stability: float | None = None,
        alpha: float = 0.602,
        gamma: float = 0.101,
        rng: random.Random | None = None,
        np_rng: np.random.Generator | None = None,
    ) -> None:
        """Create an SPSA optimizer.

        The schedule is pair-indexed (1 pair = 2 games). `a_stability` is the
        standard SPSA stability constant (often written as A in the literature).
        If None, we set A = DEFAULT_STABILITY_A_FRAC * total_pairs.
        """
        self.games = games
        self.num_params = int(num_params)
        self.num_batches = int(num_batches)
        self.r_end = float(r_end)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.rng = rng if rng is not None else random.Random()  # noqa: S311
        self.np_rng = np_rng if np_rng is not None else np.random.default_rng()

        if self.num_params <= 0:
            msg = "num_params must be > 0"
            raise ValueError(msg)
        if self.num_batches <= 0:
            msg = "num_batches must be > 0"
            raise ValueError(msg)

        self.batch_size_pairs = int(self.games.batch_size_pairs)
        if self.batch_size_pairs <= 0:
            msg = "batch_size_pairs must be > 0"
            raise ValueError(msg)

        self.total_pairs = self.num_batches * self.batch_size_pairs
        if self.total_pairs <= 0:
            msg = "total_pairs must be > 0"
            raise ValueError(msg)

        if a_stability is None:
            self.a_stability = DEFAULT_STABILITY_A_FRAC * float(self.total_pairs)
        else:
            self.a_stability = float(a_stability)
            if not np.isfinite(self.a_stability) or self.a_stability < 0.0:
                msg = "a_stability must be a finite float >= 0"
                raise ValueError(msg)

        if c_end is None:
            # Choose c_end so that the actual SPSA perturbation at the peak
            # (x +/- c*delta) corresponds to approximately `c_diag_elo_drop` Elo.
            #
            # For a quadratic objective elo(x) = ELO_PEAK - 0.5 * d^T G d,
            # E[DeltaE_diag] = 0.5 * (c/X_SCALE)^2 * trace(G).
            # When we don't have an objective-specific trace(G) available here,
            # use the script's default total curvature budget.
            trace_g_default = float(DEFAULT_TRACE_G)
            c_target = _c_end_from_c_diag_elo_drop(
                c_diag_elo_drop=float(c_diag_elo_drop),
                trace_g=float(trace_g_default),
            )
            self.c_end_vec = [float(c_target) for _ in range(self.num_params)]
        else:
            c_end_list = [float(v) for v in c_end]
            if len(c_end_list) != self.num_params:
                msg = "c_end must have length == num_params"
                raise ValueError(msg)
            self.c_end_vec = c_end_list

        # Precompute schedule bases so that c_k(total_pairs)=c_end and
        # r_k(total_pairs)=r_end, where r_k := a_k / c_k^2.
        self.c_base = np.asarray(
            [c * (self.total_pairs**self.gamma) for c in self.c_end_vec],
            dtype=np.float64,
        )

        # In fishtest, r_end is the end-of-run learning rate in phi-space.
        # We precompute r_base so that r_k(total_pairs) == r_end under the
        # usual power-law schedules.
        self.r_base = (
            self.r_end
            * ((self.a_stability + float(self.total_pairs)) ** self.alpha)
            / (float(self.total_pairs) ** (2.0 * self.gamma))
        )

        # No extra normalization: grad_signal = net_score / 2.

    def _auto_stride(self) -> int:
        # Store at most MAX_STORED_POINTS points (including step 0).
        if self.num_batches + 1 <= MAX_STORED_POINTS:
            return 1
        return int(np.ceil((self.num_batches + 1) / float(MAX_STORED_POINTS)))

    def _schedule(self, pair_index: float) -> tuple[FloatArray, float]:
        c_k = self.c_base / (pair_index**self.gamma)
        r_k = (
            self.r_base
            * (pair_index ** (2.0 * self.gamma))
            / ((self.a_stability + pair_index) ** self.alpha)
        )
        return c_k, float(r_k)

    def _sample_delta(self) -> FloatArray:
        return np.where(
            self.np_rng.random(self.num_params) < P_DELTA_POS,
            1.0,
            -1.0,
        ).astype(np.float64, copy=False)

    def _init_storage(
        self,
        *,
        x: FloatArray,
        start_value: float,
    ) -> tuple[list[int], FloatArray, FloatArray, FloatArray, int, int]:
        stride = self._auto_stride()
        store_steps = list(range(0, self.num_batches + 1, stride))
        if store_steps[-1] != self.num_batches:
            store_steps.append(self.num_batches)
        store_len = len(store_steps)

        trajectory = np.empty((store_len, self.num_params), dtype=np.float64)
        pairs_history = np.empty((store_len,), dtype=np.float64)
        value_history = np.empty((store_len,), dtype=np.float64)

        trajectory[0] = x
        pairs_history[0] = 0.0
        value_history[0] = float(start_value)

        next_store_step = store_steps[1] if store_len > 1 else -1
        return store_steps, trajectory, pairs_history, value_history, 1, next_store_step

    def _batch_step(self, *, batch_idx: int, x: FloatArray) -> tuple[FloatArray, float]:
        pair_index = float(batch_idx * self.batch_size_pairs + 1)
        c_k, r_k = self._schedule(pair_index)
        base_seed = self.rng.randrange(1 << 63)
        delta = self._sample_delta()

        x_plus = x + c_k * delta
        x_minus = x - c_k * delta
        net_score = self.games.match_net_wins(x_plus, x_minus, base_seed=base_seed)

        grad_signal = float(net_score) / 2.0
        x_new = x + (c_k * (r_k * grad_signal)) * delta
        val = float(self.games.objective_function(x_new))
        return x_new, val

    def run(self, init: Iterable[float]) -> SPSAResult:
        """Run SPSA starting from `init` and return the full trajectory."""
        x0_list = [float(v) for v in init]
        if len(x0_list) != self.num_params:
            msg = "init must have length == num_params"
            raise ValueError(msg)

        x = np.asarray(x0_list, dtype=np.float64)

        best = tuple(float(v) for v in x)
        best_val = float(self.games.objective_function(x))
        best_pairs: float = 0.0

        (
            store_steps,
            trajectory,
            pairs_history,
            value_history,
            store_i,
            next_store_step,
        ) = self._init_storage(x=x, start_value=best_val)

        for batch_idx in range(self.num_batches):
            x, val = self._batch_step(batch_idx=batch_idx, x=x)
            if val > best_val:
                best_val = val
                best = tuple(float(v) for v in x)
                best_pairs = float((batch_idx + 1) * self.batch_size_pairs)

            step = batch_idx + 1
            if step == next_store_step:
                trajectory[store_i] = x
                pairs_history[store_i] = float(step * self.batch_size_pairs)
                value_history[store_i] = float(val)
                store_i += 1
                next_store_step = (
                    store_steps[store_i] if store_i < len(store_steps) else -1
                )

        if store_i != len(store_steps):
            msg = "internal error: stored history size mismatch"
            raise RuntimeError(msg)

        return SPSAResult(
            best_params=best,
            best_value=float(best_val),
            best_pairs=float(best_pairs),
            trajectory=trajectory,
            pairs_history=pairs_history,
            value_history=value_history,
        )


def toy_elo_function(x: FloatArray) -> float:
    """Backward-compatible wrapper for the default spectrum-based toy Elo."""
    x_arr = np.asarray(x, dtype=np.float64)
    model = _make_toy_objective(num_params=int(x_arr.shape[0]), seed=0)
    return float(model.objective(x_arr))


@dataclass(frozen=True, slots=True)
class PlotInputs:
    """Inputs for the 2D contour + trajectory plot."""

    plot_bounds: Sequence[tuple[float, float]]
    g_sub_2d: np.ndarray
    trajectory: np.ndarray
    start: tuple[float, float]
    expected_peak: tuple[float, float]
    best: tuple[float, float]


def plot_contours_and_trajectory(inputs: PlotInputs) -> None:
    """Plot a 2D contour slice (dims 0/1) and overlay the trajectory."""
    if len(inputs.plot_bounds) < MIN_DIMS_FOR_PLOT_2D:
        return

    # Auto-zoom: fit the complete trajectory with a small margin.
    # Also enforce symmetric axes around the peak (X_PEAK, X_PEAK).
    traj_xy = np.asarray(inputs.trajectory[:, :2], dtype=np.float64)
    dx_max = float(np.max(np.abs(traj_xy[:, 0] - float(X_PEAK))))
    dy_max = float(np.max(np.abs(traj_xy[:, 1] - float(X_PEAK))))
    r = max(dx_max, dy_max)
    if not np.isfinite(r) or r <= 0.0:
        # Degenerate trajectory (or numeric issues): fall back to plot bounds.
        x_low0, x_high0 = (
            float(inputs.plot_bounds[0][0]),
            float(inputs.plot_bounds[0][1]),
        )
        y_low0, y_high0 = (
            float(inputs.plot_bounds[1][0]),
            float(inputs.plot_bounds[1][1]),
        )
        r = 0.5 * max(abs(x_high0 - x_low0), abs(y_high0 - y_low0), 1.0)

    margin_frac = 0.06
    margin_abs = 1.5
    r = float(r) * (1.0 + margin_frac) + float(margin_abs)

    x_low, x_high = float(X_PEAK) - r, float(X_PEAK) + r
    y_low, y_high = float(X_PEAK) - r, float(X_PEAK) + r

    grid_n = 200
    xs = np.linspace(x_low, x_high, grid_n)
    ys = np.linspace(y_low, y_high, grid_n)
    x_grid, y_grid = np.meshgrid(xs, ys)
    # Quadratic slice in normalized coords d=(x-X_PEAK)/X_SCALE using the
    # top-left 2x2 block of G.
    d0 = (x_grid - X_PEAK) / X_SCALE
    d1 = (y_grid - X_PEAK) / X_SCALE
    g00 = float(inputs.g_sub_2d[0, 0])
    g01 = float(inputs.g_sub_2d[0, 1])
    g11 = float(inputs.g_sub_2d[1, 1])
    z_grid = float(ELO_PEAK) - 0.5 * (
        g00 * d0 * d0 + 2.0 * g01 * d0 * d1 + g11 * d1 * d1
    )

    _fig, ax = plt.subplots(figsize=(9, 7))
    z_min = float(np.percentile(z_grid, 5.0))
    levels = np.linspace(z_min, 0.0, 20)
    ax.contour(x_grid, y_grid, z_grid, levels=levels, linewidths=1.0, alpha=0.9)

    final_xy = (float(inputs.trajectory[-1, 0]), float(inputs.trajectory[-1, 1]))

    ax.plot(
        inputs.trajectory[:, 0],
        inputs.trajectory[:, 1],
        "-k",
        lw=1.5,
        label="Trajectory",
    )
    ax.scatter(
        [inputs.start[0]],
        [inputs.start[1]],
        c="tab:green",
        s=80,
        label="Start",
    )
    ax.scatter(
        [inputs.expected_peak[0]],
        [inputs.expected_peak[1]],
        c="tab:blue",
        s=80,
        label="Expected peak",
    )
    ax.scatter(
        [inputs.best[0]],
        [inputs.best[1]],
        c="tab:red",
        s=80,
        label="Best found",
    )
    ax.scatter([final_xy[0]], [final_xy[1]], c="tab:orange", s=80, label="Final")

    ax.set_title("SPSA trajectory on engine Elo surface (maximize to 0)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(x_low, x_high)
    ax.set_ylim(y_low, y_high)
    ax.grid(visible=True)
    ax.legend(loc="best")


def plot_value_vs_pairs(
    *,
    pairs_history: np.ndarray,
    value_history: np.ndarray,
    best_pairs: float,
    best_value: float,
) -> None:
    """Plot objective value vs total pairs processed."""
    _fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(pairs_history, value_history, "-k", lw=1.5)

    # Mark start / best (true best, not downsampled argmax) / final explicitly.
    if pairs_history.size > 0 and value_history.size == pairs_history.size:
        start_pairs = float(pairs_history[0])
        start_val = float(value_history[0])
        final_pairs = float(pairs_history[-1])
        final_val = float(value_history[-1])

        # Keep colors consistent with the contour plot:
        # Start=green, Expected peak=blue, Best found=red, Final=orange.
        ax.scatter(
            [start_pairs],
            [start_val],
            c="tab:green",
            s=70,
            zorder=5,
            label="Start",
        )
        ax.scatter(
            [float(best_pairs)],
            [float(best_value)],
            c="tab:red",
            s=70,
            zorder=5,
            label="Best found",
        )
        ax.scatter(
            [final_pairs],
            [final_val],
            c="tab:orange",
            s=70,
            zorder=5,
            label="Final",
        )

        # Expected peak value is 0.0 for this toy objective.
        ax.axhline(ELO_PEAK, color="tab:blue", lw=1.2, alpha=0.9, label="Expected peak")

        ax.annotate(
            f"start={start_val:.3g}",
            xy=(start_pairs, start_val),
            xytext=(8, 8),
            textcoords="offset points",
            color="tab:green",
        )
        ax.annotate(
            f"best={float(best_value):.3g}",
            xy=(float(best_pairs), float(best_value)),
            xytext=(8, 8),
            textcoords="offset points",
            color="tab:red",
        )
        ax.annotate(
            f"final={final_val:.3g}",
            xy=(final_pairs, final_val),
            xytext=(8, 8),
            textcoords="offset points",
            color="tab:orange",
        )

    ax.set_title("Engine Elo vs pairs (maximize)")
    ax.set_xlabel("Pairs processed")
    ax.set_ylabel("Engine Elo")
    ax.grid(visible=True)
    ax.legend(loc="best")


def plot_active_eigenspectrum(
    *,
    eigvals: np.ndarray,
    eigvals_dev: np.ndarray | None = None,
    dev_label: str = "dev",
) -> None:
    """Plot the eigenvalue spectrum for active (non-zero) eigenvalues.

    If eigvals_dev is provided, overlays the (possibly noisy) dev-predicted spectrum.
    """
    eig = np.asarray(eigvals, dtype=np.float64).reshape(-1)
    active = eig[eig > 0.0]
    if active.size == 0:
        return

    xs = np.arange(1, int(active.size) + 1, dtype=np.float64)
    _fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(xs, active, "-o", lw=1.5, ms=4, label="true")

    if eigvals_dev is not None:
        eig_dev = np.asarray(eigvals_dev, dtype=np.float64).reshape(-1)
        active_dev = eig_dev[eig_dev > 0.0]
        if active_dev.size > 0:
            xs_dev = (
                xs
                if int(active_dev.size) == int(active.size)
                else np.arange(1, int(active_dev.size) + 1, dtype=np.float64)
            )
            ax.plot(xs_dev, active_dev, "--s", lw=1.2, ms=3.5, label=str(dev_label))

    # Always keep the spectrum chart linear (no auto log scaling).
    ax.set_yscale("linear")

    ax.set_title(f"Curvature eigenvalue spectrum (active K={int(active.size)})")
    ax.set_xlabel("Eigenvalue index i (1..K)")
    ax.set_ylabel("lambda_i")
    ax.grid(visible=True)
    if eigvals_dev is not None:
        ax.legend(loc="best")


def _parse_init_vector(init_str: str, num_params: int) -> tuple[float, ...]:
    """Parse --init into an N-length vector.

    Accepts either:
    - a single float (broadcast to all dims)
    - a comma-separated list of N floats
    """
    raw = str(init_str).strip()
    if raw == "":
        msg = "--init must be a number or a comma-separated list"
        raise ValueError(msg)

    parts = [p.strip() for p in raw.split(",")]
    vals = [float(p) for p in parts if p != ""]
    if len(vals) == 1:
        return tuple([float(vals[0]) for _ in range(num_params)])
    if len(vals) == num_params:
        return tuple(float(v) for v in vals)
    msg = (
        f"--init must be a single float or {num_params} comma-separated floats; "
        f"got {len(vals)}"
    )
    raise ValueError(msg)


def _init_vector_for_diagonal_elo_drop(
    *,
    num_params: int,
    elo_drop_from_peak: float,
    trace_g: float,
) -> tuple[float, ...]:
    """Build an init vector on the peak->origin diagonal from an Elo drop.

    We interpret "diagonal toward the origin" as x_i = X_PEAK - u for all i.
    Under the quadratic objective elo(x) = ELO_PEAK - 0.5 * d^T G d with
    d=(x-X_PEAK)/X_SCALE and a rotation that keeps the all-ones direction,
    the peak->origin diagonal corresponds to d = -(u/X_SCALE) * 1 and:

    elo_drop_from_peak = 0.5 * trace(G) * (u/X_SCALE)^2

    Solve for u, then set x_i = X_PEAK - u.
    """
    n = int(num_params)
    if n <= 0:
        msg = "num_params must be > 0"
        raise ValueError(msg)

    drop = float(elo_drop_from_peak)
    if not np.isfinite(drop) or drop < 0.0:
        msg = "init_elo_drop_from_peak must be a finite float >= 0"
        raise ValueError(msg)

    trace_f = float(trace_g)
    if not np.isfinite(trace_f) or trace_f <= 0.0:
        msg = "trace_g must be a positive finite float"
        raise ValueError(msg)

    u = float(X_SCALE) * float(np.sqrt((2.0 * drop) / trace_f))
    x_val = float(X_PEAK) - u

    return tuple([x_val for _ in range(n)])


def _configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="optimize-spsa-toy",
        description=(
            "Toy SPSA optimizer demo (engine Elo maximization). Supports arbitrary "
            "num-params; plots always use params 0 and 1 and are skipped if N<2."
        ),
    )
    parser.add_argument("--num-params", type=int, default=2)
    parser.add_argument("--batch-size-pairs", type=int, default=64)
    parser.add_argument("--num-batches", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--r-end",
        type=float,
        default=0.001,
        help=(
            "End-of-run learning rate r_end (i.e., r_k at the last pair index). "
            "In fishtest terms this is the phi-space learning rate; the per-axis "
            "theta step scales with c_k."
        ),
    )
    init_group = parser.add_mutually_exclusive_group()
    init_group.add_argument(
        "--init",
        type=str,
        default="0",
        help=(
            "Starting point. Either a single float (broadcast to all params) "
            "or a comma-separated list of N floats. Default: 0 (origin)."
        ),
    )
    init_group.add_argument(
        "--init-elo-drop-from-peak",
        type=float,
        default=None,
        help=(
            "Alternative init mode: choose the start point on the peak->origin "
            "diagonal such that the (deterministic) Elo is "
            "`-init_elo_drop_from_peak`. "
            "Example: --init-elo-drop-from-peak 3 starts ~3 Elo below the peak. "
            "Name uses 'drop-from-peak' to avoid confusion with --c-diag-elo-drop "
            "(SPSA c calibration)."
        ),
    )
    parser.add_argument(
        "--c-diag-elo-drop",
        type=float,
        default=0.1,
        help=(
            "Calibrate c_end so that the SPSA diagonal perturbation theta +/- "
            "c_end*delta at the peak causes about this many Elo drop (default: 0.1)."
        ),
    )

    parser.add_argument(
        "--spectrum-shape",
        type=str,
        default="power",
        choices=("power", "linear", "uniform"),
        help=(
            "Eigenvalue spectrum family for G. 'power' uses lambda_i ~ 1/i^p "
            "(controlled by --spectrum-exponent; this is the 'gamma' case). "
            "'linear' uses a decreasing linear spectrum on active dimensions "
            "(single canonical slope). 'uniform' is flat. "
            "All shapes are normalized to hit --spectrum-trace."
        ),
    )

    parser.add_argument(
        "--spectrum-exponent",
        type=float,
        default=1.0,
        help=(
            "Eigenvalue spectrum exponent p for lambda_i ~ 1/i^p (i=1..K). "
            "Higher p means a steeper spectrum (more ill-conditioned)."
        ),
    )
    parser.add_argument(
        "--spectrum-active",
        type=int,
        default=None,
        help=(
            "If set, only the first K eigenvalues are non-zero (others are 0). "
            "This simulates inactive parameters."
        ),
    )

    # Developer prediction (in the eigenbasis): a wrong spectrum shape parameterization
    # that typically *overestimates tail curvature*.
    parser.add_argument(
        "--dev-spectrum-exponent",
        type=float,
        default=None,
        help=(
            "Developer's predicted exponent p_dev for the power/gamma spectrum. "
            "Defaults to --spectrum-exponent. For a true power spectrum, choosing "
            "p_dev < p_true makes the tail heavier (larger lambdas) and thus "
            "overestimates tail curvature."
        ),
    )
    parser.add_argument(
        "--dev-linear-tilt",
        type=float,
        default=1.0,
        help=(
            "Developer's linear-spectrum misprediction factor (only for --spectrum-shape linear). "
            "This continuously tilts/rotates the canonical linear weights about their midpoint while preserving trace. "
            "1.0 keeps the true slope; <1 flattens (heavier tail / larger lambdas); >1 steepens."
        ),
    )
    parser.add_argument(
        "--dev-log-sigma",
        type=float,
        default=0.0,
        help=(
            "Stddev sigma_dev for i.i.d. lognormal noise applied to the developer's "
            "predicted eigen-spectrum on the active subspace: "
            "lambda_dev_i *= exp(sigma_dev * z_i), z_i~N(0,1). "
            "After applying noise, lambda_dev is renormalized to preserve the "
            "trace/budget (--spectrum-trace)."
        ),
    )
    parser.add_argument(
        "--dev-seed",
        type=int,
        default=None,
        help=(
            "Seed for developer spectrum noise (defaults to --seed). Only used when "
            "--dev-log-sigma > 0."
        ),
    )
    parser.add_argument(
        "--spectrum-trace",
        type=float,
        default=None,
        help=(
            "Target sum of eigenvalues trace(G). Implies elo(0,...,0) = -0.5*trace(G) "
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
            "trace(G)=2*drop. Mutually exclusive with --spectrum-trace."
        ),
    )
    parser.add_argument(
        "--u-identity",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, uses U=I (no rotation / no coupling between axes).",
    )
    parser.add_argument(
        "--u-seed",
        type=int,
        default=None,
        help="Seed for generating the orthonormal rotation U (defaults to --seed).",
    )
    return parser


def _validate_elo_calibration(
    *,
    expected_peak: Sequence[float],
    objective_function: ObjectiveFn,
    expected_lower_bounds_elo: float,
) -> None:
    num_params = len(expected_peak)
    corner = tuple([0.0 for _ in range(num_params)])

    elo_at_corner = float(objective_function(np.asarray(corner, dtype=np.float64)))
    elo_at_peak = float(objective_function(np.asarray(expected_peak, dtype=np.float64)))

    if not np.isfinite(elo_at_corner) or (
        abs(elo_at_corner - float(expected_lower_bounds_elo)) > ELO_CALIBRATION_EPS
    ):
        msg = (
            f"Elo calibration broken: elo(corner)={elo_at_corner}"
            f" (expected {float(expected_lower_bounds_elo)})"
        )
        raise RuntimeError(msg)

    if not np.isfinite(elo_at_peak) or abs(elo_at_peak - ELO_PEAK) > ELO_PEAK_EPS:
        msg = f"Elo calibration broken: elo(peak)={elo_at_peak} (expected {ELO_PEAK})"
        raise RuntimeError(msg)


def _is_interactive_matplotlib_backend(backend: str) -> bool:
    """Return True if the current Matplotlib backend supports GUI windows."""
    backend_l = backend.lower()
    interactive = {
        name.lower()
        for name in backend_registry.list_builtin(BackendFilter.INTERACTIVE)
    }
    return backend_l in interactive


def main() -> int:  # noqa: C901, PLR0912, PLR0915
    """CLI entry point."""
    _configure_logging()

    parser = _build_parser()
    args = parser.parse_args()

    num_params = int(args.num_params)
    if num_params < 1:
        msg = "--num-params must be >= 1"
        raise SystemExit(msg)

    batch_size_pairs = int(args.batch_size_pairs)
    num_batches = int(args.num_batches)
    seed = int(args.seed)
    r_end = float(args.r_end)

    if not np.isfinite(r_end) or r_end <= 0.0:
        msg = "--r-end must be a positive finite float"
        raise SystemExit(msg)

    if args.spectrum_trace is not None and (
        (not np.isfinite(float(args.spectrum_trace)))
        or float(args.spectrum_trace) <= 0.0
    ):
        msg = "--spectrum-trace must be a positive finite float"
        raise SystemExit(msg)
    if args.lower_bounds_elo_drop is not None and (
        (not np.isfinite(float(args.lower_bounds_elo_drop)))
        or float(args.lower_bounds_elo_drop) <= 0.0
    ):
        msg = "--lower-bounds-elo-drop must be a positive finite float"
        raise SystemExit(msg)
    if args.spectrum_trace is not None and args.lower_bounds_elo_drop is not None:
        msg = "Use only one of --spectrum-trace or --lower-bounds-elo-drop"
        raise SystemExit(msg)

    spectrum_trace = (
        float(args.spectrum_trace)
        if args.spectrum_trace is not None
        else (
            2.0 * float(args.lower_bounds_elo_drop)
            if args.lower_bounds_elo_drop is not None
            else float(DEFAULT_TRACE_G)
        )
    )

    u_seed = int(args.u_seed) if args.u_seed is not None else int(seed)

    spectrum_shape = str(args.spectrum_shape)
    spectrum_params = SpectrumParams(
        shape=str(spectrum_shape),
        exponent=float(args.spectrum_exponent),
        active=(
            int(args.spectrum_active) if args.spectrum_active is not None else None
        ),
        trace=float(spectrum_trace),
        use_identity_u=bool(args.u_identity),
        u_seed=int(u_seed),
    )
    objective_model = _make_spectrum_objective(
        num_params=num_params,
        params=spectrum_params,
    )
    objective_fn = objective_model.objective

    # Plot bounds: used only for contour axes; objective/optimizer are unbounded.
    plot_bounds: list[tuple[float, float]] = [
        PLOT_BOUNDS_DEFAULT for _ in range(num_params)
    ]

    # Peak at (100, 100, ..., 100).
    expected_peak = tuple([X_PEAK for _ in range(num_params)])

    expected_lower_bounds_elo = -0.5 * float(objective_model.trace_g)
    _validate_elo_calibration(
        expected_peak=expected_peak,
        objective_function=objective_fn,
        expected_lower_bounds_elo=float(expected_lower_bounds_elo),
    )

    # Start point (CLI).
    if args.init_elo_drop_from_peak is not None:
        init = _init_vector_for_diagonal_elo_drop(
            num_params=num_params,
            elo_drop_from_peak=float(args.init_elo_drop_from_peak),
            trace_g=float(objective_model.trace_g),
        )
        logger.info(
            "Init mode: diagonal elo-drop-from-peak=%.6g -> x_i=%.6g",
            float(args.init_elo_drop_from_peak),
            float(init[0]) if len(init) > 0 else float("nan"),
        )
    else:
        init = _parse_init_vector(str(args.init), num_params)

    logger.info(
        "Starting point: %s | engine_elo: %.6g",
        _format_params(init),
        float(objective_fn(np.asarray(init, dtype=np.float64))),
    )
    logger.info(
        "Expected peak (maximum): %s | engine_elo: %.6g",
        _format_params(expected_peak),
        float(objective_fn(np.asarray(expected_peak, dtype=np.float64))),
    )
    logger.info(
        (
            "Objective: quadratic spectrum | trace(G)=%.6g | "
            "shape=%s | exponent=%.6g | "
            "active=%s | "
            "U=identity=%s | u_seed=%s | "
            "elo_lower_bounds=%.6g"
        ),
        float(objective_model.trace_g),
        str(spectrum_shape),
        float(args.spectrum_exponent),
        (int(args.spectrum_active) if args.spectrum_active is not None else "all"),
        bool(args.u_identity),
        int(u_seed),
        float(expected_lower_bounds_elo),
    )

    games = MatchSimulator(
        objective_fn,
        batch_size_pairs=batch_size_pairs,
        rng=random.Random(seed),  # noqa: S311
    )

    g_diag = np.diag(objective_model.g_matrix)

    dev_exponent = (
        float(args.dev_spectrum_exponent)
        if args.dev_spectrum_exponent is not None
        else float(args.spectrum_exponent)
    )
    dev_linear_tilt = float(args.dev_linear_tilt)
    if str(spectrum_shape) != "linear" and abs(dev_linear_tilt - 1.0) > 0.0:
        msg = "--dev-linear-tilt is only valid when --spectrum-shape linear"
        raise SystemExit(msg)

    if str(spectrum_shape) == "linear":
        active_k = _spectrum_active_dim(
            n_total=num_params,
            active_k=(
                int(args.spectrum_active) if args.spectrum_active is not None else None
            ),
        )
        w_true = _spectrum_raw_weights(
            k=int(active_k),
            shape="linear",
            uniform=False,
            exponent=0.0,
        )
        mid_idx = (int(active_k) - 1) // 2
        w_mid = float(w_true[int(mid_idx)])
        w_dev = w_mid + float(dev_linear_tilt) * (w_true - w_mid)
        # Keep the dev spectrum strictly positive on the active subspace.
        w_dev = np.maximum(w_dev, 1e-12)
        w_sum = float(np.sum(w_dev))
        eigvals_dev = w_dev * (float(spectrum_trace) / w_sum)
        if int(active_k) < int(num_params):
            eigvals_dev = np.concatenate(
                [
                    eigvals_dev,
                    np.zeros((int(num_params) - int(active_k),), dtype=np.float64),
                ],
                axis=0,
            )
    else:
        eigvals_dev = _make_spectrum_eigenvalues(
            num_params=num_params,
            shape=str(spectrum_shape),
            exponent=float(dev_exponent),
            active=(
                int(args.spectrum_active) if args.spectrum_active is not None else None
            ),
            trace_target=float(spectrum_trace),
        )

    dev_log_sigma = float(args.dev_log_sigma)
    dev_seed = int(args.dev_seed) if args.dev_seed is not None else int(seed)
    if dev_log_sigma > 0.0:
        eig_noisy = np.asarray(eigvals_dev, dtype=np.float64).reshape(-1).copy()
        active = eig_noisy > 0.0
        if np.any(active):
            dev_rng = np.random.default_rng(np.uint64(int(dev_seed)))
            z = dev_rng.standard_normal(size=eig_noisy.shape)
            eig_noisy[active] = eig_noisy[active] * np.exp(dev_log_sigma * z[active])

            sum_active = float(np.sum(eig_noisy[active]))
            if np.isfinite(sum_active) and sum_active > 0.0:
                eig_noisy[active] = eig_noisy[active] * (
                    float(spectrum_trace) / sum_active
                )
                eigvals_dev = eig_noisy

    c_end_arr, c_shape_arr = _c_end_vec_from_dev_eigenbasis(
        c_diag_elo_drop=float(args.c_diag_elo_drop),
        g_diag_true=np.asarray(g_diag, dtype=np.float64),
        u_matrix=np.asarray(objective_model.u_matrix, dtype=np.float64),
        eigvals_dev=np.asarray(eigvals_dev, dtype=np.float64),
    )
    c_end_vec = [float(v) for v in c_end_arr]
    diag_drop_calibrated = _diag_elo_drop_at_peak_from_c_end(
        c_end=np.asarray(c_end_arr, dtype=np.float64),
        g_diag=np.asarray(g_diag, dtype=np.float64),
    )
    c_shape_pos = np.asarray(c_shape_arr, dtype=np.float64)
    c_shape_pos = c_shape_pos[c_shape_pos > 0]
    shape_min = float(np.min(c_shape_pos)) if c_shape_pos.size > 0 else 0.0
    shape_max = float(np.max(c_shape_pos)) if c_shape_pos.size > 0 else 0.0
    logger.info(
        (
            "c_end: dev-eigenbasis | dev_exponent=%.6g | dev_linear_tilt=%s | "
            "dev_log_sigma=%.6g | dev_seed=%s | "
            "diag_drop calibrated=%.6g | c_shape_minmax=%s"
        ),
        float(dev_exponent),
        (f"{float(dev_linear_tilt):.6g}" if str(spectrum_shape) == "linear" else "-"),
        float(dev_log_sigma),
        (int(dev_seed) if dev_log_sigma > 0.0 else "-"),
        float(diag_drop_calibrated),
        _format_minmax(shape_min, shape_max),
    )

    optimizer = SPSAOptimizer(
        games,
        num_params=num_params,
        num_batches=num_batches,
        c_diag_elo_drop=float(args.c_diag_elo_drop),
        r_end=float(r_end),
        c_end=c_end_vec,
        rng=random.Random(seed + 1),  # noqa: S311
        np_rng=np.random.default_rng(seed + 2),
    )

    total_pairs = int(num_batches) * int(batch_size_pairs)
    a_frac = float(optimizer.a_stability) / float(max(total_pairs, 1))
    logger.info(
        "Schedule: alpha=%.6g gamma=%.6g A=%.6g (%.3g of total_pairs=%s)",
        float(optimizer.alpha),
        float(optimizer.gamma),
        float(optimizer.a_stability),
        a_frac,
        total_pairs,
    )
    logger.info("Update scaling: grad_signal = net_score / 2")

    preview_dims = min(2, num_params)
    c_preview = tuple(float(v) for v in optimizer.c_end_vec[:preview_dims])
    c_suffix = " ..." if num_params > preview_dims else ""

    trace_g = float(objective_model.trace_g)
    c_vec = np.asarray(optimizer.c_end_vec, dtype=np.float64)
    c_rms = float(np.sqrt(float(np.mean(c_vec * c_vec)))) if c_vec.size > 0 else 0.0
    c_min = float(np.min(c_vec)) if c_vec.size > 0 else 0.0
    c_max = float(np.max(c_vec)) if c_vec.size > 0 else 0.0

    # True expected diagonal/SPSA Elo drop at the peak for vector c_end.
    # For delta_i in {-1,+1}: E[ΔE_diag] = 0.5*(1/X_SCALE^2)*sum_i diag(G)_i*c_i^2.
    achieved_diag_drop = float(
        0.5
        * float(np.sum(np.asarray(g_diag, dtype=np.float64) * (c_vec * c_vec)))
        / (float(X_SCALE) * float(X_SCALE)),
    )

    # Effective dimension in the coordinate system SPSA operates in.
    # Participation ratio of diagonal curvature (depends on U and the spectrum).
    diag_sum = float(np.sum(g_diag))
    diag_sq_sum = float(np.sum(g_diag * g_diag))
    n_eff_diag = (
        (diag_sum * diag_sum) / diag_sq_sum if diag_sq_sum > 0.0 else float(num_params)
    )
    n_eff_diag = float(min(float(num_params), max(1.0, float(n_eff_diag))))
    logger.info(
        "Diag participation ratio: N_eff_diag=%.6g",
        float(n_eff_diag),
    )

    # Spectrum stiffness metrics (eigenvalues, not diagonal entries).
    eig = np.asarray(objective_model.eigvals, dtype=np.float64).reshape(-1)
    active_eig = eig[eig > 0.0]
    if active_eig.size > 0:
        lam_max = float(np.max(active_eig))
        lam_min = float(np.min(active_eig))
        cond = float(lam_max / lam_min) if lam_min > 0.0 else float("inf")
        n_stiff = float(trace_g / lam_max) if lam_max > 0.0 else float("inf")
    else:
        lam_max = 0.0
        lam_min = 0.0
        cond = float("inf")
        n_stiff = float("inf")

    logger.info(
        (
            "Spectrum metrics: lambda_max=%.6g | lambda_min_active=%.6g | "
            "cond=%.6g | trace/lambda_max=%.6g"
        ),
        lam_max,
        lam_min,
        cond,
        n_stiff,
    )

    achieved_axis_gaps = [
        0.5 * (float(c) / float(X_SCALE)) ** 2 * float(g_diag[i])
        for i, c in enumerate(optimizer.c_end_vec)
    ]
    achieved_axis_min = float(min(achieved_axis_gaps))
    achieved_axis_max = float(max(achieved_axis_gaps))

    logger.info(
        (
            "c_end preview (first %s dims): %s%s | c_rms=%.6g | c_minmax=%s | "
            "achieved diag drop=%.6g | implied axis drop=%s"
        ),
        preview_dims,
        c_preview,
        c_suffix,
        c_rms,
        _format_minmax(c_min, c_max),
        achieved_diag_drop,
        _format_minmax(achieved_axis_min, achieved_axis_max),
    )

    # Phi-space effective curvature (theta = c * phi). This is the curvature
    # SPSA operates against when using a single r_end in phi.
    h_phi_diag = (
        np.asarray(g_diag, dtype=np.float64)
        * (c_vec * c_vec)
        / (float(X_SCALE) * float(X_SCALE))
    )
    h_phi_sum = float(np.sum(h_phi_diag))
    h_phi_sum2 = float(np.sum(h_phi_diag * h_phi_diag))
    if h_phi_sum > 0.0 and h_phi_sum2 > 0.0:
        n_eff_phi = float((h_phi_sum * h_phi_sum) / h_phi_sum2)
    else:
        n_eff_phi = float("nan")

    logger.info(
        "Phi curvature participation ratio: N_eff_phi=%.6g",
        n_eff_phi,
    )

    # First-step scale diagnostic: uses the real schedule values and the match model.
    x0 = np.asarray([float(v) for v in init], dtype=np.float64)
    c_k_l2, r_k_1, net_mean, net_rms, dx_l2_rms = _estimate_first_update_l2_scale(
        x0=x0,
        games=games,
        optimizer=optimizer,
        seed=seed + 999,
    )
    dx_rms_per_param = dx_l2_rms / float(np.sqrt(float(max(num_params, 1))))
    logger.info(
        (
            "First-step scale est (pair_index=1): ||c_k||_2=%.6g | r_k=%.6g | "
            "net_score mean=%.6g rms=%.6g | ||Δx||_2 rms=%.6g (per-param rms=%.6g)"
        ),
        c_k_l2,
        r_k_1,
        net_mean,
        net_rms,
        dx_l2_rms,
        dx_rms_per_param,
    )

    # Near-peak infinite-time noise-drop diagnostic at fixed (c_end, r_end).
    elo_drop_inf = _estimate_noise_drop_end_proxy(
        optimizer=optimizer,
        h_phi_trace=h_phi_sum,
        n_eff_phi=n_eff_phi,
    )
    logger.info(
        "Noise-drop est (infinite, phi-space fixed r_end): E_inf=%.6g",
        float(elo_drop_inf),
    )

    run_t0 = time.perf_counter()
    result = optimizer.run(init)
    run_s = max(time.perf_counter() - run_t0, 0.0)
    final_params = tuple(float(v) for v in result.trajectory[-1, :num_params])
    best_params = tuple(float(v) for v in result.best_params)
    final_value = float(result.value_history[-1])
    start_value = float(objective_fn(np.asarray(init, dtype=np.float64)))
    best_value = float(result.best_value)

    final_improvement = final_value - start_value

    final_elo_per_1k_pairs = 1000.0 * final_improvement / float(max(total_pairs, 1))

    best_rms_dist = _rms_normalized_distance_to_peak(best_params)
    final_rms_dist = _rms_normalized_distance_to_peak(final_params)

    # Late-run KPI: summarize the tail of the stored history to reduce
    # sensitivity to single-step noise. Note: history may be downsampled.
    tail_len = int(max(10, round(0.10 * float(result.value_history.size))))
    tail_vals = np.asarray(result.value_history[-tail_len:], dtype=np.float64)
    tail_mean = float(np.mean(tail_vals)) if tail_vals.size > 0 else float("nan")
    tail_median = float(np.median(tail_vals)) if tail_vals.size > 0 else float("nan")

    pairs_per_s = float(total_pairs) / run_s if run_s > 0.0 else float("inf")
    logger.info(
        "KPI: start=%.6g final=%.6g (Δ=%.6g) | final_eff=%.6g Elo/1k pairs",
        start_value,
        final_value,
        final_improvement,
        final_elo_per_1k_pairs,
    )
    logger.info(
        "KPI (tail): mean_last_10%%=%.6g | median_last_10%%=%.6g",
        tail_mean,
        tail_median,
    )
    logger.info(
        "KPI: rms_dist_to_peak final=%.6g | runtime=%.3gs | throughput=%.3g pairs/s",
        final_rms_dist,
        run_s,
        pairs_per_s,
    )

    logger.info(
        "After %s batches (%s pairs/batch, %s pairs total, seed=%s)...",
        num_batches,
        batch_size_pairs,
        total_pairs,
        seed,
    )
    logger.info("Final parameters: %s", _format_params(final_params))
    logger.info("Final engine Elo: %.6g", final_value)

    # Best-so-far is an extreme statistic and is not a fishtest-like objective
    # (fishtest keeps the final parameters). Keep it available for debugging.
    logger.debug("Best parameters (diagnostic): %s", _format_params(best_params))
    logger.debug("Best engine Elo (diagnostic): %.6g", float(best_value))
    logger.debug("Best rms_dist_to_peak (diagnostic): %.6g", float(best_rms_dist))

    if num_params >= MIN_DIMS_FOR_PLOT_2D:
        plot_contours_and_trajectory(
            PlotInputs(
                plot_bounds=plot_bounds,
                g_sub_2d=np.asarray(objective_model.g_matrix[:2, :2], dtype=np.float64),
                trajectory=result.trajectory,
                start=(float(init[0]), float(init[1])),
                expected_peak=(float(expected_peak[0]), float(expected_peak[1])),
                best=(float(best_params[0]), float(best_params[1])),
            ),
        )
    else:
        logger.info("Skipping 2D trajectory plot (num_params < 2)")
    plot_value_vs_pairs(
        pairs_history=result.pairs_history,
        value_history=result.value_history,
        best_pairs=float(result.best_pairs),
        best_value=float(result.best_value),
    )
    plot_active_eigenspectrum(
        eigvals=np.asarray(objective_model.eigvals, dtype=np.float64),
        eigvals_dev=np.asarray(eigvals_dev, dtype=np.float64),
        dev_label=("dev (noisy)" if float(dev_log_sigma) > 0.0 else "dev"),
    )
    backend = str(plt.get_backend())
    if _is_interactive_matplotlib_backend(backend):
        plt.show()
    else:
        logger.info(
            "Non-interactive matplotlib backend (%s); skipping plt.show()",
            backend,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
