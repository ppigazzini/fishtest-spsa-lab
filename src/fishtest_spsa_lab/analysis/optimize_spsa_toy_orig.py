"""Toy SPSA optimizer demo using the vendored pentamodel.

This is an engine-Elo maximization problem:
- Peak (best) is 0 Elo at (100, 100, ..., 100).
- The corner at the lower bounds (default (0, 0, ..., 0)) is -5 Elo.

Supports arbitrary dimensionality; 2D plots always use parameters 0 and 1 and are
skipped if there are fewer than 2 parameters.
"""

from __future__ import annotations

import argparse
import logging
import random
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.backends import BackendFilter, backend_registry
import numpy as np
from numpy.typing import NDArray

from fishtest_spsa_lab.vendor.pentamodel.pentamodel import PentaModel

logger = logging.getLogger(__name__)


PEAK_PARAM: float = 100.0
PARAM_SCALE: float = 100.0
PEAK_ELO: float = 0.0
CORNER_ELO: float = -5.0
DEFAULT_PLOT_BOUNDS: tuple[float, float] = (0.0, 200.0)

DELTA_PROB: float = 0.5
MIN_DIMS_FOR_2D_PLOT: int = 2

CORNER_ELO_EPS: float = 1e-9
PEAK_ELO_EPS: float = 1e-12

# SPSA stability constant A: fishtest typically uses a fraction of total pairs.
DEFAULT_A_STABILITY_FRAC: float = 0.1

# Automatic trajectory downsampling: keep at most this many points (including start).
MAX_STORED_STEPS: int = 10_000


type FloatArray = NDArray[np.float64]
type ObjectiveFn = Callable[[FloatArray], float]


def _c_end_for_elo_gap(
    *,
    elo_gap: float,
    num_params: int,
) -> list[float]:
    """Compute a per-dimension `c_end` vector calibrated to an Elo gap.

    Calibration matches a 1D slice: perturbing exactly one coordinate by +/-c at
    the peak corresponds to approximately `elo_gap` Elo loss.

    For the toy objective this implies c scales like sqrt(N).
    """
    elo_gap_f = float(elo_gap)
    if not np.isfinite(elo_gap_f) or elo_gap_f <= 0.0:
        msg = "c_elo_gap must be a positive finite float"
        raise ValueError(msg)

    corner_mag = float(abs(CORNER_ELO))
    if corner_mag <= 0.0:
        msg = "CORNER_ELO must be non-zero to calibrate c_end"
        raise ValueError(msg)

    if num_params <= 0:
        msg = "num_params must be > 0"
        raise ValueError(msg)

    c_target = float(PARAM_SCALE) * float(
        np.sqrt(elo_gap_f * float(num_params) / corner_mag),
    )

    # No clipping: objective is defined on (-inf, +inf) and bounds are for plotting.
    return [c_target for _ in range(num_params)]


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
    deltas = (x - PEAK_PARAM) / PARAM_SCALE
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


class GameProvider:
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
        if abs(prob_sum - 1.0) > PEAK_ELO_EPS:
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
        games: GameProvider,
        *,
        num_params: int,
        num_batches: int,
        normalize_sqrt_n: bool = True,
        c_elo_gap: float = 0.1,
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
        If None, we set A = DEFAULT_A_STABILITY_FRAC * total_pairs.
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
            self.a_stability = DEFAULT_A_STABILITY_FRAC * float(self.total_pairs)
        else:
            self.a_stability = float(a_stability)
            if not np.isfinite(self.a_stability) or self.a_stability < 0.0:
                msg = "a_stability must be a finite float >= 0"
                raise ValueError(msg)

        if c_end is None:
            # Choose c_end so that a +/-c perturbation in ONE coordinate at the peak
            # corresponds to approximately `c_elo_gap` Elo.
            #
            # For the toy objective:
            #   elo(x) = CORNER_ELO * mean(((x_i - PEAK_PARAM) / PARAM_SCALE)^2)
            self.c_end_vec = _c_end_for_elo_gap(
                elo_gap=float(c_elo_gap),
                num_params=self.num_params,
            )
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

        # Optional SPSA hypercube normalization: divide gradient estimate by sqrt(N).
        self.normalize_sqrt_n = bool(normalize_sqrt_n)
        if self.normalize_sqrt_n:
            self.signal_scale = 1.0 / (float(self.num_params) ** 0.5)
        else:
            self.signal_scale = 1.0

    def _auto_stride(self) -> int:
        # Store at most MAX_STORED_STEPS points (including step 0).
        if self.num_batches + 1 <= MAX_STORED_STEPS:
            return 1
        return int(np.ceil((self.num_batches + 1) / float(MAX_STORED_STEPS)))

    def _schedule(self, iter_local: float) -> tuple[FloatArray, float]:
        c_k = self.c_base / (iter_local**self.gamma)
        r_k = (
            self.r_base
            * (iter_local ** (2.0 * self.gamma))
            / ((self.a_stability + iter_local) ** self.alpha)
        )
        return c_k, float(r_k)

    def _sample_delta(self) -> FloatArray:
        return np.where(
            self.np_rng.random(self.num_params) < DELTA_PROB,
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
        iter_local = float(batch_idx * self.batch_size_pairs + 1)
        c_k, r_k = self._schedule(iter_local)
        base_seed = self.rng.randrange(1 << 63)
        delta = self._sample_delta()

        x_plus = x + c_k * delta
        x_minus = x - c_k * delta
        net_wins = self.games.match_net_wins(x_plus, x_minus, base_seed=base_seed)

        grad_signal = (float(net_wins) * self.signal_scale) / 2.0
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
    """Toy engine Elo for N parameters.

    Quadratic peak at 100 in every dimension.

    Normalization:
    - Each coordinate uses the normalized delta (x_i - 100) / 100.
    - The sum of squared deltas is averaged over N.

    This guarantees:
    - elo([100, 100, ..., 100]) = 0
    - elo([0,   0,   ...,   0]) = -5
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        msg = "toy_elo_function expects a 1D parameter vector"
        raise ValueError(msg)

    n = int(x.shape[0])
    if n <= 0:
        msg = "Need at least 1 parameter"
        raise ValueError(msg)

    deltas = (x - PEAK_PARAM) / PARAM_SCALE
    mean_sq = float(np.mean(deltas * deltas))
    return CORNER_ELO * mean_sq


def plot_contours_and_trajectory(
    *,
    plot_bounds: Sequence[tuple[float, float]],
    trajectory: np.ndarray,
    start: tuple[float, float],
    expected_peak: tuple[float, float],
    best: tuple[float, float],
) -> None:
    """Plot a 2D contour slice (dims 0/1) and overlay the trajectory."""
    if len(plot_bounds) < MIN_DIMS_FOR_2D_PLOT:
        return

    x_low, x_high = float(plot_bounds[0][0]), float(plot_bounds[0][1])
    y_low, y_high = float(plot_bounds[1][0]), float(plot_bounds[1][1])

    grid_n = 200
    xs = np.linspace(x_low, x_high, grid_n)
    ys = np.linspace(y_low, y_high, grid_n)
    x_grid, y_grid = np.meshgrid(xs, ys)
    # Plot only the first two dimensions as a slice where the remaining
    # coordinates are fixed at their peak (100). For the normalized objective,
    # that means the Elo surface is scaled by 5/N.
    n = int(trajectory.shape[1]) if trajectory.ndim == 2 else MIN_DIMS_FOR_2D_PLOT
    z_grid = (CORNER_ELO / float(max(n, 1))) * (
        ((x_grid - PEAK_PARAM) / PARAM_SCALE) ** 2
        + ((y_grid - PEAK_PARAM) / PARAM_SCALE) ** 2
    )

    _fig, ax = plt.subplots(figsize=(9, 7))
    z_min = float(np.percentile(z_grid, 5.0))
    levels = np.linspace(z_min, 0.0, 20)
    ax.contour(x_grid, y_grid, z_grid, levels=levels, linewidths=1.0, alpha=0.9)

    final_xy = (float(trajectory[-1, 0]), float(trajectory[-1, 1]))

    ax.plot(trajectory[:, 0], trajectory[:, 1], "-k", lw=1.5, label="Trajectory")
    ax.scatter([start[0]], [start[1]], c="tab:green", s=80, label="Start")
    ax.scatter(
        [expected_peak[0]],
        [expected_peak[1]],
        c="tab:blue",
        s=80,
        label="Expected peak",
    )
    ax.scatter([best[0]], [best[1]], c="tab:red", s=80, label="Best found")
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
        ax.axhline(PEAK_ELO, color="tab:blue", lw=1.2, alpha=0.9, label="Expected peak")

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
        "--init",
        type=str,
        default="0",
        help=(
            "Starting point. Either a single float (broadcast to all params) "
            "or a comma-separated list of N floats. Default: 0 (origin)."
        ),
    )
    parser.add_argument(
        "--normalize-sqrt-n",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If set, normalizes the SPSA update signal by 1/sqrt(N).",
    )
    parser.add_argument(
        "--c-elo-gap",
        type=float,
        default=0.1,
        help=(
            "Calibrate c_end so that a +/-c perturbation in one coordinate at the peak "
            "is about this many Elo (default: 0.1). Scaling is ~sqrt(N)."
        ),
    )
    return parser


def _validate_elo_calibration(
    *,
    expected_peak: Sequence[float],
) -> None:
    num_params = len(expected_peak)
    corner = tuple([0.0 for _ in range(num_params)])

    corner_elo = float(toy_elo_function(np.asarray(corner, dtype=np.float64)))
    peak_elo = float(toy_elo_function(np.asarray(expected_peak, dtype=np.float64)))

    if not np.isfinite(corner_elo) or abs(corner_elo - CORNER_ELO) > CORNER_ELO_EPS:
        msg = (
            f"Elo calibration broken: elo(corner)={corner_elo} (expected {CORNER_ELO})"
        )
        raise RuntimeError(msg)

    if not np.isfinite(peak_elo) or abs(peak_elo - PEAK_ELO) > PEAK_ELO_EPS:
        msg = f"Elo calibration broken: elo(peak)={peak_elo} (expected {PEAK_ELO})"
        raise RuntimeError(msg)


def _is_interactive_matplotlib_backend(backend: str) -> bool:
    """Return True if the current Matplotlib backend supports GUI windows."""
    backend_l = backend.lower()
    interactive = {
        name.lower()
        for name in backend_registry.list_builtin(BackendFilter.INTERACTIVE)
    }
    return backend_l in interactive


def main() -> int:
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

    # Plot bounds: used only for contour axes; objective/optimizer are unbounded.
    plot_bounds: list[tuple[float, float]] = [
        DEFAULT_PLOT_BOUNDS for _ in range(num_params)
    ]

    # Peak at (100, 100, ..., 100).
    expected_peak = tuple([PEAK_PARAM for _ in range(num_params)])

    _validate_elo_calibration(expected_peak=expected_peak)

    # Start point (CLI).
    init = _parse_init_vector(str(args.init), num_params)

    logger.info(
        "Starting point: %s | engine_elo: %.6g",
        _format_params(init),
        float(toy_elo_function(np.asarray(init, dtype=np.float64))),
    )
    logger.info(
        "Expected peak (maximum): %s | engine_elo: %.6g",
        _format_params(expected_peak),
        float(toy_elo_function(np.asarray(expected_peak, dtype=np.float64))),
    )

    games = GameProvider(
        toy_elo_function,
        batch_size_pairs=batch_size_pairs,
        rng=random.Random(seed),  # noqa: S311
    )

    optimizer = SPSAOptimizer(
        games,
        num_params=num_params,
        num_batches=num_batches,
        normalize_sqrt_n=bool(args.normalize_sqrt_n),
        c_elo_gap=float(args.c_elo_gap),
        r_end=0.001,
        c_end=None,
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

    preview_dims = min(2, num_params)
    c_preview = tuple(float(v) for v in optimizer.c_end_vec[:preview_dims])
    c_suffix = " ..." if num_params > preview_dims else ""

    corner_mag = float(abs(CORNER_ELO))
    c_target = float(PARAM_SCALE) * float(
        np.sqrt(float(args.c_elo_gap) * float(num_params) / corner_mag),
    )
    achieved_gaps = [
        corner_mag * (float(c) / float(PARAM_SCALE)) ** 2 / float(num_params)
        for c in optimizer.c_end_vec
    ]
    achieved_min = float(min(achieved_gaps))
    achieved_max = float(max(achieved_gaps))

    logger.info(
        "c_end preview (first %s dims): %s%s | target=%.6g | achieved Elo gap=%.6g..%.6g",
        preview_dims,
        c_preview,
        c_suffix,
        c_target,
        achieved_min,
        achieved_max,
    )

    run_t0 = time.perf_counter()
    result = optimizer.run(init)
    run_s = max(time.perf_counter() - run_t0, 0.0)
    final_params = tuple(float(v) for v in result.trajectory[-1, :num_params])
    best_params = tuple(float(v) for v in result.best_params)
    final_value = float(result.value_history[-1])
    start_value = float(toy_elo_function(np.asarray(init, dtype=np.float64)))
    best_value = float(result.best_value)

    best_improvement = best_value - start_value
    final_improvement = final_value - start_value
    elo_per_1k_pairs = 1000.0 * best_improvement / float(max(total_pairs, 1))

    best_rms_dist = _rms_normalized_distance_to_peak(best_params)
    final_rms_dist = _rms_normalized_distance_to_peak(final_params)
    max_dd = _max_drawdown(result.value_history)
    record_highs = _count_record_highs(result.value_history)

    pairs_per_s = float(total_pairs) / run_s if run_s > 0.0 else float("inf")
    best_at_frac = float(result.best_pairs) / float(max(total_pairs, 1))

    logger.info(
        "KPI: start=%.6g best=%.6g (Δ=%.6g) final=%.6g (Δ=%.6g)",
        start_value,
        best_value,
        best_improvement,
        final_value,
        final_improvement,
    )
    logger.info(
        "KPI: best@%.0f pairs (%.1f%%) | eff=%.6g Elo/1k pairs | drawdown=%.6g | record_highs=%s",
        float(result.best_pairs),
        100.0 * best_at_frac,
        elo_per_1k_pairs,
        max_dd,
        record_highs,
    )
    logger.info(
        "KPI: rms_dist_to_peak best=%.6g final=%.6g | runtime=%.3gs | throughput=%.3g pairs/s",
        best_rms_dist,
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
    logger.info("Best parameters: %s", _format_params(best_params))
    logger.info("Best engine Elo: %.6g", float(result.best_value))

    if num_params >= MIN_DIMS_FOR_2D_PLOT:
        plot_contours_and_trajectory(
            plot_bounds=plot_bounds,
            trajectory=result.trajectory,
            start=(float(init[0]), float(init[1])),
            expected_peak=(float(expected_peak[0]), float(expected_peak[1])),
            best=(float(best_params[0]), float(best_params[1])),
        )
    else:
        logger.info("Skipping 2D trajectory plot (num_params < 2)")
    plot_value_vs_pairs(
        pairs_history=result.pairs_history,
        value_history=result.value_history,
        best_pairs=float(result.best_pairs),
        best_value=float(result.best_value),
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
