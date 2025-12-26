"""Validate pentanomial sampling noise for fixed or varying Elo differences.

This script simulates batched pentanomial outcomes and tracks:
- EMA asymmetry A and EMA mean outcome (mu)
- Cumulative asymmetry A and cumulative mean outcome (mu)
- Penta coefficient r and gain scale factor

Default behavior (entrypoint `validate-penta`) prints a Monte Carlo summary and
shows charts from a single representative run.

Disable plotting (e.g., headless):
    uv run validate-penta --no-plot

Single-trial mode (plots + totals):
    uv run validate-penta --trials 1

Monte Carlo summary + one-run frequencies (no plots):
    uv run validate-penta --trials 500 --no-plot
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from fishtest_spsa_lab.simulator.config import SPSAConfig
from fishtest_spsa_lab.vendor.pentamodel.pentamodel import PentaModel

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

OUTCOMES = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=float)
POINTS = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=float)
DEBIAS_EPS = 1e-6

# Used only for plot titles.
OPTIMIZER_NAME = "validate-penta"

# Penta gain defaults
BETA_PG = 0.999
MU_WEIGHT = 0.5
R_SMALL = 0.002
R_LARGE = 0.02
MIN_SCALE = 0.2
MAX_SCALE = 1.5


def _mean_var_points_from_probs(p: np.ndarray) -> tuple[float, float]:
    p = np.asarray(p, dtype=float)
    mean_y = float(np.dot(p, POINTS))
    mean_y2 = float(np.dot(p, POINTS * POINTS))
    var_y = float(max(0.0, mean_y2 - mean_y * mean_y))
    return mean_y, var_y


def _score_from_freq(freq: np.ndarray) -> float:
    """Convert pentanomial frequencies to match score in [0, 1]."""
    return float(np.dot(np.asarray(freq, dtype=float), POINTS) / 2.0)


def _start_diff_from_score_oracle(score: float) -> float:
    """Oracle Elo estimate from score, using the vendored pentamodel convention.

    PentaModel.elo_diff_from_score(score) returns opponentElo, so we negate to
    map back to this script's start_diff convention (theta_plus - theta_minus).
    """
    s = float(score)
    eps = 1e-12
    s = float(max(eps, min(1.0 - eps, s)))
    opponent_elo = float(PentaModel.elo_diff_from_score(s))
    return float(-opponent_elo)


def _delo_dscore(score: float) -> float:
    """Compute the derivative of opponent Elo with respect to score.

    For f(s) = 400 * log10(1/s - 1),
        f'(s) = -400 / ln(10) * 1/(s(1-s)).
    """
    s = float(score)
    eps = 1e-12
    s = float(max(eps, min(1.0 - eps, s)))
    return float((-400.0 / math.log(10.0)) * (1.0 / (s * (1.0 - s))))


def _fmt_freq(freq: np.ndarray, *, decimals: int = 6) -> list[float]:
    """Format a 5-vector [LL, DL, DD, WD, WW] as rounded floats for logging."""
    return np.round(np.asarray(freq, dtype=float), decimals).tolist()


def _default_batch_bounds_pairs() -> tuple[int, int]:
    """Return (min_pairs, max_pairs) for async variable batching.

    Mirrors `AsyncSpsaRunner._schedule_job()`:
        tc_ratio = max(1, round(config.tc_ratio))
        batch_size_games = concurrency * 4 * tc_ratio
        batch_size_pairs = max(1, batch_size_games // 2)

    Using `SPSAConfig` defaults keeps this script in sync with
    `worker_concurrency_min/max` and `tc_ratio`.
    """
    config = SPSAConfig()
    tc_ratio = max(1, round(config.tc_ratio))

    min_pairs = max(1, (config.worker_concurrency_min * 4 * tc_ratio) // 2)
    max_pairs = max(1, (config.worker_concurrency_max * 4 * tc_ratio) // 2)
    return int(min_pairs), int(max_pairs)


@dataclass(frozen=True, slots=True)
class Args:
    """Parsed CLI arguments."""

    start_diff: float
    m: int
    min_batch: int
    max_batch: int
    trials: int
    seed: int
    varying: bool
    plot: bool


@dataclass(slots=True)
class SingleRun:
    """Single-run histories used for plotting."""

    steps: list[int]
    asym_ema: list[float]
    mu_ema: list[float]
    asym_cum: list[float]
    mu_cum: list[float]
    penta_coeff: list[float]
    gain_scale: list[float]
    total: np.ndarray
    net_wins: int


def batch_sequence(
    m: int,
    min_b: int,
    max_b: int,
    rng: np.random.Generator,
) -> list[int]:
    """Generate a random batch-size sequence summing to `m` pairs."""
    sizes: list[int] = []
    remaining = m
    while remaining > 0:
        b = int(rng.integers(min_b, max_b + 1))
        b = min(b, remaining)
        sizes.append(b)
        remaining -= b
    return sizes


def penta_stats(p: np.ndarray) -> tuple[float, float]:
    """Return (asymmetry A, mu) for pentanomial distribution p."""
    p_ll, p_dl, _p_dd, p_wd, p_ww = p
    asym = float(abs(p_ww - p_ll) + 0.5 * abs(p_wd - p_dl))
    mu = float(np.dot(p, OUTCOMES))
    return asym, mu


def _debias_ema(p_ema: np.ndarray, beta_prod: float) -> np.ndarray:
    denom = 1.0 - beta_prod
    if denom <= DEBIAS_EPS:
        return p_ema.copy()
    return p_ema / denom


def _gain_scale_from_r(r_value: float) -> float:
    # Linear map from (R_SMALL, R_LARGE) -> (MIN_SCALE, MAX_SCALE),
    # intentionally *not* clamped so experiments can observe extrapolation.
    t = (r_value - R_SMALL) / (R_LARGE - R_SMALL)
    return float(MIN_SCALE + (MAX_SCALE - MIN_SCALE) * t)


def _elo_at_fraction(start_diff: float, frac: float) -> float:
    return float(start_diff * (1.0 - frac))


def _penta_probs(opponent_elo: float) -> np.ndarray:
    return np.asarray(
        PentaModel(opponentElo=opponent_elo).pentanomialProbs,
        dtype=float,
    )


class _SizesDoNotSumToTotalPairsError(AssertionError):
    """Raised when a batch schedule does not sum to the requested total."""

    def __init__(self) -> None:
        super().__init__("Batch sizes must sum to the total number of pairs.")


class _SimulationProcessedWrongTotalError(AssertionError):
    """Raised when a simulation processes a different number of pairs than expected."""

    def __init__(self) -> None:
        super().__init__(
            "Simulation must process exactly the requested number of pairs.",
        )


class _FixedModeProbabilitiesNotInitializedError(AssertionError):
    """Raised when fixed-mode probabilities are unexpectedly missing."""

    def __init__(self) -> None:
        super().__init__(
            "Fixed-mode probabilities must be initialized before sampling.",
        )


class _MinBatchMustBeAtLeastOneError(ValueError):
    """Raised when --min-batch is less than 1."""

    def __init__(self) -> None:
        super().__init__("--min-batch must be >= 1")


class _MaxBatchMustBeAtLeastOneError(ValueError):
    """Raised when --max-batch is less than 1."""

    def __init__(self) -> None:
        super().__init__("--max-batch must be >= 1")


class _MinBatchMustNotExceedMaxBatchError(ValueError):
    """Raised when --min-batch is greater than --max-batch."""

    def __init__(self) -> None:
        super().__init__("--min-batch must be <= --max-batch")


class PentaProbCache:
    """Cache for pentanomial probabilities keyed by rounded opponent Elo."""

    def __init__(self, *, decimals: int = 6) -> None:
        """Create the cache.

        Args:
            decimals: Number of decimals to round opponent Elo values to for caching.

        """
        self._decimals = int(decimals)
        self._cache: dict[float, np.ndarray] = {}

    def probs(self, *, opponent_elo: float) -> np.ndarray:
        """Return cached probabilities for the given opponent Elo."""
        key = float(round(float(opponent_elo), self._decimals))
        if key not in self._cache:
            self._cache[key] = _penta_probs(opponent_elo=key)
        return self._cache[key]


def _expected_freq_for_varying_schedule(
    *,
    start_diff: float,
    m: int,
    sizes: list[int],
    cache: PentaProbCache,
) -> np.ndarray:
    """Deterministic expected frequency for a specific batch schedule.

    Uses the varying Elo schedule (start_diff -> 0 over M pairs) and computes:
        E[freq] = sum_b (b/M) * p_batch(processed_at_batch_start)
    """
    expected_freq = np.zeros(5, dtype=float)
    processed = 0
    denom = float(max(1, m))

    for b in sizes:
        frac = processed / denom
        elo = _elo_at_fraction(start_diff, frac)
        p_batch = cache.probs(opponent_elo=-elo)
        expected_freq += (float(b) / denom) * p_batch
        processed += b

    return expected_freq


def _update_histories(
    *,
    counts: np.ndarray,
    n: int,
    p_ema: np.ndarray,
    beta_prod: float,
    cum_counts: np.ndarray,
) -> tuple[np.ndarray, float, float, float, float, float, float, float]:
    beta_eff = float(BETA_PG**n)
    p_batch = counts.astype(float) / float(max(n, 1))

    p_ema = beta_eff * p_ema + (1.0 - beta_eff) * p_batch
    beta_prod *= beta_eff

    p_hat = _debias_ema(p_ema, beta_prod)
    asym_ema, mu_ema = penta_stats(p_hat)

    cum_counts += counts.astype(float)
    total_cum = float(np.sum(cum_counts))
    p_cum = (cum_counts / total_cum) if total_cum > 0.0 else np.zeros(5, dtype=float)
    asym_cum, mu_cum = penta_stats(p_cum)

    r_value = abs(asym_cum) + float(MU_WEIGHT) * abs(mu_cum)
    scale = _gain_scale_from_r(r_value)

    return p_ema, beta_prod, asym_ema, mu_ema, asym_cum, mu_cum, r_value, scale


def _simulate_single(
    *,
    probs_for_batch: Callable[[int, int], np.ndarray],
    m: int,
    sizes: list[int],
    rng: np.random.Generator,
) -> SingleRun:
    if sum(sizes) != m:
        raise _SizesDoNotSumToTotalPairsError

    steps: list[int] = [0]
    processed = 0

    p_ema = np.zeros(5, dtype=float)
    beta_prod = 1.0
    cum_counts = np.zeros(5, dtype=float)

    asym_ema_hist: list[float] = []
    mu_ema_hist: list[float] = []
    asym_cum_hist: list[float] = []
    mu_cum_hist: list[float] = []
    penta_coeff_hist: list[float] = []
    gain_scale_hist: list[float] = []

    total = np.zeros(5, dtype=int)

    for b in sizes:
        probs = probs_for_batch(processed, m)
        counts = rng.multinomial(b, probs).astype(int)

        total += counts
        processed += b
        steps.append(processed)

        (
            p_ema,
            beta_prod,
            asym_ema,
            mu_ema,
            asym_cum,
            mu_cum,
            r_value,
            scale,
        ) = _update_histories(
            counts=counts,
            n=b,
            p_ema=p_ema,
            beta_prod=beta_prod,
            cum_counts=cum_counts,
        )

        asym_ema_hist.append(asym_ema)
        mu_ema_hist.append(mu_ema)
        asym_cum_hist.append(asym_cum)
        mu_cum_hist.append(mu_cum)
        penta_coeff_hist.append(r_value)
        gain_scale_hist.append(scale)

    if processed != m:
        raise _SimulationProcessedWrongTotalError

    net_wins = int((2 * total[4] + total[3]) - (2 * total[0] + total[1]))
    return SingleRun(
        steps=steps,
        asym_ema=asym_ema_hist,
        mu_ema=mu_ema_hist,
        asym_cum=asym_cum_hist,
        mu_cum=mu_cum_hist,
        penta_coeff=penta_coeff_hist,
        gain_scale=gain_scale_hist,
        total=total,
        net_wins=net_wins,
    )


def _plot_asym_mu(
    step_array: np.ndarray,
    asym_ema: np.ndarray,
    asym_cum: np.ndarray,
    mu_ema: np.ndarray,
    mu_cum: np.ndarray,
) -> None:
    max_k = max(0, step_array.size - 1)
    if max_k <= 0:
        return

    if (
        asym_ema.size == 0
        and asym_cum.size == 0
        and mu_ema.size == 0
        and mu_cum.size == 0
    ):
        return

    fig, (ax_asym, ax_mu) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    k = min(asym_ema.size, max_k)
    x_vals = step_array[1 : k + 1]
    ax_asym.plot(x_vals, asym_ema[:k], label="EMA Asymmetry A")

    k = min(asym_cum.size, max_k)
    x_vals = step_array[1 : k + 1]
    ax_asym.plot(x_vals, asym_cum[:k], label="Cumulative Asymmetry A")

    ax_asym.set_ylabel("Asymmetry")
    ax_asym.legend(loc="best")
    ax_asym.grid(visible=True)

    k = min(mu_ema.size, max_k)
    x_vals = step_array[1 : k + 1]
    ax_mu.plot(x_vals, mu_ema[:k], label="EMA Mean outcome mu")

    k = min(mu_cum.size, max_k)
    x_vals = step_array[1 : k + 1]
    ax_mu.plot(x_vals, mu_cum[:k], label="Cumulative Mean outcome mu")

    ax_mu.set_xlabel("Pairs processed")
    ax_mu.set_ylabel("Mean outcome")
    ax_mu.legend(loc="best")
    ax_mu.grid(visible=True)

    fig.suptitle(f"Pentanomial asymmetry/mean (optimizer={OPTIMIZER_NAME})")
    plt.tight_layout()
    plt.show()


def _plot_coeff_scale(
    step_array: np.ndarray,
    coeff: np.ndarray,
    scale: np.ndarray,
) -> None:
    max_k = max(0, step_array.size - 1)
    if max_k <= 0:
        return

    if coeff.size == 0 and scale.size == 0:
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    if coeff.size > 0:
        k = min(coeff.size, max_k)
        x_vals = step_array[1 : k + 1]
        ax.plot(x_vals, coeff[:k], label="Cumulative penta r")

    if scale.size > 0:
        k = min(scale.size, max_k)
        x_vals = step_array[1 : k + 1]
        ax.plot(x_vals, scale[:k], label="Gain scale factor")

    ax.set_xlabel("Pairs processed")
    ax.set_ylabel("Value")
    ax.legend(loc="best")
    ax.grid(visible=True)
    fig.suptitle(f"Penta coefficient / gain scale (optimizer={OPTIMIZER_NAME})")
    plt.tight_layout()
    plt.show()


def run_single(
    args: Args,
    *,
    sizes: list[int],
    cache: PentaProbCache,
    expected_freq_schedule: np.ndarray | None,
    log_theory: bool = True,
) -> None:
    """Run one simulation.

    Always prints the single-run frequencies. Charts are shown only when
    `args.plot` is enabled.
    """
    logger.info(
        "Single-run workload: pairs=%d, batches=%d (min=%d, max=%d)",
        args.m,
        len(sizes),
        min(sizes) if sizes else 0,
        max(sizes) if sizes else 0,
    )
    rng = np.random.default_rng(args.seed)

    if args.varying:

        def probs_for_batch(processed: int, m: int) -> np.ndarray:
            frac = processed / float(max(1, m))
            elo = _elo_at_fraction(args.start_diff, frac)
            return cache.probs(opponent_elo=-elo)

    else:
        probs_fixed = cache.probs(opponent_elo=-args.start_diff)

        def probs_for_batch(_processed: int, _m: int) -> np.ndarray:
            return probs_fixed

    if args.varying and log_theory:
        freq_start = cache.probs(opponent_elo=-args.start_diff)
        freq_end = cache.probs(opponent_elo=-0.0)
        logger.info(
            "Theory freq start [LL, DL, DD, WD, WW]: %s",
            _fmt_freq(freq_start, decimals=8),
        )
        logger.info(
            "Theory freq end   [LL, DL, DD, WD, WW]: %s",
            _fmt_freq(freq_end, decimals=8),
        )

        if expected_freq_schedule is not None:
            logger.info(
                (
                    "Expected freq (representative schedule, seed=%d) "
                    "[LL, DL, DD, WD, WW]: %s"
                ),
                args.seed,
                _fmt_freq(expected_freq_schedule, decimals=6),
            )

    res = _simulate_single(
        probs_for_batch=probs_for_batch,
        m=args.m,
        sizes=sizes,
        rng=rng,
    )

    # Frequency-only logging: theory (optional), then the single empirical realization.
    if log_theory and not args.varying:
        freq_theory = cache.probs(opponent_elo=-args.start_diff)
        logger.info(
            "Theory freq [LL, DL, DD, WD, WW]: %s",
            _fmt_freq(freq_theory, decimals=8),
        )

    total_pairs = int(np.sum(res.total))
    freq_single = (
        res.total.astype(float) / float(total_pairs)
        if total_pairs > 0
        else np.zeros(5, dtype=float)
    )
    logger.info(
        "Single-run freq [LL, DL, DD, WD, WW]: %s",
        _fmt_freq(freq_single, decimals=6),
    )

    if args.plot:
        step_array = np.asarray(res.steps, dtype=float)
        _plot_asym_mu(
            step_array,
            np.asarray(res.asym_ema, dtype=float),
            np.asarray(res.asym_cum, dtype=float),
            np.asarray(res.mu_ema, dtype=float),
            np.asarray(res.mu_cum, dtype=float),
        )
        _plot_coeff_scale(
            step_array,
            np.asarray(res.penta_coeff, dtype=float),
            np.asarray(res.gain_scale, dtype=float),
        )


def _monte_carlo_totals(
    *,
    args: Args,
    rng_master: np.random.Generator,
    cache: PentaProbCache,
) -> np.ndarray:
    totals = np.zeros((args.trials, 5), dtype=int)

    probs_fixed: np.ndarray | None = None
    if not args.varying:
        probs_fixed = cache.probs(opponent_elo=-args.start_diff)

    for t in range(args.trials):
        rng = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
        sizes = batch_sequence(args.m, args.min_batch, args.max_batch, rng)
        if sum(sizes) != args.m:
            raise _SizesDoNotSumToTotalPairsError

        total = np.zeros(5, dtype=int)
        if not args.varying:
            probs = probs_fixed
            if probs is None:
                raise _FixedModeProbabilitiesNotInitializedError
            for b in sizes:
                total += rng.multinomial(b, probs)
        else:
            processed = 0
            for b in sizes:
                frac = processed / float(max(1, args.m))
                elo = _elo_at_fraction(args.start_diff, frac)
                probs = cache.probs(opponent_elo=-elo)
                total += rng.multinomial(b, probs)
                processed += b

        totals[t, :] = total

    return totals


def _log_expected_freq_schedule(
    *,
    args: Args,
    expected_freq_schedule: np.ndarray | None,
) -> None:
    if expected_freq_schedule is None:
        return
    logger.info(
        ("Expected freq (representative schedule, seed=%d) [LL, DL, DD, WD, WW]: %s"),
        args.seed,
        _fmt_freq(expected_freq_schedule, decimals=6),
    )


def _log_oracle_elo_noise(*, args: Args, totals: np.ndarray) -> None:
    """Log Elo noise for the oracle score->Elo estimator."""
    freq_trials = totals.astype(float) / float(args.m)
    score_hats = (freq_trials @ POINTS) / 2.0
    elo_hats = np.array(
        [_start_diff_from_score_oracle(float(s)) for s in score_hats],
        dtype=float,
    )

    elo_mean = float(np.mean(elo_hats))
    elo_std = float(np.std(elo_hats))
    elo_p05, elo_p50, elo_p95 = np.percentile(elo_hats, [5.0, 50.0, 95.0]).tolist()

    logger.info(
        (
            "Elo_hat (oracle score->Elo): mean=%.4f, std=%.4f, "
            "p05=%.4f, p50=%.4f, p95=%.4f"
        ),
        elo_mean,
        elo_std,
        float(elo_p05),
        float(elo_p50),
        float(elo_p95),
    )

    score_mean = float(np.mean(score_hats))
    score_std = float(np.std(score_hats))
    d_at_mean = _delo_dscore(score_mean)
    sigma_elo_from_score = abs(d_at_mean) * score_std
    logger.info(
        ("score_hat std=%.6f; |dElo/dScore|@mean=%.2f (score=%.6f) => sigma_Elo≈%.4f"),
        score_std,
        abs(float(d_at_mean)),
        score_mean,
        float(sigma_elo_from_score),
    )


def _log_delta_method_theory(
    *,
    args: Args,
    cache: PentaProbCache,
    expected_freq_schedule: np.ndarray | None,
    rep_sizes: list[int] | None,
) -> None:
    """Log delta-method theory for the oracle estimator."""
    if not args.varying:
        elo_true = float(args.start_diff)
        p_true = cache.probs(opponent_elo=-elo_true)
        mean_y, var_y = _mean_var_points_from_probs(p_true)
        score_true = float(mean_y / 2.0)
        sigma_score = math.sqrt(var_y / (4.0 * float(args.m)))
        d = _delo_dscore(score_true)
        sigma_elo = abs(d) * sigma_score
        logger.info(
            "Theory Elo_hat std (delta method, fixed): %.4f (score=%.6f)",
            float(sigma_elo),
            float(score_true),
        )
        return

    if expected_freq_schedule is None or rep_sizes is None or not rep_sizes:
        return

    score_sched = _score_from_freq(expected_freq_schedule)
    processed = 0
    var_y_weighted = 0.0
    denom = float(max(1, args.m))
    for b in rep_sizes:
        frac = processed / denom
        elo = _elo_at_fraction(args.start_diff, frac)
        p_b = cache.probs(opponent_elo=-elo)
        _mean_y_b, var_y_b = _mean_var_points_from_probs(p_b)
        var_y_weighted += (float(b) / denom) * float(var_y_b)
        processed += int(b)

    sigma_score = math.sqrt(var_y_weighted / (4.0 * float(args.m)))
    d = _delo_dscore(score_sched)
    sigma_elo = abs(d) * sigma_score
    logger.info(
        (
            "Theory Elo_hat std (delta method, varying, rep schedule): %.4f "
            "(score_equiv=%.6f)"
        ),
        float(sigma_elo),
        float(score_sched),
    )


def run_monte_carlo(
    args: Args,
    *,
    cache: PentaProbCache,
    expected_freq_schedule: np.ndarray | None = None,
    rep_sizes: list[int] | None = None,
) -> None:
    """Run many trials and print a summary."""
    rng_master = np.random.default_rng(args.seed)
    totals = _monte_carlo_totals(args=args, rng_master=rng_master, cache=cache)

    tot_mean = np.mean(totals, axis=0)
    tot_std = np.std(totals, axis=0)

    freq_mean = tot_mean / float(args.m)
    freq_std = tot_std / float(args.m)

    nets = (2 * totals[:, 4] + totals[:, 3]) - (2 * totals[:, 0] + totals[:, 1])

    logger.info(
        "Summary: start_diff=%.3f, pairs=%d, trials=%d",
        args.start_diff,
        args.m,
        args.trials,
    )
    logger.info("Mode: %s", "varying -> 0" if args.varying else "fixed")

    # Frequency-only logging: theory first, then MC mean ± std.
    if args.varying:
        freq_start = cache.probs(opponent_elo=-args.start_diff)
        freq_end = cache.probs(opponent_elo=-0.0)
        logger.info(
            "Theory freq start [LL, DL, DD, WD, WW]: %s",
            _fmt_freq(freq_start, decimals=8),
        )
        logger.info(
            "Theory freq end   [LL, DL, DD, WD, WW]: %s",
            _fmt_freq(freq_end, decimals=8),
        )

        _log_expected_freq_schedule(
            args=args,
            expected_freq_schedule=expected_freq_schedule,
        )
    else:
        freq_theory = cache.probs(opponent_elo=-args.start_diff)
        logger.info(
            "Theory freq [LL, DL, DD, WD, WW]: %s",
            _fmt_freq(freq_theory, decimals=8),
        )

    logger.info(
        "MC freq [LL, DL, DD, WD, WW]: %s ± %s",
        _fmt_freq(freq_mean, decimals=6),
        _fmt_freq(freq_std, decimals=6),
    )

    net_mean = float(np.mean(nets))
    net_std = float(np.std(nets))
    snr = abs(net_mean) / (net_std + 1e-12) if net_std > 0.0 else math.inf

    logger.info("net_wins: mean=%.2f, std=%.2f", net_mean, net_std)
    logger.info("empirical SNR for net_wins across trials: %.3f", float(snr))

    _log_oracle_elo_noise(args=args, totals=totals)
    _log_delta_method_theory(
        args=args,
        cache=cache,
        expected_freq_schedule=expected_freq_schedule,
        rep_sizes=rep_sizes,
    )


def parse_args(argv: list[str] | None = None) -> Args:
    """Parse CLI args into an `Args` dataclass."""
    default_min_batch, default_max_batch = _default_batch_bounds_pairs()

    parser = argparse.ArgumentParser(
        description="Validate pentanomial sampling noise (batched).",
    )
    parser.add_argument(
        "--start-diff",
        type=float,
        default=-0.5,
        help="Starting Elo diff (theta_plus - theta_minus).",
    )
    parser.add_argument(
        "--M",
        type=int,
        default=10_000,
        help="Total number of pairs to simulate (across all random batches).",
    )
    parser.add_argument(
        "--min-batch",
        type=int,
        default=default_min_batch,
        help=(
            "Minimum batch size (pairs). Default matches simulator async "
            "variable-batch min under SPSAConfig defaults."
        ),
    )
    parser.add_argument(
        "--max-batch",
        type=int,
        default=default_max_batch,
        help=(
            "Maximum batch size (pairs). Default matches simulator async "
            "variable-batch max under SPSAConfig defaults."
        ),
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1_000,
        help="Number of Monte Carlo trials.",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    parser.add_argument(
        "--varying",
        action="store_true",
        help="If set, Elo diff moves linearly to 0 over M.",
    )
    parser.add_argument(
        "--plot",
        dest="plot",
        action="store_true",
        default=True,
        help="Show charts (default).",
    )
    parser.add_argument(
        "--no-plot",
        dest="plot",
        action="store_false",
        help="Do not show charts.",
    )

    ns = parser.parse_args(argv)

    min_batch = int(ns.min_batch)
    max_batch = int(ns.max_batch)
    if min_batch < 1:
        raise _MinBatchMustBeAtLeastOneError
    if max_batch < 1:
        raise _MaxBatchMustBeAtLeastOneError
    if min_batch > max_batch:
        raise _MinBatchMustNotExceedMaxBatchError

    return Args(
        start_diff=float(ns.start_diff),
        m=int(ns.M),
        min_batch=min_batch,
        max_batch=max_batch,
        trials=int(ns.trials),
        seed=int(ns.seed),
        varying=bool(ns.varying),
        plot=bool(ns.plot),
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    args = parse_args(argv)

    cache = PentaProbCache(decimals=6)

    rep_rng = np.random.default_rng(args.seed)
    rep_sizes = batch_sequence(args.m, args.min_batch, args.max_batch, rep_rng)

    expected_freq_schedule: np.ndarray | None = None
    if args.varying:
        expected_freq_schedule = _expected_freq_for_varying_schedule(
            start_diff=args.start_diff,
            m=args.m,
            sizes=rep_sizes,
            cache=cache,
        )

    if args.trials > 1:
        run_monte_carlo(
            args,
            cache=cache,
            expected_freq_schedule=expected_freq_schedule,
            rep_sizes=rep_sizes,
        )

    # Always log a representative single run; charts are gated by `--plot/--no-plot`.
    run_single(
        args,
        sizes=rep_sizes,
        cache=cache,
        expected_freq_schedule=expected_freq_schedule,
        log_theory=(args.trials <= 1),
    )


if __name__ == "__main__":
    main()
