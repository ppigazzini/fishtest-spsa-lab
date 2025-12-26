"""Simulation runner and game provider logic."""

from __future__ import annotations

import heapq
import logging
import time
from dataclasses import dataclass, field

import numpy as np

from fishtest_spsa_lab.simulator.config import (
    ELO_CLIP_RANGE,
    LOG_INTERVAL,
    SPSAConfig,
    objective_function,
)
from fishtest_spsa_lab.simulator.optimizer import (
    OPTIMIZER_REGISTRY,
    PentaStatsMixin,
)
from fishtest_spsa_lab.vendor.pentamodel.pentamodel import PentaModel

logger = logging.getLogger(__name__)


class GameProvider:
    """Provides methods to generate game outcomes using the Pentamodel."""

    def __init__(self, config: SPSAConfig) -> None:
        """Initialize the GameProvider with the given configuration."""
        self.config = config

    def simulate_match(
        self,
        theta_plus: np.ndarray,
        theta_minus: np.ndarray,
        rng: np.random.Generator,
        batch_size_pairs: int | None = None,
    ) -> tuple[int, np.ndarray]:
        """Simulate a match between theta_plus and theta_minus."""
        elo_plus = objective_function(theta_plus, self.config)
        elo_minus = objective_function(theta_minus, self.config)

        # Elo difference from perspective of theta_plus
        elo_diff = elo_plus - elo_minus

        # PentaModel(opponentElo=X) means Hero is 0, Opponent is X.
        input_elo = np.clip(-elo_diff, -ELO_CLIP_RANGE, ELO_CLIP_RANGE)

        model = PentaModel(opponentElo=input_elo)
        probs = model.pentanomialProbs

        # multinomial requires a sequence, probs is a list
        if batch_size_pairs is None:
            n_pairs = self.config.batch_size
        else:
            n_pairs = batch_size_pairs
        counts = rng.multinomial(n_pairs, probs)

        # Net wins calculation: (2*WW + WD) - (2*LL + DL)
        # counts indices: 0=LL, 1=DL, 2=DD, 3=WD, 4=WW
        net_wins = (2 * counts[4] + counts[3]) - (2 * counts[0] + counts[1])
        return int(net_wins), counts


def collect_convergence_metrics(
    config: SPSAConfig,
    trajectory: list[np.ndarray] | np.ndarray,
    total_pairs: int,
) -> dict[str, float | bool]:
    """Compute convergence metrics given a trajectory and total pairs.

    This helper is shared by both synchronous and asynchronous runners.
    """
    # Ensure we have a NumPy array for vectorized ops
    traj_arr = np.array(trajectory)
    final_theta = traj_arr[-1]

    # Step-size proxy from last 100 steps
    if len(traj_arr) > 1:
        tail = traj_arr[-100:]
        final_distances = np.linalg.norm(np.diff(tail, axis=0), axis=1)
        avg_step_size = float(np.mean(final_distances))
    else:
        avg_step_size = 0.0

    # Distance to target using only active params
    mask = config.w_true > 0
    if np.any(mask):
        active_theta = final_theta[mask]
        target_theta = config.theta_peak[mask]
        dist_to_target = float(np.linalg.norm(active_theta - target_theta))
    else:
        dist_to_target = 0.0

    return {
        "avg_step_size_last_100": avg_step_size,
        "dist_to_target": dist_to_target,
        "total_pairs": float(total_pairs),
        "start_elo": float(config.start_elo),
        "final_elo": float(objective_function(final_theta, config)),
    }


@dataclass(order=True)
class JobEvent:
    """Represents a completed job returning from a worker."""

    finish_time: float
    worker_id: int = field(compare=False)
    # Payload (not used for sorting)
    theta_snapshot: np.ndarray = field(compare=False)
    flip: np.ndarray = field(compare=False)
    c_k: np.ndarray = field(compare=False)
    iter_local: int = field(compare=False)
    batch_size_pairs: int = field(compare=False)


@dataclass
class SimWorker:
    """Simulated worker with fixed concurrency and speed factor."""

    concurrency: int
    speed_factor: float


def make_workers(config: SPSAConfig, rng: np.random.Generator) -> list[SimWorker]:
    """Create a heterogeneous worker pool based on the configuration."""
    workers: list[SimWorker] = []

    # Powers of two within configured bounds
    candidates = [1, 2, 4, 8, 16, 32, 64]
    candidates = [
        c
        for c in candidates
        if config.worker_concurrency_min <= c <= config.worker_concurrency_max
    ]
    if not candidates:
        # Fallback to single-core workers if misconfigured
        candidates = [1]

    # Uniform weighting: all allowed concurrencies equally likely
    weights = np.ones(len(candidates), dtype=float)
    weights /= weights.sum()

    for _ in range(config.num_workers):
        concurrency = int(rng.choice(candidates, p=weights))
        speed_factor = float(
            rng.uniform(config.worker_speed_min, config.worker_speed_max),
        )
        workers.append(
            SimWorker(concurrency=concurrency, speed_factor=speed_factor),
        )

    return workers


class SpsaRunner:
    """Manages the simulation loop."""

    def __init__(self, config: SPSAConfig) -> None:
        """Initialize the SpsaRunner with the given configuration."""
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.game_provider = GameProvider(config)

        # Initialize optimizer
        optimizer_cls = OPTIMIZER_REGISTRY.get(
            config.optimizer,
            OPTIMIZER_REGISTRY["spsa"],
        )
        self.optimizer = optimizer_cls(config)

        # History tracking
        self.trajectory = [self.optimizer.get_params().copy()]
        self.spsa_signal_history = []
        # Cumulative pairs processed after each optimizer step
        self.pairs_history: list[int] = [0]

    def run(self) -> dict:
        """Run the SPSA optimization simulation."""
        start_time = time.time()
        num_batches = self.config.num_pairs // self.config.batch_size

        signal_scale = float(self.config.gradient_scale_factor)

        for batch_idx in range(num_batches):
            # k is the number of pairs processed so far
            k = batch_idx * self.config.batch_size
            iter_local = k + 1
            batch_size_pairs = self.config.batch_size

            # 1. Get Perturbation Scale
            c_k = self.optimizer.get_perturbation_scale(iter_local)

            # 2. Generate Perturbation
            current_theta = self.optimizer.get_params()
            flip = self.rng.choice([-1, 1], size=self.config.num_params)

            theta_plus = np.clip(
                current_theta + c_k * flip,
                self.config.theta_min,
                self.config.theta_max,
            )
            theta_minus = np.clip(
                current_theta - c_k * flip,
                self.config.theta_min,
                self.config.theta_max,
            )

            # 3. Evaluation (Oracle)
            net_wins, counts = self.game_provider.simulate_match(
                theta_plus,
                theta_minus,
                self.rng,
                batch_size_pairs=batch_size_pairs,
            )

            # SPSA gradient signal in phi-space uses (result / 2), where result is the
            # sum of per-pair outcomes in {-2,-1,0,1,2}. Keep the factor 1/2 in the
            # gradient estimate (not folded into the learning rate).
            #
            # Optional dimensionality compensation scales the *update signal* only
            # (do NOT change the perturbation used to evaluate y(θ±cδ)).
            scaled_net_wins = (float(net_wins) * signal_scale) / 2.0

            # 4. Optional pentanomial stats hook
            if isinstance(self.optimizer, PentaStatsMixin):
                self.optimizer.update_penta_stats(counts)

            # 5. Optimizer Step
            self.optimizer.step(
                iter_local,
                scaled_net_wins,
                flip,
                c_k,
                batch_size_pairs,
            )

            # Logging & History
            self.trajectory.append(self.optimizer.get_params().copy())
            self.spsa_signal_history.append(scaled_net_wins * flip)
            self.pairs_history.append(iter_local + self.config.batch_size - 1)

            if batch_idx % LOG_INTERVAL == 0:
                current_elo = objective_function(
                    self.optimizer.get_params(),
                    self.config,
                )
                logger.debug(
                    "Batch %d: Elo=%.2f, Theta[0]=%.3f",
                    batch_idx,
                    current_elo,
                    self.optimizer.get_params()[0],
                )

        return self._collect_results(start_time)

    def _collect_results(self, start_time: float) -> dict:
        """Package results."""
        elapsed_time = time.time() - start_time
        # In the synchronous runner we always consume the full budget.
        total_pairs = int(self.config.num_pairs)

        convergence_metrics = collect_convergence_metrics(
            self.config,
            self.trajectory,
            total_pairs=total_pairs,
        )

        final_theta = self.optimizer.get_params()

        result: dict[str, object] = {
            "config": self.config,
            "trajectory": np.array(self.trajectory),
            "pairs_history": np.array(self.pairs_history, dtype=float),
            "cumulative_spsa_signal": np.cumsum(
                self.spsa_signal_history,
                axis=0,
            ),
            "final_params": final_theta,
            "convergence_metrics": convergence_metrics,
            "elapsed_time": elapsed_time,
        }

        if isinstance(self.optimizer, PentaStatsMixin):
            result["asym_history"] = np.array(self.optimizer.asym_history, dtype=float)
            result["mu_history"] = np.array(self.optimizer.mu_history, dtype=float)
            result["asym_cum_history"] = np.array(
                self.optimizer.asym_cum_history,
                dtype=float,
            )
            result["mu_cum_history"] = np.array(
                self.optimizer.mu_cum_history,
                dtype=float,
            )
            result["penta_coeff_history"] = np.array(
                self.optimizer.penta_coeff_history,
                dtype=float,
            )
            result["gain_scale_history"] = np.array(
                self.optimizer.gain_scale_history,
                dtype=float,
            )

        return result


class AsyncSpsaRunner(SpsaRunner):
    """Manages the simulation loop with asynchronous workers."""

    def __init__(
        self,
        config: SPSAConfig,
        workers: list[SimWorker] | None = None,
    ) -> None:
        """Initialize the AsyncSpsaRunner."""
        super().__init__(config)

        # Optional heterogeneous worker pool (can be shared across runs)
        if workers is None:
            self.workers: list[SimWorker] = make_workers(self.config, self.rng)
        else:
            if len(workers) != self.config.num_workers:
                msg = (
                    "Length of workers list must match config.num_workers "
                    f"({len(workers)} != {self.config.num_workers})"
                )
                raise ValueError(msg)
            # Copy to avoid accidental mutation by caller
            self.workers = list(workers)

        # Priority Queue for events
        self.event_queue: list[JobEvent] = []
        self.current_time = 0.0

        # Track how many pairs have been scheduled and processed
        self.pairs_scheduled = 0
        self.pairs_processed = 0

        # Track how many pairs each worker processes (for diagnostics)
        self.worker_pairs_processed: list[int] = [0 for _ in range(len(self.workers))]

        # Track per-job batch sizes (in pairs) for diagnostics
        self.batch_sizes: list[int] = []

        # Out-of-order diagnostics
        self.lags: list[int] = []
        self.normalized_lags: list[float] = []
        self.weighted_abs_lag_pairs: float = 0.0

        # Cache the update-signal scaling factor once per run.
        self._signal_scale = float(self.config.gradient_scale_factor)

    def _schedule_job(self, worker_id: int) -> None:
        """Dispatch a new job to the given worker."""
        # Stop if we already scheduled all required pairs
        if self.pairs_scheduled >= self.config.num_pairs:
            return

        worker = self.workers[worker_id]

        # Determine batch size in pairs for this worker
        if self.config.variable_batch_size:
            tc_ratio = max(1, round(self.config.tc_ratio))
            batch_size_games = worker.concurrency * 4 * tc_ratio
            batch_size_pairs = max(1, batch_size_games // 2)
        else:
            batch_size_pairs = self.config.batch_size

        # Truncate so we don't exceed total num_pairs
        remaining_pairs = self.config.num_pairs - self.pairs_scheduled
        batch_size_pairs = min(batch_size_pairs, remaining_pairs)
        if batch_size_pairs <= 0:
            return

        # Record batch size for stats
        self.batch_sizes.append(batch_size_pairs)

        # In Fishtest, iter is global and counts PAIRS.
        iter_local = self.pairs_scheduled + 1
        self.pairs_scheduled += batch_size_pairs

        # Get Perturbation Scale (based on when job starts)
        c_k = self.optimizer.get_perturbation_scale(iter_local)

        # Generate Perturbation (Snapshot of current theta)
        current_theta = self.optimizer.get_params().copy()
        flip = self.rng.choice([-1, 1], size=self.config.num_params)

        # Calculate Duration: model per-game durations and true parallelism.
        #
        # - A "pair" is two games; batch_size_pairs counts pairs.
        # - game durations are sampled i.i.d. from a log-normal distribution.
        # - the worker runs up to `concurrency` games in parallel; job wall-time is
        #   the makespan (max over lanes of summed game durations).
        # - faster workers (higher speed_factor) complete the same work sooner.
        mu, sigma = self.config.get_lognormal_params()
        batch_size_games = int(2 * batch_size_pairs)
        lane_count = max(1, min(worker.concurrency, batch_size_games))

        # Greedy list scheduling using a min-heap of lane accumulated times.
        lane_heap = [0.0] * lane_count
        heapq.heapify(lane_heap)
        max_lane_time = 0.0
        for duration in self.rng.lognormal(mu, sigma, size=batch_size_games):
            t = heapq.heappop(lane_heap) + float(duration)
            max_lane_time = max(max_lane_time, t)
            heapq.heappush(lane_heap, t)

        job_duration = max_lane_time / worker.speed_factor
        finish_time = self.current_time + job_duration

        event = JobEvent(
            finish_time=finish_time,
            worker_id=worker_id,
            theta_snapshot=current_theta,
            flip=flip,
            c_k=c_k,
            iter_local=iter_local,
            batch_size_pairs=batch_size_pairs,
        )
        heapq.heappush(self.event_queue, event)

    def run(self) -> dict:
        """Run the Async SPSA optimization simulation."""
        start_real_time = time.time()

        logger.info(
            "Starting async optimization (%s) with %d workers.",
            self.config.optimizer,
            self.config.num_workers,
        )

        # Initial dispatch to all workers
        for w_id in range(self.config.num_workers):
            self._schedule_job(w_id)

        # Event Loop
        while self.pairs_processed < self.config.num_pairs:
            if not self.event_queue:
                break

            self._process_next_event()

        elapsed_time = time.time() - start_real_time
        convergence_metrics, lag_stats, final_theta = self._finalize_async_run()

        result: dict[str, object] = {
            "config": self.config,
            "trajectory": np.array(self.trajectory),
            "pairs_history": np.array(self.pairs_history, dtype=float),
            "cumulative_spsa_signal": np.cumsum(
                self.spsa_signal_history,
                axis=0,
            ),
            "final_params": final_theta,
            "convergence_metrics": convergence_metrics,
            "elapsed_time": elapsed_time,
            "out_of_order_stats": lag_stats,
        }

        if isinstance(self.optimizer, PentaStatsMixin):
            result["asym_history"] = np.array(self.optimizer.asym_history, dtype=float)
            result["mu_history"] = np.array(self.optimizer.mu_history, dtype=float)
            result["asym_cum_history"] = np.array(
                self.optimizer.asym_cum_history,
                dtype=float,
            )
            result["mu_cum_history"] = np.array(
                self.optimizer.mu_cum_history,
                dtype=float,
            )
            result["penta_coeff_history"] = np.array(
                self.optimizer.penta_coeff_history,
                dtype=float,
            )
            result["gain_scale_history"] = np.array(
                self.optimizer.gain_scale_history,
                dtype=float,
            )

        return result

    def _process_next_event(self) -> None:
        """Handle the next completed job event from the queue."""
        # Pop earliest finishing job
        event = heapq.heappop(self.event_queue)
        self.current_time = event.finish_time

        # Out-of-order tracking: how far ahead/behind this job completes
        expected_iter = self.pairs_processed + 1
        lag = event.iter_local - expected_iter

        # 1. Simulate Match (Oracle)
        # Using the parameters from the SNAPSHOT (when job started)
        theta_plus = np.clip(
            event.theta_snapshot + event.c_k * event.flip,
            self.config.theta_min,
            self.config.theta_max,
        )
        theta_minus = np.clip(
            event.theta_snapshot - event.c_k * event.flip,
            self.config.theta_min,
            self.config.theta_max,
        )

        batch_size_pairs = event.batch_size_pairs

        # Accumulate out-of-order metrics
        self.lags.append(lag)
        self.normalized_lags.append(lag / max(1, batch_size_pairs))
        self.weighted_abs_lag_pairs += abs(lag) * batch_size_pairs
        net_wins, counts = self.game_provider.simulate_match(
            theta_plus,
            theta_minus,
            self.rng,
            batch_size_pairs=batch_size_pairs,
        )

        scaled_net_wins = float(net_wins) * float(self._signal_scale)
        scaled_net_wins /= 2.0

        # 2. Optional pentanomial stats hook
        if isinstance(self.optimizer, PentaStatsMixin):
            self.optimizer.update_penta_stats(counts)

        # 3. Optimizer Step
        # Applies gradient to the CURRENT global theta (which may have changed)
        self.optimizer.step(
            event.iter_local,
            scaled_net_wins,
            event.flip,
            event.c_k,
            batch_size_pairs,
        )
        self.pairs_processed += batch_size_pairs

        # Track per-worker work share
        self.worker_pairs_processed[event.worker_id] += batch_size_pairs

        # Logging & History
        self.trajectory.append(self.optimizer.get_params().copy())
        self.spsa_signal_history.append(scaled_net_wins * event.flip)
        self.pairs_history.append(self.pairs_processed)

        # Log approximately every batch_size pairs, scaled by LOG_INTERVAL
        log_interval_pairs = max(1, self.config.batch_size * LOG_INTERVAL)
        if self.pairs_processed % log_interval_pairs == 0:
            current_elo = objective_function(
                self.optimizer.get_params(),
                self.config,
            )
            logger.debug(
                "Processed Pairs %d: Elo=%.2f, Time=%.1fs",
                self.pairs_processed,
                current_elo,
                self.current_time,
            )

        # 3. Re-dispatch worker
        self._schedule_job(event.worker_id)

    def _finalize_async_run(
        self,
    ) -> tuple[
        dict[str, float | bool],
        dict[str, float],
        np.ndarray,
    ]:
        """Compute convergence metrics, KPIs and final theta for async runs."""
        total_pairs = int(self.pairs_processed)

        convergence_metrics = collect_convergence_metrics(
            self.config,
            self.trajectory,
            total_pairs=total_pairs,
        )
        convergence_metrics["simulated_duration"] = self.current_time

        lag_stats = self._log_worker_and_ooo_kpis()
        final_theta = self.optimizer.get_params()

        return convergence_metrics, lag_stats, final_theta

    def _log_worker_and_ooo_kpis(self) -> dict[str, float]:
        """Log worker pool KPIs and out-of-order statistics."""

        def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
            if values.size == 0:
                return float("nan")
            if values.size != weights.size:
                msg = "values and weights must have same shape"
                raise ValueError(msg)
            total_w = float(weights.sum())
            if total_w <= 0.0:
                return float("nan")

            order = np.argsort(values)
            v_sorted = values[order]
            w_sorted = weights[order]
            cumsum = np.cumsum(w_sorted)
            cutoff = 0.5 * total_w
            idx = int(np.searchsorted(cumsum, cutoff, side="left"))
            idx = max(0, min(idx, v_sorted.size - 1))
            return float(v_sorted[idx])

        # --- Worker pool KPIs (static pool description) ---
        if self.workers:
            concs = np.array([w.concurrency for w in self.workers], dtype=int)
            speeds = np.array([w.speed_factor for w in self.workers], dtype=float)

            # 1) Concurrency distribution
            uniq, counts = np.unique(concs, return_counts=True)
            dist_str = ", ".join(
                f"{int(c)}: {int(n)}" for c, n in zip(uniq, counts, strict=True)
            )
            logger.info(
                "Worker concurrency distribution (concurrency: count): %s",
                dist_str,
            )

            # 2) Speed-factor distribution
            logger.info(
                "Worker speed_factor stats: min=%.2f max=%.2f mean=%.2f median=%.2f",
                float(speeds.min()),
                float(speeds.max()),
                float(speeds.mean()),
                float(np.median(speeds)),
            )

        # Log batch size statistics if available
        if self.batch_sizes:
            bs_arr = np.array(self.batch_sizes, dtype=float)
            # Pair-weighted view: each job contributes proportionally to its size
            # (i.e. how many pairs it actually processes).
            bs_weights = bs_arr.copy()
            bs_pair_weighted_mean = float(
                (bs_arr * bs_weights).sum() / bs_weights.sum(),
            )
            bs_pair_weighted_median = _weighted_median(bs_arr, bs_weights)
            logger.info(
                (
                    "Batch size stats (pairs): min=%d max=%d job_mean=%.1f "
                    "job_median=%.1f pair_mean=%.1f pair_median=%.1f"
                ),
                int(bs_arr.min()),
                int(bs_arr.max()),
                float(bs_arr.mean()),
                float(np.median(bs_arr)),
                bs_pair_weighted_mean,
                bs_pair_weighted_median,
            )

        # --- Worker pool KPIs (work share after run) ---
        if self.workers:
            wp = np.array(self.worker_pairs_processed, dtype=float)
            total_wp = float(wp.sum())
            if total_wp > 0.0:
                share = 100.0 * wp / total_wp
                logger.info(
                    "Worker pairs processed stats: min=%d max=%d mean=%.1f median=%.1f",
                    int(wp.min()),
                    int(wp.max()),
                    float(wp.mean()),
                    float(np.median(wp)),
                )
                logger.debug(
                    "Worker pairs processed (pairs, share%%): %s",
                    list(zip(wp.astype(int), share, strict=True)),
                )

        # --- Out-of-order KPIs (one-line) ---
        lag_stats: dict[str, float] = {}
        if self.lags:
            lags_array = np.array(self.lags, dtype=float)
            normalized_lags_array = np.array(
                self.normalized_lags,
                dtype=float,
            )
            lag_stats = {
                "mean": float(lags_array.mean()),
                "abs_mean": float(np.abs(lags_array).mean()),
                "std": float(lags_array.std()),
                "min": int(lags_array.min()),
                "max": int(lags_array.max()),
                "p50": float(np.percentile(lags_array, 50)),
                "p90": float(np.percentile(lags_array, 90)),
                "p99": float(np.percentile(lags_array, 99)),
                "norm_p50": float(np.percentile(normalized_lags_array, 50)),
                "norm_p90": float(np.percentile(normalized_lags_array, 90)),
                "norm_p99": float(np.percentile(normalized_lags_array, 99)),
                "weighted_abs_share_pct": float(
                    100.0 * self.weighted_abs_lag_pairs / max(1, self.pairs_processed),
                ),
            }
            logger.info(
                "Out-of-order: share=%.2f%% p50=%+.0f p90=%+.0f p99=%+.0f (pairs); "
                "norm p50=%+.2f p90=%+.2f p99=%+.2f (batches)",
                lag_stats["weighted_abs_share_pct"],
                lag_stats["p50"],
                lag_stats["p90"],
                lag_stats["p99"],
                lag_stats["norm_p50"],
                lag_stats["norm_p90"],
                lag_stats["norm_p99"],
            )

        return lag_stats
