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
from fishtest_spsa_lab.simulator.optimizer import SFSGD, ClassicSPSA, SFAdam
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
        counts = rng.multinomial(self.config.batch_size, probs)

        # Net wins calculation: (2*WW + WD) - (2*LL + DL)
        # counts indices: 0=LL, 1=DL, 2=DD, 3=WD, 4=WW
        net_wins = (2 * counts[4] + counts[3]) - (2 * counts[0] + counts[1])
        return int(net_wins), counts


@dataclass(order=True)
class JobEvent:
    """Represents a completed job returning from a worker."""

    finish_time: float
    worker_id: int = field(compare=False)
    # Payload (not used for sorting)
    theta_snapshot: np.ndarray = field(compare=False, default=None)
    flip: np.ndarray = field(compare=False, default=None)
    c_k: np.ndarray = field(compare=False, default=None)
    iter_local: int = field(compare=False, default=0)


class SpsaRunner:
    """Manages the simulation loop."""

    def __init__(self, config: SPSAConfig) -> None:
        """Initialize the SpsaRunner with the given configuration."""
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.game_provider = GameProvider(config)

        # Initialize optimizer
        if config.optimizer == "sf-sgd":
            self.optimizer = SFSGD(config)
        elif config.optimizer == "sf-adam":
            self.optimizer = SFAdam(config)
        else:
            self.optimizer = ClassicSPSA(config)

        # History tracking
        self.trajectory = [self.optimizer.get_params().copy()]
        self.spsa_signal_history = []

    def run(self) -> dict:
        """Run the SPSA optimization simulation."""
        start_time = time.time()
        num_batches = self.config.num_pairs // self.config.batch_size

        # Calculate number of active params for logging
        num_active = np.count_nonzero(self.config.param_sensitivity)
        logger.info(
            "Target parameters: %s (active: %s)",
            self.config.param_target[0],  # Just show first one as example
            num_active,
        )
        logger.info("Initial Theta (first 5): %s", self.optimizer.get_params()[:5])
        logger.info(
            "Initial Elo: %.2f",
            objective_function(self.optimizer.get_params(), self.config),
        )

        for batch_idx in range(num_batches):
            # k is the number of pairs processed so far
            k = batch_idx * self.config.batch_size
            iter_local = k + 1

            # 1. Get Perturbation Scale
            c_k = self.optimizer.get_perturbation_scale(iter_local)

            # 2. Generate Perturbation
            current_theta = self.optimizer.get_params()
            flip = self.rng.choice([-1, 1], size=self.config.num_params)

            theta_plus = np.clip(
                current_theta + c_k * flip,
                self.config.param_min,
                self.config.param_max,
            )
            theta_minus = np.clip(
                current_theta - c_k * flip,
                self.config.param_min,
                self.config.param_max,
            )

            # 3. Evaluation (Oracle)
            net_wins, _ = self.game_provider.simulate_match(
                theta_plus,
                theta_minus,
                self.rng,
            )

            # 4. Optimizer Step
            self.optimizer.step(iter_local, net_wins, flip, c_k)

            # Logging & History
            self.trajectory.append(self.optimizer.get_params().copy())
            self.spsa_signal_history.append(net_wins * flip)

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
        final_theta = self.optimizer.get_params()

        # Calculate convergence metrics
        final_distances = np.linalg.norm(
            np.diff(self.trajectory[-100:], axis=0),
            axis=1,
        )

        # Calculate distance to target (ground truth check)
        # Only for active parameters (sensitivity > 0)
        mask = self.config.param_sensitivity > 0
        if np.any(mask):
            active_theta = final_theta[mask]
            target_theta = self.config.param_target[mask]
            dist_to_target = np.linalg.norm(active_theta - target_theta)
        else:
            dist_to_target = 0.0

        convergence_metrics = {
            "avg_step_size_last_100": np.mean(final_distances),
            "dist_to_target": dist_to_target,
            "total_batches": len(self.trajectory) - 1,
            "final_elo": objective_function(final_theta, self.config),
        }

        return {
            "config": self.config,
            "trajectory": np.array(self.trajectory),
            "cumulative_spsa_signal": np.cumsum(self.spsa_signal_history, axis=0),
            "final_params": final_theta,
            "convergence_metrics": convergence_metrics,
            "elapsed_time": elapsed_time,
        }


class AsyncSpsaRunner(SpsaRunner):
    """Manages the simulation loop with asynchronous workers."""

    def __init__(self, config: SPSAConfig) -> None:
        """Initialize the AsyncSpsaRunner."""
        super().__init__(config)

        # Worker setup
        # Generate a unique "duration factor" for each worker to simulate hardware
        # differences. We model this as a multiplier on the base duration.
        # Some workers are faster (factor < 1.0), some slower (factor > 1.0).
        # We'll use a log-normal distribution so factors are always positive.
        self.worker_duration_factors = self.rng.lognormal(
            mean=0.0,
            sigma=0.5,
            size=config.num_workers,
        )

        # Priority Queue for events
        self.event_queue: list[JobEvent] = []
        self.current_time = 0.0

        # Track how many batches have been *scheduled* (dispatched)
        # and how many have been *processed* (optimizer step applied)
        self.batches_scheduled = 0
        self.batches_processed = 0

    def _schedule_job(self, worker_id: int) -> None:
        """Dispatch a new job to the given worker."""
        # If we have already scheduled enough batches, don't schedule more.
        num_batches = self.config.num_pairs // self.config.batch_size
        if self.batches_scheduled >= num_batches:
            return

        # 1. Determine iteration index
        # In Fishtest, iter is global and counts PAIRS.
        iter_local = self.batches_scheduled * self.config.batch_size + 1
        self.batches_scheduled += 1

        # 2. Get Perturbation Scale (based on when job starts)
        c_k = self.optimizer.get_perturbation_scale(iter_local)

        # 3. Generate Perturbation (Snapshot of current theta)
        current_theta = self.optimizer.get_params().copy()
        flip = self.rng.choice([-1, 1], size=self.config.num_params)

        # 4. Calculate Duration
        # Use Log-Normal distribution for strictly positive, right-skewed durations
        mu, sigma = self.config.get_lognormal_params()
        base_duration = self.rng.lognormal(mu, sigma)

        # Adjust for worker speed and batch size
        job_duration = (
            base_duration
            * self.config.batch_size
            * self.worker_duration_factors[worker_id]
        )
        finish_time = self.current_time + job_duration

        # 5. Push to Queue
        event = JobEvent(
            finish_time=finish_time,
            worker_id=worker_id,
            theta_snapshot=current_theta,
            flip=flip,
            c_k=c_k,
            iter_local=iter_local,
        )
        heapq.heappush(self.event_queue, event)

    def run(self) -> dict:
        """Run the Async SPSA optimization simulation."""
        start_real_time = time.time()

        # Calculate number of active params for logging
        num_active = np.count_nonzero(self.config.param_sensitivity)

        logger.info(
            "Starting Async SPSA with %d workers.",
            self.config.num_workers,
        )
        logger.info(
            "Target parameters: %s (active: %s)",
            self.config.param_target[0],
            num_active,
        )

        # Initial dispatch to all workers
        for w_id in range(self.config.num_workers):
            self._schedule_job(w_id)

        # Event Loop
        num_batches = self.config.num_pairs // self.config.batch_size
        while self.batches_processed < num_batches:
            if not self.event_queue:
                break

            # Pop earliest finishing job
            event = heapq.heappop(self.event_queue)
            self.current_time = event.finish_time

            # 1. Simulate Match (Oracle)
            # Using the parameters from the SNAPSHOT (when job started)
            theta_plus = np.clip(
                event.theta_snapshot + event.c_k * event.flip,
                self.config.param_min,
                self.config.param_max,
            )
            theta_minus = np.clip(
                event.theta_snapshot - event.c_k * event.flip,
                self.config.param_min,
                self.config.param_max,
            )

            net_wins, _ = self.game_provider.simulate_match(
                theta_plus,
                theta_minus,
                self.rng,
            )

            # 2. Optimizer Step
            # Applies gradient to the CURRENT global theta (which may have changed)
            self.optimizer.step(event.iter_local, net_wins, event.flip, event.c_k)
            self.batches_processed += 1

            # Logging & History
            self.trajectory.append(self.optimizer.get_params().copy())
            self.spsa_signal_history.append(net_wins * event.flip)

            if self.batches_processed % LOG_INTERVAL == 0:
                current_elo = objective_function(
                    self.optimizer.get_params(),
                    self.config,
                )
                logger.debug(
                    "Processed Batch %d: Elo=%.2f, Time=%.1fs",
                    self.batches_processed,
                    current_elo,
                    self.current_time,
                )

            # 3. Re-dispatch worker
            self._schedule_job(event.worker_id)

        # Collect results
        # Note: We need to return the same structure as SpsaRunner
        # But we need to add simulated_duration to metrics
        result = self._collect_results(start_real_time)
        result["convergence_metrics"]["simulated_duration"] = self.current_time
        return result
