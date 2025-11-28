"""Configuration and data models for SPSA simulation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# --- Constants ---
ELO_CLIP_RANGE: float = 599.0
EPSILON: float = 1e-9
TINY_EPSILON: float = 1e-16
SQRT_EPSILON: float = 1e-12
LOG_INTERVAL: int = 100
MAX_INACTIVE_PLOTS: int = 5


@dataclass
class ParamGroup:
    """Defines a group of parameters with shared properties."""

    count: int
    start_val: float
    min_val: float
    max_val: float
    true_target: float
    sensitivity: float = 1.0


@dataclass
class SPSAConfig:
    """Configuration for SPSA optimization simulation."""

    num_pairs: int = 50_000  # Total number of game pairs
    batch_size: int = 10  # Batch size for each SPSA step

    # SPSA Hyperparameters
    A: float = 5000.0
    alpha: float = 0.602
    gamma: float = 0.101
    c_end_fraction: float = 0.05  # c_end as percentage of range as set in fishtest
    r_end: float = 0.002  # Target learning rate (phi-space) at end

    # Schedule-Free SGD Hyperparameters
    sf_sgd_lr: float = 0.005  # Constant learning rate
    sf_sgd_beta1: float = 0.90  # Polyak averaging parameter
    sf_sgd_warmup_fraction: float = 0.0  # Fraction of total games for warmup
    sf_sgd_c_fraction: float = 0.05  # Constant perturbation as fraction of range

    # Schedule-Free Adam Hyperparameters
    sf_adam_lr: float = 0.001  # Constant learning rate
    sf_adam_beta1: float = 0.90  # Polyak averaging parameter
    sf_adam_beta2: float = 0.999  # Adam second moment parameter
    sf_adam_eps: float = 1e-8  # Adam epsilon
    sf_adam_warmup_fraction: float = 0.0  # Fraction of total games for warmup
    sf_adam_c_fraction: float = 0.05  # Constant perturbation as fraction of range

    # Optimizer selection
    optimizer: str = "classic"  # "classic", "sf-sgd", or "sf-adam"

    # Parameter Groups
    # Note on calculating bounds for different sensitivities:
    # To maintain a consistent maximum Elo risk at the boundaries, the range should be
    # scaled inversely to the square root of the sensitivity relative to the
    # reference group (S=1).
    #
    # Formula: Range_S = Range_ref / sqrt(S)  # noqa: ERA001
    # Bounds: Start +/- (Range_S / 2)
    #
    # Example (Reference S=1, Range=2000, Start=1000):
    # - S=4.0: Range = 2000 / 2 = 1000. Bounds = 1000 +/- 500 = [500, 1500]
    # - S=2.0: Range = 2000 / 1.414 = 1414. Bounds = 1000 +/- 707 = [293, 1707]
    param_groups: list[ParamGroup] = field(
        default_factory=lambda: [
            ParamGroup(
                count=2,
                start_val=1000.0,
                min_val=0.0,
                max_val=2000.0,
                true_target=1100.0,
                sensitivity=4.0,
            ),
            ParamGroup(
                count=2,
                start_val=1000.0,
                min_val=500.0,
                max_val=1500.0,
                true_target=1100.0,
                sensitivity=4.0,
            ),
            ParamGroup(
                count=2,
                start_val=1000.0,
                min_val=0.0,
                max_val=2000.0,
                true_target=1100.0,
                sensitivity=2.0,
            ),
            ParamGroup(
                count=2,
                start_val=1000.0,
                min_val=293.0,
                max_val=1707.0,
                true_target=1100.0,
                sensitivity=2.0,
            ),
            ParamGroup(
                count=2,
                start_val=1000.0,
                min_val=0.0,
                max_val=2000.0,
                true_target=1100.0,
                sensitivity=1.0,
            ),
            ParamGroup(
                count=10,
                start_val=1000.0,
                min_val=0.0,
                max_val=2000.0,
                true_target=1100.0,
                sensitivity=0.0,
            ),
        ],
    )

    # Objective function configuration
    # F(theta) = peak_elo - k_elo * ||weighted_dist||^2
    k_elo: float = 0.0  # Placeholder, calculated in __post_init__
    peak_elo: float = 0.0  # Max Elo achievable at theta_star
    initial_elo_gap: float = 500.0  # Target Elo gap at the starting position

    seed: int | None = None  # Random seed

    # Async / Parallel Simulation
    # Number of parallel workers (placeholder, usually set in main())
    num_workers: int = 1
    # Duration configuration (Log-Normal distribution)
    game_duration_median: float = 180.0  # Typical duration (50th percentile)
    game_duration_95th: float = 540.0  # Slow duration (95th percentile)

    @property
    def num_params(self) -> int:
        """Total number of parameters."""
        return sum(g.count for g in self.param_groups)

    @property
    def param_start(self) -> np.ndarray:
        """Vector of starting values."""
        return np.concatenate(
            [np.full(g.count, g.start_val) for g in self.param_groups],
        )

    @property
    def param_min(self) -> np.ndarray:
        """Vector of minimum values."""
        return np.concatenate([np.full(g.count, g.min_val) for g in self.param_groups])

    @property
    def param_max(self) -> np.ndarray:
        """Vector of maximum values."""
        return np.concatenate([np.full(g.count, g.max_val) for g in self.param_groups])

    @property
    def param_target(self) -> np.ndarray:
        """Vector of target values (ground truth)."""
        return np.concatenate(
            [np.full(g.count, g.true_target) for g in self.param_groups],
        )

    @property
    def param_sensitivity(self) -> np.ndarray:
        """Vector of sensitivities."""
        return np.concatenate(
            [np.full(g.count, g.sensitivity) for g in self.param_groups],
        )

    def get_lognormal_params(self) -> tuple[float, float]:
        """Convert median/95th percentile to log-normal mu/sigma."""
        # mu is simply ln(median)
        mu = np.log(self.game_duration_median)

        # 95th percentile = exp(mu + 1.645 * sigma)
        # sigma = (ln(95th) - mu) / 1.645
        sigma = (np.log(self.game_duration_95th) - mu) / 1.645
        return mu, sigma

    def __post_init__(self) -> None:
        """Calculate k_elo based on the problem geometry."""
        # Distance squared from start to target, weighted by sensitivity
        diff = self.param_start - self.param_target
        # Weighted distance squared: sum( sensitivity * (theta - target)^2 )
        # Note: sensitivity is applied as a weight to the squared distance component
        weighted_dist_sq = np.sum(self.param_sensitivity * (diff**2))

        if weighted_dist_sq > EPSILON:
            self.k_elo = self.initial_elo_gap / weighted_dist_sq
        else:
            self.k_elo = 0.0


@dataclass
class SPSAResult:
    """Container for SPSA optimization results."""

    config: SPSAConfig
    trajectory: np.ndarray
    cumulative_spsa_signal: np.ndarray
    final_params: np.ndarray
    convergence_metrics: dict[str, float | bool]
    elapsed_time: float


def objective_function(theta: np.ndarray, config: SPSAConfig) -> float:
    """Calculate the ground-truth Elo for a given parameter vector."""
    target = config.param_target
    sensitivity = config.param_sensitivity

    # Weighted distance squared
    # Inactive params have sensitivity 0, so they contribute 0 to the loss
    dist_sq = np.sum(sensitivity * (theta - target) ** 2)

    return config.peak_elo - config.k_elo * dist_sq
