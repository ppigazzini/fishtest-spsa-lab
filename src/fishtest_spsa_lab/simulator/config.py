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

# Default number of parameters per *active* group in SPSAConfig.param_groups.
# Change this single value to scale the active dimensionality.
DEFAULT_ACTIVE_GROUP_SIZE: int = 2
DEFAULT_INACTIVE_GROUP_SIZE: int = 10


@dataclass
class ParamGroup:
    """Defines a group of parameters in geometry terms.

    This is the per-group view of the true Elo bowl and the
    developer's model of it:

    - theta_start: starting point for this group (true + dev start)
    - theta_peak: ground-truth optimum for this group
    - w_true: true curvature / sensitivity for this group in the bowl
    - w_dev: developer-believed curvature for this group (may differ
      from w_true to model mis-compensation)
    - min_val/max_val: developer-visible bounds, derived in SPSAConfig.
    """

    count: int
    theta_start: float
    theta_peak: float
    w_true: float = 1.0
    w_dev: float | None = None
    min_val: float | None = None
    max_val: float | None = None


@dataclass
class SPSAScheduleConfig:
    """SPSA schedule hyperparameters (power-law a_k and c_k)."""

    A: float = 5000.0
    alpha: float = 0.602
    gamma: float = 0.101
    # Target learning rate (phi-space) at end.
    r_end: float = 0.002


@dataclass
class SPSACWDConfig:
    """Cautious weight decay (CWD) settings for SPSA-based optimizers."""

    # When set to 0.0, cautious weight decay is disabled.
    lambda_: float = 0.2


@dataclass
class SPSAPentaConfig:
    """Pentanomial-derived gain scaling settings for SPSA-penta."""

    # EMA decay per *pair* (effective decay per report is beta_pg**n)
    beta_pg: float = 0.999
    # r = |asym| + mu_weight * |mu| drives gain scale interpolation
    mu_weight: float = 0.5
    # r_small/r_large define the interpolation region
    r_small: float = 0.002
    r_large: float = 0.02
    # Clamp the resulting gain multiplier
    min_scale: float = 0.2
    max_scale: float = 1.5


@dataclass
class AcceleratedSPSAConfig:
    """Accelerated SPSA hyperparameters (accelerated SGD framework)."""

    beta: float = 0.90
    beta_mode: str = "inv_time"  # "constant" or "inv_time"
    beta_k: float = 1.0
    eta_scale: float = 0.09
    alpha_scale: float = 0.10


@dataclass
class SFSGDConfig:
    """Schedule-Free SGD hyperparameters."""

    lr: float = 0.01
    beta: float = 0.90
    warmup_fraction: float = 0.0


@dataclass
class SFAdamMu2Config:
    """Warm-start aggregates for the μ2 estimator used by SFAdamBlock."""

    init: float = 1.0
    reports: float = 0.0
    sum_n: float = 0.0
    sum_s: float = 0.0
    sum_s2_over_n: float = 0.0


@dataclass
class SFAdamConfig:
    """Schedule-Free Adam hyperparameters."""

    lr: float = 0.001
    beta1: float = 0.90
    beta2: float = 0.999
    eps: float = 1e-8
    warmup_fraction: float = 0.0
    mu2: SFAdamMu2Config = field(default_factory=SFAdamMu2Config)


@dataclass
class AdamConfig:
    """Classic Adam hyperparameters."""

    # Constant learning rate (typically higher than sf_adam.lr).
    lr: float = 0.04
    beta1: float = 0.90
    beta2: float = 0.999
    eps: float = 1e-8
    warmup_fraction: float = 0.00


@dataclass
class AdEMAMixConfig:
    """Full AdEMAMix hyperparameters."""

    lr: float = 0.04
    beta1: float = 0.90
    beta2: float = 0.999
    beta3: float = 0.9999
    alpha: float = 5.0
    eps: float = 1e-8
    eps_root: float = 0.0
    warmup_fraction: float = 0.00


@dataclass
class SPSAConfig:
    """Configuration for SPSA optimization simulation."""

    # --- Simulation budget ---
    num_pairs: int = 30_000  # Total number of game pairs
    batch_size: int = 10  # Batch size for each SPSA step

    # --- Dimensionality compensation (optional) ---
    # If enabled, the simulator scales the SPSA update signal by 1/sqrt(N),
    # where N is the total number of parameters (active + inactive).
    #
    # This compensates the growth of the raw two-point SPSA signal with
    # dimensionality when using simultaneous Rademacher perturbations.
    scale_gradient_by_sqrt_num_params: bool = True

    # --- Optimizer-specific knobs ---
    # These are grouped into nested dataclasses so the top-level config stays
    # readable as new optimizers are added.
    spsa: SPSAScheduleConfig = field(default_factory=SPSAScheduleConfig)
    spsa_cwd: SPSACWDConfig = field(default_factory=SPSACWDConfig)
    spsa_penta: SPSAPentaConfig = field(default_factory=SPSAPentaConfig)
    accelerated_spsa: AcceleratedSPSAConfig = field(
        default_factory=AcceleratedSPSAConfig,
    )
    sf_sgd: SFSGDConfig = field(default_factory=SFSGDConfig)
    sf_adam: SFAdamConfig = field(default_factory=SFAdamConfig)
    adam: AdamConfig = field(default_factory=AdamConfig)
    ademamix: AdEMAMixConfig = field(default_factory=AdEMAMixConfig)

    # --- Optimizer selection ---
    # Options: "spsa", "spsa-block", "spsa-penta", "accelerated-spsa",
    #          "sf-sgd", "sf-sgd-block",
    #          "sf-adam", "sf-adam-block", "adam", "adam-block",
    #          "ademamix"
    optimizer: str = "spsa"

    # --- Parameter groups (true geometry inputs) ---
    # These encode theta_start, theta_peak and w_true at group granularity.
    # w_true controls the sensitivity of the Elo objective, making the bowl
    # anisotropic, while w_dev encodes the developer's believed sensitivity.
    # ΔE_true_i = c_elo_gap * (w_true/w_dev) for a 1D step of size c_dev_i.
    # - w_true > w_dev: dev underestimates curvature, so c_dev is
    #   too large and true Elo drops per step are larger than intended.
    # - w_true < w_dev: dev overestimates curvature, so c_dev is
    #   too small and true Elo drops per step are smaller than intended.
    # Developer-level ranges are derived from (w_dev, c_elo_gap, c_fraction)
    # in __post_init__ and exposed via theta_min/theta_max.
    param_groups: list[ParamGroup] = field(
        default_factory=lambda: [
            ParamGroup(
                count=DEFAULT_ACTIVE_GROUP_SIZE,
                theta_start=900.0,
                theta_peak=1000.0,
                w_true=2.0,
                w_dev=1.0,
            ),
            ParamGroup(
                count=DEFAULT_ACTIVE_GROUP_SIZE,
                theta_start=1100.0,
                theta_peak=1000.0,
                w_true=2.0,
                w_dev=1.0,
            ),
            ParamGroup(
                count=DEFAULT_ACTIVE_GROUP_SIZE,
                theta_start=900.0,
                theta_peak=1000.0,
                w_true=1.0,
                w_dev=1.0,
            ),
            ParamGroup(
                count=DEFAULT_ACTIVE_GROUP_SIZE,
                theta_start=1100.0,
                theta_peak=1000.0,
                w_true=1.0,
                w_dev=1.0,
            ),
            ParamGroup(
                count=DEFAULT_ACTIVE_GROUP_SIZE,
                theta_start=900.0,
                theta_peak=1000.0,
                w_true=0.5,
                w_dev=1.0,
            ),
            ParamGroup(
                count=DEFAULT_ACTIVE_GROUP_SIZE,
                theta_start=1100.0,
                theta_peak=1000.0,
                w_true=0.5,
                w_dev=1.0,
            ),
            # Inactive parameters: w_true = 0 but the dev believes they
            # have curvature 1.0, so they still get non-zero c_dev.
            ParamGroup(
                count=DEFAULT_INACTIVE_GROUP_SIZE,
                theta_start=1000.0,
                theta_peak=1000.0,
                w_true=0.0,
                w_dev=1.0,
            ),
        ],
    )

    # --- True Elo geometry ---
    # F(theta) = peak_elo - k_elo * ||weighted_dist||^2
    k_elo: float = 0.0  # Derived in __post_init__
    peak_elo: float = 0.0  # Elo at theta_peak
    # Elo at the true starting point under the simulator's ground truth.
    # k_elo is derived from (peak_elo - start_elo) and the weighted
    # squared distance between theta_start and theta_peak.
    start_elo: float = -0.5

    # --- Developer model (c and ranges) ---
    # c_elo_gap and c_fraction define the intended Elo loss for a 1D step
    # under the developer's model and how that maps to ranges.
    c_elo_gap: float = 2.0
    c_fraction: float = 0.05
    # If True, __post_init__ overwrites ParamGroup.min_val/max_val based
    # on the developer model; if False, the initial ranges are kept.
    auto_dev_ranges: bool = True

    # Internal, derived developer-level perturbation scale. This is
    # computed in __post_init__ from (w_true, w_dev, c_elo_gap,
    # c_fraction, k_elo) and used by optimizers as their canonical c_i.
    c_dev: np.ndarray | None = field(init=False, repr=False, default=None)

    seed: int | None = None  # Random seed

    # --- Async / Parallel Simulation ---
    # Number of parallel workers (placeholder, usually set in main())
    num_workers: int = 1
    # Duration configuration (Log-Normal distribution)
    game_duration_median: float = 180.0  # Typical duration (50th percentile)
    game_duration_95th: float = 540.0  # Slow duration (95th percentile)

    # Heterogeneous worker pool and batch sizing (for async runner)
    # If False, all workers use the same global batch_size as in SpsaRunner.
    variable_batch_size: bool = False
    # Concurrency range (in cores) for simulated workers
    worker_concurrency_min: int = 1
    worker_concurrency_max: int = 64
    # Run-level TC ratio (e.g. 1 for 60+0.6, 2 for 30+0.3)
    tc_ratio: float = 1.0
    # Worker speed heterogeneity (relative to 1.0 baseline)
    worker_speed_min: float = 0.5
    worker_speed_max: float = 2.0

    @property
    def num_params(self) -> int:
        """Total number of parameters."""
        return sum(g.count for g in self.param_groups)

    @property
    def gradient_scale_factor(self) -> float:
        """Scale factor applied to the SPSA update signal.

        When scale_gradient_by_sqrt_num_params is True, this returns
        1/sqrt(N) with N = num_params. Otherwise returns 1.0.
        """
        if not self.scale_gradient_by_sqrt_num_params:
            return 1.0
        n = int(self.num_params)
        if n <= 0:
            return 1.0
        return float(1.0 / np.sqrt(float(n)))

    @property
    def theta_start(self) -> np.ndarray:
        """Vector of starting values (true and dev start)."""
        return np.concatenate(
            [np.full(g.count, g.theta_start) for g in self.param_groups],
        )

    @property
    def theta_min(self) -> np.ndarray:
        """Vector of minimum values (developer-visible bounds).

        These are derived from the developer model in __post_init__ and
        default to theta_start if no dev ranges are set.
        """
        return np.concatenate(
            [
                np.full(
                    g.count,
                    g.min_val if g.min_val is not None else g.theta_start,
                )
                for g in self.param_groups
            ],
        )

    @property
    def theta_max(self) -> np.ndarray:
        """Vector of maximum values (developer-visible bounds).

        These are derived from the developer model in __post_init__ and
        default to theta_start if no dev ranges are set.
        """
        return np.concatenate(
            [
                np.full(
                    g.count,
                    g.max_val if g.max_val is not None else g.theta_start,
                )
                for g in self.param_groups
            ],
        )

    @property
    def theta_peak(self) -> np.ndarray:
        """Vector of target values (ground truth optimum)."""
        return np.concatenate(
            [np.full(g.count, g.theta_peak) for g in self.param_groups],
        )

    @property
    def w_true(self) -> np.ndarray:
        """Vector of true sensitivities/curvatures."""
        return np.concatenate(
            [np.full(g.count, g.w_true) for g in self.param_groups],
        )

    @property
    def w_dev(self) -> np.ndarray:
        """Vector of developer-believed sensitivities/curvatures.

        If a ParamGroup does not specify w_dev, we fall back to w_true
        for active dimensions and 1.0 for inactive ones.
        """
        vals: list[float] = []
        for g in self.param_groups:
            if g.w_dev is not None and g.w_dev > EPSILON:
                wd = float(g.w_dev)
            elif g.w_true > EPSILON:
                wd = float(g.w_true)
            else:
                wd = 1.0
            vals.extend([wd] * g.count)
        return np.asarray(vals, dtype=float)

    def get_lognormal_params(self) -> tuple[float, float]:
        """Convert median/95th percentile to log-normal mu/sigma."""
        # mu is simply ln(median)
        mu = np.log(self.game_duration_median)

        # 95th percentile = exp(mu + 1.645 * sigma)
        # sigma = (ln(95th) - mu) / 1.645
        sigma = (np.log(self.game_duration_95th) - mu) / 1.645
        return mu, sigma

    def __post_init__(self) -> None:
        """Derive k_elo (true geometry) and developer-level c_dev/ranges.

        Geometry layer (ground truth):
        - theta_peak  = param_target
        - theta_start = param_start
        - w_true      = param_sensitivity
        - peak_elo, start_elo

        k_elo is set so that Elo(theta_start) = start_elo and
        Elo(theta_peak) = peak_elo.

        Developer layer:
        - w_dev encodes the developer's believed anisotropy.
        - c_elo_gap and c_fraction define c_dev and, when
            auto_dev_ranges is True, the dev ranges as described
            in docs/SIMULATOR.md.
        """
        sensitivity = self.w_true
        active_mask = sensitivity > EPSILON

        # --- True geometry: compute k_elo from (peak_elo, start_elo) ---
        if not np.any(active_mask):
            self.k_elo = 0.0
            self.c_dev = None
        else:
            theta_start = self.theta_start
            theta_peak = self.theta_peak
            delta = theta_start - theta_peak

            weighted_sq = sensitivity * (delta**2)
            w_start = float(np.sum(weighted_sq[active_mask]))

            if w_start <= EPSILON or abs(self.peak_elo - self.start_elo) <= EPSILON:
                self.k_elo = 0.0
            else:
                self.k_elo = (self.peak_elo - self.start_elo) / w_start

            # --- Developer layer: derive c_dev and dev ranges ---
            # w_dev comes from ParamGroup configuration and models the
            # developer's believed anisotropy.
            if self.k_elo > EPSILON and self.c_elo_gap > 0.0 and self.c_fraction > 0.0:
                w_dev_vec = self.w_dev

                c_vec = np.zeros_like(sensitivity, dtype=float)
                valid_mask = w_dev_vec > EPSILON
                denom = self.k_elo * w_dev_vec[valid_mask]
                c_vec_valid = np.sqrt(self.c_elo_gap / denom)
                c_vec[valid_mask] = c_vec_valid

                # Map per-dimension ranges back into param_groups if enabled
                if self.auto_dev_ranges:
                    range_dev = c_vec / self.c_fraction
                    theta_start_full = theta_start  # already computed above

                    idx = 0
                    for group in self.param_groups:
                        for _ in range(group.count):
                            r = float(range_dev[idx])
                            if r > 0.0:
                                center = float(theta_start_full[idx])
                                group.min_val = center - 0.5 * r
                                group.max_val = center + 0.5 * r
                            idx += 1

                self.c_dev = c_vec
            else:
                self.c_dev = None

        # Provide a sensible default warm-start for the μ2 estimator used by
        # SFAdamBlock, mirroring validate_sf_adam's use of a small symmetric
        # pentanomial prior when no explicit aggregates are provided.
        if self.sf_adam.mu2.reports <= 0.0:
            # Prior over outcomes in {-2, -1, 0, 1, 2}
            # p = (0.05, 0.20, 0.50, 0.20, 0.05) ⇒ mu = 0, var ≈ 0.8
            prior_reports = 5.0
            prior_mean_n = float(self.batch_size)
            mu_p = 0.0
            var_p = 0.8

            self.sf_adam.mu2.reports = prior_reports
            self.sf_adam.mu2.sum_n = prior_reports * prior_mean_n
            self.sf_adam.mu2.sum_s = prior_reports * prior_mean_n * mu_p
            self.sf_adam.mu2.sum_s2_over_n = prior_reports * (
                var_p + prior_mean_n * (mu_p * mu_p)
            )


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
    theta_peak = config.theta_peak
    w_true = config.w_true

    # Weighted distance squared
    # Inactive params have sensitivity 0, so they contribute 0 to the loss
    dist_sq = np.sum(w_true * (theta - theta_peak) ** 2)

    return config.peak_elo - config.k_elo * dist_sq
