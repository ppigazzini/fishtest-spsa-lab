"""Configuration and data models for SPSA simulation."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

logger = logging.getLogger(__name__)

#: Geometry warnings are emitted once per distinct message per process. A sweep
#: builds one config per arm per seed -- 192 of them -- and a warning repeated
#: 192 times is a warning nobody reads.
_WARNED: set[str] = set()


def _warn_once(message: str, *args: object) -> None:
    """Log a warning the first time this exact message is produced."""
    key = message % args if args else message
    if key in _WARNED:
        return
    _WARNED.add(key)
    logger.warning("%s", key)


# --- Constants ---
ELO_CLIP_RANGE: float = 599.0
EPSILON: float = 1e-9
TINY_EPSILON: float = 1e-16
LOG_INTERVAL: int = 100

# Default number of parameters per *active* group in SPSAConfig.param_groups.
# Change this single value to scale the active dimensionality.
DEFAULT_ACTIVE_GROUP_SIZE: int = 2
DEFAULT_INACTIVE_GROUP_SIZE: int = 10

# Game/pair conversion: each game pair consists of two games.
GAMES_PER_PAIR: int = 2
# Fishtest heuristic: games dispatched per concurrency unit per TC ratio unit.
GAMES_PER_CONCURRENCY_UNIT: int = 4
# Normal-distribution z-score for 95th percentile (log-normal calibration).
Z_95: float = 1.645


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

    # Stability offset. Fishtest derives it from the budget as
    # A = A_ratio * num_games / 2 = A_ratio * num_pairs, with the submission
    # form defaulting A_ratio to 0.1; the toy and docs/Analysis.md agree. Set
    # A_ratio to use that form; A is then derived in SPSAConfig.__post_init__.
    # A fixed absolute A does not scale with the run: at 30k pairs, A = 5000 made
    # a_k/c_k non-monotonic, rising from 1.70 at k=1 to 2.96 near k=361 before
    # decaying to 1.50.
    A_ratio: float | None = 0.1
    A: float = 5000.0
    alpha: float = 0.602
    gamma: float = 0.101
    # Target learning rate (phi-space) at end.
    r_end: float = 0.002


@dataclass
class SPSACWDConfig:
    """Cautious weight decay (CWD) settings for SPSA-based optimizers."""

    # When set to 0.0, cautious weight decay is disabled, which is the
    # default: the decay centre is theta_start, which sits away from the
    # optimum, so the term is a constant pull in the wrong direction. Measured
    # monotone over 6 seeds (0.0 -> -0.1685, 0.2 -> -0.1861, 5.0 -> -0.4104).
    lambda_: float = 0.0


@dataclass
class SPSAPentaConfig:
    """Pentanomial-derived gain scaling settings for SPSA-penta."""

    # EMA decay per *pair* (effective decay per report is beta_pg**n)
    beta_pg: float = 0.999
    # r = |asym| + mu_weight * |mu| drives gain scale interpolation
    mu_weight: float = 0.5
    # r_small/r_large define the interpolation region. Calibrated to the range
    # the cumulative statistic actually attains (p10 0.0008, p90 0.0046); the
    # previous r_large of 0.02 was ~4x above anything observed, so the gain sat
    # pinned at min_scale for 38% of a run.
    r_small: float = 0.001
    r_large: float = 0.005
    # Clamp the resulting gain multiplier
    min_scale: float = 0.2
    max_scale: float = 1.5


@dataclass
class AcceleratedSPSAConfig:
    """Accelerated SPSA hyperparameters (accelerated SGD framework)."""

    beta: float = 0.90
    beta_mode: str = "inv_time"  # "constant" or "inv_time"
    beta_k: float = 1.0
    # eta_scale/(1 - beta) + alpha_scale must NOT equal 1: at that point the
    # constant-beta steady-state gain is identical to plain SPSA, making this
    # entry a duplicate of it. The previous 0.09/0.10 pair summed to exactly
    # 1.0000 and, under inv_time, left the momentum term ~55x too small.
    eta_scale: float = 5.0
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

    # lr copied from adam.lr was 4x too large here; alpha = 5.0 is the
    # AdEMAMix paper's language-model value and measured worse than alpha = 0.
    lr: float = 0.01
    beta1: float = 0.90
    beta2: float = 0.999
    beta3: float = 0.9999
    alpha: float = 1.0
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
    # Valid names are the keys of simulator.optimizer.OPTIMIZER_REGISTRY.
    # Do not restate them here; the hand-maintained copy drifted and omitted
    # "spsa-cwd".
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

    # Developer-believed overall Elo curvature. None means "the developer knows
    # the true scale", which is what the lab did unconditionally until now: c_dev
    # was derived from k_elo, itself a function of peak_elo, start_elo,
    # theta_peak and w_true. Moving the true optimum therefore silently moved the
    # "developer's" perturbation scale -- theta_peak 1000 -> 902 changed c_dev
    # from 748.33 to 1047.77 -- so the developer was scale-oracular by
    # construction and only ever wrong about anisotropy. Set this to sweep
    # scale misspecification, which is a different failure from getting the
    # per-axis ratios wrong.
    k_elo_dev: float | None = None

    # Internal, derived developer-level perturbation scale. This is
    # computed in __post_init__ from (w_dev, c_elo_gap, c_fraction,
    # k_elo_dev) and used by optimizers as their canonical c_i.
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
    # Run-level TC ratio (e.g. 1 for 60+0.6, 2 for 30+0.3).
    # NOTE: this scales batch sizes only. It does NOT change the per-pair
    # outcome distribution; use `time_control` for that.
    tc_ratio: float = 1.0

    # Oracle selection: "STC", "LTC", "VLTC", or None for the vendored
    # PentaModel. See simulator/oracle.py.
    #
    # The default is LTC, calibrated to the four real 60+0.6 tunes in
    # __DEV/260809-1-REPORT.md section 7 -- per-pair variance 0.2274 and a 77.5%
    # draw rate, both matched. The vendored model draws 50% of its games where
    # the real ones draw 75-79%, which misstates the entire Elo-to-outcome
    # mapping; it is retained as `None` for reproducing pre-2026-08-16 results.
    #
    # This default was flipped on 2026-08-16 and every absolute Elo figure in the
    # lab moved with it: the predicted stationary noise floor drops 9.1% at LTC
    # relative to the vendored oracle. Comparisons between arms are unaffected in
    # kind -- see the report's section 10.3, where the ranking is noise under
    # either oracle.
    time_control: str | None = "LTC"
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

    @cached_property
    def theta_start(self) -> np.ndarray:
        """Vector of starting values (true and dev start)."""
        return np.concatenate(
            [np.full(g.count, g.theta_start, dtype=float) for g in self.param_groups],
        )

    @cached_property
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
                    dtype=float,
                )
                for g in self.param_groups
            ],
        )

    @cached_property
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
                    dtype=float,
                )
                for g in self.param_groups
            ],
        )

    @cached_property
    def theta_peak(self) -> np.ndarray:
        """Vector of target values (ground truth optimum)."""
        return np.concatenate(
            [np.full(g.count, g.theta_peak, dtype=float) for g in self.param_groups],
        )

    @cached_property
    def w_true(self) -> np.ndarray:
        """Vector of true sensitivities/curvatures."""
        return np.concatenate(
            [np.full(g.count, g.w_true, dtype=float) for g in self.param_groups],
        )

    @cached_property
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

        # 95th percentile = exp(mu + Z_95 * sigma)
        # sigma = (ln(95th) - mu) / Z_95
        sigma = (np.log(self.game_duration_95th) - mu) / Z_95
        return mu, sigma

    def _warn_on_incoherent_geometry(self) -> None:
        """Log when the probe scale does not describe a tune.

        Two independent checks, both of which the shipped defaults fail.

        The probe must not cost more Elo than the tune is worth: `c_elo_gap` is
        the intended Elo loss of a one-axis probe step and `peak_elo -
        start_elo` is the entire depth of the bowl. The defaults ask each probe
        to cost 2.0 Elo in a bowl 0.5 Elo deep.

        The probe must also be comparable to the distance it has to cover. The
        defaults put `c_dev` at 748.33 against a start-to-peak distance of 100,
        so every probe straddles the whole basin and the quadratic model is
        evaluated far outside where it holds.

        This warns rather than raises, because the tension is structural and not
        a typo. `c_j**2 * eps_j = c_elo_gap * w_true_j / w_dev_j`, so
        `lambda_j = C / (8 * r * c_elo_gap) * (w_dev_j / w_true_j)` -- the
        convergence budget depends on `c_elo_gap` and the gain, and on nothing
        else. Shrinking the probe to a physically sensible `c_dev/distance` of
        0.3 costs 625x the budget: 127 million games against the 60 thousand the
        sweep runs. Choosing where to sit on that trade-off is a decision for
        whoever is running the experiment; `design_budget()` reports the cost.
        """
        depth = abs(self.peak_elo - self.start_elo)
        if depth > EPSILON and self.c_elo_gap > 0.5 * depth:
            _warn_once(
                "geometry: c_elo_gap=%.4g asks each probe to cost more than half "
                "the %.4g Elo the whole tune is worth",
                self.c_elo_gap,
                depth,
            )

        if self.c_dev is None:
            return
        active = self.w_true > EPSILON
        if not np.any(active):
            return
        distance = np.abs(self.theta_start - self.theta_peak)[active]
        ratio = self.c_dev[active][distance > EPSILON] / distance[distance > EPSILON]
        if ratio.size and (ratio.max() > 1.0 or ratio.min() < 0.1):
            _warn_once(
                "geometry: c_dev/distance-to-optimum spans %.3g..%.3g, outside "
                "the sane 0.1..1.0; probes are not sized to the basin they search",
                float(ratio.min()),
                float(ratio.max()),
            )

    def design_budget(self) -> DesignBudget | None:
        """Return the game budget this configuration needs to converge.

        ``None`` when the geometry is degenerate (no active axes, or no derived
        perturbation scale), because there is then nothing to size.
        """
        if self.c_dev is None or self.k_elo <= EPSILON:
            return None
        active = self.w_true > EPSILON
        if not np.any(active):
            return None

        r_eff = (self.spsa.r_end / 2.0) * self.gradient_scale_factor
        if r_eff <= 0.0:
            return None

        eps = self.k_elo * self.w_true[active]
        c_j = self.c_dev[active]
        lam = ELO_C / (8.0 * r_eff * (c_j**2) * eps)
        slowest = float(np.max(lam))
        return DesignBudget(
            lambda_per_axis=lam,
            slowest_axis_games=slowest,
            recommended_games=LAMBDA_RATIO * 2.0 * slowest,
            budget_games=int(self.num_pairs) * GAMES_PER_PAIR,
            effective_r=r_eff,
        )

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
            in docs/Simulator.md.
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
            k_dev = float(self.k_elo_dev) if self.k_elo_dev is not None else self.k_elo
            if k_dev > EPSILON and self.c_elo_gap > 0.0 and self.c_fraction > 0.0:
                w_dev_vec = self.w_dev

                c_vec = np.zeros_like(sensitivity, dtype=float)
                valid_mask = w_dev_vec > EPSILON
                denom = k_dev * w_dev_vec[valid_mask]
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

        # Derive A from the budget when A_ratio is set, matching Fishtest.
        if self.spsa.A_ratio is not None:
            self.spsa.A = float(self.spsa.A_ratio) * float(self.num_pairs)

        self._warn_on_incoherent_geometry()

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

            # Express the prior in the units the optimizer is actually fed.
            # runner.py hands the step `net_wins * gradient_scale_factor / 2`,
            # not a raw outcome sum, so a prior stated in raw outcome units
            # over-estimates the second moment by 1/signal_scale**2 -- a factor
            # of 88 at the default 22 parameters. Measured before this fix:
            # _mu2_hat() started at 0.8 and, after 105 real reports, still drew
            # 84% of its value (4.0 of 4.75) from the prior, holding the Adam
            # denominator about 2.5x too large for the whole run.
            #
            # Measured effect on the outcome: none detectable. 8 paired seeds at
            # 30,000 pairs gave +0.0065 +- 0.0283 Elo, which does not separate
            # from zero. This is a units fix, not a performance fix, and it is
            # recorded that way so nobody later credits it with a gain.
            signal_scale = self.gradient_scale_factor / 2.0
            var_p *= signal_scale * signal_scale
            mu_p *= signal_scale

            self.sf_adam.mu2.reports = prior_reports
            self.sf_adam.mu2.sum_n = prior_reports * prior_mean_n
            self.sf_adam.mu2.sum_s = prior_reports * prior_mean_n * mu_p
            self.sf_adam.mu2.sum_s2_over_n = prior_reports * (
                var_p + prior_mean_n * (mu_p * mu_p)
            )


@dataclass(frozen=True, slots=True)
class DesignBudget:
    """How many games a configuration needs before its arms can be compared.

    From the design equations of Van den Bergh's ``spsa_simul``, reproduced in
    ``__DEV/260809-0-REPORT.md`` Appendix C:

    ```text
    eps_j     = Elo curvature along axis j = k_elo * w_true_j
    lambda_j  = C / (8 * r * c_j**2 * eps_j)          games for axis j
    num_games = lambda_ratio * 2 * max_j lambda_j     lambda_ratio = 3
    ```

    ``r`` is the *effective* gain, which in this lab is ``r_end / 2`` from the
    halved signal and a further ``1/sqrt(N)`` when dimensionality compensation is
    on. Both are deliberate deviations from Fishtest's gain and both slow
    convergence, so both belong in the sizing.
    """

    lambda_per_axis: np.ndarray
    slowest_axis_games: float
    recommended_games: float
    budget_games: int
    effective_r: float

    @property
    def fraction_of_recommended(self) -> float:
        """Budget as a fraction of the recommended one. Below 1 means unconverged."""
        if self.recommended_games <= 0.0:
            return math.inf
        return self.budget_games / self.recommended_games

    @property
    def is_sufficient(self) -> bool:
        """Whether the run can be expected to have converged."""
        return self.fraction_of_recommended >= 1.0

    def summary(self) -> str:
        """One line stating whether this budget can answer a comparison."""
        frac = self.fraction_of_recommended
        if self.is_sufficient:
            return (
                f"budget {self.budget_games:,} games = {frac:.2f}x the "
                f"recommended {self.recommended_games:,.0f}"
            )
        return (
            f"budget {self.budget_games:,} games is 1/{1 / frac:,.0f} of the "
            f"{self.recommended_games:,.0f} games the design equation asks for; "
            f"no arm has converged and differences between arms are not "
            f"attributable to the optimizers"
        )


#: Van den Bergh's default: run 3x the two-sided convergence time constant.
LAMBDA_RATIO: float = 3.0

#: 800 / ln(10), the Elo-to-logit constant.
ELO_C: float = 800.0 / math.log(10.0)


def objective_function(theta: np.ndarray, config: SPSAConfig) -> float:
    """Calculate the ground-truth Elo for a given parameter vector."""
    theta_peak = config.theta_peak
    w_true = config.w_true

    # Weighted distance squared
    # Inactive params have sensitivity 0, so they contribute 0 to the loss
    dist_sq = np.sum(w_true * (theta - theta_peak) ** 2)

    return config.peak_elo - config.k_elo * dist_sq
