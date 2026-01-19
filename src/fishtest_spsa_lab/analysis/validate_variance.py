"""Statistical estimator validation for Schedule-Free Adam.

This script validates the `OnlineReportStats` class, which is responsible for
estimating the second moment of the gradient (E[g^2]) in the Schedule-Free Adam
optimizer using only block-level summaries.

The Problem:
- Adam requires an estimate of the uncentered variance v = E[g^2].
- In a distributed setting (Fishtest), we only receive batch summaries:
  sum(s) and sum(N) for a batch of games.
- We cannot compute the exact sample variance of the individual outcomes
  because we don't see the individual wins/losses/draws, only the totals.

The Solution:
- We use the identity: Var(X) = E[X^2] - (E[X])^2
- We can compute the exact block-averaged variance using:
  sigma^2_hat = mean(s^2/N) - mean(s/N)^2 * mean(N)
  (Note: This is a simplified conceptual view; the actual math handles
   the expectations over blocks correctly).
- `OnlineReportStats` accumulates the necessary sufficient statistics:
  reports, sum_n, sum_s, sum_s2_over_n.

This script:
1. Defines a "True" pentanomial distribution (Win/Loss/Draw probabilities).
2. Calculates the theoretical mean and variance of this distribution.
3. Simulates a stream of batches (reports) drawn from this distribution.
4. Feeds these batches into `OnlineReportStats`.
5. Verifies that the online estimate converges to the theoretical truth.
6. Prints a JSON block with "warm-start" statistics that can be pasted
   into the main simulator configuration.
"""

import math
import random
from dataclasses import dataclass

# ----- exact helpers copied from validate_sf_adam helpers -----


def compute_pentanomial_moments(
    p5: tuple[float, float, float, float, float],
) -> tuple[float, float, float]:
    """Compute mean, second moment, and variance for a pentanomial distribution."""
    # Values correspond to [-2, -1, 0, +1, +2]
    vals = (-2.0, -1.0, 0.0, 1.0, 2.0)
    mu = sum(p * v for p, v in zip(p5, vals, strict=True))
    mu2 = sum(p * (v * v) for p, v in zip(p5, vals, strict=True))
    var = mu2 - mu * mu
    return mu, mu2, var


def gen_pentanomial_outcomes(
    seed: int,
    n: int,
    p5: tuple[float, float, float, float, float],
) -> list[int]:
    """Generate n outcomes from a pentanomial distribution."""
    rng = random.Random(seed)  # noqa: S311
    vals = [-2, -1, 0, +1, +2]
    outs = rng.choices(vals, weights=p5, k=n)
    rng.shuffle(outs)
    return outs


# ----- init aggregates (same math as in Adam script, but no p5 leaks) -----


@dataclass(slots=True)
class InitStats:
    """Initial statistics for warm-starting the estimator."""

    reports: float = 0.0
    sum_n: float = 0.0
    sum_s: float = 0.0
    sum_s2_over_n: float = 0.0


def compute_init_stats_from_prior(
    p5: tuple[float, float, float, float, float],
    reports: float,
    mean_n: float,
) -> InitStats:
    """Compute initial statistics from a prior distribution."""
    if reports <= 0.0 or mean_n <= 0.0:
        return InitStats()
    mu_p, _mu2_p, var_p = compute_pentanomial_moments(p5)
    return InitStats(
        reports=reports,
        sum_n=reports * mean_n,
        sum_s=reports * mean_n * mu_p,
        sum_s2_over_n=reports * (var_p + mean_n * (mu_p * mu_p)),
    )


# ----- online estimator using only (s, N) -----


class OnlineReportStats:
    """Online estimator using only block-level summaries (s, N) per report.

    Maintains exact block-averaged aggregates (no EMA).
    """

    def __init__(self) -> None:
        """Initialize the online estimator."""
        self.reports: float = 0.0
        self.sum_n: float = 0.0
        self.sum_s: float = 0.0
        self.sum_s2_over_n: float = 0.0

    def apply_init_stats(self, init: InitStats) -> None:
        """Apply initial statistics to the estimator."""
        # Warm-start by adding externally computed aggregates.
        if init.reports <= 0.0:
            return
        self.reports += float(init.reports)
        self.sum_n += float(init.sum_n)
        self.sum_s += float(init.sum_s)
        self.sum_s2_over_n += float(init.sum_s2_over_n)

    def update(self, s: float, n: int) -> None:
        """Update the estimator with a new report."""
        if n <= 0:
            return
        self.reports += 1.0
        self.sum_n += float(n)
        self.sum_s += float(s)
        self.sum_s2_over_n += (float(s) * float(s)) / float(n)

    # Exact block-averaged estimates

    def mean(self) -> float:
        """Compute the mean estimate."""
        return (self.sum_s / self.sum_n) if self.sum_n > 0.0 else float("nan")

    def variance_block_avg(self) -> float:
        """Compute the block-averaged variance estimate."""
        if self.reports == 0.0 or self.sum_n == 0.0:
            return float("nan")
        e_s2_over_n = self.sum_s2_over_n / self.reports
        e_n = self.sum_n / self.reports
        mu = self.mean()
        sigma2 = e_s2_over_n - (mu * mu) * e_n
        return max(sigma2, 0.0)

    def second_moment_block_avg(self) -> float:
        """Compute the block-averaged second moment estimate."""
        mu = self.mean()
        sigma2 = self.variance_block_avg()
        if math.isnan(mu) or math.isnan(sigma2):
            return float("nan")
        return mu * mu + sigma2


def main() -> None:
    """Run the main simulation."""
    # True generator pentanomial (WL domain), same as in Adam script
    p5_true: tuple[float, float, float, float, float] = (
        0.025,
        0.20,
        0.55,
        0.20,
        0.025,
    )

    # External warm-start (adjust or set reports to 0.0 to disable)
    prior_p5: tuple[float, float, float, float, float] = (
        0.05,
        0.20,
        0.50,
        0.20,
        0.05,
    )
    prior_reports: float = 10.0  # 0.0 disables
    n_min, n_max = 1, 32
    prior_mean_n: float = (n_min + n_max) / 2.0

    # Theoretical per-pair stats
    mu_th, mu2_th, var_th = compute_pentanomial_moments(p5_true)
    print(  # noqa: T201
        "=== Theoretical per-pair statistics (from p5_true, WL domain) ===",
    )
    print(f"Mean (μ)              : {mu_th:.6f}")  # noqa: T201
    print(f"Variance (σ̂^2)        : {var_th:.6f}")  # noqa: T201, RUF001
    print(f"Second moment (μ2)    : {mu2_th:.6f}")  # noqa: T201
    print()  # noqa: T201

    # Build external init aggregates once
    init_stats = compute_init_stats_from_prior(prior_p5, prior_reports, prior_mean_n)

    # Print suggested μ2 init and aggregates for spsa_handler
    mu_prior, mu2_prior, var_prior = compute_pentanomial_moments(prior_p5)
    print(  # noqa: T201
        "=== Suggested μ2 init and aggregates for spsa_handler (from prior_p5) ===",
    )
    print(f"Prior Mean (μ_prior)        : {mu_prior:.6f}")  # noqa: T201
    print(f"Prior Variance (σ̂^2_prior)  : {var_prior:.6f}")  # noqa: T201, RUF001
    print(  # noqa: T201
        f"Prior Second moment (μ2_prior = E[x^2]) : {mu2_prior:.6f}",
    )
    print()  # noqa: T201
    print("Paste this block into your run['args']['spsa'] to seed μ2:")  # noqa: T201
    print("{")  # noqa: T201
    print(f'  "mu2_init": {mu2_prior:.12f},')  # noqa: T201
    print(f'  "mu2_reports": {init_stats.reports:.12f},')  # noqa: T201
    print(f'  "mu2_sum_N": {init_stats.sum_n:.12f},')  # noqa: T201
    print(f'  "mu2_sum_s": {init_stats.sum_s:.12f},')  # noqa: T201
    print(f'  "mu2_sum_s2_over_N": {init_stats.sum_s2_over_n:.12f}')  # noqa: T201
    print("}")  # noqa: T201
    print()  # noqa: T201

    # Simulate reports
    seed = 42
    n_reports = 1000
    rng = random.Random(seed)  # noqa: S311

    stats = OnlineReportStats()
    stats.apply_init_stats(init_stats)

    for _ in range(n_reports):
        n = rng.randint(n_min, n_max)
        outs = gen_pentanomial_outcomes(rng.randint(0, 10**9), n, p5_true)
        s = float(sum(outs))
        stats.update(s, n)

    # Exact block-averaged estimates
    mu_exact = stats.mean()
    var_exact = stats.variance_block_avg()
    mu2_exact = stats.second_moment_block_avg()

    print(  # noqa: T201
        "=== Online estimated per-pair statistics (exact block-avg) ===",
    )
    print(f"Mean (μ̂)             : {mu_exact:.6f}")  # noqa: T201
    print(f"Variance (σ̂^2)       : {var_exact:.6f}")  # noqa: T201, RUF001
    print(f"Second moment (μ̂2)   : {mu2_exact:.6f}")  # noqa: T201


if __name__ == "__main__":
    main()
