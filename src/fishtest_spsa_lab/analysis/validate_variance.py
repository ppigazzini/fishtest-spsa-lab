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

from .gate import Gate
from .pentanomial import (
    OnlineReportStats,
    compute_init_stats_from_prior,
    compute_pentanomial_moments,
    gen_pentanomial_outcomes,
)

#: Replications used to give the convergence claim a standard error.
N_TRIALS: int = 40

#: Stride between per-trial seeds, so consecutive trials cannot share a stream.
PRIME_STRIDE: int = 7919


# ----- exact helpers copied from validate_sf_adam helpers -----


def _mean_and_se(sample: list[float]) -> tuple[float, float]:
    """Return the sample mean and the standard error OF THE MEAN.

    The distinction matters: quoting the per-trial spread as an error bar makes
    the interval sqrt(N) times too wide, which hides exactly the small bias this
    script exists to detect.
    """
    n = len(sample)
    mean = sum(sample) / n
    if n < 2:  # noqa: PLR2004
        return mean, float("inf")
    var = sum((x - mean) ** 2 for x in sample) / (n - 1)
    return mean, math.sqrt(var / n)


def main() -> int:
    """Estimate per-pair moments online and gate the convergence claim."""
    gate = Gate(
        "validate-variance",
        "the online block-averaged estimator converges to the true moments",
    )

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

    # Replicate. The docstring claims this script "verifies that the online
    # estimate converges"; it previously ran ONE seed, printed 0.590112 against a
    # truth of 0.600000 and stopped. The single-run standard deviation is around
    # 0.027, so -1.6% is 0.36 sd -- that run could neither confirm nor refute its
    # own claim. Replicating turns the print into a measurement with a standard
    # error, and the tolerance is derived from that error rather than picked.
    base_seed = 42
    n_reports = 1000

    mus: list[float] = []
    variances: list[float] = []
    mu2s: list[float] = []
    for trial in range(N_TRIALS):
        rng = random.Random(base_seed + trial * PRIME_STRIDE)  # noqa: S311
        stats = OnlineReportStats()
        stats.apply_init_stats(init_stats)
        for _ in range(n_reports):
            n = rng.randint(n_min, n_max)
            outs = gen_pentanomial_outcomes(rng.randint(0, 10**9), n, p5_true)
            stats.update(float(sum(outs)), n)
        mus.append(stats.mean())
        variances.append(stats.variance_block_avg())
        mu2s.append(stats.second_moment_block_avg())

    print(  # noqa: T201
        "=== Online estimated per-pair statistics (exact block-avg) ===",
    )
    print(f"Trials                : {N_TRIALS} x {n_reports} reports")  # noqa: T201
    for label, sample, truth in (
        ("Mean (μ̂)", mus, mu_th),
        ("Variance (σ̂^2)", variances, var_th),  # noqa: RUF001
        ("Second moment (μ̂2)", mu2s, mu2_th),
    ):
        mean, se = _mean_and_se(sample)
        print(f"{label:<22}: {mean:.6f} +- {se:.6f} (truth {truth:.6f})")  # noqa: T201
    print()  # noqa: T201

    gate.note("trials", N_TRIALS)
    gate.note("reports per trial", n_reports)
    gate.note("pairs per report", f"{n_min}..{n_max}")
    gate.note("base seed", base_seed)
    gate.note("warm-start prior mu2", mu2_prior)
    gate.note("true mu2", mu2_th)

    for label, sample, truth in (
        ("mean", mus, mu_th),
        ("variance", variances, var_th),
        ("second moment", mu2s, mu2_th),
    ):
        mean, se = _mean_and_se(sample)
        gate.check_within_se(
            f"estimated {label} converges to the truth",
            mean,
            truth,
            se,
        )

    return gate.report()


if __name__ == "__main__":
    raise SystemExit(main())
