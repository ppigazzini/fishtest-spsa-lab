"""Size an SPSA run from a target precision, and derive the r_end it implies.

This is the concrete, Fishtest-facing item of MS8 (8d). Fishtest's submission
form offers a single per-row ``r_end`` with no guidance, and the folklore value
is 0.002 regardless of how many parameters are being tuned. The design equations
of Van den Bergh's ``spsa_simul`` say that is wrong in a specific, quantified
way: the stationary Elo loss grows with the number of Elo-active parameters, so
holding ``r_end`` fixed holds the *gain* fixed while the *cost* of that gain
grows linearly.

```text
r          = 8 * precision / (C * chi2_ppf(confidence, n_active) * sigma2)
E[Elo]     = -(r / 8) * C * n_active * sigma2          the stationary noise ball
P(e > -x)  = gamma at x = (r/8) * C * chi2_ppf(gamma, n) * sigma2
lambda_j   = C / (8 * r * c_j**2 * eps_j)              games to converge axis j
num_games  = lambda_ratio * 2 * max_j lambda_j         lambda_ratio = 3
```

``chi2_ppf`` grows roughly linearly in ``n``, so the design ``r`` falls roughly
as ``1/n``. The folklore 0.002 is right at about 14 parameters and wrong
everywhere else: six times too conservative at one parameter, where it buys a
noise ball far tighter than anyone asked for at the price of convergence speed,
and about 3.6 times too hot at 64, where the tune sits 0.38 Elo deep in noise
while appearing converged.

Run ``spsa-design`` for the table. The Fishtest-facing proposal it supports is a
form change: compute the suggested ``r_end`` from (precision, confidence,
n_active, draw ratio) rather than defaulting to a constant.

Note the convention. ``sigma2`` here is ``var(net pair outcome) / 2``, matching
``simulator/oracle.py``; the calibrated LTC value is 0.2274 and the vendored
model gives 0.250225. The lab's own effective gain carries two further factors
that Fishtest does not have -- the halved signal and ``1/sqrt(N)`` -- so
``SPSAConfig.design_budget()`` is the right tool for sizing a *lab* run, and this
module is the right tool for advising *Fishtest*.
"""

from __future__ import annotations

import argparse
import math

__all__ = [
    "ELO_C",
    "chi2_ppf",
    "design_r",
    "games_per_axis",
    "main",
    "noise_ball_elo",
    "quantile_elo",
]

#: 800 / ln(10).
ELO_C: float = 800.0 / math.log(10.0)

#: Van den Bergh's default: run 3x the two-sided convergence time constant.
LAMBDA_RATIO: float = 3.0

_GAMMA_MAX_ITER: int = 500
_GAMMA_EPS: float = 3.0e-16
_TINY: float = 1.0e-300


def _lower_gamma_series(a: float, x: float) -> float:
    """Regularized lower incomplete gamma P(a, x) by series; good for x < a+1."""
    ap = a
    total = 1.0 / a
    delta = total
    for _ in range(_GAMMA_MAX_ITER):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * _GAMMA_EPS:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _upper_gamma_cf(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x) by continued fraction."""
    b = x + 1.0 - a
    c = 1.0 / _TINY
    d = 1.0 / b
    h = d
    for i in range(1, _GAMMA_MAX_ITER + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _TINY:
            d = _TINY
        c = b + an / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _GAMMA_EPS:
            break
    return h * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _chi2_cdf(x: float, k: int) -> float:
    """CDF of the chi-square distribution with ``k`` degrees of freedom."""
    if x <= 0.0:
        return 0.0
    a = 0.5 * k
    z = 0.5 * x
    if z < a + 1.0:
        return _lower_gamma_series(a, z)
    return 1.0 - _upper_gamma_cf(a, z)


def chi2_ppf(p: float, k: int) -> float:
    """Quantile of the chi-square distribution, by bisection on the CDF.

    Bisection rather than a rational approximation: this is called a handful of
    times to build a table, and an exactly-monotone inverse of the CDF above
    cannot disagree with it.
    """
    if not (0.0 < p < 1.0) or k <= 0:
        return math.nan
    lo, hi = 0.0, max(1.0, float(k))
    while _chi2_cdf(hi, k) < p:
        hi *= 2.0
        if hi > 1e12:  # noqa: PLR2004
            return hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _chi2_cdf(mid, k) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def design_r(
    precision: float,
    confidence: float,
    n_active: int,
    sigma2: float,
) -> float:
    """The gain that reaches ``precision`` Elo of the optimum with ``confidence``."""
    chi2 = chi2_ppf(confidence, n_active)
    if not math.isfinite(chi2) or chi2 <= 0.0:
        return math.nan
    return 8.0 * precision / (ELO_C * chi2 * sigma2)


def noise_ball_elo(r: float, n_active: int, sigma2: float) -> float:
    """Stationary expected Elo loss at gain ``r``. Negative."""
    return -(r / 8.0) * ELO_C * n_active * sigma2


def quantile_elo(r: float, n_active: int, sigma2: float, gamma: float) -> float:
    """Elo loss not exceeded with probability ``gamma``. Negative."""
    return -(r / 8.0) * ELO_C * chi2_ppf(gamma, n_active) * sigma2


def games_per_axis(r: float, c_j: float, eps_j: float) -> float:
    """Games needed to converge one axis with curvature ``eps_j`` and probe ``c_j``."""
    if r <= 0.0 or c_j <= 0.0 or eps_j <= 0.0:
        return math.inf
    return ELO_C / (8.0 * r * c_j * c_j * eps_j)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Derive the SPSA gain r_end from a target precision and the number "
            "of Elo-active parameters, and show what a fixed 0.002 costs."
        ),
    )
    parser.add_argument(
        "--precision",
        type=float,
        default=0.5,
        help="target Elo below the optimum (default: 0.5)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="probability of achieving it (default: 0.95)",
    )
    parser.add_argument(
        "--sigma2",
        type=float,
        default=0.2274,
        help=(
            "var(net pair outcome)/2. Default 0.2274 is the calibrated LTC value; "
            "0.2654 is STC, 0.2133 VLTC, 0.250225 the vendored model."
        ),
    )
    parser.add_argument(
        "--folklore",
        type=float,
        default=0.002,
        help="the constant r_end Fishtest defaults to (default: 0.002)",
    )
    parser.add_argument(
        "--n",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 12, 14, 22, 32, 64],
        help="parameter counts to tabulate",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Print the r_end-vs-n_active design table."""
    args = _parse_args(argv)

    print(  # noqa: T201
        f"SPSA design table: precision {args.precision} Elo at "
        f"{args.confidence:.0%} confidence, sigma2 = {args.sigma2}",
    )
    print(  # noqa: T201
        "sigma2 is var(net pair outcome)/2; see simulator/oracle.py\n",
    )
    header = (
        f"{'n_active':>9}{'chi2':>10}{'design r':>12}{'ball @ design':>15}"
        f"{'ball @ ' + str(args.folklore):>15}{'folklore is':>14}"
    )
    print(header)  # noqa: T201
    print("-" * len(header))  # noqa: T201

    for n in args.n:
        chi2 = chi2_ppf(args.confidence, n)
        r = design_r(args.precision, args.confidence, n, args.sigma2)
        ball_design = noise_ball_elo(r, n, args.sigma2)
        ball_folklore = noise_ball_elo(args.folklore, n, args.sigma2)
        ratio = args.folklore / r if r > 0.0 else math.inf
        print(  # noqa: T201
            f"{n:9d}{chi2:10.2f}{r:12.6f}{ball_design:15.3f}"
            f"{ball_folklore:15.3f}{ratio:13.2f}x",
        )

    print()  # noqa: T201
    print(  # noqa: T201
        "'folklore is' is the constant r_end divided by the design value: above "
        "1 means too hot (a deeper noise ball than requested), below 1 means "
        "over-conservative (convergence speed given away for precision nobody "
        "asked for).",
    )
    print(  # noqa: T201
        "Fishtest proposal: compute the suggested r_end from (precision, "
        "confidence, n_active, draw ratio) on the submission form instead of "
        "defaulting to a constant. r scales as 1/chi2(confidence, n), i.e. "
        "roughly 1/n.",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
