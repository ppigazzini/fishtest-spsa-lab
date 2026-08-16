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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "ELO_C",
    "FOLKLORE_C_DIVISOR",
    "adiabatic_ratio",
    "annealed_noise_ball",
    "chi2_ppf",
    "curvature_from_elo",
    "design_r",
    "folklore_c",
    "games_per_axis",
    "gauss_newton_c",
    "main",
    "noise_ball_elo",
    "quantile_elo",
    "relaxation_pairs",
]

#: The folklore rule for a Fishtest submission row: c_end = range / 20.
FOLKLORE_C_DIVISOR: float = 20.0

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


def curvature_from_elo(elo_over_range: float, param_range: float) -> float:
    """Elo curvature ``eps_j`` for a parameter worth ``E`` Elo over its range.

    With ``Elo(theta) = peak - (eps/2) * (theta - theta*)**2`` and the optimum at
    the centre of the range, the drop at the range edge is ``eps * R**2 / 8``.
    Setting that equal to ``E`` gives ``eps = 8E / R**2``.
    """
    if param_range <= 0.0:
        return math.nan
    return 8.0 * elo_over_range / (param_range * param_range)


def folklore_c(param_range: float) -> float:
    """The `c_end = range / 20` rule Fishtest submissions use by convention."""
    return param_range / FOLKLORE_C_DIVISOR


def gauss_newton_c(
    param_range: float,
    elo_over_range: float,
    scale: float = 1.0,
) -> float:
    """Per-axis probe satisfying the Gauss-Newton condition ``c_j**2 * eps_j = const``.

    The Remark in ``spsa_simul``'s ``theoretical_basis.tex`` states that when
    ``E(c c^T) Hess(e) = mu * I``, the average SPSA update is proportional to
    ``Hess^-1 grad`` -- the Newton direction. Diagonally that is
    ``c_j**2 * eps_j = const``, i.e. ``c_j`` proportional to
    ``1 / sqrt(curvature)``. Substituting ``eps_j = 8E_j / R_j**2`` gives

    ```text
    c_j  proportional to  R_j / sqrt(E_j)
    ```

    so the developer needs one extra number per row -- a rough Elo estimate --
    and nothing else. ``scale`` sets the overall aggressiveness, which the
    condition leaves free.
    """
    if param_range <= 0.0 or elo_over_range <= 0.0:
        return math.nan
    return scale * param_range / math.sqrt(elo_over_range)


def relaxation_pairs(r: float, mu: float) -> float:
    """Pairs for the mean iterate to relax by 1/e, the time constant ``C/(2*r*mu)``.

    ``mu`` is ``E(c c^T) Hess(e)`` -- under the Gauss-Newton condition a single
    scalar, which is the case this is written for.
    """
    if r <= 0.0 or mu <= 0.0:
        return math.inf
    return ELO_C / (2.0 * r * mu)


def annealed_noise_ball(
    gains: Sequence[float],
    n_active: int,
    sigma2: float,
    mu: float,
    initial_drop: float = 0.0,
) -> list[float]:
    """Track the noise floor under a *decaying* gain schedule.

    ``docs/Noise_ball.md`` derives the stationary floor at a FIXED gain,
    ``D(r) = (r/8)*C*n*sigma2``. Under a schedule the floor is a moving target:
    the process relaxes toward ``D(r_k)`` with time constant
    ``lambda_k = C/(2*r_k*mu)``, so

    ```text
    dD/dk = (D(r_k) - D) / lambda_k
    ```

    This integrates that. It is the quantitative form of the correction to
    ``spsa_simul``'s "decay only buys unreachable asymptotic convergence": the
    floor a decaying schedule relaxes toward shrinks with the iterations, so
    decay pays on any horizon long enough to track it -- and the second clause is
    what :func:`adiabatic_ratio` measures.

    Returns the drop after each step, as a positive Elo magnitude.
    """
    drop = float(initial_drop)
    out: list[float] = []
    for r in gains:
        target = -noise_ball_elo(r, n_active, sigma2)
        lam = relaxation_pairs(r, mu)
        if math.isfinite(lam) and lam > 0.0:
            drop += (target - drop) / lam
        out.append(drop)
    return out


def adiabatic_ratio(gains: Sequence[float], mu: float, index: int) -> float:
    """How fast the gain moves relative to how fast the process can follow it.

    The floor is only a floor if the schedule changes slowly compared with the
    relaxation time. With ``lambda_k = C/(2*r_k*mu)``,

    ```text
    adiabatic ratio = |d ln r / dk| * lambda_k
    ```

    Well below 1, the process sits at the instantaneous floor ``D(r_k)`` and
    ``Noise_ball.md``'s fixed-gain formula applies pointwise. Near or above 1, it
    lags and the fixed-gain formula overstates how tight the ball is.

    For the Fishtest schedule ``r_k`` proportional to ``k**(2*gamma)/(A+k)**alpha``,
    ``d ln r/dk -> (2*gamma - alpha)/k`` at large k, which is -0.4/k at the
    defaults -- so the condition is ``k >> 0.4 * lambda_k``.
    """
    if index <= 0 or index >= len(gains) - 1:
        return math.nan
    r = gains[index]
    dr = (gains[index + 1] - gains[index - 1]) / 2.0
    if r <= 0.0:
        return math.nan
    return abs(dr / r) * relaxation_pairs(r, mu)


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
    parser.add_argument(
        "--ranges",
        type=float,
        nargs="+",
        default=None,
        help="per-parameter ranges, for the c_end comparison",
    )
    parser.add_argument(
        "--elos",
        type=float,
        nargs="+",
        default=None,
        help=(
            "per-parameter Elo estimates over those ranges. Given with --ranges, "
            "compares c_end = range/20 against the Gauss-Newton rule."
        ),
    )
    return parser.parse_args(argv)


def _report_c_end(ranges: list[float], elos: list[float], r_eff: float) -> None:
    """Compare the folklore c_end rule against the Gauss-Newton one."""
    eps = [curvature_from_elo(e, rng) for e, rng in zip(elos, ranges, strict=True)]
    c_folk = [folklore_c(rng) for rng in ranges]

    # The condition fixes the shape and leaves the scale free; match the mean of
    # the folklore rule so the two are compared on aggressiveness, not on an
    # arbitrary constant.
    raw = [gauss_newton_c(rng, e) for rng, e in zip(ranges, elos, strict=True)]
    scale = sum(c_folk) / sum(raw)
    c_gn = [x * scale for x in raw]

    lam_folk = [games_per_axis(r_eff, c, e) for c, e in zip(c_folk, eps, strict=True)]
    lam_gn = [games_per_axis(r_eff, c, e) for c, e in zip(c_gn, eps, strict=True)]

    print()  # noqa: T201
    print(  # noqa: T201
        f"Per-axis c_end, at r = {r_eff:.3e}. "
        f"E_j is the Elo the parameter is worth over its range.",
    )
    header = (
        f"{'range':>10}{'E_j':>8}{'eps_j':>12}{'c=R/20':>10}{'c=GN':>10}"
        f"{'games @ R/20':>15}{'games @ GN':>15}"
    )
    print(header)  # noqa: T201
    print("-" * len(header))  # noqa: T201
    for rng, e, ep, cf, cg, lf, lg in zip(
        ranges, elos, eps, c_folk, c_gn, lam_folk, lam_gn, strict=True
    ):
        print(  # noqa: T201
            f"{rng:10.1f}{e:8.2f}{ep:12.3e}{cf:10.2f}{cg:10.2f}{lf:15,.0f}{lg:15,.0f}",
        )

    spread_folk = max(lam_folk) / min(lam_folk)
    print()  # noqa: T201
    print(  # noqa: T201
        f"c_end = range/20 : slowest axis {max(lam_folk):,.0f} games, "
        f"spread across axes {spread_folk:.1f}x",
    )
    print(  # noqa: T201
        f"Gauss-Newton     : slowest axis {max(lam_gn):,.0f} games, "
        f"spread across axes {max(lam_gn) / min(lam_gn):.1f}x",
    )
    print(  # noqa: T201
        f"The run is set by its slowest axis, so this is a "
        f"{max(lam_folk) / max(lam_gn):.2f}x saving.",
    )
    print()  # noqa: T201
    print(  # noqa: T201
        "Under c_end = range/20, c_j**2 * eps_j = E_j / 50: the RANGE CANCELS, "
        "and games-to-converge is inversely proportional to the Elo the "
        "parameter is worth. The slowest axis is the least valuable parameter, "
        "and the spread is exactly max(E)/min(E). The folklore rule is "
        "self-consistent only if every parameter is worth the same Elo.",
    )


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

    if args.ranges and args.elos:
        if len(args.ranges) != len(args.elos):
            print("--ranges and --elos must have the same length")  # noqa: T201
            return 1
        r_eff = design_r(args.precision, args.confidence, len(args.ranges), args.sigma2)
        _report_c_end(args.ranges, args.elos, r_eff)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
