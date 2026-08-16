"""Pin the SPSA design equations against published values and the prior audit."""

from __future__ import annotations

import math

import pytest

from fishtest_spsa_lab.analysis.design import (
    ELO_C,
    adiabatic_ratio,
    annealed_noise_ball,
    chi2_ppf,
    curvature_from_elo,
    design_r,
    folklore_c,
    games_per_axis,
    gauss_newton_c,
    main,
    noise_ball_elo,
    quantile_elo,
    relaxation_pairs,
)

#: Two-sided 95% chi-square quantiles from published tables.
PUBLISHED_CHI2 = {
    1: 3.841,
    2: 5.991,
    3: 7.815,
    4: 9.488,
    8: 15.507,
    12: 21.026,
    14: 23.685,
    22: 33.924,
    32: 46.194,
    64: 83.675,
}

#: __DEV/260809-0-REPORT.md section 5.1, at sigma2 = 0.2502 (the vendored model),
#: precision 0.5 Elo, confidence 0.95. Reproducing an independently computed
#: table is what makes this module trustworthy.
AUDIT_TABLE = {
    1: (0.011977, -0.130),
    2: (0.007679, -0.167),
    4: (0.004849, -0.211),
    8: (0.002967, -0.258),
    12: (0.002188, -0.285),
    14: (0.001943, -0.296),
    22: (0.001356, -0.324),
    32: (0.000996, -0.346),
    64: (0.000550, -0.382),
}

VENDORED_SIGMA2 = 0.2502


@pytest.mark.parametrize(("k", "expected"), sorted(PUBLISHED_CHI2.items()))
def test_chi2_matches_published_tables(k: int, expected: float) -> None:
    """Agreement is to the published rounding, 5e-4."""
    assert chi2_ppf(0.95, k) == pytest.approx(expected, abs=1e-3)


def test_chi2_is_monotone_in_both_arguments() -> None:
    """A quantile that is not monotone would make the design table meaningless."""
    for k in (1, 4, 32):
        assert chi2_ppf(0.5, k) < chi2_ppf(0.95, k) < chi2_ppf(0.99, k)
    for p in (0.5, 0.95):
        assert chi2_ppf(p, 1) < chi2_ppf(p, 4) < chi2_ppf(p, 32)


@pytest.mark.parametrize(("n", "expected"), sorted(AUDIT_TABLE.items()))
def test_design_table_reproduces_the_audit(
    n: int, expected: tuple[float, float]
) -> None:
    """Both the derived gain and the noise ball it buys."""
    expected_r, expected_ball = expected
    r = design_r(0.5, 0.95, n, VENDORED_SIGMA2)
    assert r == pytest.approx(expected_r, rel=1e-3)
    assert noise_ball_elo(r, n, VENDORED_SIGMA2) == pytest.approx(
        expected_ball, abs=1e-3
    )


def test_the_folklore_r_end_is_right_at_about_fourteen_parameters() -> None:
    """The claim the whole module exists to support."""
    folklore = 0.002
    ratios = {
        n: folklore / design_r(0.5, 0.95, n, VENDORED_SIGMA2) for n in AUDIT_TABLE
    }
    # Crosses 1 between 12 and 14 parameters.
    assert ratios[12] < 1.0 < ratios[14]
    # Six times over-conservative at one parameter, 3.6x too hot at 64.
    assert ratios[1] == pytest.approx(0.17, abs=0.02)
    assert ratios[64] == pytest.approx(3.64, abs=0.05)


def test_the_design_gain_falls_roughly_as_one_over_n() -> None:
    """chi2 grows about linearly in n, so r must fall about as 1/n."""
    r1 = design_r(0.5, 0.95, 1, VENDORED_SIGMA2)
    r32 = design_r(0.5, 0.95, 32, VENDORED_SIGMA2)
    # Exactly 1/n would give 32x; chi2's offset makes it milder.
    assert 8.0 < r1 / r32 < 16.0


def test_the_design_gain_hits_its_own_precision_target() -> None:
    """Self-consistency: at the design r, the confidence-quantile loss is the target.

    This is the definition of the design equation, so a mismatch means the two
    formulas have drifted apart.
    """
    for n in (1, 4, 12, 64):
        r = design_r(0.5, 0.95, n, VENDORED_SIGMA2)
        assert quantile_elo(r, n, VENDORED_SIGMA2, 0.95) == pytest.approx(
            -0.5, abs=1e-6
        )


def test_games_per_axis_matches_the_lab_configuration() -> None:
    """Cross-check against SPSAConfig.design_budget on the default geometry."""
    from fishtest_spsa_lab.simulator.config import SPSAConfig

    config = SPSAConfig(num_pairs=833 * 36, batch_size=36)
    budget = config.design_budget()
    assert budget is not None
    assert config.c_dev is not None

    active = config.w_true > 0
    eps = config.k_elo * config.w_true[active]
    c_j = config.c_dev[active]
    by_hand = max(
        games_per_axis(budget.effective_r, float(c), float(e))
        for c, e in zip(c_j, eps, strict=True)
    )
    assert by_hand == pytest.approx(budget.slowest_axis_games, rel=1e-9)


def test_degenerate_inputs_do_not_raise() -> None:
    """A design tool must fail visibly, not with a traceback."""
    assert math.isnan(chi2_ppf(0.0, 4))
    assert math.isnan(chi2_ppf(0.95, 0))
    assert math.isnan(design_r(0.5, 0.95, 0, VENDORED_SIGMA2))
    assert games_per_axis(0.0, 1.0, 1.0) == math.inf


def test_the_entry_point_runs(capsys: pytest.CaptureFixture) -> None:
    """spsa-design prints a table and exits 0."""
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "n_active" in out
    assert "design r" in out


def test_the_folklore_rule_makes_the_range_cancel() -> None:
    """The result that makes `c_end = range/20` indefensible for unequal Elo.

    c_j**2 * eps_j = (R**2/400) * (8E/R**2) = E/50. The range drops out entirely,
    so games-to-converge depends only on the Elo the parameter is worth.
    """
    for param_range in (10.0, 100.0, 1000.0):
        for elo in (0.2, 2.0):
            product = folklore_c(param_range) ** 2 * curvature_from_elo(
                elo, param_range
            )
            assert product == pytest.approx(elo / 50.0, rel=1e-12)


def test_gauss_newton_equalises_curvature_times_probe_squared() -> None:
    """The condition itself: c_j**2 * eps_j constant across axes."""
    ranges = [100.0, 40.0, 400.0, 60.0]
    elos = [2.0, 0.5, 4.0, 0.2]
    products = [
        gauss_newton_c(rng, e) ** 2 * curvature_from_elo(e, rng)
        for rng, e in zip(ranges, elos, strict=True)
    ]
    assert max(products) == pytest.approx(min(products), rel=1e-12)


def test_the_spread_under_the_folklore_rule_is_exactly_max_over_min_elo() -> None:
    """And Gauss-Newton makes it 1, so the run is set by no single weak axis."""
    ranges = [100.0, 40.0, 400.0, 60.0]
    elos = [2.0, 0.5, 4.0, 0.2]
    r = 2.132e-04

    eps = [curvature_from_elo(e, rng) for e, rng in zip(elos, ranges, strict=True)]
    lam_folk = [
        games_per_axis(r, folklore_c(rng), e)
        for rng, e in zip(ranges, eps, strict=True)
    ]
    lam_gn = [
        games_per_axis(r, gauss_newton_c(rng, el), e)
        for rng, el, e in zip(ranges, elos, eps, strict=True)
    ]

    assert max(lam_folk) / min(lam_folk) == pytest.approx(
        max(elos) / min(elos), rel=1e-9
    )
    assert max(lam_gn) / min(lam_gn) == pytest.approx(1.0, rel=1e-9)


def test_the_two_rules_coincide_when_every_parameter_is_worth_the_same() -> None:
    """The folklore rule is self-consistent exactly under that assumption."""
    ranges = [10.0, 100.0, 1000.0]
    elos = [2.0, 2.0, 2.0]
    raw = [gauss_newton_c(rng, e) for rng, e in zip(ranges, elos, strict=True)]
    folk = [folklore_c(rng) for rng in ranges]
    scale = sum(folk) / sum(raw)
    for got, want in zip((x * scale for x in raw), folk, strict=True):
        assert got == pytest.approx(want, rel=1e-12)


def test_the_c_end_comparison_runs(capsys: pytest.CaptureFixture) -> None:
    """The CLI path that produces the comparison table."""
    code = main(["--n", "4", "--ranges", "100", "40", "--elos", "2.0", "0.5"])
    out = capsys.readouterr().out
    assert code == 0
    assert "RANGE CANCELS" in out


def test_mismatched_ranges_and_elos_are_rejected() -> None:
    """A silent zip truncation would produce a plausible, wrong table."""
    assert main(["--ranges", "100", "40", "--elos", "2.0"]) == 1


def test_a_constant_gain_relaxes_to_the_closed_form_floor() -> None:
    """The correctness check for the annealed integrator.

    With a constant gain the moving target is stationary, so the tracked floor
    must converge to docs/Noise_ball.md's fixed-gain result exactly.
    """
    n, sigma2, mu, r = 12, 0.2274, 2.3333, 0.002
    steps = int(20 * relaxation_pairs(r, mu))
    tracked = annealed_noise_ball([r] * steps, n, sigma2, mu)
    assert tracked[-1] == pytest.approx(-noise_ball_elo(r, n, sigma2), rel=1e-6)


def test_a_decaying_gain_tracks_its_shrinking_floor_once_adiabatic() -> None:
    """8b: the floor is a moving target, and the condition for treating it as one.

    Early in the run the schedule outruns the process and the fixed-gain formula
    does not apply pointwise; once the adiabatic ratio falls well below 1 the
    tracked floor sits on the instantaneous one.
    """
    n, sigma2, mu, r_end = 12, 0.2274, 2.3333, 0.002
    total = 600_000
    a_stability, alpha, gamma = 0.1 * total, 0.602, 0.101
    norm = (total ** (2 * gamma)) / ((a_stability + total) ** alpha)
    gains = [
        r_end * ((k ** (2 * gamma)) / ((a_stability + k) ** alpha)) / norm
        for k in range(1, total + 1)
    ]

    tracked = annealed_noise_ball(gains, n, sigma2, mu)
    instantaneous = [-noise_ball_elo(g, n, sigma2) for g in gains]

    # Early: not adiabatic, and the process lags far behind.
    assert adiabatic_ratio(gains, mu, 5_000) > 0.5
    assert tracked[5_000] / instantaneous[5_000] < 0.3

    # Late: adiabatic, and tracking to within a few percent.
    assert adiabatic_ratio(gains, mu, 200_000) < 0.1
    assert tracked[200_000] / instantaneous[200_000] == pytest.approx(1.0, abs=0.1)

    # And the floor genuinely shrinks over the tail, which is the whole point:
    # a decaying gain lowers the target it is relaxing toward.
    assert instantaneous[-1] < instantaneous[50_000]


def test_relaxation_time_is_the_documented_constant() -> None:
    """lambda = C / (2 * r * mu), from spsa_simul's Appendix C."""
    assert relaxation_pairs(0.002, 2.3333) == pytest.approx(
        ELO_C / (2 * 0.002 * 2.3333), rel=1e-12
    )
    assert relaxation_pairs(0.0, 1.0) == math.inf
