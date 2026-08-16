"""Pin the SPSA design equations against published values and the prior audit."""

from __future__ import annotations

import math

import pytest

from fishtest_spsa_lab.analysis.design import (
    chi2_ppf,
    design_r,
    games_per_axis,
    main,
    noise_ball_elo,
    quantile_elo,
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
