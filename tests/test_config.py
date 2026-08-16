"""Smoke tests for SPSAConfig initialization, derived geometry, and sizing."""

import numpy as np
import pytest

from fishtest_spsa_lab.simulator.config import ParamGroup, SPSAConfig


def test_default_config_initializes() -> None:
    cfg = SPSAConfig()
    assert cfg.num_params > 0
    assert cfg.k_elo != 0.0, "k_elo should be derived in __post_init__"


def test_theta_vectors_shape() -> None:
    cfg = SPSAConfig()
    n = cfg.num_params
    assert cfg.theta_start.shape == (n,)
    assert cfg.theta_peak.shape == (n,)
    assert cfg.theta_min.shape == (n,)
    assert cfg.theta_max.shape == (n,)
    assert cfg.w_true.shape == (n,)
    assert cfg.w_dev.shape == (n,)


def test_c_dev_derived() -> None:
    cfg = SPSAConfig()
    assert cfg.c_dev is not None
    assert cfg.c_dev.shape == (cfg.num_params,)
    assert np.all(cfg.c_dev >= 0.0)


def test_gradient_scale_factor() -> None:
    cfg = SPSAConfig()
    assert 0.0 < cfg.gradient_scale_factor < 1.0

    cfg_noscale = SPSAConfig(scale_gradient_by_sqrt_num_params=False)
    assert cfg_noscale.gradient_scale_factor == 1.0


def test_lognormal_params() -> None:
    cfg = SPSAConfig()
    mu, sigma = cfg.get_lognormal_params()
    assert mu > 0.0
    assert sigma > 0.0


def test_design_budget_reproduces_the_measured_figures() -> None:
    """The sizing helper must agree with the analysis it was written from.

    __DEV/260816-0-REPORT.md section 10.4 measured these by hand before the
    helper existed; if they drift, one of the two is wrong.
    """
    budget = SPSAConfig(num_pairs=833 * 36, batch_size=36).design_budget()
    assert budget is not None

    per_axis = sorted({round(float(x)) for x in budget.lambda_per_axis})
    assert per_axis == [50926, 101851, 203702]
    assert round(budget.slowest_axis_games) == 203702
    assert round(budget.recommended_games) == 1222213
    assert budget.budget_games == 59976
    # The lab's effective gain: r_end/2 from the halved signal, times 1/sqrt(22).
    assert budget.effective_r == pytest.approx(2.132e-04, rel=1e-3)


def test_the_default_sweep_budget_is_declared_insufficient() -> None:
    """The whole point: an undersized run must say so rather than rank."""
    budget = SPSAConfig(num_pairs=833 * 36, batch_size=36).design_budget()
    assert budget is not None
    assert not budget.is_sufficient
    assert budget.fraction_of_recommended == pytest.approx(0.0491, abs=1e-3)
    assert "1/20" in budget.summary()
    assert "no arm has converged" in budget.summary()


def test_a_sufficient_budget_is_recognised() -> None:
    """And a run at the recommended size must not carry the warning."""
    budget = SPSAConfig(num_pairs=700_000, batch_size=36).design_budget()
    assert budget is not None
    assert budget.is_sufficient
    assert "no arm has converged" not in budget.summary()


def test_design_budget_is_none_for_degenerate_geometry() -> None:
    """No active axes means there is nothing to size."""
    config = SPSAConfig(
        param_groups=[ParamGroup(count=2, theta_start=1.0, theta_peak=1.0, w_true=0.0)],
    )
    assert config.design_budget() is None
