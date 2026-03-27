"""Smoke tests for SPSAConfig initialization and derived geometry."""

import numpy as np

from fishtest_spsa_lab.simulator.config import SPSAConfig


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
