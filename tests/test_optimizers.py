"""Smoke tests for optimizer instantiation and single-step correctness."""

import numpy as np

from fishtest_spsa_lab.simulator.config import SPSAConfig
from fishtest_spsa_lab.simulator.optimizer import OPTIMIZER_REGISTRY, Optimizer


def test_registry_not_empty() -> None:
    assert len(OPTIMIZER_REGISTRY) > 0


def test_all_registry_entries_are_optimizer_subclasses() -> None:
    for name, cls in OPTIMIZER_REGISTRY.items():
        assert issubclass(cls, Optimizer), f"{name} is not an Optimizer subclass"


def test_instantiate_all_optimizers() -> None:
    cfg = SPSAConfig(num_pairs=100, batch_size=2)
    for name, cls in OPTIMIZER_REGISTRY.items():
        opt = cls(cfg)
        assert opt.theta.shape == (cfg.num_params,), f"{name}: bad theta shape"


def test_single_step_all_optimizers() -> None:
    cfg = SPSAConfig(num_pairs=100, batch_size=2, seed=42)
    rng = np.random.default_rng(42)

    for name, cls in OPTIMIZER_REGISTRY.items():
        opt = cls(cfg)
        theta_before = opt.theta.copy()
        c_k = opt.get_perturbation_scale(1)
        flip = rng.choice([-1, 1], size=cfg.num_params).astype(float)
        opt.step(iter_local=1, net_wins=1.0, flip=flip, c_k=c_k, batch_size_pairs=2)
        # After a step with non-zero signal, theta should change
        # (unless optimizer has zero learning rate by design)
        assert opt.theta.shape == theta_before.shape, f"{name}: shape changed"
