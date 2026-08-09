"""Regressions for defects that produced silently wrong numbers.

Each test here corresponds to a measured defect. They are cheap and they fail
loudly if the fix is reverted.
"""

from __future__ import annotations

import numpy as np
import pytest

from fishtest_spsa_lab.simulator.config import ParamGroup, SPSAConfig
from fishtest_spsa_lab.simulator.optimizer import OPTIMIZER_REGISTRY, SFSGD
from fishtest_spsa_lab.simulator.runner import SpsaRunner


def test_get_params_returns_a_copy() -> None:
    """A stored result must not change when the optimizer steps again."""
    config = SPSAConfig(num_pairs=100, batch_size=10, seed=1)
    runner = SpsaRunner(config)
    result = runner.run()

    final = result["final_params"]
    snapshot = final.copy()

    optimizer = runner.optimizer
    optimizer.step(
        1,
        5.0,
        np.ones(config.num_params),
        optimizer.get_perturbation_scale(1),
        10,
    )

    assert np.array_equal(final, snapshot), "final_params aliased live state"


def test_perturbation_scale_is_not_shared_between_optimizers() -> None:
    """Each optimizer owns its c vector; the config must not be writable."""
    config = SPSAConfig(num_pairs=100, batch_size=10)
    first = SFSGD(config)
    second = SFSGD(config)

    assert first.c_constant is not second.c_constant
    assert first.c_constant is not config.c_dev

    first.c_constant[0] = -999.0
    assert second.c_constant[0] != -999.0
    assert config.c_dev is not None
    assert config.c_dev[0] != -999.0


def test_integer_param_group_fields_produce_a_float_theta() -> None:
    """Integer literals in a ParamGroup must not make theta unsteppable."""
    config = SPSAConfig(
        num_pairs=100,
        batch_size=10,
        seed=1,
        param_groups=[
            ParamGroup(count=2, theta_start=900, theta_peak=1000, w_true=1),
        ],
    )
    assert config.theta_start.dtype == np.float64
    SpsaRunner(config).run()


def test_same_seed_is_bit_identical() -> None:
    """Reproducibility is a precondition for every comparative claim."""

    def run(seed: int) -> np.ndarray:
        config = SPSAConfig(num_pairs=2000, batch_size=10, seed=seed)
        return SpsaRunner(config).run()["final_params"]

    assert np.array_equal(run(7), run(7))
    assert not np.array_equal(run(7), run(8))


def test_perturbations_are_common_across_optimizer_arms() -> None:
    """A seed sweep is only a paired comparison if the arms see the same flips."""
    draws = {}
    for name in ("spsa", "adam", "sf-adam"):
        config = SPSAConfig(optimizer=name, num_pairs=500, batch_size=10, seed=123)
        runner = SpsaRunner(config)
        draws[name] = [
            runner.rng_perturb.choice([-1, 1], size=config.num_params).copy()
            for _ in range(50)
        ]

    for name in ("adam", "sf-adam"):
        matches = sum(
            int(np.array_equal(a, b))
            for a, b in zip(draws["spsa"], draws[name], strict=True)
        )
        assert matches == 50, f"{name} diverged from the shared stream"


def test_total_pairs_reports_what_was_simulated() -> None:
    """num_batches truncates, so the requested budget is not always spent."""
    config = SPSAConfig(num_pairs=95, batch_size=10, seed=1)
    result = SpsaRunner(config).run()
    reported = result["convergence_metrics"]["total_pairs"]
    assert reported == pytest.approx(result["pairs_history"][-1])
    assert reported == pytest.approx(90.0)


def test_every_optimizer_moves_theta_in_the_signalled_direction() -> None:
    """The previous smoke test only asserted that theta.shape survived."""
    config = SPSAConfig(num_pairs=100, batch_size=2, seed=42)
    flip = np.ones(config.num_params)

    for name, cls in OPTIMIZER_REGISTRY.items():
        optimizer = cls(config)
        before = optimizer.get_params()
        c_k = optimizer.get_perturbation_scale(1)
        optimizer.step(1, 10.0, flip, c_k, 2)
        after = optimizer.get_params()

        moved = after - before
        assert np.any(moved != 0.0), f"{name}: theta did not move"
        active = moved[moved != 0.0]
        assert np.all(active > 0.0), f"{name}: moved against a positive signal"
