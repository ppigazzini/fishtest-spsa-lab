"""Numerical properties of the optimizer implementations.

These pin the three math defects that shipped: an Adam second moment that is
uniform across coordinates by construction, a schedule-free Adam step that
scaled with the batch size, and a block-Adam closed form missing its bias
correction.
"""

from __future__ import annotations

import numpy as np
import pytest

from fishtest_spsa_lab.simulator.config import SPSAConfig
from fishtest_spsa_lab.simulator.optimizer import (
    OPTIMIZER_REGISTRY,
    Adam,
    AdamBlock,
    ScheduleFreeCore,
)

ADAM_FAMILY = ("adam", "adam-block", "sf-adam", "sf-adam-block", "ademamix")


@pytest.mark.parametrize("name", ADAM_FAMILY)
def test_adam_second_moment_is_coordinate_uniform(name: str) -> None:
    """Document, in a test, that Adam has no per-parameter adaptivity here.

    The SPSA gradient is ``scalar * flip`` with ``flip`` in ``{-1, +1}``, so
    ``grad ** 2 == scalar ** 2`` for every coordinate and the second moment is
    identical everywhere. These optimizers are therefore normalized-momentum
    SGD with a single global step size, not Adam. If this test ever fails, the
    gradient proxy has gained per-coordinate structure and the docs claiming
    otherwise must be revisited.
    """
    config = SPSAConfig(num_pairs=3000, batch_size=36, seed=1)
    rng = np.random.default_rng(1)
    optimizer = OPTIMIZER_REGISTRY[name](config)

    for k in range(1, 60):
        flip = rng.choice([-1, 1], size=config.num_params).astype(float)
        optimizer.step(
            k,
            float(rng.normal(0.0, 5.0)),
            flip,
            optimizer.get_perturbation_scale(k),
            36,
        )

    # AdEMAMix names its second moment `nu`; the rest use `v`.
    second_moment = getattr(optimizer, "v", None)
    if second_moment is None:
        second_moment = getattr(optimizer, "nu", None)
    assert second_moment is not None, f"{name} exposes no second moment"
    assert float(second_moment.max() - second_moment.min()) == 0.0


@pytest.mark.parametrize("name", ("sf-adam", "sf-adam-block"))
def test_schedule_free_adam_step_does_not_scale_with_batch_size(name: str) -> None:
    """A fixed per-report signal must not produce a step proportional to N.

    ``sf-adam`` previously put the block mean into ``v`` and the block sum into
    the numerator, so the step grew linearly with the batch size and the fast
    iterate diverged under the default sweep.
    """
    steps = []
    for batch in (2, 36, 128):
        config = SPSAConfig(num_pairs=100_000, batch_size=batch, seed=1)
        optimizer = OPTIMIZER_REGISTRY[name](config)
        assert isinstance(optimizer, ScheduleFreeCore)
        z_before = optimizer.z.copy()
        c_k = optimizer.get_perturbation_scale(1)
        optimizer.step(1, 8.0, np.ones(config.num_params), c_k, batch)
        steps.append(float(np.abs((optimizer.z - z_before) / c_k)[0]))

    growth = steps[-1] / steps[0]
    linear_growth = 128 / 2
    assert growth < 0.5 * linear_growth, f"{name}: step grew {growth:.1f}x with N"


def test_adam_block_tracks_micro_adam_after_bias_correction() -> None:
    """The closed form must follow N textbook micro-steps to a stated bound.

    Without bias correction the first block overshot by 4.0x.
    """
    config = SPSAConfig(num_pairs=100_000, batch_size=36, seed=1)
    block = AdamBlock(config)
    micro = Adam(config)
    rng = np.random.default_rng(0)

    ratios = []
    for k in range(1, 11):
        flip = rng.choice([-1, 1], size=config.num_params).astype(float)
        net_wins = float(rng.normal(0.0, 5.0))

        block_before = block.get_params()
        micro_before = micro.get_params()
        block.step(k, net_wins, flip, block.get_perturbation_scale(k), 36)
        micro.step(k, net_wins, flip, micro.get_perturbation_scale(k), 36)

        moved_block = float(np.linalg.norm(block.get_params() - block_before))
        moved_micro = float(np.linalg.norm(micro.get_params() - micro_before))
        if moved_micro > 0.0:
            ratios.append(moved_block / moved_micro)

    assert ratios
    assert max(ratios) < 1.5, f"block/micro displacement ratio {max(ratios):.3f}"
    assert min(ratios) > 0.5, f"block/micro displacement ratio {min(ratios):.3f}"
