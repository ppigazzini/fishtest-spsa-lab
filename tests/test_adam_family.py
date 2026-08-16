"""Pin the structural facts about the Adam family under a Rademacher proxy.

The SPSA gradient proxy is ``scalar * flip`` with ``flip`` in ``{-1, +1}``, so
``grad**2 == scalar**2`` identically in every coordinate and the second moment
``v`` cannot differentiate between parameters. All five Adam-family entries
therefore reduce to normalized-momentum SGD with one global step size.

This is not a defect and there is nothing to fix. It is pinned here because it
determines what a result from those optimizers *means*, and because it has now
been rediscovered by two separate audits. ``docs/Simulator.md`` section 2.6 says
the same thing in prose.
"""

from __future__ import annotations

import numpy as np
import pytest

from fishtest_spsa_lab.simulator.config import SPSAConfig
from fishtest_spsa_lab.simulator.runner import SpsaRunner

ADAM_FAMILY = ["adam", "adam-block", "sf-adam", "sf-adam-block", "ademamix"]

#: Attribute holding the second moment, per optimizer.
SECOND_MOMENT = {"ademamix": "nu"}


@pytest.mark.parametrize("name", ADAM_FAMILY)
def test_second_moment_is_coordinate_uniform(name: str) -> None:
    """``v`` is identical in every coordinate, exactly, not approximately."""
    config = SPSAConfig(num_pairs=1800, batch_size=36, seed=3, optimizer=name)
    runner = SpsaRunner(config)
    runner.run()

    attr = SECOND_MOMENT.get(name, "v")
    second_moment = np.asarray(getattr(runner.optimizer, attr), dtype=float)

    assert second_moment.size == config.num_params
    spread = float(second_moment.max() - second_moment.min())
    assert spread == 0.0, (
        f"{name}: v spread {spread:.3e} is non-zero. If a per-coordinate signal "
        f"was introduced deliberately, update docs/Simulator.md section 2.6, "
        f"which states the Adam family has no per-parameter adaptivity here."
    )


def test_the_reason_is_the_rademacher_square() -> None:
    """Spell out the mechanism, so the test above is not mistaken for a quirk."""
    rng = np.random.default_rng(0)
    flip = rng.choice([-1, 1], size=64).astype(float)
    scalar = -3.5
    grad = scalar * flip

    assert np.all(grad**2 == scalar**2)
