"""Pin the pair-aware oracle against the vendored model and the TC targets."""

from __future__ import annotations

import numpy as np
import pytest

from fishtest_spsa_lab.simulator.config import SPSAConfig
from fishtest_spsa_lab.simulator.oracle import (
    TIME_CONTROL_SIGMA2,
    VENDORED_SPREAD,
    PairOracle,
    calibrate_spread,
    pair_sigma2,
)
from fishtest_spsa_lab.simulator.runner import GameProvider, SpsaRunner
from fishtest_spsa_lab.vendor.pentamodel.pentamodel import PentaModel

BOOK_SIGMA = 60.0


@pytest.mark.parametrize("elo", [0.0, 1.0, -2.5, 10.0, -0.5])
def test_reimplemented_logistic_reproduces_the_vendored_model(elo: float) -> None:
    """Configured identically, the two must agree to machine precision.

    The per-game logistic is reimplemented only so that the spread can move. If
    it drifted from the vendored form, every comparison against historical
    results would silently shift.
    """
    reference = np.asarray(PentaModel(opponentElo=elo).pentanomialProbs, dtype=float)
    oracle = PairOracle(
        book_sigma=100.0,
        spread=VENDORED_SPREAD,
        deterministic_exit=True,
    )
    assert np.abs(oracle.pentanomial_probs(elo) - reference).max() < 1e-15


def test_the_vendored_model_has_no_within_pair_correlation() -> None:
    """The defect this module exists to fix, asserted rather than remembered."""
    oracle = PairOracle(
        book_sigma=100.0,
        spread=VENDORED_SPREAD,
        deterministic_exit=True,
    )
    assert abs(oracle.pair_correlation(0.0)) < 1e-12


@pytest.mark.parametrize("tc", sorted(TIME_CONTROL_SIGMA2))
def test_each_time_control_is_reachable(tc: str) -> None:
    """LTC and VLTC sit below the vendored model's floor and were unreachable."""
    target = TIME_CONTROL_SIGMA2[tc]
    spread = calibrate_spread(target, book_sigma=BOOK_SIGMA)
    oracle = PairOracle(book_sigma=BOOK_SIGMA, spread=spread)
    achieved = pair_sigma2(oracle.pentanomial_probs(0.0))

    assert abs(achieved / target - 1.0) < 0.01, f"{tc}: {achieved} vs {target}"
    # The point of the whole exercise: pairing information must be present, and
    # with the RIGHT SIGN. Reversed-colour pairing reduces variance because the
    # shared opening pushes the two games in opposite directions, so the
    # correlation must be negative. Measured -0.229 (STC) to -0.271 (VLTC).
    assert oracle.pair_correlation(0.0) < -0.01


def test_the_vendored_model_cannot_reach_the_long_time_controls() -> None:
    """Pin the floor, so the motivation for this module cannot quietly evaporate.

    With a deterministic exit the pentanomial is frozen near
    [0, 0.25, 0.5, 0.25, 0] whatever the spread, because one game is a coin flip
    between win and draw and the other between loss and draw.
    """
    floor = min(
        pair_sigma2(
            PairOracle(
                book_sigma=100.0,
                spread=b,
                deterministic_exit=True,
            ).pentanomial_probs(0.0),
        )
        for b in (0.5, 5.0, 22.0, 60.0, 200.0, 800.0)
    )
    assert floor >= 0.24
    assert TIME_CONTROL_SIGMA2["LTC"] < floor
    assert TIME_CONTROL_SIGMA2["VLTC"] < floor


def test_probabilities_are_a_distribution() -> None:
    """Quadrature must not leak mass, including in the far tail."""
    for book_sigma in (1.0, 60.0, 1000.0):
        probs = PairOracle(book_sigma=book_sigma).pentanomial_probs(0.0)
        assert probs.shape == (5,)
        assert np.all(probs >= 0.0)
        assert abs(float(probs.sum()) - 1.0) < 1e-12


def test_the_default_config_keeps_the_vendored_oracle() -> None:
    """Switching the oracle changes every number, so it must be deliberate."""
    assert SPSAConfig().time_control is None
    assert GameProvider(SPSAConfig()).pair_oracle is None


def test_an_unknown_time_control_is_rejected() -> None:
    """A typo must not silently fall back to the vendored model."""
    with pytest.raises(ValueError, match="unknown time_control"):
        GameProvider(SPSAConfig(time_control="BLITZ"))


def test_the_pair_oracle_runs_end_to_end() -> None:
    """Selecting a time control must drive a full simulation."""
    config = SPSAConfig(num_pairs=360, batch_size=36, seed=1, time_control="LTC")
    result = SpsaRunner(config).run()
    assert np.isfinite(result["convergence_metrics"]["final_elo"])
