"""Pin every shipped block optimizer against its clean-room re-implementation.

This is the test that was missing. ``analysis/validate_*.py`` reimplements each
update rule independently, which is what makes those scripts a real check of the
derivation -- and also means a change to ``simulator/optimizer.py`` alone cannot
fail them. Two defects walked straight through that gap in a single commit:

- ``1bea8b1`` removed the k(N, beta2) damping from ``SFAdamBlock`` and left
  ``validate_sf_adam.py`` applying it, a divergence of up to 6.6x.
- The same commit added Adam bias correction to ``AdamBlock`` and left
  ``validate_adam.py`` without it, a divergence of 21x.

Both gates stayed green because nothing compared the two implementations.
Here each pair is driven over one scripted sequence of (N, s) reports and
required to agree to float tolerance.

Parity is asserted on the *update rule*, so each test constructs the
analysis-side schedule from the simulator's own derived constants rather than
restating them. If the simulator changes how it derives a_base or c_base, that
is a separate question and ``validate-*`` covers it.
"""

from __future__ import annotations

import numpy as np
import pytest

from fishtest_spsa_lab.analysis import validate_adam as va
from fishtest_spsa_lab.analysis import validate_sf_adam as vsa
from fishtest_spsa_lab.analysis import validate_sf_sgd as vss
from fishtest_spsa_lab.analysis import validate_spsa as vs
from fishtest_spsa_lab.simulator.config import ParamGroup, SPSAConfig
from fishtest_spsa_lab.simulator.optimizer import (
    AdamBlock,
    SFAdamBlock,
    SFSGDBlock,
    SPSABlock,
)

#: Reports as (pairs, sum-of-outcomes). Deliberately uneven: a constant batch
#: size hides every error that scales with N, which is the class both historical
#: defects belonged to.
REPORTS: list[tuple[int, float]] = [
    (4, 2.0),
    (36, -5.0),
    (1, 1.0),
    (128, 11.0),
    (7, 0.0),
    (64, -18.0),
    (2, -1.0),
    (100, 4.0),
]

TOLERANCE: float = 1e-10


def _config(**overrides: object) -> SPSAConfig:
    """A one-parameter config with bounds wide enough that clipping never fires.

    Clipping is a real part of the Fishtest protocol and is covered elsewhere; it
    is not part of the update rule under test here, and letting it engage would
    mask a divergence rather than reveal one.
    """
    return SPSAConfig(
        num_pairs=1000,
        batch_size=10,
        param_groups=[
            ParamGroup(
                count=1,
                theta_start=1000.0,
                theta_peak=1000.0 + 100.0,
                w_true=1.0,
                w_dev=1.0,
                min_val=-1.0e9,
                max_val=1.0e9,
            ),
        ],
        auto_dev_ranges=False,
        **overrides,  # ty: ignore
    )


def _drive_simulator(optimizer: object) -> list[float]:
    """Run the scripted reports through a simulator optimizer, one block each."""
    flip = np.ones(1, dtype=float)
    thetas: list[float] = [float(optimizer.get_params()[0])]  # ty: ignore
    iter_local = 1
    for n, s in REPORTS:
        c_k = optimizer.get_perturbation_scale(iter_local)  # ty: ignore
        optimizer.step(iter_local, s, flip, c_k, n)  # ty: ignore
        thetas.append(float(optimizer.get_params()[0]))
        iter_local += n
    return thetas


def _outcomes_by_report() -> list[list[int]]:
    """Express REPORTS as outcome lists whose per-block sums match.

    The analysis runners take outcomes, not (N, s), but every path under test
    consumes only the block sum and the count, so any list with the right sum
    and length drives the identical update.
    """
    blocks: list[list[int]] = []
    for n, s in REPORTS:
        total = int(s)
        outs = [0] * n
        i = 0
        while total != 0:
            step = 1 if total > 0 else -1
            outs[i % n] += step
            total -= step
            i += 1
        blocks.append(outs)
    return blocks


def test_scripted_reports_are_faithful() -> None:
    """The outcome lists must reproduce REPORTS exactly, or every test below lies."""
    for (n, s), outs in zip(REPORTS, _outcomes_by_report(), strict=True):
        assert len(outs) == n
        assert float(sum(outs)) == s


def test_spsa_block_matches_validate_spsa() -> None:
    """SPSABlock == the mean-gain macro of validate_spsa."""
    config = _config()
    optimizer = SPSABlock(config)
    simulated = _drive_simulator(optimizer)

    # Take the schedule from the simulator's own derived constants.
    sched = vs.SpsaSchedule(
        a=float(optimizer.a_base[0]),
        a_stability=config.spsa.A,
        alpha=config.spsa.alpha,
        c=float(optimizer.c_base[0]),
        gamma=config.spsa.gamma,
    )
    reference = vs.run_macro_corrected(_outcomes_by_report(), sched=sched)
    expected = [config.theta_start[0] + th for th in reference.theta]

    assert len(simulated) == len(expected)
    assert (
        max(abs(a - b) for a, b in zip(simulated, expected, strict=True)) <= TOLERANCE
    )


def test_sf_sgd_block_matches_validate_sf_sgd() -> None:
    """SFSGDBlock == the triangular block macro of validate_sf_sgd."""
    config = _config()
    optimizer = SFSGDBlock(config)
    c = float(optimizer.c_constant[0])
    simulated = _drive_simulator(optimizer)

    reference = vss.run_macro(
        _outcomes_by_report(),
        lr=config.sf_sgd.lr,
        beta=config.sf_sgd.beta,
        c=c,
    )
    expected = [config.theta_start[0] + th for th in reference.theta]

    assert (
        max(abs(a - b) for a, b in zip(simulated, expected, strict=True)) <= TOLERANCE
    )


def test_adam_block_matches_validate_adam() -> None:
    """AdamBlock == the bias-corrected closed form of validate_adam.

    This equality did not hold before 90a6d32; the two diverged by 21x.
    """
    config = _config()
    optimizer = AdamBlock(config)
    simulated = _drive_simulator(optimizer)

    reference = va.run_macro_block_adam(
        _outcomes_by_report(),
        lr=config.adam.lr,
        beta1=config.adam.beta1,
        beta2=config.adam.beta2,
        eps=config.adam.eps,
    )
    # The Adam family negates: SPSA maximizes Elo while textbook Adam minimizes a
    # loss, so simulator/optimizer.py feeds -net_wins/n and validate_adam.py feeds
    # +s/n into the same subtract-lr*step rule. The trajectories are mirror
    # images, and a parity test that ignored that would be comparing nothing.
    expected = [config.theta_start[0] - th for th in reference.theta]

    assert (
        max(abs(a - b) for a, b in zip(simulated, expected, strict=True)) <= TOLERANCE
    )


def test_sf_adam_block_matches_validate_sf_adam() -> None:
    """SFAdamBlock == the macro update of validate_sf_adam.

    This equality did not hold before f3a3738; the analysis side still applied
    the k(N, beta2) damping factor the simulator had dropped.
    """
    config = _config()
    optimizer = SFAdamBlock(config)
    c = float(optimizer.c_constant[0])
    simulated = _drive_simulator(optimizer)

    mu2 = config.sf_adam.mu2
    reference = vsa.run_macro(
        _outcomes_by_report(),
        lr=config.sf_adam.lr,
        beta1=config.sf_adam.beta1,
        beta2=config.sf_adam.beta2,
        eps=config.sf_adam.eps,
        c=c,
        mu2_init=mu2.init,
        init_stats=vsa.InitStats(
            reports=mu2.reports,
            sum_n=mu2.sum_n,
            sum_s=mu2.sum_s,
            sum_s2_over_n=mu2.sum_s2_over_n,
        ),
    )
    expected = [config.theta_start[0] + th for th in reference.theta]

    assert (
        max(abs(a - b) for a, b in zip(simulated, expected, strict=True)) <= TOLERANCE
    )


def test_the_k_factor_has_not_come_back() -> None:
    """The k(N, beta2) damping factor must not reappear on the analysis side.

    Redundant with the parity test above and cheap to keep: it names the defect,
    so a reintroduction fails with the reason rather than with a float gap.
    """
    assert not hasattr(vsa, "adam_k"), (
        "validate_sf_adam.adam_k is back; SFAdamBlock applies no such factor"
    )


@pytest.mark.parametrize(
    ("name", "cls"),
    [
        ("spsa-block", SPSABlock),
        ("sf-sgd-block", SFSGDBlock),
        ("adam-block", AdamBlock),
        ("sf-adam-block", SFAdamBlock),
    ],
)
def test_every_block_optimizer_is_covered_here(name: str, cls: type) -> None:
    """Each -block optimizer has a parity test above; this pins the roster.

    A new -block entry in the registry without a parity test is the exact gap
    that let two divergences ship, so adding one should break this list loudly.
    """
    from fishtest_spsa_lab.simulator.optimizer import OPTIMIZER_REGISTRY

    block_entries = {n for n in OPTIMIZER_REGISTRY if n.endswith("-block")}
    assert name in block_entries
    assert OPTIMIZER_REGISTRY[name] is cls
    assert block_entries == {
        "spsa-block",
        "sf-sgd-block",
        "adam-block",
        "sf-adam-block",
    }, "a -block optimizer was added without a parity test in this file"
