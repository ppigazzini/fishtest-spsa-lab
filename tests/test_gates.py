"""Run every ``validate-*`` gate in-process and require a zero exit code.

AGENTS.md designates these scripts as the real regression suite, but nothing ran
them: a broken update rule shipped green because `pytest` never invoked the gate
that covers it. That is exactly how the k(N, beta2) factor survived in
`validate_sf_adam.py` after `1bea8b1` removed it from `simulator/optimizer.py`.

These tests are slower than the rest of the suite (about 5 s in total, dominated
by `validate-penta`'s 1000-trial Monte Carlo). That is the price of the suite
actually covering the math, and it is paid once per run.
"""

from __future__ import annotations

import matplotlib
import pytest

# Force a non-interactive backend before any pyplot import, so a gate invoked
# here can never block on a window.
matplotlib.use("Agg")

from fishtest_spsa_lab.analysis import (  # noqa: E402
    validate_adam,
    validate_pentanomial,
    validate_sf_adam,
    validate_sf_sgd,
    validate_spsa,
    validate_spsa_u2,
    validate_variance,
)
from fishtest_spsa_lab.analysis.gate import NO_PLOT_ENV, Gate, plots_enabled  # noqa: E402

# The argparse-based entry points must be handed an explicit empty argv. With
# argv=None they read sys.argv, which under pytest is pytest's own command line
# and makes the gate exit 2 on "unrecognized arguments: -q".
GATES = [
    ("validate-spsa", validate_spsa.main),
    ("validate-sf-sgd-block", validate_sf_sgd.main),
    ("validate-sf-adam-block", validate_sf_adam.main),
    ("validate-adam", validate_adam.main),
    ("validate-spsa-u2", validate_spsa_u2.main),
    ("validate-variance", validate_variance.main),
    ("validate-penta", lambda: validate_pentanomial.main([])),
]


@pytest.fixture(autouse=True)
def _no_plot(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every gate with plotting suppressed."""
    monkeypatch.setenv(NO_PLOT_ENV, "1")


@pytest.mark.parametrize(("name", "entry_point"), GATES, ids=[g[0] for g in GATES])
def test_gate_passes(
    name: str, entry_point: object, capsys: pytest.CaptureFixture
) -> None:
    """Each validator returns 0 and says something while doing it."""
    code = entry_point()  # ty: ignore
    captured = capsys.readouterr()
    combined = captured.out + captured.err

    assert code == 0, f"{name} returned {code}\n{combined}"
    # The other half of the gate: five of seven used to print nothing at all, so
    # "byte-identical output" compared empty against empty.
    assert "VERDICT: PASS" in captured.out, f"{name} printed no verdict\n{combined}"


def test_no_plot_env_suppresses_display() -> None:
    """The fixture above is only meaningful if the env var is honoured."""
    assert not plots_enabled()


def test_a_gate_with_no_checks_fails() -> None:
    """A validator that asserts nothing must not report success."""
    assert Gate("validate-nothing", "asserts nothing").report() == 1


def test_a_failed_check_sets_the_exit_code() -> None:
    """A single FAIL is enough to fail the run."""
    gate = Gate("validate-demo", "one good check and one bad one")
    gate.check_le("passes", 0.0, 1.0)
    gate.check_le("fails", 1.0, 0.0)
    assert gate.report() == 1
    assert len(gate.failures) == 1
