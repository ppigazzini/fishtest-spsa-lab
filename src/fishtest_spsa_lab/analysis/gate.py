"""Pass/fail accounting and stable stdout for the ``validate-*`` entry points.

AGENTS.md makes these scripts the regression suite -- "these are the real
regression suite; ``tests/`` is thinner than it looks" -- and defines the refactor
gate as "every ``validate-*`` produces byte-identical output", to be judged by the
exit code. Measured on 2026-08-16, five of seven wrote nothing at all to stdout
and none of the seven had a path to a non-zero exit, so both halves of that gate
were vacuous: empty output is byte-identical to empty output, and a script that
cannot fail cannot report a failure.

This module supplies the missing half. A validator declares each invariant it
claims, the measured value, the tolerance it is asserted at, and a verdict; the
run exits non-zero if any verdict is FAIL. The output is deterministic and
free of timings and paths, so comparing two runs byte for byte is meaningful.

Plotting is suppressed under a non-interactive backend and by setting
``SPSA_LAB_NO_PLOT``, so one entry point serves both as a gate and as the
exploratory tool it was written as.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.registry import BackendFilter, backend_registry

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = [
    "Check",
    "Gate",
    "plots_enabled",
    "show",
]

#: Set to any non-empty value to run a validator as a pure gate.
NO_PLOT_ENV: str = "SPSA_LAB_NO_PLOT"


def plots_enabled() -> bool:
    """Whether figures should be displayed.

    False under ``SPSA_LAB_NO_PLOT`` and under any non-interactive backend, so a
    validator invoked from a test or from CI neither blocks nor emits the
    "FigureCanvasAgg is non-interactive" warning.
    """
    if os.environ.get(NO_PLOT_ENV):
        return False
    interactive = {
        name.lower()
        for name in backend_registry.list_builtin(BackendFilter.INTERACTIVE)
    }
    return matplotlib.get_backend().lower() in interactive


def show(fig: Figure | None = None) -> None:
    """Display a figure when plotting is enabled, and always close it.

    Closing is unconditional. The analysis scripts opened 14 figures across the
    package and closed none, which trips matplotlib's ``figure.max_open_warning``
    once several of them run in one process -- as they now do under pytest.
    """
    if plots_enabled():
        plt.show()
    if fig is None:
        plt.close("all")
    else:
        plt.close(fig)


@dataclass(frozen=True, slots=True)
class Check:
    """One asserted invariant, its measurement, and its verdict."""

    name: str
    measured: float
    tolerance: float
    passed: bool
    note: str = ""


@dataclass(slots=True)
class Gate:
    """Collects checks for one validator and renders the verdict.

    Usage::

        gate = Gate("validate-spsa", "SPSA macro-vs-micro equivalence")
        gate.check_le("mean-gain macro == const-mean micro", gap, 1e-12)
        return gate.report()
    """

    entry_point: str
    title: str
    checks: list[Check] = field(default_factory=list)
    notes: list[tuple[str, str]] = field(default_factory=list)

    # --- recording ---

    def note(self, label: str, value: object) -> None:
        """Record an informational quantity that carries no verdict."""
        self.notes.append((label, _fmt(value)))

    def check_le(
        self,
        name: str,
        measured: float,
        tolerance: float,
        note: str = "",
    ) -> bool:
        """Assert ``measured <= tolerance``; the usual form for a gap."""
        passed = bool(measured <= tolerance)
        self.checks.append(
            Check(
                name=name,
                measured=float(measured),
                tolerance=float(tolerance),
                passed=passed,
                note=note,
            ),
        )
        return passed

    def check_close(
        self,
        name: str,
        measured: float,
        expected: float,
        tolerance: float,
        note: str = "",
    ) -> bool:
        """Assert ``|measured - expected| <= tolerance``."""
        detail = note or f"measured {_fmt(measured)} vs expected {_fmt(expected)}"
        return self.check_le(
            name,
            abs(float(measured) - float(expected)),
            tolerance,
            detail,
        )

    def check_within_se(
        self,
        name: str,
        measured: float,
        expected: float,
        se: float,
        k: float = 4.0,
        note: str = "",
    ) -> bool:
        """Assert a Monte Carlo estimate lies within ``k`` standard errors.

        A stochastic quantity needs a tolerance derived from its own sampling
        error, not a hand-picked constant. ``k = 4`` keeps the false-failure rate
        per check near 6e-5, which is what a gate run on every commit needs.
        """
        detail = note or (
            f"measured {_fmt(measured)} vs expected {_fmt(expected)}, "
            f"SE {_fmt(se)}, k={k:g}"
        )
        return self.check_le(
            name, abs(float(measured) - float(expected)), k * se, detail
        )

    # --- rendering ---

    @property
    def failures(self) -> list[Check]:
        """The checks that did not pass."""
        return [c for c in self.checks if not c.passed]

    def report(self) -> int:
        """Print the result table and return a process exit code."""
        header = f"{self.entry_point}: {self.title}"
        print(header)
        print("=" * len(header))

        if self.notes:
            width = max(len(label) for label, _ in self.notes)
            for label, value in self.notes:
                print(f"  {label:<{width}}  {value}")
            print()

        if not self.checks:
            print("  no checks declared")
            print()
            print("VERDICT: FAIL (a validator that asserts nothing validates nothing)")
            return 1

        width = max(len(c.name) for c in self.checks)
        for c in self.checks:
            verdict = "PASS" if c.passed else "FAIL"
            print(
                f"  {c.name:<{width}}  {_fmt(c.measured)} <= {_fmt(c.tolerance)}"
                f"  {verdict}",
            )
            if c.note:
                print(f"  {'':<{width}}  ({c.note})")

        failed = self.failures
        print()
        print(
            f"VERDICT: {'FAIL' if failed else 'PASS'} "
            f"({len(self.checks) - len(failed)}/{len(self.checks)} checks passed)",
        )
        return 1 if failed else 0


def _fmt(value: object) -> str:
    """Format a value for the stable result table."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value == 0.0:
            return "0"
        if 1e-4 <= abs(value) < 1e6:
            return f"{value:.6g}"
        return f"{value:.6e}"
    return str(value)
