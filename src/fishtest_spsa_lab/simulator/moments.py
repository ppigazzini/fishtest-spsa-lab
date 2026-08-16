"""Online second-moment estimator from report-level summaries.

Fishtest never sees individual game outcomes, only per-report totals ``(N, s)``.
Adam needs ``E[g**2]``, and this recovers it exactly from those totals:

```text
mu    = sum(s_i) / sum(N_i)
sigma2 = E[s_i**2 / N_i] - mu**2 * E[N_i]
mu2    = mu**2 + sigma2
```

This lived in three places at once -- ``analysis/common.py``,
``simulator/optimizer.py :: SFAdamBlock`` and a local closure in
``validate_sf_adam.py`` -- identical down to the ``min(max(mu2, 1e-12), 4.0)``
clamp, so changing one silently diverged the others. It sits in ``simulator/``
because the shipped optimizer needs it and ``simulator`` must not import from
``analysis``; the analysis side re-exports it.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = ["HasMu2Stats", "MU2_MAX", "MU2_MIN", "mu2_hat", "update_mu2_stats"]

#: Clamp on the estimate. The bounds are in RAW outcome units, where a per-pair
#: outcome lies in {-2..2} and so ``E[g**2] <= 4``. The simulator feeds a signal
#: scaled by ``1/(2*sqrt(N))``, which puts its estimates two orders of magnitude
#: below the upper bound, so the clamp is inert there and active only for the
#: analysis scripts, which work in raw units.
MU2_MIN: float = 1e-12
MU2_MAX: float = 4.0


@runtime_checkable
class HasMu2Stats(Protocol):
    """Object exposing the online mu2 aggregates."""

    reports: float
    sum_n: float
    sum_s: float
    sum_s2_over_n: float
    mu2_init: float


def mu2_hat(state: HasMu2Stats) -> float:
    """Block-averaged ``E[g**2]`` from report-level ``(N, s)`` aggregates.

    Before any reports arrive, returns ``state.mu2_init``.
    """
    if state.reports <= 0.0:
        return state.mu2_init
    mu = (state.sum_s / state.sum_n) if state.sum_n > 0.0 else 0.0
    e_s2_over_n = state.sum_s2_over_n / state.reports
    e_n = state.sum_n / state.reports
    sigma2 = max(e_s2_over_n - (mu * mu) * e_n, 0.0)
    return min(max(mu * mu + sigma2, MU2_MIN), MU2_MAX)


def update_mu2_stats(state: HasMu2Stats, n: int, s: float) -> None:
    """Fold one report in, AFTER its estimate has been used for that report."""
    if n <= 0:
        return
    state.reports += 1.0
    state.sum_n += float(n)
    state.sum_s += float(s)
    state.sum_s2_over_n += (float(s) * float(s)) / float(n)
