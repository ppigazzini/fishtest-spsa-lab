"""Pair-aware pentanomial oracle.

The vendored ``PentaModel`` fixes the book exit at a deterministic ``v = +-100``
centipawns and plays the two reversed-colour games at ``+v`` and ``-v``.
Conditional on a *deterministic* v the two games are independent, and since there
is nothing to marginalise over they are independent outright. Measured on the
vendored model at equal strength:

```text
game A (hero on the +100cp side): W 0.5000  D 0.4999  L 0.0001
game B (hero on the -100cp side): W 0.0001  D 0.4999  L 0.5000
cov(g_a, g_b) = 6.9e-18
pentanomial [LL, DL, DD, WD, WW] = [5.6e-05, 0.25, 0.4999, 0.25, 5.6e-05]
```

Two consequences, and they are the reason this module exists.

**The pentanomial carries no pairing information.** Zero correlation means the
entire variance-reduction rationale for reversed-colour pairs -- the reason
Fishtest scores in pentanomials at all -- is absent from the model every noise
number in this repository is derived from.

**The distribution is frozen.** Because one game is a coin flip between win and
draw while the other is a coin flip between loss and draw, ``p_WD`` and ``p_DL``
sit at 0.25 whatever the spread ``b`` is set to. So ``var(net outcome)/2`` is
pinned near 0.2502 and cannot reach the draw-heavier values real long time
controls show.

The fix is the one thing the model is missing: **draw the book exit once per pair
and share it between the two games.** A balanced exit (small |v|) draws both
games; a sharp exit decides both, in opposite directions. Marginalising over v
induces the negative within-pair correlation that reversed-colour pairing exists
to exploit, and it unfreezes the distribution, because the draw rate is now
governed by how often the book hands out a balanced position.

Integration over v is done by Gauss-Hermite quadrature rather than by sampling,
so the returned probabilities are deterministic and exact to quadrature order --
a sampled book exit would add a second noise source the caller cannot control.

The per-game logistic is reimplemented here rather than called on ``PentaModel``,
for one reason: the spread ``b`` is hardcoded at 22 in the vendored file, and a
shared book exit alone cannot reach the targets. Measured range of
``var(net)/2`` at equal strength:

```text
deterministic +-100cp exit, b = 22    0.2502   (and >= 0.25 for every b)
shared random exit,         b = 22    0.021 .. 0.106, peaking near sigma = 100
```

The two mechanisms bracket the real values from opposite sides, so both knobs are
needed. ``b`` controls how much per-game randomness survives a given evaluation;
``book_sigma`` controls how often the book hands out a decisive one. The vendored
file is not edited, and ``b = 22`` remains the default so the reimplementation is
verified against it.
"""

from __future__ import annotations

import numpy as np

from fishtest_spsa_lab.vendor.pentamodel.pentamodel import PentaModel

__all__ = [
    "TIME_CONTROL_SIGMA2",
    "VENDORED_SPREAD",
    "PairOracle",
    "calibrate_spread",
    "pair_sigma2",
]

#: Centipawn scale at which self-play reaches a 50% win chance. The vendored
#: model's ``a``; kept identical so the two agree when configured the same.
EVAL_SCALE: float = 100.0

#: The vendored model's spread. Retained as the default so that
#: ``PairOracle(book_sigma=..., spread=22, deterministic_exit=True)`` reproduces
#: ``PentaModel`` exactly, which is how this reimplementation is tested.
VENDORED_SPREAD: float = 22.0

#: Target ``var(net pair outcome) / 2`` per time control, the convention used
#: throughout this repository and in ``__DEV/260809-0-REPORT.md`` Appendix C.
#: Longer time controls draw more often and are therefore quieter.
#:
#: The vendored model sits at 0.2502 and cannot go below about 0.25, so the two
#: longer controls were unreachable before this module existed.
TIME_CONTROL_SIGMA2: dict[str, float] = {
    "STC": 0.2654,
    "LTC": 0.2274,
    "VLTC": 0.2133,
}

#: Pentanomial outcome scale, in net units. Index order is
#: ``[LL, LD+DL, DD+WL+LW, WD+DW, WW]``, matching ``PentaModel``.
NET_OUTCOMES: np.ndarray = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=float)

#: Quadrature nodes for the book-exit integral. 48 is far past convergence for a
#: smooth logistic integrand; the cost is paid once per distinct Elo difference.
DEFAULT_QUADRATURE_NODES: int = 48

_CALIBRATION_ITERATIONS: int = 200
_SPREAD_LOWER: float = 1.0
_SPREAD_UPPER: float = 5.0e3

#: Book exits are clipped to +-1500 cp before reaching the logistic. With the
#: vendored spread b = 22 the win probability is 1.0 to machine precision beyond
#: about +-300 cp, so this changes no probability; it only stops the far
#: quadrature tail from overflowing math.exp, which it did at 20,000 cp.
_EVAL_CLIP: float = 1500.0


class PairOracle:
    """Pentanomial probabilities under a shared, randomly drawn book exit.

    ``book_sigma`` is the standard deviation, in centipawns, of the evaluation
    the opening book hands to the pair. It is the knob that sets the draw rate
    and therefore the per-pair variance; use :func:`calibrate_book_sigma` to
    solve for the value matching a time control.

    Probabilities are cached by rounded Elo difference. The runner previously
    rebuilt a ``PentaModel`` on every batch, which profiled at 60% of a 6,000-
    pair run (0.059 s of 0.099 s); a tuning run visits few distinct Elo gaps, so
    the cache turns that into a handful of constructions.
    """

    def __init__(
        self,
        *,
        book_sigma: float,
        spread: float = VENDORED_SPREAD,
        nodes: int = DEFAULT_QUADRATURE_NODES,
        cache_decimals: int = 4,
        deterministic_exit: bool = False,
    ) -> None:
        """Build an oracle with the given book spread and quadrature order.

        ``deterministic_exit`` reproduces the vendored behaviour -- a fixed
        ``+-book_sigma`` exit rather than a distribution -- and exists so the
        reimplemented logistic can be checked against ``PentaModel``.
        """
        if book_sigma <= 0.0:
            msg = f"book_sigma must be positive, got {book_sigma}"
            raise ValueError(msg)
        if spread <= 0.0:
            msg = f"spread must be positive, got {spread}"
            raise ValueError(msg)
        self.book_sigma = float(book_sigma)
        self.spread = float(spread)
        self._decimals = int(cache_decimals)
        self._cache: dict[float, np.ndarray] = {}
        self._moment_cache: dict[float, tuple[float, float, float, float]] = {}

        # Gauss-Hermite integrates against exp(-x**2); v = sqrt(2)*sigma*x turns
        # that into a Normal(0, sigma**2) expectation, and the weights are
        # normalised so they sum to 1.
        if deterministic_exit:
            self._v_nodes = np.array([self.book_sigma], dtype=float)
            self._v_weights = np.array([1.0], dtype=float)
        else:
            raw_nodes, raw_weights = np.polynomial.hermite_e.hermegauss(int(nodes))
            self._v_nodes = np.clip(
                self.book_sigma * raw_nodes,
                -_EVAL_CLIP,
                _EVAL_CLIP,
            )
            self._v_weights = raw_weights / raw_weights.sum()

    @staticmethod
    def _logistic(x: float) -> float:
        """``1 / (1 + exp(x))``, without overflowing for large positive x.

        The naive form raises OverflowError at x > 709 and the vendored model
        never sees such inputs because its exit is fixed; a distribution of exits
        reaches them in the quadrature tail.
        """
        if x >= 0.0:
            z = np.exp(-x)
            return float(z / (1.0 + z))
        return float(1.0 / (1.0 + np.exp(x)))

    def _win(self, v: float, shift: float) -> float:
        return self._logistic((EVAL_SCALE + shift - v) / self.spread)

    def _loss(self, v: float, shift: float) -> float:
        return self._logistic((EVAL_SCALE - shift + v) / self.spread)

    def _draw(self, v: float, shift: float) -> float:
        return max(0.0, 1.0 - self._win(v, shift) - self._loss(v, shift))

    def pentanomial_probs(self, elo_diff: float) -> np.ndarray:
        """Return ``[LL, DL, DD, WD, WW]`` for an opponent-Elo difference.

        The sign convention follows ``PentaModel(opponentElo=...)``: positive
        means the opponent is stronger.
        """
        key = float(round(float(elo_diff), self._decimals))
        cached = self._cache.get(key)
        if cached is None:
            cached = self._compute(key)
            self._cache[key] = cached
        return cached

    def _compute(self, elo_diff: float) -> np.ndarray:
        # The Elo-to-shift map is the vendored one; only the per-game logistic
        # is reimplemented, and only so that `spread` can move.
        shift = PentaModel(opponentElo=elo_diff).s

        probs = np.zeros(5, dtype=float)
        for v, weight in zip(self._v_nodes, self._v_weights, strict=True):
            # One draw of the book exit, shared by both games of the pair: the
            # hero plays the +v side in one game and the -v side in the other.
            w_a = self._win(v, shift)
            d_a = self._draw(v, shift)
            l_a = self._loss(v, shift)
            w_b = self._win(-v, shift)
            d_b = self._draw(-v, shift)
            l_b = self._loss(-v, shift)

            # Conditional on this exit the two games are independent; the
            # correlation appears only after summing over the exits.
            probs[4] += weight * (w_a * w_b)
            probs[3] += weight * (w_a * d_b + d_a * w_b)
            probs[2] += weight * (d_a * d_b + w_a * l_b + l_a * w_b)
            probs[1] += weight * (d_a * l_b + l_a * d_b)
            probs[0] += weight * (l_a * l_b)

        total = probs.sum()
        if total > 0.0:
            probs /= total
        return probs

    def _game_moments(self, elo_diff: float) -> tuple[float, float, float, float]:
        """Return ``(mean_a, var_a, mean_b, var_b)`` for the two games.

        Computed from the quadrature, NOT from the pentanomial. The pentanomial
        collapses WL and LW into its middle bucket, so the per-game marginals
        are not recoverable from it -- a first attempt to do so produced a
        spurious correlation of 4.5e-04 where the truth is 0.
        """
        key = float(round(float(elo_diff), self._decimals))
        cached = self._moment_cache.get(key)
        if cached is not None:
            return cached

        shift = PentaModel(opponentElo=key).s
        m_a = m_b = e2_a = e2_b = 0.0
        for v, weight in zip(self._v_nodes, self._v_weights, strict=True):
            s_a = self._win(v, shift) + 0.5 * self._draw(v, shift)
            s_b = self._win(-v, shift) + 0.5 * self._draw(-v, shift)
            q_a = self._win(v, shift) + 0.25 * self._draw(v, shift)
            q_b = self._win(-v, shift) + 0.25 * self._draw(-v, shift)
            m_a += weight * s_a
            m_b += weight * s_b
            e2_a += weight * q_a
            e2_b += weight * q_b

        moments = (m_a, e2_a - m_a * m_a, m_b, e2_b - m_b * m_b)
        self._moment_cache[key] = moments
        return moments

    def pair_correlation(self, elo_diff: float = 0.0) -> float:
        """Correlation between the two games of a pair.

        ``var(g_a + g_b) = var(g_a) + var(g_b) + 2*cov``, with the pair variance
        read off the pentanomial and the two marginal variances taken from the
        same quadrature.
        """
        probs = self.pentanomial_probs(elo_diff)
        pair_scores = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        mean_pair = float(probs @ pair_scores)
        var_pair = float(probs @ (pair_scores**2)) - mean_pair**2

        _m_a, var_a, _m_b, var_b = self._game_moments(elo_diff)
        if var_a <= 0.0 or var_b <= 0.0:
            return 0.0
        cov = 0.5 * (var_pair - var_a - var_b)
        return cov / np.sqrt(var_a * var_b)


def pair_sigma2(probs: np.ndarray) -> float:
    """Return ``var(net pair outcome) / 2`` for a pentanomial vector.

    This is the ``sigma2`` the noise-ball derivations take as input.
    """
    probs = np.asarray(probs, dtype=float)
    mean = float(probs @ NET_OUTCOMES)
    var = float(probs @ (NET_OUTCOMES**2)) - mean**2
    return var / 2.0


def calibrate_spread(
    target_sigma2: float,
    *,
    book_sigma: float,
    nodes: int = DEFAULT_QUADRATURE_NODES,
) -> float:
    """Solve for the per-game spread giving ``target_sigma2`` at equal strength.

    ``book_sigma`` is held fixed and ``spread`` is solved for, because sigma2 is
    monotone in ``spread`` (more per-game randomness means more decisive pairs)
    but NOT in ``book_sigma`` -- it peaks near 100 cp and falls on both sides,
    since a very balanced book draws everything and a very sharp one decides both
    games in opposite directions, and both give a 1-1 pair.

    That non-monotonicity is why the first attempt at this function, which
    bisected on ``book_sigma`` assuming it increased sigma2, ran to its upper
    bound and returned nonsense.
    """

    def sigma2_at(spread: float) -> float:
        return pair_sigma2(
            PairOracle(
                book_sigma=book_sigma,
                spread=spread,
                nodes=nodes,
            ).pentanomial_probs(0.0),
        )

    lo, hi = _SPREAD_LOWER, _SPREAD_UPPER
    if target_sigma2 <= sigma2_at(lo):
        return lo
    if target_sigma2 >= sigma2_at(hi):
        return hi
    for _ in range(_CALIBRATION_ITERATIONS):
        mid = 0.5 * (lo + hi)
        if sigma2_at(mid) < target_sigma2:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
