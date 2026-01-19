"""Plot naive Fishtest-style SPSA schedules (c_k, a_k, r_k) vs iteration.

This follows the scalar 1D formulas used in the documentation. For a given
configuration (num_pairs, A, alpha, gamma, c_end, r_end) it computes

- c_k: perturbation scale
- a_k: theta-space step size
- r_k = a_k / c_k**2: effective learning rate in the rescaled space

and plots them as functions of the pair index k.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt

from .common import Line, plot_many


@dataclass(slots=True)
class NaiveSpsaSchedule:
    """Naive Fishtest SPSA schedule parameters in 1D.

    Uses the standard SPSA notation where ``A`` is the stability constant
    in the denominator of the a_k schedule.
    """

    num_pairs: int
    A: float
    alpha: float
    gamma: float
    c_end: float
    r_end: float

    def sequences(self) -> tuple[list[int], list[float], list[float], list[float]]:
        """Compute k, c_k, a_k, and r_k for k = 1..num_pairs."""
        if self.num_pairs <= 0:
            msg = "num_pairs must be positive"
            raise ValueError(msg)

        num_iter = self.num_pairs
        ks = list(range(1, num_iter + 1))

        c_base = self.c_end * (num_iter**self.gamma)
        a_end = self.r_end * (self.c_end**2)
        a_base = a_end * ((self.A + num_iter) ** self.alpha)

        c_seq: list[float] = []
        a_seq: list[float] = []
        r_seq: list[float] = []

        for k in ks:
            k_float = float(k)
            c_k = c_base / (k_float**self.gamma)
            a_k = a_base / ((self.A + k_float) ** self.alpha)
            r_k = a_k / (c_k**2) if c_k != 0.0 else 0.0

            c_seq.append(c_k)
            a_seq.append(a_k)
            r_seq.append(r_k)

        return ks, c_seq, a_seq, r_seq


def main() -> None:
    """Plot c_k, a_k, and r_k for a chosen naive SPSA schedule."""
    num_pairs = 30_000
    a_param = 5_000.0
    alpha = 0.602
    gamma = 0.101
    c_end = 1.0
    r_end = 0.002

    schedule = NaiveSpsaSchedule(
        num_pairs=num_pairs,
        A=a_param,
        alpha=alpha,
        gamma=gamma,
        c_end=c_end,
        r_end=r_end,
    )
    ks, c_seq, a_seq, r_seq = schedule.sequences()

    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    plot_many(
        axs[0],
        Line(ks, c_seq, "c_k"),
        y_label="c_k",
        legend_ncol=1,
    )
    axs[0].set_yscale("log")
    axs[0].set_title("Naive SPSA c_k schedule (log scale)")

    plot_many(
        axs[1],
        Line(ks, a_seq, "a_k"),
        y_label="a_k",
        legend_ncol=1,
    )
    axs[1].set_yscale("log")
    axs[1].set_title("Naive SPSA a_k schedule (log scale)")

    plot_many(
        axs[2],
        Line(ks, r_seq, "r_k = a_k / c_k^2"),
        y_label="r_k",
        legend_ncol=1,
    )
    axs[2].set_yscale("log")
    axs[2].set_xlabel("pair index k")
    axs[2].set_title("Naive SPSA r_k schedule (log scale)")

    fig.suptitle(
        (
            f"Naive Fishtest SPSA schedules "
            f"(num_pairs={num_pairs}, A={a_param}, alpha={alpha}, "
            f"gamma={gamma}, c_end={c_end}, r_end={r_end})"
        ),
        y=0.98,
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
