"""Where a decaying gain overtakes a constant one, measured across budgets.

MS8 item 8a. ``spsa_simul`` argues that since convergence is unreachable on a
real budget, the decaying ``c_k``/``R_k`` schedules should be replaced by
constants chosen for a target precision. The premise is right and the conclusion
does not follow: the stationary loss is proportional to the *current* gain, so a
decaying schedule shrinks the floor it is relaxing toward, on any horizon long
enough to track it. ``analysis/design.py :: adiabatic_ratio`` bounds "long
enough"; this measures the consequence.

Two arms, following ``__DEV/260809-0-REPORT.md`` E6:

* **decay** -- classic SPSA, the Fishtest schedule, ``r_end`` at the horizon.
* **constant** -- ``sf-sgd`` with ``beta = 0``, which reduces the schedule-free
  update to a plain constant-gain SPSA step at ``lr``. Setting ``lr = r_end``
  gives the two arms the same FINAL gain, so the comparison is of the schedule
  and not of its level.

``--from-optimum`` starts both arms at ``theta_peak``, which removes the approach
transient and isolates the noise ball. That is generous to the constant arm --
its whole advantage is early -- so a crossover measured that way is a lower
bound on the decaying arm's case.

Seeds are shared between arms and the difference is paired, because the
between-seed spread is comparable to the effect.
"""

from __future__ import annotations

import argparse
import logging
import sys


from fishtest_spsa_lab.simulator.config import SPSAConfig
from fishtest_spsa_lab.simulator.runner import SpsaRunner
from fishtest_spsa_lab.simulator.stats import mean_ci, paired_diff

logger = logging.getLogger(__name__)

__all__ = ["main", "run_arm"]

#: Budgets in pairs. The default ladder brackets the crossover E6 saw near
#: 90,000 pairs while staying inside a few minutes of runtime.
DEFAULT_BUDGETS: tuple[int, ...] = (6_000, 18_000, 36_000, 60_000, 90_000, 120_000)

#: Batch size, matching the sweep in simulator/main.py.
BATCH: int = 36


def run_arm(
    *,
    arm: str,
    seed: int,
    num_pairs: int,
    from_optimum: bool,
    time_control: str | None,
) -> float:
    """Run one arm at one seed and return the final Elo."""
    config = SPSAConfig(
        optimizer="spsa" if arm == "decay" else "sf-sgd",
        seed=seed,
        num_pairs=num_pairs,
        batch_size=BATCH,
        time_control=time_control,
    )
    if arm == "constant":
        # beta = 0 collapses the schedule-free update to z, i.e. a constant-gain
        # SPSA step; lr = r_end matches the decaying arm's FINAL gain.
        config.sf_sgd.beta = 0.0
        config.sf_sgd.lr = config.spsa.r_end

    runner = SpsaRunner(config)
    if from_optimum:
        # Start at the optimum so the noise ball is measured without the
        # approach transient. Schedule-free state must move with theta or the
        # exported iterate is reconstructed from a stale pair.
        peak = config.theta_peak.copy()
        runner.optimizer.theta = peak.copy()
        for attr in ("z", "x"):
            if hasattr(runner.optimizer, attr):
                setattr(runner.optimizer, attr, peak.copy())
        runner.trajectory = [runner.optimizer.get_params()]

    return float(runner.run()["convergence_metrics"]["final_elo"])


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure where a decaying SPSA gain overtakes a constant one, "
            "as a function of budget."
        ),
    )
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=list(DEFAULT_BUDGETS),
        help="budgets in pairs",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=6,
        help="paired seeds per budget (default: 6)",
    )
    parser.add_argument(
        "--from-optimum",
        action="store_true",
        help="start both arms at theta_peak, isolating the noise ball",
    )
    parser.add_argument(
        "--time-control",
        default="LTC",
        help="oracle time control, or 'none' for the vendored model",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the crossover experiment and print a table."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    args = _parse_args(argv)
    time_control = None if args.time_control.lower() == "none" else args.time_control
    seeds = range(1, args.seeds + 1)

    print(  # noqa: T201
        f"Decay vs constant gain, {args.seeds} paired seeds, batch {BATCH}, "
        f"oracle {time_control or 'vendored'}, "
        f"{'from the optimum' if args.from_optimum else 'from theta_start'}",
    )
    print(  # noqa: T201
        "constant = sf-sgd with beta=0 and lr=r_end: same FINAL gain, no decay\n",
    )
    header = (
        f"{'pairs':>10}{'decay':>12}{'constant':>12}"
        f"{'paired (decay - constant)':>30}{'winner':>12}"
    )
    print(header)  # noqa: T201
    print("-" * len(header))  # noqa: T201

    crossover: int | None = None
    previous_sign: int | None = None
    for num_pairs in args.budgets:
        pairs = (num_pairs // BATCH) * BATCH
        decay = [
            run_arm(
                arm="decay",
                seed=s,
                num_pairs=pairs,
                from_optimum=args.from_optimum,
                time_control=time_control,
            )
            for s in seeds
        ]
        constant = [
            run_arm(
                arm="constant",
                seed=s,
                num_pairs=pairs,
                from_optimum=args.from_optimum,
                time_control=time_control,
            )
            for s in seeds
        ]
        diff = paired_diff(decay, constant)
        if diff.separated_from_zero:
            winner = "decay" if diff.mean > 0 else "constant"
        else:
            winner = "(ns)"
        sign = 1 if diff.mean > 0 else -1
        if previous_sign is not None and sign != previous_sign and crossover is None:
            crossover = pairs
        previous_sign = sign

        print(  # noqa: T201
            f"{pairs:10,}{mean_ci(decay).mean:12.4f}{mean_ci(constant).mean:12.4f}"
            f"{f'{diff.mean:+.4f} +- {diff.half_width:.4f}':>30}{winner:>12}",
        )

    print()  # noqa: T201
    if crossover is not None:
        print(  # noqa: T201
            f"Sign of the paired difference changes at about {crossover:,} pairs. "
            "Below it the constant gain is ahead; above it the decaying schedule "
            "is, and keeps descending because the floor it relaxes toward shrinks "
            "with the gain.",
        )
    else:
        print(  # noqa: T201
            "No sign change over these budgets: one arm leads throughout. Widen "
            "--budgets to bracket the crossover.",
        )
    print(  # noqa: T201
        "spsa_simul recommends replacing the decaying schedule with a constant. "
        "That is right only on the short side of this crossover.",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
