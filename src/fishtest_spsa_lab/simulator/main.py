"""Entry point for the cross-optimizer comparison sweep.

``run-simulation`` compares every registered optimizer over a shared set of
seeds and prints one table: mean final Elo, a 95% interval, and a paired
difference against the ``spsa`` baseline. It draws no plots.
"""

from __future__ import annotations

import logging
import sys

import numpy as np

from fishtest_spsa_lab.simulator.config import SPSAConfig
from fishtest_spsa_lab.simulator.optimizer import OPTIMIZER_REGISTRY
from fishtest_spsa_lab.simulator.runner import (
    AsyncSpsaRunner,
    SpsaRunner,
    make_workers,
)
from fishtest_spsa_lab.simulator.stats import mean_ci, paired_diff

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


BASELINE_OPTIMIZER = "spsa"
DEFAULT_SEEDS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)


def _run_one(optimizer: str, seed: int, **kwargs: object) -> float:
    """Run one optimizer at one seed and return the final Elo."""
    config = SPSAConfig(optimizer=optimizer, seed=seed, **kwargs)  # ty: ignore
    runner: SpsaRunner
    if config.num_workers > 1:
        workers = make_workers(config, np.random.default_rng(seed))
        runner = AsyncSpsaRunner(config, workers=workers)
    else:
        runner = SpsaRunner(config)
    return float(runner.run()["convergence_metrics"]["final_elo"])


def main() -> int:
    """Compare every registered optimizer over a shared set of seeds."""
    num_pairs = 30_000
    batch_size = 36
    num_workers = 20
    seeds = DEFAULT_SEEDS

    optimizers = list(OPTIMIZER_REGISTRY)
    if BASELINE_OPTIMIZER not in optimizers:
        logger.error("baseline %r is not registered", BASELINE_OPTIMIZER)
        return 1

    logger.info(
        "Comparing %d optimizers over %d seeds (%d pairs, batch %d, %d workers)",
        len(optimizers),
        len(seeds),
        num_pairs,
        batch_size,
        num_workers,
    )

    # Same seeds, same worker pool, same match noise for every arm: the
    # comparison is paired, so the between-seed variance cancels.
    results: dict[str, list[float]] = {name: [] for name in optimizers}
    for seed in seeds:
        for name in optimizers:
            results[name].append(
                _run_one(
                    name,
                    seed,
                    num_pairs=num_pairs,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    variable_batch_size=True,
                ),
            )
        logger.info("seed %d complete", seed)

    baseline = results[BASELINE_OPTIMIZER]
    start_elo = SPSAConfig().start_elo

    logger.info("")
    logger.info(
        "Final Elo, mean +- 95%% CI over %d seeds (start %.3f)", len(seeds), start_elo
    )
    header = (
        f"{'optimizer':<18}{'mean':>9}{'95% CI':>22}"
        f"{'paired vs ' + BASELINE_OPTIMIZER:>28}"
    )
    logger.info(header)
    logger.info("-" * len(header))

    ranked = sorted(optimizers, key=lambda n: -mean_ci(results[n]).mean)
    for name in ranked:
        est = mean_ci(results[name])
        if name == BASELINE_OPTIMIZER:
            verdict = "(baseline)"
        else:
            diff = paired_diff(results[name], baseline)
            mark = "*" if diff.separated_from_zero else "(ns)"
            verdict = f"{diff.mean:+.4f} +- {diff.half_width:.4f} {mark}"
        ci = f"[{est.low:+.4f}, {est.high:+.4f}]"
        logger.info("%-18s%9.4f%22s%28s", name, est.mean, ci, verdict)

    logger.info("")
    logger.info(
        "'*' marks a paired difference whose 95%s interval excludes zero; "
        "'(ns)' means the arms are not separated at this sample size.",
        "%",
    )
    return 0
