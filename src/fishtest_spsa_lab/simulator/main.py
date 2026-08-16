"""Entry point for the cross-optimizer comparison sweep.

``run-simulation`` compares every registered optimizer over a shared set of
seeds and prints one table: mean final Elo, a 95% interval, and a paired
difference against the ``spsa`` baseline. It draws no plots.
"""

from __future__ import annotations

import logging
import math
import sys

import numpy as np

from fishtest_spsa_lab.simulator.config import SPSAConfig
from fishtest_spsa_lab.simulator.optimizer import OPTIMIZER_REGISTRY
from fishtest_spsa_lab.simulator.runner import (
    AsyncSpsaRunner,
    SpsaRunner,
    make_workers,
)
from fishtest_spsa_lab.simulator.stats import (
    holm_adjusted,
    mean_ci,
    paired_diff,
    t95,
    t_two_sided_p,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


BASELINE_OPTIMIZER = "spsa"
DEFAULT_SEEDS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)

#: Family-wise significance level for the whole comparison table.
ALPHA: float = 0.05


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
    batch_size = 36
    # A multiple of batch_size. 30_000 // 36 leaves 12 pairs unspent and made
    # runner.py warn once per run -- 96 times per sweep, for every arm at every
    # seed. The budget is the round number here, not the pair count.
    num_pairs = 833 * batch_size  # 29_988
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

    # One baseline, many treatments: eleven comparisons at 95% each would give a
    # family-wise error rate of 1 - 0.95**11 = 43%, so marking every interval
    # independently overstates the evidence badly. Holm-Bonferroni controls the
    # family-wise rate, assumes nothing about the dependence between arms, and is
    # uniformly more powerful than plain Bonferroni.
    contenders = [n for n in optimizers if n != BASELINE_OPTIMIZER]
    diffs = {n: paired_diff(results[n], baseline) for n in contenders}
    raw_p: list[float] = []
    for name in contenders:
        est = diffs[name]
        sem = est.half_width / t95(est.n - 1) if est.n > 1 else math.inf
        if sem > 0.0:
            raw_p.append(t_two_sided_p(est.mean / sem, est.n - 1))
        elif est.mean == 0.0:
            # Every seed gave an identical result, so the arm is a duplicate of
            # the baseline, not a discovery. `spsa-cwd` at its default
            # lambda_ = 0.0 is exactly this: the decay term is switched off and
            # the update rule reduces to SPSA. Treating 0/0 as an infinite t
            # statistic would print p = 0 for two bit-identical arms.
            raw_p.append(1.0)
        else:
            raw_p.append(0.0)
    adjusted = dict(zip(contenders, holm_adjusted(raw_p), strict=True))
    raw = dict(zip(contenders, raw_p, strict=True))

    logger.info("")
    logger.info(
        "Final Elo over %d seeds (start %.3f), %d pairs, batch %d",
        len(seeds),
        start_elo,
        num_pairs,
        batch_size,
    )
    header = (
        f"{'optimizer':<18}{'mean':>9}{'95% CI':>22}"
        f"{'paired vs ' + BASELINE_OPTIMIZER:>22}{'p':>10}{'p_holm':>10}"
    )
    logger.info(header)
    logger.info("-" * len(header))

    ranked = sorted(optimizers, key=lambda n: -mean_ci(results[n]).mean)
    for name in ranked:
        est = mean_ci(results[name])
        ci = f"[{est.low:+.4f}, {est.high:+.4f}]"
        if name == BASELINE_OPTIMIZER:
            logger.info(
                "%-18s%9.4f%22s%22s%10s%10s", name, est.mean, ci, "(baseline)", "", ""
            )
            continue
        diff = diffs[name]
        p_adj = adjusted[name]
        mark = "*" if p_adj < ALPHA else " "
        logger.info(
            "%-18s%9.4f%22s%21s%1s%10.4f%10.4f",
            name,
            est.mean,
            ci,
            f"{diff.mean:+.4f} +- {diff.half_width:.4f}",
            mark,
            raw[name],
            p_adj,
        )

    separated = sum(1 for n in contenders if adjusted[n] < ALPHA)
    logger.info("")
    logger.info(
        "'*' marks a paired difference significant at family-wise alpha=%.2f after "
        "Holm-Bonferroni over %d comparisons: %d of %d separate from the baseline. "
        "The 95%% CI column is per-arm and uncorrected; read p_holm, not the CI.",
        ALPHA,
        len(contenders),
        separated,
        len(contenders),
    )
    return 0
