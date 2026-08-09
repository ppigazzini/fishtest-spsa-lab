"""Main entry point and plotting logic."""

from __future__ import annotations

import logging
import sys

import matplotlib.pyplot as plt
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


def plot_basic_results(result: dict) -> None:  # noqa: C901, PLR0912, PLR0915
    """Plot parameter trajectories vs cumulative pairs."""
    config = result["config"]
    trajectory = result["trajectory"]
    steps = result.get("pairs_history")
    if steps is None:
        # Fallback: approximate with linspace if old results lack pairs_history
        total_pairs = config.num_pairs
        num_steps = len(trajectory)
        if num_steps > 1:
            steps = np.linspace(0, total_pairs, num_steps, dtype=float)
        else:
            steps = np.array([0.0])

    _fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    _fig.suptitle(f"Optimizer: {config.optimizer}", fontsize=16)

    # Identify active and inactive parameters based on sensitivity
    sensitivity = config.w_true
    active_indices = np.where(sensitivity > 0)[0]
    inactive_indices = np.where(sensitivity == 0)[0]
    max_inactive_plots = 5
    max_active_legend = 10

    # Plot 1: Active Parameters vs Pairs
    for i, idx in enumerate(active_indices):
        label = f"Param {idx}" if i < max_active_legend else "_nolegend_"
        ax1.plot(steps, trajectory[:, idx], label=label)

    # Plot target line
    targets = config.theta_peak
    if len(active_indices) > 0:
        ref_target = targets[active_indices[0]]
        ax1.axhline(
            y=ref_target,
            color="r",
            linestyle="--",
            label=f"Target (~{ref_target})",
            linewidth=2,
        )

    ax1.set_title(f"Active Parameters Trajectory (Count: {len(active_indices)})")
    ax1.set_ylabel("Parameter Value")
    ax1.legend()
    ax1.grid(visible=True)
    ax1.ticklabel_format(useOffset=False, style="plain")

    # Plot 2: Inactive Parameters vs Pairs
    if len(inactive_indices) > max_inactive_plots:
        inactive_indices = inactive_indices[:max_inactive_plots]

    if len(inactive_indices) > 0:
        for idx in inactive_indices:
            ax2.plot(steps, trajectory[:, idx], label=f"Param {idx}")

        ax2.set_title(f"Inactive Parameters Trajectory (First {len(inactive_indices)})")
        ax2.set_xlabel("Pairs processed")
        ax2.set_ylabel("Parameter Value")
        ax2.legend()
        ax2.grid(visible=True)
        ax2.ticklabel_format(useOffset=False, style="plain")

    plt.tight_layout()
    plt.show()
    plt.close(_fig)

    # Optional asymmetry / mean plots for optimizers that expose them
    # (currently SPSA-penta).
    asym_hist = result.get("asym_history")
    mu_hist = result.get("mu_history")
    asym_cum_hist = result.get("asym_cum_history")
    mu_cum_hist = result.get("mu_cum_history")

    # Optional penta coefficient (cumulative r) / gain scale history.
    penta_coeff_hist = result.get("penta_coeff_history")
    gain_scale_hist = result.get("gain_scale_history")

    if (
        asym_hist is not None
        or mu_hist is not None
        or asym_cum_hist is not None
        or mu_cum_hist is not None
        or penta_coeff_hist is not None
        or gain_scale_hist is not None
    ) and steps is not None:
        step_array = np.asarray(steps, dtype=float)

        if asym_hist is not None:
            asym_hist = np.asarray(asym_hist, dtype=float)
        if mu_hist is not None:
            mu_hist = np.asarray(mu_hist, dtype=float)
        if asym_cum_hist is not None:
            asym_cum_hist = np.asarray(asym_cum_hist, dtype=float)
        if mu_cum_hist is not None:
            mu_cum_hist = np.asarray(mu_cum_hist, dtype=float)

        if penta_coeff_hist is not None:
            penta_coeff_hist = np.asarray(penta_coeff_hist, dtype=float)
        if gain_scale_hist is not None:
            gain_scale_hist = np.asarray(gain_scale_hist, dtype=float)

        max_k = max(0, step_array.size - 1)
        if max_k > 0:
            fig_penta, (ax_asym, ax_mu) = plt.subplots(
                2,
                1,
                figsize=(8, 6),
                sharex=True,
            )

            # Asymmetry panel: EMA vs cumulative
            if asym_hist is not None and asym_hist.size > 0:
                k = min(asym_hist.size, max_k)
                x_vals = step_array[1 : k + 1]
                ax_asym.plot(x_vals, asym_hist[:k], label="EMA Asymmetry A")

            if asym_cum_hist is not None and asym_cum_hist.size > 0:
                k = min(asym_cum_hist.size, max_k)
                x_vals = step_array[1 : k + 1]
                ax_asym.plot(x_vals, asym_cum_hist[:k], label="Cumulative Asymmetry A")

            if ax_asym.lines:
                ax_asym.set_ylabel("Asymmetry")
                ax_asym.legend(loc="best")
                ax_asym.grid(visible=True)

            # Mean-outcome panel: EMA vs cumulative
            if mu_hist is not None and mu_hist.size > 0:
                k = min(mu_hist.size, max_k)
                x_vals = step_array[1 : k + 1]
                ax_mu.plot(x_vals, mu_hist[:k], label="EMA Mean outcome μ")

            if mu_cum_hist is not None and mu_cum_hist.size > 0:
                k = min(mu_cum_hist.size, max_k)
                x_vals = step_array[1 : k + 1]
                ax_mu.plot(x_vals, mu_cum_hist[:k], label="Cumulative Mean outcome μ")

            if ax_mu.lines:
                ax_mu.set_xlabel("Pairs processed")
                ax_mu.set_ylabel("Mean outcome")
                ax_mu.legend(loc="best")
                ax_mu.grid(visible=True)

            fig_penta.suptitle(
                f"Pentanomial asymmetry/mean (optimizer={config.optimizer})",
            )
            plt.tight_layout()
            plt.show()
            plt.close(fig_penta)

            # --- Penta coefficient / gain scale history ---
            if (penta_coeff_hist is not None and penta_coeff_hist.size > 0) or (
                gain_scale_hist is not None and gain_scale_hist.size > 0
            ):
                fig_coeff, ax_coeff = plt.subplots(1, 1, figsize=(8, 4))

                if penta_coeff_hist is not None and penta_coeff_hist.size > 0:
                    k = min(penta_coeff_hist.size, max_k)
                    x_vals = step_array[1 : k + 1]
                    ax_coeff.plot(
                        x_vals,
                        penta_coeff_hist[:k],
                        label="Cumulative penta r (used for gain)",
                    )

                if gain_scale_hist is not None and gain_scale_hist.size > 0:
                    k = min(gain_scale_hist.size, max_k)
                    x_vals = step_array[1 : k + 1]
                    ax_coeff.plot(
                        x_vals,
                        gain_scale_hist[:k],
                        label="Gain scale factor (from cumulative r)",
                    )

                ax_coeff.set_xlabel("Pairs processed")
                ax_coeff.set_ylabel("Value")
                ax_coeff.legend(loc="best")
                ax_coeff.grid(visible=True)
                fig_coeff.suptitle(
                    f"Penta coefficient / gain scale (optimizer={config.optimizer})",
                )
                plt.tight_layout()
                plt.show()
                plt.close(fig_coeff)


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
