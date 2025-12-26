"""Main entry point and plotting logic."""

from __future__ import annotations

import logging
import sys

import matplotlib.pyplot as plt
import numpy as np

from fishtest_spsa_lab.simulator.config import SPSAConfig
from fishtest_spsa_lab.simulator.runner import (
    AsyncSpsaRunner,
    SpsaRunner,
    make_workers,
)

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


def main() -> None:
    """Run the main entry point."""
    optimizers = [
        "spsa",
        "spsa-block",
        "spsa-penta",
        "spsa-cwd",
        "accelerated-spsa",
        "sf-sgd",
        "sf-sgd-block",
        "sf-adam",
        "sf-adam-block",
        "adam",
        "adam-block",
        "ademamix",
    ]

    # Common configuration parameters
    num_pairs = 30_000
    batch_size = 36
    num_workers = 20
    variable_batch_size = True

    # Build a shared worker pool (same concurrencies and speeds for all runs)
    base_worker_config = SPSAConfig(
        num_pairs=num_pairs,
        batch_size=batch_size,
        num_workers=num_workers,
        variable_batch_size=variable_batch_size,
    )
    worker_rng = np.random.default_rng(base_worker_config.seed)
    workers = make_workers(base_worker_config, worker_rng)

    for opt_name in optimizers:
        logger.info("=" * 60)
        logger.info("Running Simulation with Optimizer: %s", opt_name)
        logger.info("=" * 60)

        config = SPSAConfig(
            optimizer=opt_name,
            num_pairs=num_pairs,
            batch_size=batch_size,
            num_workers=num_workers,
            variable_batch_size=variable_batch_size,
        )

        logger.info(
            "Starting optimization (%s) with config:\n%s",
            opt_name,
            config,
        )

        # Select runner based on workers
        if config.num_workers > 1:
            runner = AsyncSpsaRunner(config, workers=workers)
        else:
            runner = SpsaRunner(config)

        result = runner.run()
        # config is already in result from runner.run()

        logger.info("\n--- Simulation Summary (%s) ---", opt_name)
        logger.info(
            "Simulation complete in %.2f seconds.",
            result["elapsed_time"],
        )
        # Parameter summary (active / inactive / total)
        num_active = int(np.count_nonzero(config.w_true))
        num_inactive = int(config.num_params - num_active)
        logger.info(
            "Params summary: active=%d inactive=%d (total=%d)",
            num_active,
            num_inactive,
            config.num_params,
        )
        # Theta snapshots around the run (config-level start/target + final)
        logger.info(
            "Initial Theta (first 5): %s",
            config.theta_start[:5],
        )
        logger.info(
            "Target Theta (first 5) : %s",
            config.theta_peak[:5],
        )
        logger.info(
            "Final Theta (first 5)  : %s",
            result["final_params"][:5],
        )

        # Convergence metrics are already calculated in runner
        metrics = result["convergence_metrics"]
        logger.info("Convergence metrics: %s", metrics)

        # Plot
        if hasattr(sys, "ps1") or "ipykernel" in sys.modules:
            plot_basic_results(result)
        else:
            # Fallback for non-interactive environments if needed,
            # but user requested interactive charts.
            # We'll try to show it anyway.
            try:
                plot_basic_results(result)
            except Exception as e:  # noqa: BLE001
                logger.warning("Could not display plot: %s", e)

    logger.info("\nAll simulations complete.")
