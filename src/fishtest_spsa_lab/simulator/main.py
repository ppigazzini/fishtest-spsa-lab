"""Main entry point and plotting logic."""

from __future__ import annotations

import logging
import sys

import matplotlib.pyplot as plt
import numpy as np

from fishtest_spsa_lab.simulator.config import SPSAConfig
from fishtest_spsa_lab.simulator.runner import AsyncSpsaRunner, SpsaRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def plot_basic_results(result: dict) -> None:
    """Plot parameter trajectories."""
    config = result["config"]
    trajectory = result["trajectory"]
    steps = np.arange(len(trajectory))

    _fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    _fig.suptitle(f"Optimizer: {config.optimizer}", fontsize=16)

    # Identify active and inactive parameters based on sensitivity
    sensitivity = config.param_sensitivity
    active_indices = np.where(sensitivity > 0)[0]
    inactive_indices = np.where(sensitivity == 0)[0]
    max_inactive_plots = 5

    # Plot 1: Active Parameters vs Batch
    for idx in active_indices:
        ax1.plot(steps, trajectory[:, idx], label=f"Param {idx}")

    # Plot target line
    targets = config.param_target
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

    # Plot 2: Inactive Parameters vs Batch
    if len(inactive_indices) > max_inactive_plots:
        inactive_indices = inactive_indices[:max_inactive_plots]

    if len(inactive_indices) > 0:
        for idx in inactive_indices:
            ax2.plot(steps, trajectory[:, idx], label=f"Param {idx}")

        ax2.set_title(f"Inactive Parameters Trajectory (First {len(inactive_indices)})")
        ax2.set_xlabel("Batch Number")
        ax2.set_ylabel("Parameter Value")
        ax2.legend()
        ax2.grid(visible=True)
        ax2.ticklabel_format(useOffset=False, style="plain")

    plt.tight_layout()
    plt.show()


def main() -> None:
    """Run the main entry point."""
    optimizers = ["classic", "sf-sgd", "sf-adam"]

    # Common configuration parameters
    num_pairs = 50_000
    batch_size = 10
    num_workers = 50

    for opt_name in optimizers:
        logger.info("=" * 60)
        logger.info("Running Simulation with Optimizer: %s", opt_name)
        logger.info("=" * 60)

        config = SPSAConfig(
            optimizer=opt_name,
            num_pairs=num_pairs,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        logger.info("Starting SPSA optimization with config:\n%s", config)

        # Select runner based on workers
        if config.num_workers > 1:
            runner = AsyncSpsaRunner(config)
        else:
            runner = SpsaRunner(config)

        result = runner.run()
        # config is already in result from runner.run()

        logger.info("\n--- Simulation Summary (%s) ---", opt_name)
        logger.info(
            "Simulation complete in %.2f seconds.",
            result["elapsed_time"],
        )
        logger.info(
            "Final parameters (first 5): %s",
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
