# Fishtest SPSA Lab

This project implements a standalone laboratory for the SPSA (Simultaneous Perturbation Stochastic Approximation) tuning algorithm used in [Fishtest](https://github.com/official-stockfish/fishtest). It provides a controlled environment to test, compare, and validate different optimization algorithms before deploying them to the distributed testing framework.

## Overview

The simulator mirrors the logic of the Fishtest server but runs locally with a simulated oracle ("GameProvider") instead of distributed workers. This allows for rapid experimentation with hyperparameters, schedules, and new optimizer variants.

### Key Features

*   **Realistic Noise Model**: Uses `PentaModel` to generate game outcomes (Win/Draw/Loss) based on Elo differences, simulating the variance found in real engine testing.
*   **Multiple Optimizers**:
    *   **Classic SPSA**: The standard algorithm currently used in Fishtest with decaying learning rates.
    *   **Schedule-Free SGD**: An adaptation of Schedule-Free SGD for SPSA, eliminating the need for complex decay schedules.
    *   **Schedule-Free Adam**: A variant combining Schedule-Free updates with Adam's adaptive moments for better handling of parameters with different sensitivities.
*   **Asynchronous Simulation**: Supports simulating multiple workers with variable latencies and out-of-order reporting, mimicking the distributed nature of Fishtest.
*   **Variable Sensitivity**: Can define parameter groups with different sensitivities to the objective function (Elo), allowing tests of how well optimizers handle heterogeneous parameters.

## Documentation Map

*   **[Core Simulator Internals](docs/SIMULATOR.md)**: Detailed math and logic of the `simulator` module (Optimizers, Oracle, Runner).
*   **[Analysis Framework Internals](docs/ANALYSIS.md)**: Explanation of the validation scripts and statistical tools in the `analysis` module.
*   **[Algorithms Guide](docs/ALGORITHMS.md)**: Theory guide for SPSA, SF-SGD, and SF-Adam.
*   **[Macro vs Micro Analysis](docs/SPSA_MACRO_VS_MICRO.md)**: Analysis of batching effects and aggregation bias.

## Project Structure

The project is organized into the following modules:

*   **`src/fishtest_spsa_lab/simulator/`**: The production simulation engine.
    *   `main.py`: Simulation entry point and configuration wiring.
    *   `runner.py`: Simulation loop and `GameProvider`.
    *   `optimizer.py`: SPSA, SF-SGD, SF-Adam, and Adam implementations.
    *   `config.py`: Elo geometry and developer-model configuration.
*   **`src/fishtest_spsa_lab/analysis/`**: Validation and research tools.
    *   `validate_*.py`: Scripts to mathematically verify update rules.
    *   `common.py`: Shared testing utilities.
*   **`src/fishtest_spsa_lab/vendor/`**: Third-party libraries.
    *   `pentamodel/`: Chess outcome probability model.

## Design Principles

1.  **Modularity**: The `Optimizer` is decoupled from the `GameProvider`, allowing us to swap optimization algorithms without changing the simulation physics.
2.  **Fishtest Parity**: The simulator is designed to replicate the exact mathematical behavior of the distributed Fishtest framework, including "Phi-space" scaling and asynchronous batching.
3.  **Verifiability**: The `analysis` module ensures that optimizations (like batching updates) do not deviate from the theoretical sequential updates.

## Usage

You can run the validation scripts directly using `uv`:

```bash
# Run the main simulator
uv run run-simulation

# Run SPSA validation
uv run validate-spsa

# Run SGD validation
uv run validate-sgd

# Run Adam validation
uv run validate-adam

# Run Variance validation
uv run validate-variance
```

### Configuration

Currently, simulation parameters are defined directly in the code.

*   **Simulator**: Edit `src/fishtest_spsa_lab/simulator/main.py` to change `num_pairs`, `batch_size`, or `num_workers`.
*   **Validation**: Edit the `main()` function in the respective `validate_*.py` scripts in `src/fishtest_spsa_lab/analysis/`.

These scripts will:
1.  Initialize the simulation with the specific optimizer.
2.  Run the simulation for a specified number of game pairs.
3.  Log progress and Elo estimates to the console.
4.  Generate plots showing the trajectory of parameters over time.

## Requirements

*   Python 3.13+
*   `numpy`
*   `matplotlib`

## Acknowledgments

*   Thanks to [@vondele](https://github.com/vondele) for the [pentamodel](https://github.com/vondele/pentamodel) library, which provides the realistic game outcome probabilities used in this simulator.
