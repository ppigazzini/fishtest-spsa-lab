# Fishtest SPSA Lab

This project implements a standalone laboratory for the SPSA (Simultaneous Perturbation Stochastic Approximation) tuning algorithm used in [Fishtest](https://github.com/official-stockfish/fishtest). It provides a controlled environment to test, compare, and validate different optimization algorithms before deploying them to the distributed testing framework.

## Overview

The simulator mirrors the logic of the Fishtest server but runs locally with a simulated oracle ("GameProvider") instead of distributed workers. This allows for rapid experimentation with hyperparameters, schedules, and new optimizer variants.

### Key Features

*   **Pentanomial Noise Model**: Uses `PentaModel` to generate game-pair outcomes based on Elo differences. Note the modelling limits documented in [Simulator.md](docs/Simulator.md) section 1 before quoting any noise figure from it.
*   **Multiple Optimizers**:
    *   **Classic SPSA**: The standard algorithm currently used in Fishtest with decaying learning rates.
    *   **Schedule-Free SGD**: An adaptation of Schedule-Free SGD for SPSA, eliminating the need for complex decay schedules.
    *   **Schedule-Free Adam**: A variant combining Schedule-Free updates with Adam's adaptive moments for better handling of parameters with different sensitivities.
*   **Asynchronous Simulation**: Supports simulating multiple workers with variable latencies and out-of-order reporting, mimicking the distributed nature of Fishtest.
*   **Variable Sensitivity**: Can define parameter groups with different sensitivities to the objective function (Elo), allowing tests of how well optimizers handle heterogeneous parameters.

## Documentation Map

*   **[Core Simulator Internals](docs/Simulator.md)**: Detailed math and logic of the `simulator` module (Optimizers, Oracle, Runner).
*   **[Analysis Framework Internals](docs/Analysis.md)**: Explanation of the validation scripts and statistical tools in the `analysis` module.
*   **[Algorithms Guide](docs/Algorithms.md)**: Theory guide for SPSA, SF-SGD, and SF-Adam.
*   **[SF-SGD Derivation](docs/SF_SGD_derivation.md)**: Schedule-free SPSA with SGD backend derivation.
*   **[SF-Adam Derivation](docs/SF_Adam_derivation.md)**: Schedule-free SPSA with AdamW backend derivation.
*   **[Macro vs Micro Analysis](docs/SPSA_macro_micro.md)**: Analysis of batching effects and aggregation bias.
*   **[Elo Function](docs/Elo_function.md)**: Latent Elo surface and measurement model for SPSA scaling.
*   **[Noise Ball](docs/Noise_ball.md)**: Elo loss from match-outcome noise (noise-ball diagnostics).
*   **[Rademacher Scaling](docs/Rademacher.md)**: Why `sqrt(N)` and `1/sqrt(N)` show up in SPSA.

## Project Structure

The project is organized into the following modules:

*   **`src/fishtest_spsa_lab/simulator/`**: The production simulation engine.
    *   `main.py`: Simulation entry point and configuration wiring.
    *   `runner.py`: Simulation loop and `GameProvider`.
    *   `optimizer.py`: SPSA, SF-SGD, SF-Adam, and Adam implementations.
    *   `config.py`: Elo geometry and developer-model configuration.
*   **`src/fishtest_spsa_lab/analysis/`**: Validation and research tools.
    *   `validate_*.py`: Scripts to mathematically verify update rules.
    *   `noise_ball.py`: Stationary Elo loss from match-outcome noise.
    *   `rademacher.py`: Monte Carlo checks of `sqrt(N)` scaling.
    *   `spsa_gradient.py`: SPSA gradient estimator diagnostics.
    *   `optimize_spsa_toy.py`: Toy optimizer with eigen-spectrum tooling.
    *   `common.py`: Shared testing utilities.
    *   `crossover.py`: measures where a decaying gain overtakes a constant
        one, as a function of budget (`uv run spsa-crossover`).
    *   `design.py`: SPSA design equations -- the `r_end` and game budget a
        target precision and parameter count imply. Fishtest defaults `r_end` to
        a constant 0.002, which is right at about 14 parameters and wrong
        elsewhere; see `uv run spsa-design`.
    *   `plot_spsa_schedule.py`: Plots the naive Fishtest SPSA internal schedules
        (c_k, a_k, r_k) over iterations for a given (num_pairs, A, alpha,
        gamma, c_end, r_end).
*   **`src/fishtest_spsa_lab/vendor/`**: Third-party libraries.
    *   `pentamodel/`: Chess outcome probability model.
*   **`tests/`**: Pytest suite validating macro-vs-micro correctness, config
    initialization, and optimizer single-step sanity.

## Design Principles

1.  **Modularity**: The `Optimizer` is decoupled from the `GameProvider`, allowing us to swap optimization algorithms without changing the simulation physics.
2.  **Fishtest Protocol Parity**: The simulator reproduces Fishtest's *protocol* exactly -- dispatch snapshot, flip transport, batched arrival, clipping, and asynchronous out-of-order updates. It deliberately differs in *gain*: the update signal is halved and scaled by `1/sqrt(N)` (see [Algorithms.md](docs/Algorithms.md) ch. 4 and [Rademacher.md](docs/Rademacher.md)). That divergence is a proposal under test, not an accident.
3.  **Verifiability**: The `analysis` module ensures that optimizations (like batching updates) do not deviate from the theoretical sequential updates.

## Usage

This repo exposes its main tools as console-script entry points (see `[project.scripts]` in `pyproject.toml`).

You can run them directly using `uv`:

```bash
# Simulator
uv run run-simulation

# Validator / analysis entry points
uv run validate-spsa
uv run validate-spsa-u2
uv run validate-penta
uv run validate-variance
uv run validate-sf-sgd-block
uv run validate-sf-adam-block
uv run validate-adam

# Toy SPSA optimizer demo (pentamodel-driven noise)
uv run optimize-spsa-toy

# Stationary noise-ball estimate
uv run noise-ball

# Rademacher scaling and SPSA gradient Monte Carlo
uv run rademacher
uv run spsa-gradient

# Size a run: the r_end a given parameter count actually calls for
uv run spsa-design

# Where a decaying gain overtakes a constant one
uv run spsa-crossover --seeds 4 --budgets 6000 36000 120000
uv run spsa-design --precision 0.5 --confidence 0.95 --sigma2 0.2274 --n 1 8 32

# Plot naive SPSA internal schedules (c_k, a_k, r_k)
uv run plot-spsa-schedule
```

Only `noise-ball`, `optimize-spsa-toy`, `validate-penta`, `rademacher`,
`spsa-gradient`, `spsa-design` and `spsa-crossover` parse command-line
arguments. The remaining entry points take
no options and run to completion with built-in constants, so passing `--help`
to them starts the run.

### Development

Dev tools are not installed by default (`[tool.uv] default-groups = []`), so
every check needs `--group dev`:

```bash
uv sync --group dev
uv run --group dev pytest -q
uv run --group dev ruff check .
uv run --group dev ruff format --check .
uv run --group dev ty check
uv run --group dev pre-commit run --all-files
```

### Configuration

Currently, simulation parameters are defined directly in the code.

*   **Simulator**: Edit `src/fishtest_spsa_lab/simulator/main.py` to change `num_pairs`, `batch_size`, or `num_workers`.
*   **Validation**: Edit the `main()` function in the respective `validate_*.py` scripts in `src/fishtest_spsa_lab/analysis/`.

What they do:

*   `run-simulation` runs every registered optimizer over a shared set of seeds
    and prints one table: mean final Elo, a 95% interval, and a paired difference
    against the `spsa` baseline. It draws no plots.
*   The `validate-*` scripts print a result table -- each invariant, its measured
    value, the tolerance it is asserted at, and PASS or FAIL -- and **exit
    non-zero if any check fails**. Set `SPSA_LAB_NO_PLOT=1` to suppress figures;
    they are also suppressed automatically under a non-interactive matplotlib
    backend. `pytest` runs all seven.
*   The remaining tools (`noise-ball`, `optimize-spsa-toy`, `rademacher`,
    `spsa-gradient`, `plot-spsa-schedule`) are exploratory and open figures.

## Requirements

*   Python 3.14+
*   `numpy`
*   `matplotlib`

## Acknowledgments

*   Thanks to [@vondele](https://github.com/vondele) for the [pentamodel](https://github.com/vondele/pentamodel) library, which provides the realistic game outcome probabilities used in this simulator.
