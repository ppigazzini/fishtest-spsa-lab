# Analysis Framework Internals

The `fishtest_spsa_lab.analysis` module provides a rigorous mathematical verification suite for the SPSA optimizers.

## 1. The Gap: Theory vs. Reality

The State-of-the-Art (SOTA) SPSA algorithm, as described in academic literature, is a **purely sequential** process:
1.  Perturb parameters.
2.  Run **one** measurement (game).
3.  Update parameters immediately.
4.  Repeat.

**Fishtest operates differently:**
*   **Batched**: Games are grouped into batches (reports) of varying sizes (`N_i`).
*   **Parallel**: Thousands of workers run games concurrently.
*   **Out-of-Order**: Results arrive asynchronously, often out of the original scheduling order.

**Purpose of this Framework**:
This framework serves to bridge the gap between the **optimal sequential algorithm** and the **distributed Fishtest reality**. It validates that our "Macro" (batched) update logic mathematically corrects for these discrepancies, ensuring that the distributed system behaves as close to the theoretical ideal as possible.

## 2. Simulation Architecture

The analysis simulations run in a "clean room" environment, decoupled from the complex `GameProvider` logic. They use deterministic data generators to ensure reproducibility.

### 2.1. Data Generation (`common.py`)
The simulation starts by generating a **Schedule**:
1.  **Reports**: A sequence of batches. Each batch `i` has a size `N_i` (number of game pairs).
2.  **Outcomes**: For each batch, a list of `N_i` outcomes is generated using a Pentanomial distribution (Win/Loss/Draw probabilities).
3.  **Determinism**: The schedule is generated upfront using a fixed seed, ensuring that all comparison runs (Macro vs. Micro) see the exact same data.

### 2.2. Execution Modes
The framework compares three distinct execution paths:

1.  **Macro (Production / Fishtest)**:
    *   **Input**: Receives the sum of outcomes `S_i` and count `N_i` for a batch.
    *   **Logic**: Performs a **single** parameter update per batch.
    *   **Math**: Uses closed-form approximations (e.g., triangular weighting for Polyak averaging) to calculate what the parameters *would be* if we had updated them sequentially.

2.  **Micro Mean (Theoretical Bridge)**:
    *   **Input**: Receives a sequence of `N_i` identical values: `S_i / N_i`.
    *   **Logic**: Performs `N_i` sequential updates.
    *   **Purpose**: This is the "mathematical target" for the Macro update. If `Macro == Micro Mean`, then our batching logic correctly handles changing learning rates and averaging weights.

3.  **Micro Real (Ground Truth / SOTA)**:
    *   **Input**: Receives the actual sequence of `N_i` individual outcomes.
    *   **Logic**: Performs `N_i` sequential updates.
    *   **Purpose**: Represents the **optimal sequential algorithm**. The difference between `Micro Mean` and `Micro Real` represents the irreducible error introduced by batching (noise).

### 2.3. Out-of-Order Execution (Shuffling)
To simulate the distributed nature of Fishtest (where workers return results asynchronously), the framework includes a **Shuffling** phase:
*   **End-Adjacent Shuffle**: A probabilistic swap of adjacent reports in the schedule.
*   **Verification**: The framework asserts that the `Macro` update logic remains robust (i.e., `Macro == Micro Mean`) even when the order of batches is randomized.

## 3. Optimizer Internals

### 3.1. Classic SPSA (`validate_spsa.py`)
Validates the standard SPSA update rule.

*   **Gain Correction**: The learning rate `a_k` and perturbation `c_k` change *during* a batch.
*   **Uncorrected Macro**: Uses `a_start / c_start` for the whole batch. (Incorrect).
*   **Corrected Macro**: Calculates the average gain `g_bar = (1/N) * sum(a_{k+j} / c_{k+j})`.
*   **Assertion**: `Macro Corrected` trajectory must exactly match `Micro Mean`.

### 3.2. Schedule-Free SGD (`validate_sgd.py`)
Validates the Polyak averaging logic.

*   **State**: Tracks fast iterate `z` and slow average `x`.
*   **Challenge**: Updating `x` requires a weighted average of `z` at every step.
*   **Solution**: The script implements a closed-form "Triangular Update" for `x`.
    *   Inside a batch of size `N`, `z` moves linearly.
    *   The average `x` accumulates `z` with weights that form a triangular series.
    *   The macro update computes this sum in `O(1)` time.

### 3.3. Schedule-Free Adam (`validate_adam.py`)
Validates the adaptive moment estimation.

*   **State**: Tracks `z`, `x`, and the second moment estimator `v`.
*   **Online Estimation**:
    *   The optimizer needs an estimate of `E[g^2]` (expected squared gradient).
    *   **`validate_variance.py`** implements `OnlineReportStats`, which calculates the exact block-averaged variance using only batch summaries (`sum(s)`, `sum(N)`, `sum(s^2/N)`).
*   **Synchronization**: The script ensures that the Macro path (using block summaries) and the Micro path (using step-by-step updates) share the exact same estimate for `v` at every point in time.

## 4. Key Files

*   **`common.py`**:
    *   `make_schedule`: Generates the test data.
    *   `end_adjacent_shuffle`: Simulates network reordering.
    *   `plot_many`: Visualization helper.
*   **`validate_variance.py`**:
    *   `compute_pentanomial_moments`: Theoretical math for chess outcomes.
    *   `OnlineReportStats`: The accumulator class for calculating variance from batch sums.
*   **`validate_*.py`**: The executable scripts that run the comparisons and generate plots.
