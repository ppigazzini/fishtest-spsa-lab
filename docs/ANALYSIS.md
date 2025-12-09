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
Validates the standard SPSA update rule and how its batched "macro" update compares to the ideal sequential path.

- **Gain correction**: The learning rate `a_k` and perturbation `c_k` change *during* a batch.
- **Uncorrected macro**: Uses `a_start / c_start` for the whole batch (incorrect).
- **Corrected macro**: Uses the average gain, for example:
    - `mean_gain = (1.0 / N) * sum(a[k + j] / c[k + j] for j in range(N))`
- **Micro mean path**: Sequential SPSA updates with `S_i / N_i` repeated `N_i` times inside each report.
- **Micro real path**: Sequential SPSA updates with the true per-pair outcomes.

Plots in [src/fishtest_spsa_lab/analysis/validate_spsa.py](src/fishtest_spsa_lab/analysis/validate_spsa.py):

- **Single schedule (Figure 1)** — θ vs pairs, legend order:
    - `theta — micro real` (ground-truth sequential SPSA)
    - `theta — micro mean` (theoretical bridge)
    - `theta — macro` (corrected macro / production path)
    - `theta — macro (uncorrected)` (incorrect baseline)
- **Original vs shuffled (Figure 2)** — same four paths, for both original and end-adjacent–shuffled schedules:
    - `… micro real (orig)`, `… micro real (shuf)`
    - `… micro mean (orig)`, `… micro mean (shuf)`
    - `… macro (orig)`, `… macro (shuf)`
    - `… macro unc. (orig)`, `… macro unc. (shuf)`

### 3.2. Schedule-Free SGD (`validate_sf_sgd.py`)
Validates the schedule-free Polyak averaging logic.

- **State**: Tracks fast iterate `z`, Polyak surrogate `x`, and blended export `theta = (1 - beta) * z + beta * x`.
- **Challenge**: Updating `x` requires a weighted average of `z` at every step.
- **Solution (macro)**: Uses a closed-form "triangular" formula inside each report to reproduce the per-step averaging in `O(1)` time.
- **Micro mean path**: Sequential schedule-free SGD updates with `S_i / N_i` repeated `N_i` times.
- **Micro real path**: Sequential schedule-free SGD with the true outcomes.

Plots in [src/fishtest_spsa_lab/analysis/validate_sf_sgd.py](src/fishtest_spsa_lab/analysis/validate_sf_sgd.py):

- **Single schedule (Figure 1)** — three panels for `x`, `z`, and `theta` vs pairs; in each panel the legend order is:
    - `… micro real`
    - `… micro mean`
    - `… macro`
- **Original vs shuffled (Figure 2)** — same three paths for both schedules; in each panel:
    - `… micro real (orig)`, `… micro real (shuf)`
    - `… micro mean (orig)`, `… micro mean (shuf)`
    - `… macro (orig)`, `… macro (shuf)`

### 3.3. Schedule-Free Adam (`validate_sf_adam.py`)
Validates the adaptive second-moment estimation and its block-wise approximation.

- **State**: Tracks `z`, Polyak surrogate `x`, blended `theta`, and second moment `v`.
- **Online second-moment (mu2) estimation**:
    - The optimizer needs an estimate of `E[g^2]` (expected squared gradient).
    - [src/fishtest_spsa_lab/analysis/validate_variance.py](src/fishtest_spsa_lab/analysis/validate_variance.py) implements `OnlineReportStats`, which computes an exact block-averaged estimator using only `(N_i, s_i, s_i^2/N_i)`.
    - `validate_sf_adam.py` mirrors this logic via report-level aggregates plus an optional warm-start prior.
- **Macro path**: One update per report using block summaries and closed-form Adam-style `v` evolution.
- **Micro mean path**: Replays the same online μ² estimator at micro-step resolution, with `S_i / N_i` repeated `N_i` times.
- **Micro real path**: Full per-outcome schedule-free Adam with per-step `g` and `g^2`.

Plots in [src/fishtest_spsa_lab/analysis/validate_sf_adam.py](src/fishtest_spsa_lab/analysis/validate_sf_adam.py):

- **Single schedule (Figure 1)** — `x`, `z`, and `theta` vs pairs, legend order in each panel:
    - `… micro real`
    - `… micro mean`
    - `… macro`
- **Original vs shuffled (Figure 2)** — same three paths for original and shuffled schedules:
    - `… micro real (orig)`, `… micro real (shuf)`
    - `… micro mean (orig)`, `… micro mean (shuf)`
    - `… macro (orig)`, `… macro (shuf)`

### 3.4. Classic Adam (`validate_adam.py`)
Validates a block-level Adam variant against textbook (per-outcome) Adam.

- **State**: Standard Adam state `(theta, m, v)` in parameter space.
- **Update rule (textbook Adam)**:
    - At step `t` with gradient `g_t`:
        - `m_{t+1} = beta1 * m_t + (1 - beta1) * g_t`.
        - `v_{t+1} = beta2 * v_t + (1 - beta2) * g_t^2`.
        - `m_hat_{t+1} = m_{t+1} / (1 - beta1^{t+1})`, `v_hat_{t+1} = v_{t+1} / (1 - beta2^{t+1})`.
        - `theta_{t+1} = theta_t - lr * m_hat_{t+1} / (sqrt(v_hat_{t+1}) + eps)`.
- **Micro real path**: One textbook Adam step per outcome (`g_j = outcome_j`).
- **Micro const-mean path**: For each report `i`, takes `N_i` Adam steps with constant gradient `g_i = S_i / N_i` (constant-mean surrogate inside the block).
- **Macro const-mean path**: A block function that internally replays exactly those `N_i` classic-Adam steps using `g_i = S_i / N_i`; by construction it matches the micro const-mean path at report boundaries.
- **Macro block-Adam path**: A new optimizer that performs a single closed-form block update using only `(N_i, S_i)` by:
    - Updating `m` and `v` in closed form for `N_i` steps with constant `g_i`.
    - Summing the first-moment ladder `sum_m = sum_{t=1..N_i} m_t` in closed form.
    - Approximating the `N_i` θ updates with one block step using a frozen denominator `sqrt(v_{N_i}) + eps`.

Plots in [src/fishtest_spsa_lab/analysis/validate_adam.py](src/fishtest_spsa_lab/analysis/validate_adam.py):

- **Single schedule (Figure 1)** — θ vs pairs, legend order:
    - `theta — micro real`
    - `theta — micro const-mean`
    - `theta — macro const-mean`
    - `theta — macro block-Adam`
- **Original vs shuffled (Figure 2)** — same four paths for original and shuffled schedules:
    - `… micro real (orig)`, `… micro real (shuf)`
    - `… micro const-mean (orig)`, `… micro const-mean (shuf)`
    - `… macro const-mean (orig)`, `… macro const-mean (shuf)`
    - `… macro block-Adam (orig)`, `… macro block-Adam (shuf)`

## 4. Key Files

- **[src/fishtest_spsa_lab/analysis/common.py](src/fishtest_spsa_lab/analysis/common.py)**:
    - `make_schedule`: Generates the synthetic sequence of batch sizes and outcomes.
    - `end_adjacent_shuffle`: Applies the end-adjacent shuffling used in all `validate_*.py` scripts.
    - `plot_many`: Helper for consistent Matplotlib plotting and legends.
- **[src/fishtest_spsa_lab/analysis/validate_variance.py](src/fishtest_spsa_lab/analysis/validate_variance.py)**:
    - `compute_pentanomial_moments`: Theoretical mean, variance, and second moment for the pentanomial game model.
    - `OnlineReportStats`: Accumulator for calculating block-averaged mean and second moment (mu and mu2) from batch-level statistics.
- **[src/fishtest_spsa_lab/analysis/validate_*.py](src/fishtest_spsa_lab/analysis)**: Executable scripts that run the comparisons and generate the plots described above.
 - **[src/fishtest_spsa_lab/analysis/plot_spsa_schedule.py](src/fishtest_spsa_lab/analysis/plot_spsa_schedule.py)**:
    Plots the naive 1D SPSA schedules `c_k`, `a_k`, and `r_k = a_k / c_k^2` versus the pair index for a chosen `(num_pairs, A, alpha, gamma, c_end, r_end)`.
