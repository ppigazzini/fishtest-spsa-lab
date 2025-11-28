# Core Simulator Internals

This document provides a detailed technical specification of the `fishtest_spsa_lab.simulator` module. It explains the mathematical models, the optimization algorithms, and the simulation architecture.

## 1. The Oracle (`GameProvider`)

The `GameProvider` class acts as the ground-truth environment. It simulates the "Universe" where a chess engine plays against a fixed opponent.

### 1.1. Objective Function
The underlying "true" strength (Elo) of a parameter vector `theta` is defined as a quadratic loss function relative to a target parameter vector `theta_target`.

```
Elo(theta) = PeakElo - k_elo * sum(w_i * (theta[i] - theta_target[i])**2)
```

*   **`PeakElo`**: The maximum possible Elo (when `theta == theta_target`).
*   **`k_elo`**: A scaling constant calculated during initialization. It ensures that the starting parameters (`param_start`) have exactly `initial_elo_gap` (default 500 Elo) less than the target.
*   **`w_i`**: Sensitivity weights. Active parameters typically have `w_i > 0`, while inactive ones have `w_i = 0`.

### 1.2. Noise Model (PentaModel)
Instead of returning the exact Elo, the oracle returns a noisy game outcome.

1.  **Elo Difference**: `delta_elo = Elo(theta_hero) - Elo(theta_opponent)`.
2.  **PentaModel**: We use the `PentaModel` (from `vendor/pentamodel`) to calculate the probabilities of the 5 possible outcomes for a **pair** of games:
    *   **LL** (Loss-Loss): 0 points
    *   **LD** (Loss-Draw): 0.5 points
    *   **DD** (Draw-Draw): 1.0 point
    *   **WD** (Win-Draw): 1.5 points
    *   **WW** (Win-Win): 2.0 points
3.  **Sampling**: The outcome is sampled from this multinomial distribution.
4.  **Net Wins**: The optimizer receives `net_wins`, calculated as:
    ```
    net_wins = (2*WW + WD) - (2*LL + LD)
    ```
    This maps the result to the range `[-2, 2]` per pair.

## 2. Optimizers

All optimizers inherit from the `Optimizer` base class.

### 2.1. Classic SPSA (`ClassicSPSA`)
Implements the standard SPSA algorithm used in Fishtest.

*   **Schedules**:
    *   `c_k = c_base / k^gamma`
    *   `a_k = a_base / (A + k)^alpha`
*   **Update**:
    ```
    step = (a_k / c_k) * net_wins * flip
    theta_{k+1} = theta_k + step
    ```

### 2.2. Schedule-Free SGD (`SFSGD`)
Adapted from "The Road to Schedule-Free Training". It removes the learning rate schedule (`a_k`) by using Polyak averaging.

*   **State**:
    *   `z`: The "fast" iterate (current position).
    *   `x`: The "slow" iterate (averaged position).
*   **Update**:
    1.  Update `z` with the gradient.
    2.  Update `x` as a weighted average of `z` values.
    3.  **Triangular Weighting**: To handle batched updates efficiently, `x` is updated using a closed-form formula that simulates a triangular weighting scheme over the batch.

### 2.3. Schedule-Free Adam (`SFAdam`)
Combines Schedule-Free averaging with Adam's adaptive moments.

*   **State**: `z`, `x`, and `v` (second moment estimate).
*   **Damping Factor**: When updating with a batch size `N > 1`, the step size is damped by a factor `k(N, beta2)` to account for the fact that we are taking `N` steps worth of gradient in one go, but the adaptive moment `v` only updates once.
    ```
    k_damping approx 1 - (N-1)/4 * (1 - beta2)
    ```
*   **Preconditioning**: The gradient is divided by `sqrt(v) + epsilon`.

## 3. Simulation Runner

The runner orchestrates the interaction between the Optimizer and the GameProvider.

### 3.1. Synchronous Mode (`SpsaRunner`)
A simple loop that processes batches sequentially.
1.  Get perturbation `c_k`.
2.  Generate `theta_plus`, `theta_minus`.
3.  Simulate `batch_size` games.
4.  Update optimizer.

### 3.2. Asynchronous Mode (`AsyncSpsaRunner`)
Simulates the distributed, asynchronous nature of Fishtest.

*   **Event Queue**: A priority queue (`heapq`) stores `JobEvent` objects, sorted by `finish_time`. This allows the simulator to jump forward in time to the next completing job, rather than ticking every second.
*   **Worker Simulation**:
    *   **Heterogeneity**: Each worker is assigned a "speed factor" sampled from a log-normal distribution. This simulates the reality where  contributors provide both fast machines and slow machines.
    *   **Duration**: The duration of a job is calculated as `base_duration * batch_size * worker_factor`, where `base_duration` is also stochastic (log-normal).
*   **The "Snapshot" Problem (Stale Gradients)**:
    This is the critical difference between the synchronous and asynchronous modes.
    1.  **Scheduling**: When a worker requests a job, the runner takes a **snapshot** of the current global parameters (`theta_snapshot`) AND the perturbation vector (`flip`). It also records the scale `c_k` at this moment.
    2.  **Execution**: The worker "goes away" for a simulated duration. During this time, other workers may finish and update the global parameters.
    3.  **Completion**: When the worker finishes (the event pops from the queue), the game outcome is calculated based on the *old* `theta_snapshot` and `flip`.
    4.  **Update**: The optimizer calculates the gradient using this outcome and the *old* `flip`, but applies the update to the *current* global parameters. This introduces "gradient staleness," which can destabilize training if the learning rate is too high or the batch size too small.
*   **Event Loop**:
    1.  Initialize `N` workers and schedule their first jobs.
    2.  While `batches_processed < total_batches`:
        *   Pop the event with the smallest `finish_time`.
        *   Advance `current_time` to this `finish_time`.
        *   Simulate the match result using `theta_snapshot` and `flip`.
        *   Update the global optimizer state.
        *   Immediately schedule a new job for the freed worker (if batches remain).
