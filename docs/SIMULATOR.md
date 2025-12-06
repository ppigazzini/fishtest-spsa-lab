# Core Simulator Internals

This document provides a detailed technical specification of the `fishtest_spsa_lab.simulator` module. It explains the mathematical models, the optimization algorithms, and the simulation architecture.

## Terminology (batches, blocks, pairs)

- **Pair**: two games with colors swapped (one as White, one as Black). All budgets (`num_pairs`, `batch_size`) are expressed in pairs.
- **Batch / report**: a group of `N` pairs that is dispatched as one job and returned as one result; `batch_size` / `batch_size_pairs` always refer to the number of pairs in such a batch.
- **Block**: the macro optimizer update that aggregates the `N` micro-steps corresponding to a batch of `N` pairs. Optimizers whose names end in `-block` (e.g., `spsa-block`, `sf-sgd-block`, `sf-adam-block`, `adam-block`) use a block-corrected macro that matches, in expectation, the micro const-mean path over that batch.

## 1. The Oracle (`GameProvider`)

The `GameProvider` class acts as the ground-truth environment. It simulates the "Universe" where a chess engine plays against a fixed opponent.

### 1.1. Objective Function
The underlying "true" strength (Elo) of a parameter vector `theta` is defined
as a quadratic bowl relative to a target parameter vector `theta_peak`.

In code-style notation, with weights `w_true[i]`:

```text
Elo(theta) = peak_elo - k_elo * sum_i( w_true[i] * (theta[i] - theta_peak[i])**2 )
```

Where in `SPSAConfig`:

* **`theta_peak`**: true optimum / target position.
* **`theta_start`**: true starting point (and also the dev's start).
* **`w_true[i]`**: true sensitivity / curvature weights. Active parameters
    have `w_true[i] > 0`, inactive ones have `w_true[i] = 0`. Larger
    `w_true[i]` means Elo drops faster for the same displacement, making the
    bowl anisotropic.
* **`peak_elo`** and **`start_elo`**: Elo at `theta_peak` and `theta_start`
    under the simulator's ground truth.
* **`k_elo`**: global curvature scale chosen so that

    ```text
    Elo(theta_start) = start_elo
    Elo(theta_peak)  = peak_elo
    ```

    Let

    ```text
    W_start = sum_i w_true[i] * (theta_start[i] - theta_peak[i])**2
    ```

    Then, when `W_start > 0` and `peak_elo != start_elo`,

    ```text
    k_elo = (peak_elo - start_elo) / W_start
    ```

    otherwise `k_elo` is set to `0.0` and the bowl is flat.

This **geometry layer** depends only on (`theta_peak`, `theta_start`,
`w_true`, `peak_elo`, `start_elo`) and is independent of how the developer
chooses ranges or SPSA/Adam perturbation scales.

#### 1.1.1 Developer model (w_dev, elo_gap_c, c_fraction)

On the fishtest side, we model a developer who:

* Starts from the true `theta_start`.
* Believes the bowl has anisotropy `w_dev[i]` (from `ParamGroup.w_dev`).
    If `w_dev[i] = w_true[i]` they are perfectly calibrated; if
    `w_dev[i] != w_true[i]` they are mis-compensated in that dimension.
* Chooses a target Elo loss `elo_gap_c` for a "one‑step" move in a single
    dimension.
* Uses a simple "step is X% of range" rule with `c_fraction` (typically
    `0.05`).

Given the **true** `k_elo` and the dev's `w_dev[i]`, the simulator derives
per-dimension perturbation scales `c_dev[i]` and dev ranges as:

```text
c_i        = sqrt(elo_gap_c / (k_elo * w_dev[i]))
range_i    = c_i / c_fraction
min_i_dev  = theta_start[i] - range_i / 2
max_i_dev  = theta_start[i] + range_i / 2
```

Under the true bowl, the actual Elo drop from using the dev-chosen `c_i` in
dimension `i` is

```text
DeltaE_true_i = k_elo * w_true[i] * c_i**2
              = elo_gap_c * (w_true[i] / w_dev[i])
```

so:

* `w_dev[i] < w_true[i]`  ⇒ the dev underestimates curvature and steps too far
    (larger Elo risk than intended).
* `w_dev[i] > w_true[i]`  ⇒ the dev overestimates curvature and is overly
    conservative (smaller Elo change than intended).

#### 1.1.2 Geometry vs developer summary

Putting it together:

* The **true geometry** is fully defined by `theta_peak`, `theta_start`,
    `w_true`, `peak_elo`, and `start_elo`, via the derived `k_elo`.
* The **developer layer** transforms high-level knobs (`w_dev`, `elo_gap_c`,
    `c_fraction`) into per-dimension perturbation scales `c_dev` and dev ranges
    `[min_i_dev, max_i_dev]` around `theta_start`.
* Mis-compensation is controlled explicitly by the ratio `w_true[i] / w_dev[i]`
    rather than by hardcoded ranges: this ratio determines how the actual Elo
    loss per step compares to the intended `elo_gap_c`.
* Optimizers operate purely on `c_dev` and the noisy Elo outcomes; they do not
    need to know about `k_elo`, `w_true`, `theta_peak`, or `theta_start`.

### 1.2. Noise Model (PentaModel)
Instead of returning the exact Elo, the oracle returns a noisy game outcome.

1.  **Elo Difference**: `delta_elo = Elo(theta_hero) - Elo(theta_opponent)`.
2.  **PentaModel**: We use the `PentaModel` (from `vendor/pentamodel`) to calculate the probabilities of the 5 possible outcomes for a **pair** of games:
    *   **LL** (Loss-Loss): 0 points
    *   **DL** (Draw-Loss): 0.5 points
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

### 2.1. SPSA (`SPSA`)
Implements the standard SPSA algorithm used in Fishtest-like simulations.

*   **Schedules**:
    *   `c_k = c_base / k^gamma`
    *   `a_k = a_base / (A + k)^alpha`
*   **Update**:
    ```
    step = (a_k / c_k) * net_wins * flip
    theta_{k+1} = theta_k + step
    ```

There is also a corrected macro variant, `SPSABlock`, which aggregates a
batch of `N` pairs using the **mean gain** over the block instead of the
block-start gain:

*   **Mean gain over a block** starting at micro index `k_start`:
    ```
    gain_k = a_k / c_k
    mean_gain = (1.0 / N) * sum_{j=0..N-1} gain_{k_start + j}
    ```
*   **Update** for `SPSABlock` uses that mean gain:
    ```
    step = mean_gain * net_wins * flip
    theta_{k+1} = theta_k + step
    ```

In code, the `SPSAConfig.optimizer` field accepts:

*   `"spsa"` — standard SPSA macro (uses block-start gain `a_k / c_k`).
*   `"spsa-block"` — corrected SPSA macro (uses mean gain over the block; block-corrected SPSA).
*   `"sf-sgd"` — naive Schedule-Free SGD macro (no block compensation in the Polyak surrogate).
*   `"sf-sgd-block"` — Schedule-Free SGD using a block-corrected macro update (closed-form triangular weighting for `x`).
*   `"sf-adam"` — naive Schedule-Free Adam macro (per-block EMA of `g_φ_mean^2`, no `k(N, β₂)` intra-block damping).
*   `"sf-adam-block"` — Schedule-Free Adam using a μ₂-based block-corrected macro update (closed-form `v`/`k(N, β₂)` path).
*   `"adam"` — textbook Adam macro using micro const-mean steps inside each block.
*   `"adam-block"` — block-Adam macro using closed-form EMAs over block-mean SPSA signals.

### 2.2. Schedule-Free SGD (`SFSGD`)
Adapted from "The Road to Schedule-Free Training". It removes the learning rate schedule (`a_k`) by using Polyak averaging.

*   **State**:
    *   `z`: The "fast" iterate (current position).
    *   `x`: The "slow" iterate (averaged position).
*   **Naive macro (`sf-sgd`)**:
    *   `z` update matches the micro loop: one block step with total signal `result = sum outcomes`.
    *   `x` treats the whole block as a single mass at `z_new` with weight `lr * N` (no within-block correction).
*   **Block-corrected macro (`sf-sgd-block`)**:
    *   Uses a closed-form triangular formula for `x` that matches the micro const-mean path at report boundaries.
    *   This removes the aggregation bias from compressing N micro steps into one macro update.

### 2.3. Schedule-Free Adam (`SFAdam`)
Combines Schedule-Free averaging with Adam's adaptive moments.

*   **State**: `z`, `x`, and `v` (second moment estimate).
*   **Naive macro (`sf-adam`)**:
    *   Treats each report as a block with per-pair mean gradient `g_φ_mean = (net_wins / N) * flip` and updates `v` as a simple EMA of `g_φ_mean^2`.
    *   Takes a single block step `step_phi = (lr * net_wins * flip) / denom` (no intra-block damping), so effective step size grows roughly linearly with `N` and depends strongly on the batch-size distribution.
*   **Block-corrected macro (`sf-adam-block`)**:
    *   Maintains a global μ₂ estimate from report-level aggregates `(N_i, S_i, S_i^2 / N_i)` and updates `v` in closed form over each block using that μ₂, with bias correction based on the total number of processed pairs.
    *   Scales the block step by `k(N, beta2)` so one macro step matches N micro const-mean steps in expectation, making it much more stable across varying `N` and closely aligned with the macro path analyzed in the schedule-free Adam notes.

### 2.4. Classic Adam (`Adam` and `AdamBlock`)

These optimizers implement textbook Adam in the simulator, using the same
pentanomial SPSA signal as the other methods but interpreting it as a
gradient in the Adam sense.

*   **State**: `theta`, `m`, and `v` (Adam's parameter, first moment, and second moment).
*   **Gradient proxy**:
    *   For a block with `N` pairs and total signal `net_wins`, we define a
        per-pair mean signal
        `g = - net_wins / N` (minus sign because we *minimize* the quadratic loss
        while SPSA *maximizes* Elo).
    *   The per-parameter gradient is then `grad = g * flip`, where `flip` is
        the SPSA Rademacher vector.

Two macros are provided:

*   **Adam (`"adam"`)** — micro const-mean Adam:
    *   For each report, with length `N` and `net_wins`, we compute `grad` as
        above and then take **N textbook Adam steps** inside the block with
        that constant `grad`.
    *   This matches the "micro const-mean" Adam path studied in `docs/ADAM.md`
        and `analysis/validate_adam.py`: the optimizer applies Adam once per
        pair, but the per-pair gradient is the block mean.
    *   Because it truly does N micro steps, its effective behavior depends on
        N in the same way classic Adam does when you change how many updates
        you take per batch.

*   **AdamBlock (`"adam-block"`)** — block-Adam approximation:
    *   Uses the same gradient proxy `grad`, but does **one block update per
        report** instead of replaying N micro steps.
    *   Internally, it:
        *   Updates `m` and `v` in **closed form** for N Adam-style EMA steps
            with constant `grad` (this part is exact for the constant-gradient
            model).
        *   Computes the exact geometric sum of the first-moment ladder
            `sum_m = sum_{t=1..N} m_t`.
        *   Approximates the total θ change by freezing the denominator at the
            end-of-block second moment (`sqrt(v_N) + eps`) and using a single
            block step `Δtheta ≈ - lr * sum_m / (sqrt(v_N) + eps)`.
    *   This makes `adam-block` a fast, batch-aware approximation to the
        micro const-mean Adam path. It is **not** mathematically identical to
        running N textbook Adam steps; the approximation error comes entirely
        from freezing the denominator over the block.

Both `adam` and `adam-block` share their own hyperparameters (`adam_*`) in
`SPSAConfig`. All optimizers use the developer-level perturbation vector
`c_dev` derived from (`w_true`, `w_dev`, `elo_gap_c`, `c_fraction`, `k_elo`)
and fall back to a simple `0.05 * (theta_max - theta_min)` rule if `c_dev`
is not provided.

## 3. Simulation Runner

The runner orchestrates the interaction between the Optimizer and the GameProvider.

### 3.1. Synchronous Mode (`SpsaRunner`)
A simple loop that processes fixed-size batches of **pairs** sequentially.

1.  Get perturbation scale `c_k` from the optimizer.
2.  Generate `theta_plus`, `theta_minus` by adding/subtracting `c_k * flip` and clipping to parameter bounds.
3.  Simulate `batch_size` **pairs** of games, where `batch_size = config.batch_size` (in pairs).
4.  Call the optimizer's `step` method with `batch_size_pairs=batch_size`.
5.  Repeat until `config.num_pairs` total pairs have been consumed.

### 3.2. Asynchronous Mode (`AsyncSpsaRunner`)
Simulates the distributed, asynchronous nature of Fishtest with a pool of heterogeneous workers.

*   **Event Queue**: A priority queue (`heapq`) stores `JobEvent` objects, sorted by `finish_time`. This allows the simulator to jump forward in time to the next completing job, rather than ticking every second.
*   **Worker Simulation**:
    *   **Heterogeneity**: Workers are `SimWorker(concurrency, speed_factor)` instances created by `make_workers(config, rng)`. Concurrency is sampled uniformly from powers of two between `worker_concurrency_min` and `worker_concurrency_max`, and `speed_factor` is sampled uniformly from `[worker_speed_min, worker_speed_max]`, so some workers are faster and/or larger than others.
    *   **Batch size and duration**: If `variable_batch_size=False`, every job uses the global `batch_size` in pairs (like the synchronous runner). If `variable_batch_size=True`, each job's batch size in games is proportional to the worker's concurrency and `tc_ratio`, then converted to pairs and clipped by the remaining budget of `num_pairs`. Job duration is drawn from a log-normal distribution (calibrated by `game_duration_median` and `game_duration_95th`) and then scaled by `batch_size_pairs / speed_factor`.
*   **The "Snapshot" Problem (Stale Gradients)**:
    This is the critical difference between the synchronous and asynchronous modes.
    1.  **Scheduling**: When a worker requests a job, the runner takes a **snapshot** of the current global parameters (`theta_snapshot`) AND the perturbation vector (`flip`). It also records the scale `c_k` at this moment.
    2.  **Execution**: The worker "goes away" for a simulated duration. During this time, other workers may finish and update the global parameters.
    3.  **Completion**: When the worker finishes (the event pops from the queue), the game outcome for that job's `batch_size_pairs` is calculated based on the *old* `theta_snapshot` and `flip`.
    4.  **Update**: The optimizer calculates the gradient using this outcome and the *old* `flip`, but applies the update to the *current* global parameters. This introduces "gradient staleness," which can destabilize training if the learning rate is too high or the batch size too small.
*   **Event Loop**:
    1.  Initialize `N` workers and schedule their first jobs.
    2.  While the running total of processed pairs is less than `num_pairs`:
        *   Pop the event with the smallest `finish_time`.
        *   Advance `current_time` to this `finish_time`.
        *   Simulate the match result using the job's `theta_snapshot`, `flip`, and `batch_size_pairs`.
        *   Update the global optimizer state and increment the processed pairs counter.
        *   Immediately schedule a new job for the freed worker (if pairs remain).

## 4. Simulation Recipes

These recipes assume you have installed the project editable so that the
`run-simulation` console script is available (from the repo root:
`pip install -e .`).

### 4.1. Synchronous simulation (no asynchrony)

Goal: run a clean, deterministic SPSA / SF-SGD / SF-SGD-block / SF-Adam / SF-Adam-block comparison without
worker effects.

1. In `src/fishtest_spsa_lab/simulator/main.py`, set:

   * `num_pairs = 30_000`
   * `batch_size = 36`
   * `num_workers = 1`  (forces `SpsaRunner`)

2. From the project root, run:

   ```bash
   run-simulation
   ```

With `num_workers = 1`, the code uses `SpsaRunner`, processing fixed batches of
`batch_size` **pairs** until `num_pairs` pairs are consumed. Async-specific
fields in `SPSAConfig` are ignored.

### 4.2. Async simulation with fixed batch size

Goal: use the asynchronous event loop and worker model, but keep a fixed batch
size per job.

1. Keep multiple workers and disable variable batch size when you build
   `SPSAConfig` in `main()`:

   * `num_pairs = 30_000`
   * `batch_size = 36`
   * `num_workers = 20`
   * `variable_batch_size = False`

2. Ensure `AsyncSpsaRunner(config, workers=workers)` is used when
   `num_workers > 1`, and run:

   ```bash
   run-simulation
   ```

Every async job will then use `batch_size_pairs = config.batch_size` (in pairs),
independent of worker concurrency, while still modeling heterogeneous durations
and gradient staleness.

### 4.3. Async simulation with variable / random batches

Goal: mimic Fishtest-like heterogeneity where each worker has its own
concurrency and thus its own effective batch size.

1. Enable variable batch size and configure the worker concurrency range and
   `tc_ratio`:

   * `num_pairs = 30_000`
   * `batch_size = 36`  (used only if `variable_batch_size` is `False`)
   * `num_workers = 20`
   * `variable_batch_size = True`
   * `worker_concurrency_min = 4`
   * `worker_concurrency_max = 32`
   * `tc_ratio = 1.0`

2. Build a worker pool with `make_workers(config, rng)` and pass it to
   `AsyncSpsaRunner(config, workers=workers)` (as in `main()`).

3. Run:

   ```bash
   run-simulation
   ```

Each worker draws a concurrency from the allowed powers of two in
`[worker_concurrency_min, worker_concurrency_max]`, derives its batch size in
games and then in pairs, and uses log-normal durations scaled by `speed_factor`.
This produces realistic random / heterogeneous batch scheduling and gradient
staleness.

