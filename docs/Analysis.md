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

### 3.5. Pentanomial Noise Charts (`validate_pentanomial.py`)

Explores how batched pentanomial sampling noise propagates into EMA/cumulative summaries and into a
derived penta coefficient (r) and gain scale factor. This is an analysis sandbox for experimenting with
definitions and plots before (optionally) porting ideas into the simulator.

This script does **not** compare macro vs micro updates. Instead, it simulates variable-sized batches of
penta outcomes and tracks these histories:

- **Pentanomial batch sampling**: For each batch size `n`, draw `counts ~ Multinomial(n, p)` where `p` comes
    from `PentaModel(opponentElo=...)`.
- **EMA probabilities with per-pair decay**:
    `beta_eff = beta_pg**n`,
    `p_ema <- beta_eff * p_ema + (1 - beta_eff) * p_batch`,
    with debiasing by `1 - prod(beta_eff)`.
- **Asymmetry and mean outcome**:
    - `A = |p_WW - p_LL| + 0.5 * |p_WD - p_DL|`
    - `mu = E[outcome]` with outcomes `[-2, -1, 0, 1, 2]`.
- **Cumulative stats**: Maintains cumulative counts and computes the same `(A_cum, mu_cum)`.
- **Penta coefficient / gain scale**:
    `r = |A_cum| + mu_weight * |mu_cum|`, then map linearly from `(r_small, r_large)` to
    `(min_scale, max_scale)` (intentionally not clamped, so experiments can observe extrapolation).

Plots in [src/fishtest_spsa_lab/analysis/validate_pentanomial.py](src/fishtest_spsa_lab/analysis/validate_pentanomial.py):

- **Pentanomial asymmetry/mean** — two panels:
    - Asymmetry panel: `EMA Asymmetry A` and `Cumulative Asymmetry A` vs `Pairs processed`.
    - Mean panel: `EMA Mean outcome mu` and `Cumulative Mean outcome mu` vs `Pairs processed`.
- **Penta coefficient / gain scale** — one panel:
        - `Cumulative penta r` and `Gain scale factor` vs `Pairs processed`.

#### 3.5.1. validate-penta: Elo noise math (oracle score -> Elo)

`validate-penta` can report the sampling noise of an Elo estimate derived from pentanomial counts.
This answers: “If the true Elo is X, what Elo might I measure after a finite number of games?”

Units:

- `--M` is the number of pairs.
- Total games = `2*M`.

Per Monte Carlo trial, the script samples total counts in this order:

- `[LL, DL, DD, WD, WW]` (pair outcomes).

From counts it computes score_hat (score per game, in [0, 1]) using the same points mapping as the
vendored pentamodel:

- points per pair: `POINTS = [0, 0.5, 1, 1.5, 2]`
- score_hat from counts:
    - `score_hat = (0*LL + 0.5*DL + 1*DD + 1.5*WD + 2*WW) / (2*M)`

Then it converts score_hat to an Elo estimate using the pentamodel “oracle” conversion:

- `opponent_elo = PentaModel.elo_diff_from_score(score_hat)`
- `Elo_hat = -opponent_elo` (sign flip so it matches this script’s `start_diff` convention)

Why the Elo noise can look “surprisingly large”:

- score_hat is an average, so its standard deviation shrinks like `1/sqrt(M)`.
- but the score -> Elo mapping is nonlinear, and its slope depends strongly on the score.
    Near score 0.5 the slope is about 695 Elo per 1.0 score, and it becomes much steeper as score
    approaches 0 or 1.

Delta-method (first-order) standard deviation used for the “Theory Elo_hat std” log:

1) score_hat variance from points-per-pair variance:

- Let `Y` be the random “points per pair” value.
- `score_hat = mean(Y) / 2`
- `Var(score_hat) = Var(Y) / (4*M)`

2) convert score std to Elo std using the local slope:

- `sigma_Elo ~= abs(dElo/dScore at score_true) * sigma_score`

To make the relationship obvious in output, the script also logs this direct conversion using the
empirical `score_hat std` and the slope evaluated at `mean(score_hat)`:

- `score_hat std=...; |dElo/dScore|@mean=... (score=...) => sigma_Elo≈...`

CLI usage (entrypoint `validate-penta`):

- `uv run validate-penta`:
    - Runs a Monte Carlo summary (default `--trials 500`), then plots a single representative run.
- `uv run validate-penta --no-plot`:
    - Summary only (headless-friendly).
- `uv run validate-penta --trials 1`:
    - Single-run plots + totals.

## 4. Toy SPSA demo (`optimize-spsa-toy`)

The analysis module also ships a fast, standalone toy demo intended for intuition-building and
performance experiments (vectorization, multinomial sampling, plotting, logging).

File: [src/fishtest_spsa_lab/analysis/optimize_spsa_toy.py](src/fishtest_spsa_lab/analysis/optimize_spsa_toy.py)

Key semantics:

- **Unbounded objective / optimizer**: the toy objective is a smooth math function on `R^N` and the SPSA optimizer does not clamp parameters.
- **Plot bounds are plot-only**: contour axes use a fixed range (defaults to `[0, 200]`) solely to make 2D plots readable.
- **Diagonal/SPSA c calibration**: `--c-diag-elo-drop` chooses `c_end` so that the *actual SPSA perturbation* `theta ± c_end * delta` at the peak costs that many Elo (per run).
- **Phi-space learning rate**: `--r-end` is the end-of-run learning rate in phi-normalized coordinates (see `docs/ALGORITHMS.md`). The theta step scales with `c_k`.
- **c_end construction** (dev-eigenbasis): the dev predicts curvature in the eigenbasis by choosing a *predicted* spectrum `lambda_dev` (often wrong in the tail), optionally applies i.i.d. lognormal noise to `lambda_dev` (then renormalizes to preserve trace), uses eigen-probes `c_z ∝ 1/sqrt(lambda_dev)`, maps that implied variance back into axis space using `U`, and applies one global rescale so the achieved diagonal/SPSA Elo drop matches `--c-diag-elo-drop` exactly.
    - True spectrum controls: `--spectrum-shape`, `--spectrum-exponent`.
    - Dev mismatch knobs: `--dev-spectrum-exponent`, `--dev-linear-tilt`, `--dev-log-sigma`, `--dev-seed`.
- **Stability constant `A`**: the toy script mirrors common fishtest practice by defaulting to `A = 0.1 * total_pairs` (instead of a fixed constant), and logs the effective schedule.

Output and KPIs:

- The script logs a compact **schedule line**: `alpha`, `gamma`, `A`, and `total_pairs`.
- It logs a **c_end preview** (first 1–2 dims) plus the achieved diagonal Elo drop and implied per-axis drops.
- It logs **curvature participation ratios**:
    - `N_eff_diag`: participation ratio of `diag(G)` (depends on U and the spectrum).
    - `N_eff_phi`: participation ratio of `diag(H_phi)` where `H_phi = C G C / X_SCALE^2`.
- It logs a small set of **KPIs** to compare runs:
    - start/best/final Elo and improvements
    - best-at pairs (time-to-best), improvement per 1k pairs, drawdown, record highs
    - RMS normalized distance to the peak (unitless), runtime, and throughput

Note on “smooth-looking” curves:

- For very large `--num-batches`, the trajectory/value history is automatically downsampled to cap memory.
    This can visually smooth plots because intermediate noisy steps are not stored.
- Useful controls:
    - `--batch-size-pairs`, `--num-batches`, `--seed`.

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

- **[src/fishtest_spsa_lab/analysis/optimize_spsa_toy.py](src/fishtest_spsa_lab/analysis/optimize_spsa_toy.py)**:
    A sandbox demo optimizer that runs an unbounded SPSA loop against a toy N-D objective,
    using the vendored `PentaModel` to generate pentanomial match noise. Plots always use
    parameters 0 and 1.

    Entry point:

    - `uv run optimize-spsa-toy --help`
