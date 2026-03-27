# Schedule-free SPSA with SGD backend

This document derives the schedule-free SGD update path for the lean `sf-sgd`
branch. It is authoritative for tests and audits. SPSA's two-sided estimator
supplies the gradient signal, a constant learning rate `sf_lr` drives the fast
iterate `z` in θ-space, and an optional Polyak average controlled by `beta`
smooths the exported `theta`.

See [Algorithms.md](Algorithms.md) for background on SPSA and the θ ↔ φ
transform. See [SF_Adam_derivation.md](SF_Adam_derivation.md) for the AdamW
variant.

Lab implementation: [src/fishtest_spsa_lab/simulator/optimizer.py](../src/fishtest_spsa_lab/simulator/optimizer.py).

## 1. Requirements (authoritative)

- Constant learning rate `sf_lr`; no decay, no warmup.
- Raw `result = wins - losses`; never divided by `N` or otherwise normalized for the step amplitude.
- A report with `N` pairs produces the same fast-iterate total delta as `N` sequential single-pair arrivals (under the constant per-report signal convention):
  `Δz = sf_lr * c * result * flip`.
- Persist only: `z` (unclamped fast iterate) and `theta` (exported & clamped). The Polyak surrogate `x` is reconstructed per update if `beta > 0`.
- Clamp `x_new` (if used) and always clamp `theta_new`. Never clamp `z`.
- Global counters: `iter += N`; `sf_weight_sum += report_weight`, with `weight = sf_lr` and `report_weight = weight * N`.
- Legacy fallback: if a parameter lacks `"z"`, apply classic SPSA: `theta += R * c * result * flip` then clamp.

## 2. Snapshot (per report arrival)

```
result = wins - losses
N = num_games // 2
if N <= 0: abort

# Advance counters
iter += N
weight = sf_lr
report_weight = weight * N
weight_sum_prev = sf_weight_sum
weight_sum_curr = weight_sum_prev + report_weight
sf_weight_sum = weight_sum_curr
tri_factor = (N + 1) / 2
```

## 3. State structures

Global (`spsa` dict):
```
iter             # cumulative pairs (raw pair count)
sf_lr            # constant learning rate
beta             # blend coefficient (single β in SOTA notation)
sf_weight_sum    # accumulated weighted mass (Σ report_weight), currently lr * total_pairs
```
Per schedule-free parameter:
```
theta  # exported, always clamped
z      # fast iterate, unclamped
min, max, c, ...
```
Legacy parameter (classic):
```
theta, min, max, c, R, ...
```

## 4. Fast iterate aggregation (z-path, θ-space)

```
delta_total_step = sf_lr * c * result * flip   # θ-step (no division by N)
z_new = z_prev + delta_total_step              # z lives in θ-space (unclamped)
```

## 5. Surrogate averaging and blending (x/θ; clamp rules, θ-space)

Space recap (what lives where)
- θ-space: z_t, z_prev, z_new; x_prev, x_new (Polyak surrogate); θ, θ_new; s = delta_total_step / N; tri_factor contribution.
- φ-space (not used in SGD path): n/a here.

Batch-size randomness and gradient scale (why we use result, not result/N)
- Workers return a random number of pairs `N` per report (capacity varies).
- The per-pair gradient proxy in φ is `g_phi_mean = (result / N) * flip`. Over `N` pairs, the sum of identical micro-gradients is `N * g_phi_mean = result * flip`.
- To make the fast iterate `z` (θ-space) invariant to `N`, we update with the total signal `result` (not `result/N`):
  ```
  delta_total_step = sf_lr * c * result * flip   # θ-space
  z_new = z_prev + delta_total_step              # θ-space
  ```

Polyak filtering: what x is (plain words and one formula, θ-space)
- x is the Polyak (running) arithmetic mean of θ-states z over time. Think "keep the arithmetic mean of the z's you visit," but with a constant per-micro-step weight.
- Implementation model (conceptual): every micro-step contributes equally with weight `weight = sf_lr`.
  - Running numerator: `num = Σ (weight * z_t)` over all processed micro-steps t since the run started.
  - Running denominator (mass): `den = Σ weight = sf_lr * (total_pairs_so_far)`.
  - Running average: `x = num / den`.
- In code we don't loop micro-steps; we add the whole report's contribution in closed form, then divide by the new total mass.

Reconstruct Polyak surrogate (θ-space, used only if `beta > 0`)
```
x_prev = (theta_prev - (1 - beta) * z_prev) / beta   # θ-space
x_prev = clamp(x_prev)    # clamp before use
```

Triangular surrogate: closed-form, no loops (why `tri_factor = (N + 1) / 2`, θ-space)
- All quantities below are θ-space within this report.
- Goal: compute the arithmetic mean of the z "right endpoints" you would see inside this report if you expanded it into N unit micro-steps.
- Within one report of `N` pairs, the micro-step size is:
  ```
  s = delta_total_step / N   # θ-space micro-step
  ```
- Right-endpoint model for the fast iterate after `t` micro-steps (`t = 0..N`):
  ```
  z_t = z_prev + t * s       # θ-space
  ```
- Where the sum comes from (explicit breakdown)
  - The N right endpoints are: `z_prev + 1*s, z_prev + 2*s, ..., z_prev + N*s`.
  - Arithmetic mean of those N values:
  ```
  avg_right_end = (1/N) * sum(z_prev + t * s for t in range(1, N+1))
  # separate the constant and the ramp terms
  avg_right_end = (1/N) * (N * z_prev) + (1/N) * s * sum(t for t in range(1, N+1))
  # sum(range(1, N+1)) = N * (N + 1) / 2
  avg_right_end = z_prev + s * (N + 1) / 2
  # replace s = delta_total_step / N
  avg_right_end = z_prev + delta_total_step * ((N + 1) / (2 * N))
  ```
- Tiny sanity example (`N = 3`): right endpoints are `z_prev + s, z_prev + 2*s, z_prev + 3*s`; average is `z_prev + (1+2+3)*s/3 = z_prev + 2*s`. With `s = delta_total_step/3`, this is `z_prev + delta_total_step * (2/3)`.
- Report-mass contribution to the surrogate numerator (`weight = sf_lr`, `report_weight = weight * N`):
  ```
  # Add this report's micro-steps to the running numerator num = Σ weight * z_t
  # Each of the N right endpoints contributes weight * z_t. Sum them in closed form:
  #   Σ (weight * z_t) = weight * Σ (z_prev + t*s)
  #                    = weight * (N * z_prev) + weight * s * Σ t
  # Use s = delta_total_step / N and Σ t = N*(N+1)/2.
  # Here delta_total_step = sf_lr * c * result * flip and report_weight = weight * N,
  # so the triangular correction term carries an explicit sf_lr**2 factor:
  #   weight * delta_total_step * tri_factor = sf_lr**2 * c * result * flip * tri_factor.
  contrib = report_weight * z_prev + weight * delta_total_step * ((N + 1) / 2)
  tri_factor = (N + 1) / 2   # average t over 1..N; exact; midpoint intuition is N/2
  ```

Weighted-mass Polyak surrogate and clamp (θ-space)
```
x_new = (
    weight_sum_prev * x_prev
    + report_weight * z_prev
    + weight * delta_total_step * tri_factor
) / weight_sum_curr
x_new = clamp(x_new)
```

Why this formula matches "x is the running average of z" (θ-space)
- Before this report: `num_prev = weight_sum_prev * x_prev`, `den_prev = weight_sum_prev`.
- This report adds: `num_add = contrib`, `den_add = report_weight`.
- After this report: `num_curr = num_prev + num_add`, `den_curr = den_prev + den_add = weight_sum_curr`.
- Running average is `x_new = num_curr / den_curr`, which is exactly the code above.

Blend export and persist (θ-space)
```
if beta == 0:
    theta_new = z_new
else:
  theta_new = (1 - beta) * z_new + beta * x_new
theta_new = clamp(theta_new)

# Persist z_new (θ-space, unclamped) and theta_new (θ-space, clamped)
```

Sanity checks
- `N = 1` => `tri_factor = (1 + 1) / 2 = 1` (the single right endpoint).
- If `beta == 0` and no clamp: `theta_new - theta_prev == delta_total_step` (exact).
- `z` is never clamped; `x_new` and `theta_new` are clamped to `[min, max]`.

Note on bias: The right-endpoint average induces a mild bias versus an exact micro-state integral; the implementation uses this simple closed form for speed and consistency across random `N`.

## 6. History and telemetry (as implemented)

History is recorded via `_add_to_history` at a sampling cadence derived from the run-level `num_games`:
- Sampling parameters:
  ```
  n_params = len(params)
  samples = 100 if n_params < 100 else 10000 / n_params if n_params < 1000 else 1
  period = run["args"]["num_games"] / 2 / samples   # note: uses run-level num_games
  ```
- A snapshot is appended only when:
  ```
  len(param_history) + 1 > iter / period
  ```
- Stored per-parameter fields in each snapshot entry:
  - "theta": θ-space show value
    - schedule-free: `x_new` (clamped, θ) if `beta > 0`, else `theta_new` (clamped, θ)
    - classic: `theta` (clamped, θ) after update
  - "R": φ-rate used for the classic θ-update at dispatch (maps via `a = R * c**2`)
  - "c": per-axis probe scale at dispatch (θ-units)
- Note: `x` is not persisted in state; it is reconstructed per update when recorded.

## 7. Invariants and edge cases

- `iter` increases by exactly `N`.
- `sf_weight_sum` increases by `sf_lr * N`.
- If `beta == 0` and no clamp: `theta_new - theta_prev == delta_total_step`.
- Bounds: `min ≤ x_new ≤ max` (when used), `min ≤ theta_new ≤ max`; `z_new` is unconstrained.
- Update is aborted if signature mismatch or `N <= 0`.
