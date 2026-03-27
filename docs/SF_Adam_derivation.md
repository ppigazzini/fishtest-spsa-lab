# Schedule-free SPSA with AdamW backend

This document derives the schedule-free AdamW update path for the `sf-adam`
branch. It replaces the plain SGD step with a schedule-free AdamW step, where
a second moment `v` in φ-space normalizes the direction, optional micro-batch
damping keeps N-invariance, and the same Polyak/export machinery produces the
final `theta`.

See [Algorithms.md](Algorithms.md) for background on SPSA and the θ ↔ φ
transform. See [SF_SGD_derivation.md](SF_SGD_derivation.md) for the SGD
variant.

Lab implementation: [src/fishtest_spsa_lab/simulator/optimizer.py](../src/fishtest_spsa_lab/simulator/optimizer.py).

Space map at a glance:
- φ-space: g_phi_mean (conceptual per-pair gradient), global μ2 estimate → v → v_hat → denom (normalization lives here).
- θ-space: z_prev → z_new, x_prev → x_new, theta; mapping via `θ-step = c * (φ-step)`.
- Batch-size invariance: step with the total `result`; the second moment `v` uses a global μ2 estimator rather than per-parameter `(result/N * flip)**2`.
- Surrogate x: Polyak (arithmetic) mean of z via mass blend `a_k`; no triangular averaging for Adam (see "Why no triangular term here").
- Micro-batch damping k(N, β2): rescales the macro step to keep N-invariance under EMA smoothing; it is the geometric-mean analogue of the SGD triangular factor (see the k(N, β2) subsection and the α(√β2, N) note).

## 1. Requirements (authoritative)

- Constant global learning rate `sf_lr` (no decay, no warmup, no weight decay).
- Raw `result = wins - losses`; never divided by `N` for the step amplitude.
- Persist: `theta` (clamped export), `z` (fast iterate, unclamped), `v` (second moment). The Polyak surrogate `x` is reconstructed per update when `sf_beta1 > 0`.
- Reconstruction (if `sf_beta1 > 0`):
  ```
  x_prev = (theta_prev - (1 - beta1) * z_prev) / beta1
  x_prev = clamp(x_prev)
  ```
- Weighted mass accumulation:
  ```
  weight = sf_lr
  report_weight = weight * N
  sf_weight_sum += report_weight
  a_k = report_weight / sf_weight_sum   # after increment
  ```
- Second moment update for each parameter uses a **shared global μ2 estimate** of the per-pair signal variance, with a closed-form aggregation over `N` micro-steps and bias correction exponent equal to the total processed pairs after increment (φ-space):
  ```
  g2_mean = _mu2_hat(spsa)                      # global E[g^2] estimate, computed before this block
  v = (beta2**N) * v + (1 - beta2**N) * g2_mean # φ-space EMA over N
  v_hat = v / (1 - beta2**micro_steps)         # φ-space bias correction, micro_steps = iter
  denom = sqrt(v_hat) + sf_eps                 # φ-space normalization
  ```

  Why this equals N looped EMA updates (derivation)
  - Start from Adam's second-moment EMA per micro-step `t` (φ-space), with constant input `g2_mean` inside the report:
    ```
    v_{t+1} = beta2 * v_t + (1 - beta2) * g2_mean
    ```
  - Unroll N identical micro-steps (no change in `g2_mean` within the report):
    ```
    v_N = beta2^N * v_0 + (1 - beta2^N) * g2_mean
    ```
    That is exactly the closed form used above.

  - Bias correction exponent:
    Adam's bias correction uses the total number of micro-steps processed so far, call it `t`. In our setting, one pair = one micro-step, so after consuming `N` new pairs we have:
    ```
    micro_steps = t_after = (previous_total_pairs) + N = iter
    v_hat = v / (1 - beta2**micro_steps)
    ```
    This matches standard Adam, where the correction exponent is the current step count after the update.

  - Denominator for normalization (φ-space):
    ```
    denom = sqrt(v_hat) + sf_eps
    ```
    This is exactly the RMS term used to normalize the φ-step.

  Practical consequence
  - The closed form replaces an explicit N-step loop without changing the result, because within a report the μ2-based proxy `g2_mean` is treated as constant. This keeps the implementation fast and numerically consistent with sequential micro-updates under that model.

- Directional fast iterate step (φ → θ mapping; no triangular surrogate):
  ```
  step_phi = (sf_lr * result * flip) / denom   # φ-space step (batch-size invariant numerator)
  # Optional micro-batch damping (enabled in code when N>1 and 0<beta2<1):
  # k(N, beta2) = (1 - beta2**(N/2)) / (N * (1 - sqrt(beta2)))  in (0, 1]
  # Near beta2 -> 1: k ≈ 1 - ((N - 1)/4) * (1 - beta2)
  step_phi *= k(N, beta2)   # if applicable; clipped to (0, 1] in code
  z_new = z_prev + step_phi * c                 # map φ-step to θ via c
  ```
  The factor `k(N, beta2)` is bounded to `(0, 1]` in code for safety; if conditions don't hold, `k = 1`.
- Polyak surrogate averaging and blend (if `beta1 > 0`):
  ```
  x_new = (1 - a_k) * x_prev + a_k * z_new
  x_new = clamp(x_new)
  theta_new = clamp((1 - beta1) * z_new + beta1 * x_new)
  ```
  else:
  ```
  theta_new = clamp(z_new)
  ```
- Never clamp `z`. Legacy fallback: parameters lacking `"z"` are updated via classic SPSA.

## 2. Snapshot (per report arrival)

```
result = wins - losses
N = num_games // 2
if N <= 0: abort

iter += N

weight = sf_lr
report_weight = weight * N
weight_sum_prev = sf_weight_sum
weight_sum_curr = weight_sum_prev + report_weight
sf_weight_sum = weight_sum_curr
a_k = report_weight / weight_sum_curr

# μ2 estimate PRE-block (from previous reports only)
g2_mean = _mu2_hat(spsa)   # global E[g^2] estimate used for v
micro_steps = iter         # bias correction exponent for v
```

## 3. State structures

Global (`spsa` dict):
```
iter, sf_lr, sf_beta1, sf_beta2, sf_eps, sf_weight_sum
```
Per schedule-free Adam parameter:
```
theta (clamped), z (unclamped), v (second moment), min, max, c, ...
```

## 4. Batch size, second moment, step, Polyak filtering, and N-damping

Batch-size randomness and gradient scale (result vs result/N)
- Workers return a random number of pairs `N` per report.
- Per-pair φ-gradient proxy: `g_phi_mean = (result / N) * flip`. Over `N` pairs, the total signal is `result * flip`.
- Keep the step amplitude invariant to `N` by using the total `result` in the numerator of the step (φ-step, then map to θ via c):
  ```
  step_phi = (sf_lr * result * flip) / denom
  z_new = z_prev + step_phi * c
  ```
- Use `g_phi_mean` only for the second moment `v` (per-pair modeling), not for the step amplitude.

Closed-form second moment and denominator (φ-space, no loops)
```
v = (beta2**N) * v + (1 - beta2**N) * g2_mean
v_hat = v / (1 - beta2**micro_steps)   # micro_steps = total pairs after this report
denom = sqrt(v_hat) + sf_eps
```
- This aggregates the N identical micro-gradients in one shot and applies bias correction.

Optional micro-batch damping k(N, β2): what it fixes and where it comes from
- Why we need it: in Adam, the denominator (RMS) grows during the N identical micro-steps because `v` is an EMA. If we compress those N micro-steps into one macro update and use only the end-of-block denominator `denom_end`, we apply the largest denominator to the whole block. Earlier micro-steps would have used smaller denominators, so the true sequential sum is larger than the one-shot macro step. As N grows, the mismatch grows; the macro step shrinks with N.
- Back-of-the-envelope model that matches practice:
  - With constant per-pair magnitude `|g_phi_mean|` inside the block, denominators across micro-steps scale roughly geometrically by `sqrt(beta2)`.
  - Let `d_j` be the denominator at micro-step j (1..N) and `d_end` the denominator at the end of the block. Approximate:
    - `d_j ≈ d_end * beta2**((N - j)/2)`   # earlier steps see smaller denom
  - Sequential micro-steps sum:
    - `S_seq ≈ Σ_{j=1..N} (sf_lr * g_phi_mean) / d_j`
    - `= (sf_lr * g_phi_mean / d_end) * Σ_{j=1..N} beta2**((j - N)/2)`
    - `= (sf_lr * g_phi_mean / d_end) * Σ_{i=0..N-1} beta2**(i/2)`
  - Compressed macro step uses numerator `sf_lr * (N * g_phi_mean)` and denominator `d_end`:
    - `S_macro = (sf_lr * N * g_phi_mean) / d_end`
  - Match the two by multiplying the macro step with the average geometric factor:
    - `k(N, beta2) = (1/N) * Σ_{i=0..N-1} beta2**(i/2) = (1 - beta2**(N/2)) / (N * (1 - sqrt(beta2)))`
- How it plugs into the step:
  ```
  step_phi = ((sf_lr * result * flip) / denom) * k(N, beta2)
  z_new = z_prev + step_phi * c
  ```
- Guards and numerics:
  - Apply only if `N > 1` and `0 < beta2 < 1`; otherwise use `k = 1` (no damping).
  - Clip to `(0, 1]` in code for safety (geometric mean ≤ 1).
  - Near `beta2 -> 1`, use the numerically stable series:
    - `k(N, beta2) ≈ 1 - ((N - 1)/4) * (1 - beta2)`
- Sanity checks and intuition:
  - `N = 1` => `k = 1` (no change); `beta2 = 0` => `k = 1` (no smoothing); `beta2 -> 1` => `k -> 1` with a small linear correction.
  - Example: `beta2 = 0.99`, `N = 16` => `k ≈ (1 - 0.99**8) / (16 * (1 - 0.995)) ≈ 0.97` (mild reduction).
  - Takeaway: `k` compensates for the fact that "one big step with the final denom" underestimates the sum of N smaller steps that would have used a ladder of smaller denoms along the way.

Polyak filtering: x is the running arithmetic mean of z
- Think "keep the arithmetic mean of the z's you visit," with constant per-micro-step weight `weight = sf_lr`.
- Running numerator/denominator across the whole run:
  - `num = Σ (weight * z_t)`, `den = Σ weight = sf_weight_sum`.
  - Running average: `x = num / den`.
- Report-level closed form (no loops): in Adam we approximate the micro-step average by the endpoint `z_new` (no triangular term):
  - Numerator addition: `num_add = report_weight * z_new`.
  - Denominator addition: `den_add = report_weight`.
- Therefore the updated surrogate is a mass-weighted blend with `a_k = report_weight / sf_weight_sum`:
  ```
  x_new = (1 - a_k) * x_prev + a_k * z_new
  x_new = clamp(x_new)
  ```

Why no triangular term here (contrast with SGD)

What the exact surrogate would be under Adam's smoothing
- In SGD, micro-steps inside a report are equal, so the average of the N right endpoints is exactly the triangular factor `(N+1)/(2N)` times the total delta, giving the `tri_factor = (N+1)/2` in the numerator.
- In Adam, the denominator grows across the N micro-steps because `v` is an EMA, so the per-micro-step sizes shrink over the block. If we model the denominator growth as geometric with ratio
  ```
  q = sqrt(beta2)  in (0, 1]
  ```
  then the micro-step sizes are approximately a geometric sequence:
  ```
  s_j ∝ q^{j - N}     # j = 1..N, later steps are smaller only if q>1; with q<1 they are larger denominators and smaller steps earlier, larger later; the net effect is "end-heavy" change
  ```
- The exact arithmetic mean of the N right endpoints (the Polyak surrogate over micro-steps) can be written without loops as
  ```
  z_avg = z_prev + α(q, N) * Δ
  ```
  where `Δ = Σ_{j=1..N} s_j` is the total fast-iterate delta in this report, and the "Adam triangular" factor is
  ```
  α(q, N) = [1 - (N+1) q^N + N q^{N+1}] / [N (1 - q) (1 - q^N)]    # closed form
  ```
  Derivation sketch:
  - Average of right endpoints: (1/N) Σ_{t=1..N} z_t with z_t = z_prev + Σ_{j=1..t} s_j
  - Swap sums: (1/N) Σ_{j=1..N} (N - j + 1) s_j
  - With geometric steps s_j ∝ q^{j - N}, use the standard sums
    - Σ q^j = q (1 - q^N) / (1 - q)
    - Σ (N - j + 1) q^j = q [1 - (N+1) q^N + N q^{N+1}] / (1 - q)^2
  - Normalize by Δ = Σ s_j to get the α(q, N) above.

Key limits and intuition
- q = 1 (no smoothing change within the block) => α(1, N) = (N+1)/(2N)  (the SGD triangular average).
- 0 < q < 1 (Adam's usual case) => α(q, N) strictly increases toward 1 as q decreases, i.e., the average lies closer to the endpoint z_new than the triangular midpoint because more of the change happens later in the block.
- As β2 → 1 (q → 1), α(q, N) → (N+1)/(2N) and the difference from the triangular average is O(1 − q).
  A short series expansion around q = 1 gives:
  ```
  α(q, N) ≈ (N+1)/(2N) + ((N-1)/12) * (1 - q) + O((1 - q)^2)
  ```

Why we approximate with z_new (and not α(q, N))
- Accuracy vs complexity: The exact α(q, N) depends on an effective ratio q for the denominators across the block. In practice the denominator also includes bias correction and sf_eps, and g varies slightly -- so q is only approximate. Using α(q, N) adds complexity for a second-order correction.
- Magnitude of the effect: The surrogate blend uses a_k = report_weight / sf_weight_sum, which decays over the run. The difference between z_avg and z_new impacts x_new by a factor a_k * (1 − α(q, N)) * |Δ|, typically small once sf_weight_sum grows.
- Consistency with step damping: We already restore N-invariance of the macro step via k(N, β2) on the numerator. Given that, placing the surrogate at z_new (α = 1) is a simple, end-heavy approximation that aligns with the fact that under smoothing more of the change accrues toward the end of the block.

Optional: exact surrogate if you want it
- If we ever choose to match the micro-step average exactly under the geometric model, replace the report-level surrogate contribution
  ```
  # current (endpoint):
  num_add = report_weight * z_new
  # exact (geometric):
  num_add = report_weight * (z_prev + α(q, N) * Δ)
  ```
  with `q = sqrt(beta2)` and `Δ = z_new - z_prev` (after applying k(N, β2) to the step).
- We've kept the endpoint form to stay simple, fast, and robust; the empirical difference is negligible in our settings (β2 close to 1, moderate N).

## 5. History and telemetry (as implemented)

History behavior is identical to the SGD path (same cadence and stored fields):
- Sampling cadence uses the run-level `num_games` and the same `samples` heuristic and `period`.
- Stored per-parameter fields per snapshot:
  - `"theta"` = show value (`x_new` if `beta1>0`, else `theta_new`)
  - `"R"`, `"c"` as provided in `w_params` for the update
- `x` is reconstructed transiently (not persisted).

## 6. Invariants and edge cases

- `iter` increases by exactly `N`; `sf_weight_sum` increases by `sf_lr * N`.
- If `beta1 == 0` and no clamp: `theta_new - z_prev == step_phi * c`.
- Bounds: `min ≤ theta_new ≤ max`; if `beta1>0`, `min ≤ x_new ≤ max`; `z_new` is unconstrained.
- Update is aborted if signature mismatch or `N <= 0`.
