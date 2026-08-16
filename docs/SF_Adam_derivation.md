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
- Micro-batch damping: none. A geometric k(N, β2) factor was once proposed here as the analogue of the SGD triangular factor; it is not needed and is not implemented (see "Why there is no micro-batch damping factor").

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
  z_new = z_prev + step_phi * c                 # map φ-step to θ via c
  ```
  There is no damping factor on this step; the exact value is 1 for all N and
  β2 (see "Why there is no micro-batch damping factor").
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

Why there is no micro-batch damping factor
- The concern is real: in Adam the denominator grows across the N identical
  micro-steps of a block because `v` is an EMA. Compressing those N steps into
  one macro update with only the end-of-block denominator `d_end` would apply the
  largest denominator to the whole block, and the mismatch would grow with N.
- Bias correction already removes it. With `v_hat = v / (1 - beta2**t)` and a
  constant per-pair second-moment level `g2`, the in-block recursion
  `v_j = beta2**j * v_0 + (1 - beta2**j) * g2` starting from a `v_0` on the
  bias-corrected trajectory `v_0 = g2 * (1 - beta2**t0)` gives
  ```
  v_j     = g2 * (1 - beta2**(t0 + j))
  v_hat_j = v_j / (1 - beta2**(t0 + j)) = g2      # constant in j
  ```
  so `d_j = d_end` at every micro-step and the exact correction factor is
  `k = 1`, for all N and all β2. The ladder the geometric factor was invented to
  compensate does not exist once bias correction is applied.
- A geometric factor `k(N, beta2) = (1 - beta2**(N/2)) / (N * (1 - sqrt(beta2)))`
  was previously specified here and implemented. It was wrong twice over. The
  re-indexing that produced it dropped a minus sign, so it landed in `(0, 1]`
  where this page's own reasoning calls for `>= 1`; and it was then clipped to
  `(0, 1]` in code, which made the correct direction unreachable. Measured
  under-step against the exact factor of 1: 2.0x at β2 = 0.9, N = 32; **6.6x** at
  β2 = 0.9, N = 128; 1.03x at β2 = 0.999, N = 128.
- What remains is a genuinely different, smaller residual: `g2` is not constant
  across blocks, because the online μ2 estimate moves as reports arrive. That
  breaks the cancellation above by a bounded amount rather than a systematic one.
  Measured z gap 9.43e-03 on a |z| scale of 4.16 at β2 = 0.999, and 1.69e-02 at
  β2 = 0.9. `validate-sf-adam-block` asserts those bounds.

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
- Consistency with the step: the macro step needs no damping (bias correction gives an exact factor of 1), so placing the surrogate at z_new (α = 1) is a simple, end-heavy approximation that aligns with the fact that under smoothing more of the change accrues toward the end of the block.

Optional: exact surrogate if you want it
- If we ever choose to match the micro-step average exactly under the geometric model, replace the report-level surrogate contribution
  ```
  # current (endpoint):
  num_add = report_weight * z_new
  # exact (geometric):
  num_add = report_weight * (z_prev + α(q, N) * Δ)
  ```
  with `q = sqrt(beta2)` and `Δ = z_new - z_prev`.
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
