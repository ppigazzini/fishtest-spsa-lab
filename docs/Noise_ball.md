# BALL_NOISE: Elo loss from match-outcome noise (noise-ball diagnostics)

This doc derives a compact estimate for how much Elo you lose near the peak due to **evaluation noise** (finite pairs) when you keep running late-stage SPSA.

It is a diagnostic model (near-peak + constant-gain), not a full theory of schedules.

Related code:

- `src/fishtest_spsa_lab/analysis/noise_ball.py` (isotropic constant-gain estimate)
- `src/fishtest_spsa_lab/analysis/optimize_spsa_toy.py` (toy optimizer + spectrum tools)

## 1) Goal (what we compute)

We want the expected Elo drop from the local peak after running “late-stage SPSA”:

- **Infinite horizon noise floor**: `D_inf` (expected drop after many batches).
- **Finite horizon**: `D_T` (expected drop after `T` batches).

We also use these drops to form a practical “upper bound on expected improvement” when starting from an initial drop `D0`.

## 2) Inputs and conventions

### 2.1 Inputs (conceptual)

- `N`: number of parameters.
- `batch_size_pairs`: pairs per SPSA batch.
- `c`: perturbation magnitude (use end-of-run value, `c_end`).
- `r_end`: end-of-run learning rate in the fishtest-style parameterization.
- `k`: local quadratic curvature scalar (defined below).
- `slope_net_score_per_elo`: local slope of expected `net_score` with respect to Elo difference.
- `var_net_score`: variance of `net_score` per batch at zero Elo difference.

Notes (how these are obtained in the lab scripts):

- `slope_net_score_per_elo` and `var_net_score` are evaluated near Elo difference 0 using the match model, then scaled by `batch_size_pairs`.
- This document models evaluation noise only; “developer calibration noise” (uncertainty in choosing `c_end_i`) is out of scope.

### 2.2 SPSA signal convention (toy convention)

We work with a scalar signal derived from match outcomes:

```text
grad_signal := net_score / 2
```

where `net_score` is the noisy net wins aggregated over `batch_size_pairs` pairs.

### 2.3 End-of-run constant-gain proxy

In fishtest-style notation:

```text
r_end := a_end / c^2   =>   a_end = r_end * c^2
```

and we treat `c` and `r_end` as constants representative of late-stage behavior.

## 3) Assumptions (what this model is, and is not)

This estimate is intended for order-of-magnitude guidance under:

- **Near-peak quadratic** objective.
- **Small probe gap** so the match model is locally linear in Elo difference.
- **Constant-gain proxy** (late stage), ignoring the earlier schedule.
- **No clipping/bounds effects**.
- **Isotropic curvature** (collapses curvature to a single scalar `k`).

It does *not* model schedule transients, strong anisotropy, or frequent parameter clipping.

## 4) Derivation (from match noise to Elo drop)

### 4.1 Near-peak objective

Assume an isotropic quadratic near the peak:

```text
Elo(θ) = Elo_peak - k * ||e||^2
e := θ - θ_peak
```

In the isotropic toy parameterization used by the lab scripts (normalized coordinates and an isotropic curvature budget `trace(G)`), the mapping is:

```text
k = 0.5 * (trace(G) / N) / X_SCALE^2
```

SPSA chooses `δ ∈ {±1}^N` and compares `θ ± c δ`.

### 4.2 Elo difference induced by a probe

Define the Elo difference between the “plus” and “minus” probes:

```text
elo_diff := Elo(θ + cδ) - Elo(θ - cδ)
```

For small `c` (dropping O(c^3) terms),

```text
elo_diff ≈ -4 * k * c * <e, δ>
```

### 4.3 Local linearization of the match model

Near `elo_diff = 0`, approximate:

```text
E[net_score] ≈ slope_net_score_per_elo * elo_diff
Var(net_score) = var_net_score    (evaluated near 0)
```

With `grad_signal = net_score/2`,

```text
E[grad_signal] ≈ (slope_net_score_per_elo / 2) * elo_diff
Var(grad_signal) = var_net_score / 4
```

Combining with 4.2 gives the mean signal conditioned on `(e, δ)`:

```text
E[grad_signal | e,δ] ≈ -(2*k*c*slope_net_score_per_elo) * <e,δ>
```

### 4.4 Rank-1 stochastic update and effective gain

Under the constant-gain proxy, the near-peak rank-1 update can be written as:

```text
e_next ≈ e - η_eff * <e,δ> δ + η_eff * ζ * δ
```

where the effective scalar is:

```text
η_eff := (r_end * c^2) * (2*k*slope_net_score_per_elo)
```

and the canonical scalar noise term `ζ` is defined so that it enters as shown above.
A convenient variance approximation is:

```text
Var(ζ) ≈ Var(grad_signal) / ( (c * (2*k*slope_net_score_per_elo))^2 )
    = (var_net_score / 4) / ( (c * (2*k*slope_net_score_per_elo))^2 )
```

### 4.5 Second-moment recursion (noise-ball radius)

Let:

```text
S_t := E[||e_t||^2]
```

In this isotropic rank-1 approximation:

```text
S_{t+1} = a * S_t + b

a = 1 - 2*η_eff + η_eff^2 * N
b = η_eff^2 * N * Var(ζ)
```

If you start at the peak (`S_0 = 0`) and are in the stable regime `|a| < 1`:

```text
S_T = S_* * (1 - a^T)
```

and the stationary mean-square radius is:

```text
S_* = b / (1 - a)
   = (η_eff * N * Var(ζ)) / (2 - η_eff * N)
```

## 5) Final outputs (Elo drop)

In this isotropic toy:

```text
E[Elo_peak - Elo(θ)] = k * E[||e||^2] = k * S
```

Therefore:

```text
D_T   := k * S_T
D_inf := k * S_*
```

### 5.1 Stability checks (minimal)

To interpret the stationary formula you need:

```text
η_eff * N < 2
```

and for the finite-time `S_T = S_*(1 - a^T)` expression you also want:

```text
|a| < 1
```

where `a = 1 - 2*η_eff + η_eff^2 * N`.

## 6) “Upper bound” interpretation from a starting drop

Let `D0` be your current Elo drop from the local peak (in the same near-peak approximation).
Then under the same late-stage hyperparameters the model suggests:

```text
improvement_max(T)   ≈ max(0, D0 - D_T)
improvement_max(inf) ≈ max(0, D0 - D_inf)
```

This is an **expected-value** bound for this simplified system; individual runs fluctuate.

## Appendix A) Extra intuition: “bounce” picture (optional)

Ignore evaluation noise for a moment and consider the isotropic quadratic where the rank-1 update can be written as:

```text
e_next = e - η * <e, δ> δ
```

Let `u := δ / ||δ||_2` so `||δ||_2^2 = N` and `||u||_2=1`.
The component along the step direction evolves as:

```text
<u, e_next> = (1 - η*N) * <u, e>
```

So:

- `0 < η < 1/N` gives no sign flip along `u` (no bounce).
- `1/N < η < 2/N` gives a sign flip (bounce), but shrinking magnitude (damped oscillation).
- `η > 2/N` amplifies the component along `u` (unstable).

Evaluation noise prevents convergence to a point even when the deterministic dynamics are stable; it turns the late-stage behavior into a stationary cloud.
But the same `~2/N` scale is still a useful sanity check: large effective gains tend to produce large late-stage variance.

## Appendix B) Conservative curvature-aware bound (general PSD curvature, optional)

For a general quadratic with PSD curvature `H`, a conservative “don’t blow up the step-direction component” heuristic is:

```text
η * N * λ_max(H) < 2
```

This is intentionally worst-case; in practice the direction-dependent effective curvature matters.

## Appendix C) How to run the code in this repo (optional)

Run the isotropic constant-gain diagnostic:

```bash
uv run python -m fishtest_spsa_lab.analysis.noise_ball --n-max 200 --batch-size-pairs 36 --r-end 0.002
```

Interpretation (directional):

- Holding `r_end` and `batch_size_pairs` fixed, the stationary loss often grows with N in the isotropic toy.
- Increasing `batch_size_pairs` reduces the noise ball (less evaluation noise).
- Reducing `r_end` reduces the noise ball (smaller constant gain).

## Appendix D) Related: anisotropic φ-space proxy (optional)

The toy optimizer prints a φ-space fixed-`r_end` proxy (`E_inf`) that tries to keep the “SPSA normalization” consistent with `c_end`.
That proxy is not identical to the isotropic derivation here, but it is motivated by the same idea:

- treat late-stage behavior as a constant-gain system in φ-space,
- estimate a stationary drop from local slope/variance of the match model.

See `src/fishtest_spsa_lab/analysis/optimize_spsa_toy.py` for the toy spectrum utilities.
