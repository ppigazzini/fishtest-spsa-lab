# Latent Elo surface and measurement model for SPSA scaling

This document specifies the synthetic modeling assumptions used by the repo’s SPSA diagnostics as the number of parameters $N$ changes.
It keeps the discussion to the knobs that matter for cross-$N$ behavior:

- a quadratic latent objective (isotropic or spectrum+rotation),
- how end-of-run probe magnitudes are interpreted/calibrated,
- how match outcomes are represented as a noisy scalar signal,
- how the end-of-run learning-rate is normalized with $N$ to keep near-peak behavior comparable.

Dimension-dependent claims require stating the modeling choices.

## 0) Executive summary

Goal: reproduce qualitative end-of-run SPSA behavior across $N$ (e.g. $N=1/10/100$) under a fixed family of assumptions.

Cross-$N$ comparisons require stating (i) the objective family (“budget vs $N$”), (ii) what “$c$ corresponds to X Elo drop” means (axis vs diagonal), and (iii) how the end-of-run gain is normalized with $N$.

The canonical definition of the two-point SPSA numerator `deltaY_k` is in section 1.3.

## 1) Definitions and notation

### 1.1 Coordinates

- `theta ∈ R^N`: parameter vector in the coordinate system in which `f_true` is defined.
- `theta_peak`: local maximizer of `f_true` (the “good point”).
- `e := theta - theta_peak`.

### 1.2 Latent surface and match observable

- `f_true(theta)`: latent expected Elo (deterministic).
- `y(theta)`: primitive match observable.
  In the repo diagnostics, `y` is an aggregated **net score** over `n_pairs` pairs.
- `f_hat(theta)`: an Elo estimator derived from `y(theta)`.

Local Elo-estimator linearization (match-model-dependent):

- Let `s_y := d E[y] / d(elo_diff)` evaluated at `elo_diff=0`.
- Define `f_hat(theta)` in Elo units by

```text
f_hat(theta) := y(theta) / s_y
```

Then `Var(f_hat) = Var(y) / s_y^2`. Under standard match models, `Var(y) ∝ n_pairs` and `s_y ∝ n_pairs`, so `Var(f_hat) ∝ 1/n_pairs`.

### 1.3 SPSA primitives

- `delta_k ∈ {−1,+1}^N`: Rademacher direction.
- `c_k`: probe radius.
- `a_k`: gain.
- `r_k := a_k / c_k^2` (so `a_k = r_k * c_k^2`).

Two-point scalar:

```text
deltaY_k := f_hat(theta_k + c_k * delta_k) - f_hat(theta_k - c_k * delta_k)
```

### 1.4 Calibration vocabulary

- **Axis calibration**: interpret “Elo drop target” as the loss from a 1D axis move.
- **Diagonal calibration**: interpret “Elo drop target” as the loss from the SPSA probe `theta ± c*delta`.

## 2) Geometry and scaling knobs

### 2.1 Latent geometry model

Near `theta_peak`, model `f_true` as a quadratic bowl:

```text
f(theta) ~= elo_peak - 0.5 * (theta - theta_peak)^T H (theta - theta_peak)
```

where:

- H is symmetric positive semidefinite (PSD) (local curvature), typically ill-conditioned.
- Coupling between parameters is represented by off-diagonal entries in H.

If H is symmetric, diagonalize it:

```text
H = U^T Lambda U
```

with U orthonormal and Lambda = diag(lambda_1, ..., lambda_N).
In rotated coordinates z = U(theta - theta_peak), the bowl is decoupled per axis:

```text
f(theta) ~= elo_peak - 0.5 * sum_i lambda_i * z_i^2
```

This is the clean way to encode:

- **anisotropy**: a wide eigenvalue spread lambda_i,
- **inactive / weak params**: some lambda_i ~= 0,
- **coupling**: a nontrivial U.

Define per-axis scales (section 2.2) and use them to form normalized coordinates:

```text
d_i := (theta_i - theta_peak_i) / s_i
```

### 2.1.1 Anisotropy and coupling (spectrum + rotation)

To obtain a model that is more structured than a perfectly isotropic bowl while remaining controllable, build H by specifying:

1) An eigenvalue spectrum (lambda_i), e.g.

```text
lambda_i = lambda_1 / i^p    with p > 0
```

The toy implementation also supports a **linear** decreasing spectrum via `--spectrum-shape linear`:

```text
weights_i = linspace(1, 1/K, K)   for i=1..K
lambda_i = weights_i * (trace(G) / sum_j weights_j)
```

So the minimum eigenvalue on the active subspace is strictly positive (the remaining `N-K` eigenvalues are 0 if you set `--spectrum-active K`).

Optionally force a subset to be inactive by setting lambda_i = 0 for i > k_active.

Note: the linear spectrum has a single canonical slope; the only geometry/budget knob is `--spectrum-trace`.

2) A rotation U:

- U = I gives uncoupled parameters.
- Random orthonormal U gives “generic coupling” (energy spread across coordinates).

Then define:

```text
H = U^T * diag(lambda_1, ..., lambda_N) * U
```

For scaling discussions, the two key summary statistics are:

- trace(H) = sum_i lambda_i (total curvature)
- typical diagonal size H_jj (how much curvature you see on coordinate axes)

These summary statistics determine the Elo loss induced by perturbations of size c.

If f(theta) ~= elo_peak - 0.5 * (theta-theta_peak)^T H (theta-theta_peak), then at the peak:

- Axis step on coordinate j (theta +/- c * e_j):

```text
DeltaE_axis_j(c) = 0.5 * c^2 * H_jj
```

- SPSA-style diagonal step (theta +/- c * delta with delta_i in {-1,+1}):

```text
DeltaE_diag_delta(c) = 0.5 * c^2 * (delta^T H delta)
E_delta[delta^T H delta] = trace(H)
so E_delta[DeltaE_diag_delta(c)] = 0.5 * c^2 * trace(H)
```

So:

- `trace(H)` controls the typical Elo loss of the true SPSA perturbation (diagonal calibration).
- `H_jj` (or a “typical” H_jj across j) controls the Elo loss of a 1D slice move (axis calibration).

Important unit note: H depends on how theta is scaled. If you change a parameterization (or just rescale one parameter), then H, c_end,
and the implied Elo gaps change. This is an expected consequence of changing coordinates; calibration must be defined relative to the chosen bounds/ranges.

Choosing an N-family corresponds to choosing how total curvature changes (or does not change) with N, in the selected coordinate system.

---

### 2.1.2 “Budget vs N”: choosing a normalization

A normalization for how the typical magnitude of the objective scales with N is required.

For fishtest-like tuning experiments, a standard modeling choice is:

- adding “fake” / redundant parameters should not increase the available Elo budget under the chosen normalization.

Use a fixed-budget family (equivalently: fixed total curvature in normalized coordinates).
Other normalizations exist. Use exactly one normalization per analysis.

Let B > 0 be an Elo-scale constant (a curvature/budget parameter).

### Fixed-budget model (mean-normalized)

Define a simple isotropic reference bowl in normalized coordinates:

```text
f_M(theta) = elo_peak - B * mean_i(d_i^2)
```

Interpretation: adding parameters spreads a fixed total budget across more coordinates.

Connection to quadratic geometry: in normalized coordinates d (where d_i = (theta_i - theta_peak_i)/s_i), the quadratic model can be written as

```text
f(theta) ~= elo_peak - 0.5 * d^T G d,
with G = D^T H D and D = diag(s_i)
```

In the fixed-budget model above:

```text
f_M(theta) = elo_peak - (B/N) * sum_i d_i^2
          = elo_peak - 0.5 * d^T ((2B/N) * I) d
```

so G = (2B/N) * I and trace(G) = 2B is constant as N changes.

Under the repo’s toy coordinate conventions (`theta_peak_i = 100`, lower-bound origin at `theta_i=0` so `d_i=-1`), the Elo at the origin is implied by the same knob:

```text
Elo(0,...,0) = elo_peak - 0.5 * trace(G)
```

Some scripts historically exposed a “lower-bounds (corner) Elo drop” parameter `drop := -Elo(0,...,0)`; under these conventions this is just `drop = 0.5*trace(G)` (isotropic or not), so `trace(G)` is the single curvature-budget knob.

---

### 2.2 Developer parameterization

This layer maps developer conventions (bounds/units) into the `theta` coordinates used by `f_true`, and it specifies the per-axis `c_end_i` vector.
It is an external input to the optimizer.

#### 2.2.1 Developer bounds convention (signed `V_i`)

Associate each parameter `i` with a developer-provided value `V_i` (signed; sign follows the parameter encoding).
A common developer-side convention sets bounds using the two endpoints `0` and `2*V_i`:

```text
theta_i ∈ [min(0, 2*V_i), max(0, 2*V_i)]
range_i := max(0, 2*V_i) - min(0, 2*V_i) = 2*|V_i|
```

Select a per-axis scale `s_i` from the bounds. Use

```text
s_i := range_i
```

#### 2.2.2 End-of-run probe shape `c_end_i`

A common developer choice sets the end-of-run per-axis probe magnitude proportional to `|V_i|`:

```text
c_end_i := 0.10 * |V_i|
```

Only the magnitude matters: SPSA already injects sign via `delta_i ∈ {−1,+1}`.
If a tool outputs a signed value, absorb the sign into the parameter encoding and treat `c_end_i ≥ 0` without loss of generality.

### 2.3 Evaluation / report model

This section maps the match model to the primitive random variable `y(theta)` and to the Elo estimator `f_hat(theta)`.

#### 2.3.1 Primitive observable and Elo estimator

Use a match-model-dependent observable `y(theta)`.
In the repo diagnostics, `y` is an aggregated net score over `n_pairs` pairs (a sum over pairs).

The mapping from `y` to Elo units (`s_y` and `f_hat`) is defined once in section 1.2.

## 3) What SPSA sees (measurement equation)

At step `k`:

```text
y_plus  := y(theta_k + c_k*delta_k)
y_minus := y(theta_k - c_k*delta_k)
f_hat_plus  := y_plus  / s_y
f_hat_minus := y_minus / s_y
deltaY_k := f_hat_plus - f_hat_minus
```

Decomposition (conceptual; the exact form depends on the match model and on the chosen estimator):

```text
deltaY_k = signal(theta_k, c_k, delta_k) + eps_eval_k
```

Developer noise is not additive. It enters by changing `c_k` and the distribution of probe directions in `theta`-space.

Noise statements must name the object:

- `Var(y)` refers to match outcomes.
- `Var(f_hat)` refers to the Elo estimator.
- `Var(deltaY)` refers to the two-point difference.

---

## 4) Calibration: meaning of “`c_end` corresponds to an Elo drop target”

An “Elo drop target” is meaningless until you define which perturbation it refers to.

In the toy implementation (`src/fishtest_spsa_lab/analysis/optimize_spsa_toy.py`), the
main calibration target is explicitly the diagonal/SPSA meaning (section 4.B), and we refer to it
as `c_diag_elo_drop`.

### 4.A Axis / 1D calibration (slice-calibration)

Definition: choose c_end so that changing exactly one coordinate by +/- c_end at the peak costs a target axis Elo drop.

```text
DeltaE_axis(c_end) = c_axis_elo_drop
```

### 4.B Diagonal / SPSA calibration (true-perturbation calibration)

Definition: choose c_end so that the SPSA perturbation theta +/- c_end * delta at the peak costs a target diagonal Elo drop.

```text
DeltaE_diag(c_end) = c_diag_elo_drop
```

Exactly one meaning must be selected; otherwise cross-N comparisons are ill-defined.

### 4.D Axis vs diagonal Elo loss at the peak

Assume isotropic reference bowls from section 2 and equal scales s_i = s to keep formulas readable.
At theta = theta_peak:

- **Axis perturbation**: change exactly one coordinate by +/- c.
- **Diagonal perturbation**: change all coordinates by +/- c (SPSA-style).

Then the Elo loss magnitudes are:

### Fixed-budget model

```text
DeltaE_axis_M(c) = B * (c/s)^2 / N
DeltaE_diag_M(c) = B * (c/s)^2
```

These identities determine the dimension dependence. Use one calibration meaning per analysis (axis or diagonal).

---

### 4.E Required c_end under each calibration meaning

Set DeltaE(c_end) equal to your chosen Elo drop target (axis or diagonal).

### Axis calibration: DeltaE_axis(c_end) = c_axis_elo_drop

- Fixed-budget model:

```text
c_end = s * sqrt(c_axis_elo_drop * N / B)      (grows like sqrt(N))
```

Implication: the diagonal loss becomes

```text
DeltaE_diag(c_end) = N * c_axis_elo_drop       (grows like N)
```

### Diagonal calibration: DeltaE_diag(c_end) = c_diag_elo_drop

- Fixed-budget model:

```text
c_end = s * sqrt(c_diag_elo_drop / B)          (independent of N)
```

Implication: the axis loss becomes

```text
DeltaE_axis(c_end) = c_diag_elo_drop / N       (shrinks like 1/N)
```

---

## 5) N-scaling invariants (explicit)

Goal: choose the learning-rate normalization so the near-peak behavior is comparable across $N$.

Use the stability proxy from `src/fishtest_spsa_lab/analysis/noise_ball.py`:

```text
eta_eff := a_end * g_factor,
stability proxy: eta_eff * N < 2.
```

In the script parameterization:

```text
r_end := a_end / c_end^2    (so a_end = r_end * c_end^2).
```

Hold geometry and probe calibration fixed (e.g. fix `trace(G)` via `--spectrum-trace` / `--trace-g` and fix `c_end` via `--c-diag-elo-drop`), and then scale `r_end` with $N$ so that `eta_eff * N` stays constant.

Finite-time normalization depends on the run length (number of SPSA batches).
The diagnostic script supports explicit variants:

- `r_end / N` (often flattens the stationary noise-ball KPI in the isotropic toy)
- `r_end / sqrt(N)` (weaker normalization)
- `r_end / N^a` with `a` fitted separately for the stationary / after-1000 / after-10000 KPIs (`--lr-norm auto`)

---


## 6) Schedule parameters and the `a_k = r_k * c_k^2` identity

Many SPSA writeups parameterize the schedule using both a “gain” `a_k` and a probe radius `c_k`.
This document defines `r_k := a_k / c_k^2` once in section 1.3.

This separates roles:

- `c_k`: probe radius (how far you sample the objective)
- `a_k`: overall gain in parameter space
- `r_k`: “learning-rate-like” knob after factoring out the `c_k^2` scaling

In fishtest-style parameterizations, it is convenient to specify an end-of-run target `r_end` and choose the schedule so that:

```text
r_k(total_pairs) = r_end
```

### 6.1 Failure mode: mismatched calibration meaning

If you calibrate c_end using an **axis** meaning ("2 Elo for a single coordinate move") but your SPSA numerator is produced by a **diagonal** probe (theta ± c*delta), then c_end will typically scale incorrectly with N (sections 4.A–4.E).

Two bad things happen when this mismatch forces c_k to become large at high N:

1) The SPSA pseudo-gradient becomes noisy simply because the function differences become huge.
2) For nonlinear objectives, the estimator becomes biased by finite-c effects, and the Monte Carlo mean converges to the biased limit, not to grad f.

This failure mode is frequently described as a “high-dimensional SPSA” issue and then treated with ad-hoc $N$-dependent factors.
The first fix is:
- calibrate c_end to a **diagonal/SPSA** Elo drop target (`c_diag_elo_drop`) if your probe is diagonal,
- and only then decide whether you additionally want an update-size normalization.

If you add an explicit N-dependent normalization (for example using an effective dimension N_eff), treat it as choosing an update-size invariant.
It is not universally correct; it depends on which invariance you want.

The toy implementation includes diagonal (SPSA) Elo-drop calibration for `c_end` and an optional update-size normalization.

## Appendix A) Developer spectrum misprediction (eigenbasis)

---

This repo’s toy demo [src/fishtest_spsa_lab/analysis/optimize_spsa_toy.py](src/fishtest_spsa_lab/analysis/optimize_spsa_toy.py) models a developer who reasons about curvature in the **eigenbasis**.

Let the true objective curvature in normalized coordinates be:

```text
G = U^T diag(lambda_true) U
```

The developer predicts a spectrum `lambda_dev` with the same trace/budget but wrong *shape*.
Two important “realistic wrong” cases:

- **Wrong gamma (power exponent)**: for a true power spectrum, choose `p_dev < p_true`.
  This makes the tail heavier (larger eigenvalues) and therefore overestimates tail curvature.
- **Wrong linear slope**: for a true linear spectrum, choose `--dev-linear-tilt < 1`.
  This flattens the dev's predicted line (heavier tail / larger eigenvalues) and therefore overestimates tail curvature.

Optionally, the dev's predicted eigen-spectrum also includes i.i.d. **lognormal noise** on the active subspace:

```text
z_i ~ Normal(0, 1)
lambda_dev_i := lambda_dev_i * exp(sigma_dev * z_i)
```

After applying this noise, `lambda_dev` is renormalized to preserve the curvature budget:

```text
sum_i lambda_dev_i = trace(G)
```

Given `lambda_dev`, the dev chooses eigen-coordinate probe magnitudes:

```text
c_z_i ∝ 1 / sqrt(lambda_dev_i)
```

This produces an axis-aligned per-parameter `c_end` shape by matching the per-axis variance implied by independent Rademacher perturbations in the eigenbasis.
Let `d = (x - x_peak)/X_SCALE` and `z = U d`. With `z_i := c_z_i * delta_i` and `delta_i ∈ {−1,+1}` i.i.d.:

```text
Var(d_j) = sum_i U_{i,j}^2 * c_z_i^2
c_shape_x_j := X_SCALE * sqrt(Var(d_j))
```

Finally apply one global rescale so the diagonal/SPSA Elo-drop target is hit exactly:

```text
c_end := s * c_shape_x
choose s so DeltaE_diag(c_end) = c_diag_elo_drop
```

CLI knobs:

- True spectrum: `--spectrum-shape`, `--spectrum-exponent`, `--spectrum-active`, `--spectrum-trace`
- Developer prediction: `--dev-spectrum-exponent`, `--dev-linear-tilt`, `--dev-log-sigma`, `--dev-seed`

Plotting note:

- The eigen-spectrum plot is always rendered on a linear $y$ axis.

## Appendix B) Net-score to Elo mapping (implementation note)

Section 1.2 defines the local mapping `f_hat(theta) := y(theta) / s_y`.
Implementation note: `src/fishtest_spsa_lab/analysis/noise_ball.py` estimates `s_y` by a symmetric finite difference of the per-pair mean at `±elo_diff_step`, and uses the pentanomial per-pair variance at `elo_diff=0`.
