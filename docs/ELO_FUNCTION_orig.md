# Correct “engine Elo function” for SPSA scaling

This document is about writing a **correct synthetic model** of an engine for reasoning about SPSA tuning as the number of parameters N changes.

If you don’t state your modeling choices explicitly, you will mix incompatible assumptions and get nonsense scaling rules.

The two most important choices are:

1. **Objective normalization ("Elo budget vs N")**: how do you define a comparable family as N changes?
2. **Calibration meaning ("c_end corresponds to c_diag_elo_drop")**: is the Elo drop target an **axis** (1D) loss or a **diagonal/SPSA** loss?

Everything else (SNR behavior, whether `1/sqrt(N)` helps, etc.) depends on those plus the evaluation noise model.

---

## 0) What an “engine Elo function” actually is

In practice you have two layers:

1. **Latent (deterministic) Elo surface** f(theta): the “true” expected Elo of the engine with parameters theta.
2. **Measured Elo** f_hat(theta): an estimator from a finite number of games. This is noisy.

SPSA never sees f(theta) directly; it sees *differences of noisy estimates*.

So a correct simulator/model must specify both:

- a deterministic f(theta) (geometry), and
- a noise model for f_hat(theta) (evaluation).

---

## 1) Geometry: a correct local model near a good point

Near a local optimum theta_peak, a generic and useful approximation is a quadratic bowl:

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

### 1.1 Parameter scale (practical, bounds-based)

Real engines have parameters with wildly different magnitudes, and devs usually do not know a principled per-parameter scale.
That is OK.

What matters for this document is only that you pick a consistent coordinate system and remember that the curvature matrix H and all
calibration constants are defined in that coordinate system.

In practice, the only reliable per-parameter scale you typically have is the tuning range (bounds).
Let

```text
range_i := max_i - min_i
```

and choose a per-parameter scale s_i from it (any consistent convention is fine; just state it), e.g.

```text
s_i := range_i      (or s_i := 0.5 * range_i)
```

Then define the normalized coordinate for modeling convenience:

```text
d_i := (theta_i - theta_peak_i) / s_i
```

This is not claiming the engine is “dimensionless”; it is just a bookkeeping device so that “a 5% move” means the same thing across
heterogeneous parameters.

### 1.2 A more realistic way to choose H (spectrum + rotation)

If you want a model that is more engine-like than “perfectly isotropic” but still controllable, build H by specifying:

1) An eigenvalue spectrum (lambda_i), e.g.

```text
lambda_i = lambda_1 / i^p    with p > 0
```

Optionally force a subset to be inactive by setting lambda_i = 0 for i > k_active.

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

These two numbers show up immediately when you translate the quadratic bowl into “how many Elo do I lose if I perturb by c?”

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
and the implied Elo gaps change. That is not a bug; it is why practical calibration must be defined in terms of your chosen bounds/ranges.

Intuitively: choosing an N-family is choosing how the total curvature should change (or not) with N in your simplified family, in the coordinate system you decided to use.

---

## 2) “Budget vs N”: choosing a normalization

You must choose how the typical magnitude of the objective scales with N.

For fishtest-like tuning experiments, the robust default is:

- adding “fake” / redundant parameters should not magically increase the available Elo budget.

This document focuses on a fixed-budget family (equivalently: fixed total curvature in normalized coordinates).
Other normalizations exist, but mixing them into the same scaling discussion without being explicit is the main way people end up with contradictions.

Let B > 0 be an Elo-scale constant (a curvature/budget parameter).

### Fixed-budget model (mean-normalized)

Define a simple isotropic reference bowl in normalized coordinates:

```text
f_M(theta) = elo_peak - B * mean_i(d_i^2)
```

Interpretation: when you add parameters, you are spreading a fixed total “budget” across more coordinates.

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

---

## 3) Evaluation model: what SPSA actually measures

Two-point SPSA perturbs along a random Rademacher direction delta in {-1, +1}^N:

```text
y_plus  = f_hat(theta + c_k * delta)
y_minus = f_hat(theta - c_k * delta)
deltaY  = y_plus - y_minus
```

For small c_k and smooth f:

```text
deltaY ~= 2 * c_k * <grad f(theta), delta> + noise
```

The symmetric-difference factor matters:

- keep the `/2` in the gradient signal (don’t silently absorb it into the learning rate)

The noise term depends on batch size (number of game pairs) and the match-outcome model.
All that matters for scaling discussions is:

- variance shrinks like Var(f_hat) ~ 1/n with n pairs,
- so the SNR of `deltaY` depends on both c_k and batch size.

In many practical setups y_plus and y_minus are evaluated with paired randomness (same openings/seed scheme). Then the two noises are correlated and
the variance of deltaY is reduced:

```text
Var(deltaY) = Var(eps_plus - eps_minus) = Var(eps_plus) + Var(eps_minus) - 2 * Cov(eps_plus, eps_minus)
```

This is one reason “pick c_end for 1 Elo” is not a reliable rule: the detectable gap depends on how you run the experiment.

### 3.1 Separate “evaluation noise” from “developer calibration noise”

This document mostly focuses on the evaluation noise of the measured Elo estimator f_hat(theta) (finite games, pentanomial sampling, pairing, etc.).

However, in fishtest-style tuning there is a second, conceptually different noise source:

- **Developer calibration noise**: uncertainty / heuristic error in how the developer chooses per-axis perturbation scales (ranges), i.e. in `c_end_i`.

This does not change the *true* latent surface f(theta). It changes how SPSA probes that surface (and thus changes step sizes, stability, and the
effective dimension seen in φ-space).

In this repo’s toy script (`src/fishtest_spsa_lab/analysis/optimize_spsa_toy.py`), the developer calibration noise is modeled in two
distinct modes.

#### Noise mode A: curvature-estimation noise (“whiten”)

Goal: model a developer who tries to whiten per-axis curvature but only has a noisy diagonal estimate.

Let g_i be the true axis curvature proxy (the diagonal of G in the toy’s normalized coordinate system). The dev observes a noisy estimate:

```text
g_hat_i = g_i * exp(e_i),   where e_i ~ Normal(0, sigma_h^2)
```

and chooses a whitened scale

```text
c_end_i ∝ 1 / sqrt(g_hat_i)
```

with a global rescale so the *expected* diagonal/SPSA Elo drop matches `c_diag_elo_drop`.

CLI knobs:

- `--c-end-model whiten`
- `--c-hessian-log-sigma` (sigma_h)
- `--c-hessian-seed`

Interpretation: larger `--c-hessian-log-sigma` increases mis-whitening, which can reduce the effective dimension in φ-space even when the true spectrum is
high-dimensional.

#### Noise mode B: units/range heuristic noise (“units”)

Goal: model the common real workflow where per-parameter scales are chosen from “units/ranges” heuristics rather than curvature.

In real fishtest, the developer ultimately sets a per-axis perturbation vector `c_end_i` (in theta units). This mode models how that vector is often
constructed:

- start from a “range / unit scale” heuristic (per parameter),
- optionally apply per-axis tuning error,
- then apply one global multiplier so the overall perturbation is “about right”.

We represent the *shape* of the developer’s heuristic by a positive per-axis latent scale V_i.
V_i is not an extra thing the dev sets in addition to `c_end_i`; it is the simulator’s way to generate a plausible relative pattern across parameters
before the final global rescale.

Base (dimensionless) scale heterogeneity:

```text
V_i = exp(sigma_v * z_i),   where z_i ~ Normal(0, 1)
```

Optional per-axis “tuning error” on the chosen axis scales:

```text
E_i = exp(sigma_axis * z'_i),   where z'_i ~ Normal(0, 1)
```

Define the raw (pre-rescale) developer vector shape:

```text
c_raw_i := V_i * E_i
```

Then the actual perturbation used by the run is a single global rescale of that shape:

```text
c_end_i := s * c_raw_i
```

where the scalar s is chosen to hit the diagonal/SPSA Elo-drop target exactly.
For the toy quadratic model (with a single global coordinate scale `X_SCALE`, i.e. the role of `s_i` from section 1.1 is taken to be uniform):

```text
f(theta) = elo_peak - 0.5 * d^T G d,   d = (theta - theta_peak) / X_SCALE
```

the true expected diagonal/SPSA Elo drop at the peak for a vector c_end is:

```text
DeltaE_diag(c_end) = 0.5 * (1/X_SCALE^2) * sum_i g_i * c_end_i^2
```

with g_i = diag(G)_i.
So we pick s by solving:

```text
0.5 * (1/X_SCALE^2) * sum_i g_i * (s^2 * c_raw_i^2) = c_diag_elo_drop
=> s = X_SCALE * sqrt( (2*c_diag_elo_drop) / sum_i(g_i * c_raw_i^2) )
```

CLI knobs:

- `--c-end-model units`
- `--c-units-v-log-sigma` (sigma_v) and `--c-units-v-seed`
- `--c-axis-log-sigma` (sigma_axis) and `--c-axis-seed`

Key property: because the final rescale s is a single scalar, it does not change the relative shape across axes.
So any “effective dimension collapse” you see in φ-space is driven by the heterogeneity of c_raw_i (from V_i and E_i), not by the final rescale.

Practical diagnostic: the toy logs an implied `N_eff_phi` from V (pre-rescale), which is essentially the participation ratio of weights proportional to
`g_i * c_raw_i^2` over active dimensions.

### 3.2 Finite-c bias: SPSA converges to a biased limit if c is large

It is easy to get confused by plots of “Monte Carlo mean SPSA gradient” vs trial count.

- As trials → infinity, the Monte Carlo mean converges to **E[ghat]** (the expectation of the estimator).
- If your perturbation size c is not small, **E[ghat] is not equal to the true gradient**; it equals a finite-difference / smoothed quantity.

For smooth f, a standard Taylor expansion shows:

```text
E[ghat(theta)] = grad f(theta) + O(c^2)
```

The crucial point is that the bias scales like c^2 (under smoothness assumptions), so if c is large you can get a very consistent, very wrong estimate.

Concrete example (why cubic looks “wrong” in the sandbox):

- Let f(x) = sum_i x_i^3.
- SPSA uses delta in {-1,+1}^N and

```text
ghat_j := ((f(x + c*delta) - f(x - c*delta)) / (2*c)) * delta_j
```

For this specific f and Rademacher delta, you can compute the expectation exactly:

```text
E[ghat_j] = 3*x_j^2 + c^2
```

So as you increase the number of trials, the running mean converges to “true gradient + c^2”, not to the true gradient.

Practical consequence:

- If you want the SPSA pseudo-gradient to behave like the true gradient, you must keep c small.
- In optimization, this is one reason many SPSA schedules use c_k that is small (and often slowly decreasing): it controls finite-difference bias.

---

## 4) Calibration: two meanings of “c_end corresponds to an Elo drop target”

An “Elo drop target” is meaningless until you define which perturbation it refers to.

In this repo’s toy code (see `src/fishtest_spsa_lab/analysis/optimize_spsa_toy.py`), the
main calibration target is explicitly the diagonal/SPSA meaning (section 4.B), and we refer to it
as `c_diag_elo_drop`.

### 4.A Axis / 1D calibration (slice-calibration)

Definition: choose c_end so that changing exactly one coordinate by +/- c_end at the peak costs a target axis Elo drop.

```text
DeltaE_axis(c_end) = c_axis_elo_drop
```

### 4.B Diagonal / SPSA calibration (true-perturbation calibration)

Definition: choose c_end so that the actual SPSA perturbation theta +/- c_end * delta at the peak costs a target diagonal Elo drop.

```text
DeltaE_diag(c_end) = c_diag_elo_drop
```

You **must** pick one, otherwise comparing runs across N is meaningless.

### 4.C Practical calibration when you do not know a good Elo drop target

In real fishtest-like workflows, devs typically cannot set “1 Elo gap” directly. The robust practical knobs are:

1) per-parameter bounds (hence per-parameter range_i), and
2) a global multiplier that makes SPSA probes measurable but not huge.

Start with a bounds-based per-axis perturbation rule:

```text
c_end_i(base) := frac * range_i     (typical frac ~ 0.01 to 0.10)
```

Then introduce a single global scale kappa:

```text
c_end_i := kappa * c_end_i(base)
```

Now choose kappa from a short pilot that measures finite-difference SNR directly:

- Pick a representative starting point theta0 and the same evaluation protocol you will use (batch size, openings/seed pairing, etc.).
- For a few random SPSA directions delta, evaluate y_plus/y_minus and record deltaY.
- Estimate a typical signal scale (e.g. median(|deltaY|)) and a noise scale (e.g. std(deltaY)).
- Increase/decrease kappa until you reach a target like:

```text
median(|deltaY|) ~= (2 to 4) * std(deltaY)
```

This replaces the vague “set c_end to 1 Elo” guidance with a measurable criterion that automatically accounts for your actual noise and pairing.

Recommendation: if you care about SPSA behavior (not just per-axis intuition), tune kappa using diagonal/SPSA probes (random delta), because that is the perturbation the algorithm actually uses.

---

## 5) Correct math: axis vs diagonal Elo loss at the peak

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

These are the core identities. Everything that feels “paradoxical” later comes from forgetting which one you calibrated.

---

## 6) Correct math: what c_end must do under each calibration

Set DeltaE(c_end) equal to your chosen Elo drop target (axis or diagonal).

### Axis calibration: DeltaE_axis(c_end) = c_axis_elo_drop

- Fixed-budget model:

```text
c_end = s * sqrt(c_axis_elo_drop * N / B)      (grows like sqrt(N))
```

Implication (the “gotcha”): the diagonal loss becomes

```text
DeltaE_diag(c_end) = N * c_axis_elo_drop       (grows like N)
```

### Diagonal calibration: DeltaE_diag(c_end) = c_diag_elo_drop

- Fixed-budget model:

```text
c_end = s * sqrt(c_diag_elo_drop / B)          (independent of N)
```

Implication (the opposite gotcha): the axis loss becomes

```text
DeltaE_axis(c_end) = c_diag_elo_drop / N       (shrinks like 1/N)
```

---

## 7) What to keep “invariant” is not an engine property

When N changes you must define what “comparable” means.
This is **not** something the engine gives you; it is a modeling constraint you impose by choosing:

- an objective normalization (how curvature/budget scales with N),
- a calibration meaning (axis vs diagonal),
- and possibly additional scaling (batch size, `a_k`, explicit `1/sqrt(N)` normalization).

Common targets are:

1) **Parameter-space update size** (e.g. RMS-per-parameter update).
2) **Signal-to-noise (SNR)**: keep `|deltaY|` above match noise at your batch size.
3) **Elo move per perturbation**: keep the perturbation Elo loss you care about stable.

You generally cannot get all three “for free” as N changes.

---

## 8) Finite-difference signal vs SNR (clean statement)

For small c_k, the finite-difference SIGNAL amplitude obeys:

```text
|deltaY| is typically on the order of c_k * ||grad f(theta)||_2
```

At a comparable normalized distance from the peak (hold an RMS distance in d-space fixed), the gradient norm scales like:

- Fixed-budget model: `||grad f||_2 ~ 1/sqrt(N)` (up to constants and coordinate scales)

Combine that with how your calibration forces c_k to scale (section 6):


- If you keep c_k proportional to c_end:
  - Axis calibration: `c_k ~ sqrt(N)` ⇒ `|deltaY| ~ O(1)` (signal amplitude stable)
  - Diagonal calibration: `c_k ~ O(1)` ⇒ `|deltaY| ~ O(1/sqrt(N))` (signal amplitude shrinks)

To talk about true SNR you must also specify the noise of the measurement difference:

```text
SNR_deltaY ~= typical(|deltaY|) / std(deltaY_noise)
```

If `std(deltaY_noise)` is roughly constant as N changes, SNR scales like the signal amplitude above.
If instead you increase batch size (more pairs) and/or use strong plus/minus pairing that reduces Var(deltaY),
then SNR can remain acceptable even when the signal amplitude shrinks like 1/sqrt(N).

See the paired-noise identity in section 3 (Var(deltaY) = Var(eps_plus - eps_minus) = Var(eps_plus) + Var(eps_minus) - 2*Cov(eps_plus, eps_minus)) for why pairing can materially reduce std(deltaY_noise).

This is the *correct* way to talk about scaling: separate signal amplitude from noise, and state both the budget model and the calibration meaning.

---

## 9) Update-size scaling and the `a_k = r_k * c_k^2` identity

Many SPSA writeups parameterize the schedule using both a “gain” `a_k` and a probe radius `c_k`, and define:

```text
r_k := a_k / c_k^2    (so a_k = r_k * c_k^2)
```

This is useful because it cleanly separates roles:

- `c_k`: probe radius (how far you sample the objective)
- `a_k`: overall gain in parameter space
- `r_k`: “learning-rate-like” knob after factoring out the `c_k^2` scaling

In fishtest-style parameterizations, it is convenient to specify an end-of-run target `r_end` and choose the schedule so that:

```text
r_k(total_pairs) = r_end
```

### 9.1 Why sqrt(N) shows up (and RMS update math)

This document is about *choosing correct normalizations* (budget vs N, axis vs diagonal calibration, and what is meant by `r_end`).
The detailed “why `sqrt(N)` shows up” Rademacher geometry and the “noise ball from evaluation noise” derivations are useful, but they make this file too long.

Those derivations now live in dedicated docs:

- See `docs/Rademacher.md` for the Rademacher identities (`||δ||=sqrt(N)`, `cos ~ 1/sqrt(N)`) and why `1/sqrt(N)` is the natural update-size normalization.
- See `docs/BALL_NOISE.md` for the constant-gain near-peak “noise ball” model (stationary and finite-time), plus how it connects to `analysis/noise_ball.py`.

Practical summary (still worth keeping in mind here):

- “`1/sqrt(N)`” is a choice of **update-size invariant**; it is not automatically an Elo-invariant.
- Evaluation noise creates a late-stage stationary cloud around the peak whose radius depends on batch size and on the end-of-run gain.

### 9.2 Common pitfall: blaming “diagonal steps” instead of fixing calibration

If you calibrate c_end using an **axis** meaning ("2 Elo for a single coordinate move") but your SPSA numerator is produced by a **diagonal** probe (theta ± c*delta), then c_end will typically scale incorrectly with N (sections 4–6).

Two bad things happen when this mismatch forces c_k to become large at high N:

1) The SPSA pseudo-gradient becomes noisy simply because the function differences become huge.
2) For nonlinear objectives, the estimator becomes biased by finite-c effects (section 3.2), and the Monte Carlo mean converges to the biased limit, not to grad f.

This is a frequent source of confusion: people observe “SPSA is wrong in high dimension” and try to patch it with an extra 1/sqrt(N) factor.
Often the first fix should be simpler:
- calibrate c_end to a **diagonal/SPSA** Elo drop target (`c_diag_elo_drop`) if your probe is diagonal,
- and only then decide whether you additionally want an update-size normalization.

If you add an explicit 1/sqrt(N) (or 1/sqrt(N_eff)) normalization, treat it as choosing an update-size invariant.
It is not universally correct; it depends on which invariance you want.

This repo’s toy code includes both diagonal (SPSA) Elo-drop calibration for c_end and (optionally)
an update-size normalization.

---

## 10) Checklist: writing a correct synthetic engine model

To avoid broken reasoning, write down the answers to these before doing any scaling math:

1. What is the **deterministic** surface f(theta) (at least locally)? Is it quadratic? What coordinate system / bounds-based scales will you use?
2. Does the total curvature/budget stay fixed as N changes (recommended), or are you explicitly modeling additional real signal as N grows?
3. What does your Elo drop target refer to: **axis** loss (`c_axis_elo_drop`) or **diagonal** (true SPSA) loss (`c_diag_elo_drop`)?
4. What is your evaluation/noise model? How does variance scale with batch size?
5. Which invariant are you trying to preserve when N changes: update size, SNR, or perturbation Elo loss?

If you can’t answer one of these, the scaling conclusions are not defined yet.

---
