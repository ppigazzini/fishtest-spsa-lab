# SPSA in Fishtest (lean, phi-normalized)

This document explains SPSA in Fishtest using Elo-normalized coordinates `phi` and how the single learning rate `r` maps to the classic `theta`-space schedule `a = r * c**2`. Equations use simple Python-style expressions.

This document covers classic SPSA and the textbook schedule-free formulations. For the lab-specific derivations of the schedule-free variants, see:
- [SF_SGD_derivation.md](SF_SGD_derivation.md) -- schedule-free SPSA with SGD backend
- [SF_Adam_derivation.md](SF_Adam_derivation.md) -- schedule-free SPSA with AdamW backend

Lab code:
- Simulator engine: [src/fishtest_spsa_lab/simulator](../src/fishtest_spsa_lab/simulator)
- Analysis scripts: [src/fishtest_spsa_lab/analysis](../src/fishtest_spsa_lab/analysis)

## Chapter 1 -- Overview and Practical Knobs

### 1.1 At a glance
- Workers play symmetric probes around current parameters: `theta ± c * flip`.
- Each report applies one SPSA update using the total `result = wins − losses` from that report (raw result; not divided by `N`).

### 1.2 Practical knobs and defaults (SPSA/SGD/Adam)

- c_end (per axis): choose so `theta[i] ± c_end` yields a small, measurable Elo gap (a few Elo). This sets `phi`'s unit scale.
- r_end (or `sf_lr` in schedule-free): one scalar for all parameters in `phi`; tune to avoid frequent clipping and keep steady progress.
- alpha, gamma, A: `gamma` small (slow `c` decay), `alpha` moderate (stability late), `A` optional warm-up (0--20% of total pairs).
- Bounds: keep `[min, max]` wide enough to avoid constant clipping; still clamp every `theta` update.
- Defaults (good starting points):
  - Schedule-free SGD: `beta = 0.9` to enable Polyak filtering of the fast iterate; set `beta = 0` to match classic SPSA behavior.
  - AdamW: `beta1 = 0.9`, `beta2 = 0.999`, `sf_eps = 1e-8`.

### 1.3 Where things live in code

- Lab simulator engine: [src/fishtest_spsa_lab/simulator](../src/fishtest_spsa_lab/simulator).
- Lab analysis scripts: [src/fishtest_spsa_lab/analysis](../src/fishtest_spsa_lab/analysis).

## Chapter 2 -- Classic SPSA in Fishtest

### 2.1 Classic SPSA recap

#### Classic SPSA (textbook)
- Notation
  - Vectors are Python-style arrays; "*" is elementwise; "@" is matrix multiply (only used when written explicitly).
  - `F(theta)` is the objective proxy (e.g., Elo/log-odds estimated from game outcomes).
  - We maximize `F` (move along `+gradient`).

- One SPSA iteration `k`
  - Draw `Delta` with independent Rademacher entries: `Delta[i] in {-1, +1}`.
  - Pick per-axis perturbations `c_k[i]` (schedule defined below).

- Symmetric evaluations (elementwise, same scalar `deltaY`)
  - `y_plus  = F(theta_k + c_k * Delta)`      # theta_k[i] + c_k[i] * Delta[i]
  - `y_minus = F(theta_k - c_k * Delta)`
  - `deltaY = y_plus - y_minus`

- Two-sided estimator (per `i`, first-order, unbiased under Rademacher)
  - `g_hat[i] = (deltaY / (2 * c_k[i])) * Delta[i]`

- Update (maximize) with schedule `a_k`
  - `theta_{k+1}[i] = theta_k[i] + a_k * g_hat[i]`

- Canonical schedules (classic)
  - `a_k = a / (A + k)**alpha`
  - `c_k[i] = c_i / (k+1)**gamma`    # Fishtest evaluates an arriving report with k = K+1 to avoid k=0

#### Noise/SNR quick facts
- Finite-difference signal grows linearly with `c` for small gaps; over `N` pairs: `E[result] ∝ N * c`.
- Expected step (first order): `(a_k / c_k) * E[result] ∝ a_k`.
- Step noise std: `(a_k / c_k) * sqrt(N)`; hence step SNR: `SNR ∝ c_k / sqrt(N)`.

Simulator knob (optional):

- `SPSAConfig.scale_gradient_by_sqrt_num_params=True` scales the SPSA update signal by `1/sqrt(N)` using `N = config.num_params` (active + inactive). This helps explore stability when many parameters are inactive/weakly-active but still perturbed.

## Chapter 3 -- Schedule-free optimizers (textbook)

### 3.1 Schedule-free optimizers (textbook)

We minimize a differentiable objective f: R^d → R. At iteration t, let g_t be a stochastic gradient with E[g_t | θ_t] = ∇f(θ_t). Schedule-free means: constant step size (no decay), stability from Polyak averaging of the fast iterate (not from shrinking the learning rate).

Notation
- θ_t: parameter at which the gradient is computed (current iterate); by default `θ_t = (1 - ρ) * z_t + ρ * x_t`
- z_t: fast iterate (primary optimizer state)
- x_t: Polyak (running) average of z_t
- η > 0: constant learning rate
- ρ ∈ [0, 1]: export blend between z and x (ρ = 0 → export z; ρ = 1 → export x)
- t starts at 0; define α_t = 1/(t+1)

#### Schedule-free SGD
Updates (minimize):
- Gradients:
  `g_t ≈ ∇f(θ_t)`
- Fast iterate:
  `z_{t+1} = z_t - eta * g_t`
- Polyak average (arithmetic mean of visited z's):
  `x_{t+1} = (1 - alpha_t) * x_t + alpha_t * z_{t+1}`, where `alpha_t = 1/(t+1)`
- Export (optional smoothing):
  `theta_{t+1} = (1 - rho) * z_{t+1} + rho * x_{t+1}`

Notes
- With α_t = 1/(t+1), x_t is exactly the running average of z_0, z_1, ..., z_t.
- ρ is a presentation choice; it doesn't affect the internal dynamics of z.

#### Schedule-free Adam
We use only Adam's second moment (RMS) for normalization (AdamW-style), a constant `eta`, and schedule-free smoothing via Polyak averaging. There is no first-moment EMA: `m_t` is not computed. In this section `beta1` denotes the export blend with the Polyak average (i.e., the weight on `x_t`).

Hyperparameters: beta1 ∈ [0,1] (Polyak/export blend), beta2 ∈ [0,1), eps > 0, eta > 0.

State: z_t (fast iterate), x_t (Polyak average), v_t (second moment).

Updates (minimize):
- Gradients:
  `g_t ≈ ∇f(θ_t)`
- Second moment (with bias correction):
  `v_{t+1} = beta2 * v_t + (1 - beta2) * (g_t * g_t)`
  `v_hat = v_{t+1} / (1 - beta2**(t+1))`
- Normalized step and fast iterate:
  `d_{t+1} = g_t / (sqrt(v_hat) + eps)`
  `z_{t+1} = z_t - eta * d_{t+1}`
- Polyak average and export:
  `x_{t+1} = (1 - alpha_t) * x_t + alpha_t * z_{t+1}`, where `alpha_t = 1/(t+1)`
  `theta_{t+1} = (1 - beta1) * z_{t+1} + beta1 * x_{t+1}`

Defaults (common, not prescriptive)
- SGD: choose `eta` per problem; `rho ∈ {0, 0.9}` if you use `rho` as the export blend there.
- Adam (schedule-free): `beta1 = 0.9` (Polyak/export blend), `beta2 = 0.999`, `eps = 1e-8`, constant `eta`.

## Chapter 4 -- Core math: θ-space vs φ-space (maximize Elo)

This section shows the same SPSA step in two coordinate systems and why working in `phi` (Elo-normalized) is simpler and better conditioned than working in `theta`. If you only need the practical knobs for experiments, you can skim this section on a first read and return later when you want to see exactly how the φ-based learning rate `r` corresponds to the classic `a_k` in θ-space.

### Sign convention
- We maximize `F` (Elo). Updates use a plus sign (move along +gradient).

### Setup (snapshot at dispatch)
- Let `k0` be the dispatch snapshot for this report. Define `c_i = c_i(k0)` and keep it fixed within the report.
- Define normalized coordinates and the same objective in `phi`:
  - `phi[i] = theta[i] / c_i`            # elementwise, at the k0 snapshot
  - equivalently: `theta[i] = c_i * phi[i]`
  - `G(phi) = F(theta)` with `theta = C @ phi` (conceptual; `C = diag(c_i)`)

### 1) Symmetric probes (same evaluations, two views)
- In θ-space:
  - `theta_plus[i]  = theta[i] + c_i * Delta[i]`
  - `theta_minus[i] = theta[i] - c_i * Delta[i]`
  - `deltaY = F(theta_plus) - F(theta_minus)`
- In φ-space (using `theta = C @ phi`):
  - `phi_plus[i]  = phi[i] + Delta[i]`
  - `phi_minus[i] = phi[i] - Delta[i]`
  - `deltaY = G(phi_plus) - G(phi_minus)`   # same scalar as above

### 2) Two-sided gradient estimators (unbiased, first-order)
- θ-space (per `i`):
  - `g_theta[i] = (deltaY / (2 * c_i)) * Delta[i]`
  - `E[g_theta[i]] = dF/dtheta_i`
  - Update: `theta[i] = theta[i] + a_k * g_theta[i]`
- φ-space (per `i`):
  - `g_phi[i] = (deltaY / 2) * Delta[i]`
  - `E[g_phi[i]] = dG/dphi_i`
  - Update: `phi[i] = phi[i] + r_k * g_phi[i]`

### 3) Practical estimators in Fishtest (use result directly)
- We use `result = wins − losses` from the sub-batch as the finite-difference signal.
- For small Elo gaps between probes, `result` is linearly proportional to the Elo difference (constant absorbed by schedules), so you can plug it in directly:
  - `g_theta[i] ≈ (result / (2 * c_i)) * Delta[i]`
  - `g_phi[i]   ≈ (result / 2) * Delta[i]`

### 4) Exact θ ↔ φ equivalence (single equation)
- Relationships: `g_phi[i] = c_i * g_theta[i]` and `theta[i] = c_i * phi[i]`
- Map the φ-update back to θ in one line:
  - `phi[i]   = phi[i]   + r_k * g_phi[i]`
  - `theta[i] = c_i*phi[i] + r_k * c_i * g_phi[i] = theta[i] + (r_k * c_i**2) * g_theta[i]`
  - Identify the classic schedule: `a_k = r_k * c_i**2`   # exact at the same snapshot `k0`

### 5) Why φ is the better working space
- One scalar learning rate:
  - A single `r_k` works for all parameters in φ. In θ this becomes per-axis `a_{k,i} = r_k * c_i**2` automatically.
- One c, one place:
  - The same `c_i` sets both the probe separation (`theta ± c_i * Delta_i`) and the θ step via `(r_k * c_i)`.

### Notes (units and invariants)
- Units check: `phi` is unitless, `c_i` has θ-units, `r_k` has inverse "result" units; θ-step has θ-units: `delta_theta_i = (r_k * c_i) * result * Delta[i]`.
- Symbols: This chapter uses `Delta` for conceptual flips; the protocol section uses `flip` for the packed/transported bits -- same object, different names to match context.

## Chapter 5 -- Classic SPSA: inputs, schedules, and protocol

### 5.1 Inputs, schedules, and the θ ↔ φ transform

This section shows how user inputs become schedules and how θ and φ relate at dispatch and arrival.

### User inputs
- Per parameter row: `name, start, min, max, c_end, r_end`
- Global: `A, alpha, gamma, num_games`  (`num_iter = num_games // 2`)

### Derived schedules (per axis)

- c schedule (choose `c` so the last step hits `c_end` exactly):
  - `c = c_end * (num_iter**gamma)`
  - `c_k = c / k**gamma`  (evaluate an arriving report with `k = K+1`)
- a schedule (tied to `r` via `a_end = r_end * c_end**2`):
  - `a_end = r_end * (c_end**2)`
  - `a = a_end * (A + num_iter)**alpha`
  - `a_k = a / (A + k)**alpha`

- Convenience variable used by the handler and history:
  - `R_k = a_k / (c_k**2)`

### The θ ↔ φ transform, step by step

1) Dispatch snapshot (save `k0 = K`, and define `iter_local = K+1`)
- Compute the perturbation used inside the sub-batch:
  - `c_i_k0 = param.c / (iter_local**gamma)`
- Conceptual normalized coordinates at dispatch:
  - `phi[i] = theta[i] / c_i_k0`
- What the worker plays:
  - `theta_white[i] = clip(theta[i] + c_i_k0 * flip[i])`
  - `theta_black[i] = clip(theta[i] - c_i_k0 * flip[i])`
  - In φ: this is exactly `phi ± flip` (unit steps), because `theta = c_i_k0 * phi` elementwise.
- Implementation note: `k0` and the packed flips are stored in the task and sent back with the report.

2) Arrival update (classic schedule form)
- Reconstruct the same `c_i_k0` using the saved `k0`; compute:
  - `a_i_k0 = param.a / (A + iter_local)**alpha`
- Apply the θ update per parameter (maximize):
  - `step_i  = (a_i_k0 / c_i_k0) * result * flip[i]`
  - `theta[i] = clip(theta[i] + step_i)`

3) Reading the same update through φ (single `r` at the same snapshot)
- Define:
  - `r_k0 = a_i_k0 / (c_i_k0**2)`
- Then the θ step is the φ-update mapped back:
  - `delta_theta_i = r_k0 * c_i_k0 * result * flip[i]`   # identical to step_i above

### Summary

- Normalize at dispatch: `phi = theta / c(k0)`; probes are `phi ± flip`.
- Update at arrival: `theta += (a/c) * result * flip = (r * c) * result * flip` with `a = r * c**2`.

### 5.2 Classic SPSA dispatch/update protocol

Dispatch (request), using global pairs counter `K`
- `iter_local = K + 1`
- For each parameter `i`:
  - `c_i_k0 = param.c / (iter_local**gamma)`
  - `flip[i] = choice([-1, +1])`
- Return to worker:
  - `theta_white[i] = clip(theta[i] + c_i_k0 * flip[i])`
  - `theta_black[i] = clip(theta[i] - c_i_k0 * flip[i])`
- Store in task:
  - `task.spsa_params = { "iter": K, "packed_flips": pack_bits(flip) }`

Update (arrival), for a report with `num_games = 2*N`
- Reconstruct flips and `c_i_k0` using saved `k0 = task.spsa_params["iter"]`
- `result = wins - losses`
- Apply `theta` update per parameter `i` (master schedule form):
  - `a_i_k0 = param.a / (A + (k0+1))**alpha`
  - `step_i = (a_i_k0 / c_i_k0) * result * flip[i]`
  - `theta[i] = clip(theta[i] + step_i)`
- `spsa["iter"] += N`

Notes
- Multiple workers can share the same `k0`; all use the same `(a_k0, c_k0)` captured at dispatch.
- Only arrival advances `K` by `N`.

## Chapter 6 -- Quick reference and external references

### 6.1 Symbols and spaces
- `theta[i]`: parameter `i` in θ-space
- `phi[i]`: `theta[i] / c_i(k0)` (Elo-normalized at dispatch)
- `c_i(k)`: per-axis perturbation schedule; `c_i(k0)` fixed for the report
- `r_k`: φ-space LR; classic `a_k = r_k * (c_k**2)`
- `flip[i]`: Rademacher in `{-1, +1}`
- `result`: `wins - losses` over the report
- `K`: global pairs count, `k0`: dispatch snapshot, `N`: pairs in the report

### 6.2 Space map and units
- θ-space: `theta`, `z`, `x`, `delta_theta`, `tri_factor` contribution, `c` has θ-units.
- φ-space: `phi`, `g_phi_mean`, `v`, `v_hat`, `denom`, `step_phi` (unitless after multiplying by result and sf_lr).
- Mapping: `theta = c * phi`; θ-step = `c * (φ-step)`.

Units quick notes
- φ is unitless; c has θ-units; sf_lr has inverse "result" units.
- φ-step: `(sf_lr * result)` is unitless; θ-step multiplies by c to get θ-units.

### 6.3 Implementation map
- Lab simulator: [src/fishtest_spsa_lab/simulator/optimizer.py](../src/fishtest_spsa_lab/simulator/optimizer.py) -- SPSA, SF-SGD, SF-Adam, Adam implementations
- Lab runner: [src/fishtest_spsa_lab/simulator/runner.py](../src/fishtest_spsa_lab/simulator/runner.py) -- sync and async simulation loops
- Lab config: [src/fishtest_spsa_lab/simulator/config.py](../src/fishtest_spsa_lab/simulator/config.py) -- Elo geometry and developer-model configuration
- Analysis: [src/fishtest_spsa_lab/analysis/](../src/fishtest_spsa_lab/analysis/) -- validation scripts

### 6.4 External Reference Implementations

These codebases provide public implementations of schedule-free optimizers used for cross-checking semantics (fast iterate vs Polyak surrogate, weighting, second-moment handling).

- PyTorch (facebookresearch/schedule_free)
  Repository: https://github.com/facebookresearch/schedule_free

- Optax (google-deepmind/optax) schedule_free contrib module
  Source file: https://github.com/google-deepmind/optax/blob/main/optax/contrib/_schedule_free.py

- PyTorch Optimizer
  Repository: https://github.com/kozistr/pytorch_optimizer

## Bibliography

[1] J. C. Spall. "Multivariate Stochastic Approximation Using a Simultaneous Perturbation Gradient Approximation." IEEE Transactions on Automatic Control, 37(3), 1992. https://www.jhuapl.edu/spsa/PDF-SPSA/Spall_TAC92.pdf

[2] J. C. Spall. "Implementation of the Simultaneous Perturbation Algorithm for Stochastic Optimization." IEEE Transactions on Aerospace and Electronic Systems, 34(3), 1998. https://www.jhuapl.edu/SPSA/PDF-SPSA/Spall_Implementation_of_the_Simultaneous.PDF

[3] D. P. Kingma, J. Ba. "Adam: A Method for Stochastic Optimization." arXiv:1412.6980 (2014). https://arxiv.org/abs/1412.6980

[4] I. Loshchilov, F. Hutter. "Decoupled Weight Decay Regularization." arXiv:1711.05101 (2017). https://arxiv.org/abs/1711.05101

[5] X. Wang, L. Aitchison. "Batch Size Invariant Adam." arXiv:2402.18824 (February 2024). https://arxiv.org/abs/2402.18824

[6] Z. Chen, N. He, T. Ma, S. Song, Z. Wang. "The Road Less Scheduled: Schedule-Free Optimization in Deep Learning." arXiv:2405.15682 (May 2024). https://arxiv.org/abs/2405.15682

[7] K. Ahn, A. Cutkosky. "Adam with model exponential moving average is effective for nonconvex optimization." arXiv:2405.18199 (May 2024). https://arxiv.org/abs/2405.18199

[8] M. Pagliardini, P. Ablin, D. Grangier. "The AdEMAMix Optimizer: Better, Faster, Older." arXiv:2409.03137 (September 2024) https://arxiv.org/abs/2409.03137

[9] Z. Chen, N. He, T. Ma, S. Song, Z. Wang. "General Framework for Online-to-Nonconvex Conversion: Schedule-Free SGD Is Also Effective for Nonconvex Optimization." arXiv:2411.07061 (November 2024). https://arxiv.org/abs/2411.07061

[10] D. Morwani, H. Zhang, N. Vyas, S. Kakade. "Connections Between Schedule-Free Optimizers, Ademamix, and Accelerated SGD Variant." arXiv:2502.02431 (February 2025). https://arxiv.org/abs/2502.02431

[11] M. Song, K. Ahn, B. Baek, C. Yun. "Through the River: Understanding the Benefit of Schedule-Free Methods for Language Model Training." arXiv:2507.09846 (July 2025). https://arxiv.org/abs/2507.09846v1

[12] C. Brown. "Analysis of Schedule-Free Nonconvex Optimization." arXiv:2508.06743 (August 2025). https://arxiv.org/abs/2508.06743

[13] L. Chen, J. Li, K. Liang, B. Su, C. Xie, N. W. Pierse, C. Liang, N. Lao, Q. Liu "Cautious Weight Decay." arXiv:2510.12402 (October 2025) https://arxiv.org/abs/2510.12402

[14] Y.M. Pun, M. Buchholz, R. M. Gower. "Schedulers for Schedule-Free: Theoretically Inspired Hyperparameters." arXiv:2511.07767 (November 2025). https://arxiv.org/abs/2511.07767

[15] A. Defazio, K. Mishchenko, P. Raman, H.-J. M. Shi, L. Xiao. "Smoothing DiLoCo with Primal Averaging for Faster Training of LLMs." arXiv:2512.17131 (December 2025). https://arxiv.org/abs/2512.17131

[16] A. Meterez, P. A. Nair, D. Morwani, C. Pehlevan, S. Kakade. "Anytime Pretraining: Horizon-Free Learning-Rate Schedules with Weight Averaging." arXiv:2602.03702 (February 2026). https://arxiv.org/abs/2602.03702
