# Rademacher scaling (why `sqrt(N)` and `1/sqrt(N)` show up)

This document isolates the small amount of “high-dimensional geometry” math that tends to get mixed into SPSA scaling discussions.

It is deliberately independent of Elo-bowl calibration choices (axis vs diagonal) and independent of how match noise is modeled.

## 1) The Rademacher direction

In SPSA we perturb along a random sign vector:

```text
δ ∈ {−1,+1}^N,   P(δ_i=+1)=P(δ_i=−1)=1/2
```

Two basic facts drive most of the dimension-scaling intuition.

### 1.1 The direction length is deterministic

Because every coordinate has magnitude 1:

```text
||δ||_2^2 = Σ_i δ_i^2 = N
=> ||δ||_2 = sqrt(N)
```

So any update of the form “scalar × δ” has an L2 norm that grows like `sqrt(N)`.

### 1.2 Random projections have RMS that does NOT grow with N

For any fixed vector `g ∈ R^N`:

```text
E[ <g, δ>^2 ] = ||g||_2^2
```

Reason: `E[δ_i δ_j]=0` for `i≠j` and `E[δ_i^2]=1`.

So `<g,δ>` is typically of order `||g||_2`, not of order `N`.

## 2) Two different “cosines” people mix up

Let `||g||_2 > 0` and define the usual cosine between `g` and `δ`:

```text
cos_raw := <g, δ> / (||g||_2 * ||δ||_2)
```

Because `δ` is symmetric, `E[cos_raw]=0`.

But the RMS size is:

```text
sqrt(E[cos_raw^2])
= sqrt(E[<g,δ>^2]) / (||g||_2 * ||δ||_2)
= ||g||_2 / (||g||_2 * sqrt(N))
= 1/sqrt(N)
```

That is the clean statement “a random Rademacher direction is typically `~1/sqrt(N)` aligned with any fixed direction”.

### 2.1 The SPSA-chosen update direction has positive alignment

In (noise-free) 2-point SPSA, the scalar difference signal is proportional to `<g, δ>`.
So the *update direction* is effectively:

```text
u := sign(<g, δ>) * δ
```

Its cosine with `g` is:

```text
cos_update := <g, u> / (||g||_2 * ||u||_2)
           = |<g, δ>| / (||g||_2 * ||δ||_2)
```

So `E[cos_update] > 0` but still scales like `const / sqrt(N)`.
For large N (CLT intuition), `E[cos_update] ≈ sqrt(2/pi)/sqrt(N)`.

## 3) Why `1/sqrt(N)` is the natural “update-size normalization”

Many SPSA-style macros apply a single scalar signal to all coordinates with signs from `δ`.
Because `||δ||_2 = sqrt(N)`, the L2 step size naturally grows like `sqrt(N)` unless you compensate.

A principled way to keep the L2 RMS step size roughly N-invariant (holding other statistics fixed) is:

```text
scale the scalar signal by 1/||δ||_2 = 1/sqrt(N)
```

This is the “Rademacher RMS” normalization.

### Effective dimension variant

If only `K << N` coordinates are meaningfully active (or your step is effectively restricted to a K-dim subspace), then `1/sqrt(K)` is the analogous normalization.
Whether you *want* that depends on what you are trying to hold invariant (update size vs SNR vs convergence speed).

## 4) Code: Monte Carlo sanity check

This repo includes a small Monte Carlo that prints the scaling behavior directly:

```bash
uv run python -m fishtest_spsa_lab.analysis.rademacher
```

It reports:

- `E||step||` vs theory `|m*c|*sqrt(N)` for steps of the form `(m*c)*δ`.
- `RMS(cos_raw)` vs theory `1/sqrt(N)`.
- `E[cos_update]` vs theory `sqrt(2/pi)/sqrt(N)`.
