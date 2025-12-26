"""Optimizer implementations for SPSA simulation."""

from __future__ import annotations

import abc

import numpy as np

from fishtest_spsa_lab.simulator.config import (
    SQRT_EPSILON,
    TINY_EPSILON,
    SPSAConfig,
)


def _apply_linear_warmup(
    lr: float,
    *,
    iter_local: int,
    num_pairs: int,
    warmup_fraction: float,
) -> float:
    """Apply linear warmup to a learning rate.

    If warmup_fraction is 0.0, this is a no-op.
    """
    warmup_end = float(num_pairs) * float(warmup_fraction)
    if warmup_end > 0.0 and float(iter_local) < warmup_end:
        return lr * (float(iter_local) / warmup_end)
    return lr


class Optimizer(abc.ABC):
    """Abstract base class for optimizers operating on theta."""

    def __init__(self, config: SPSAConfig) -> None:
        """Initialize the optimizer with a shared SPSA configuration."""
        self.config = config
        # All optimizers operate on the same theta state.
        self.theta = config.theta_start.copy()
        self.num_params = int(self.theta.size)

    @abc.abstractmethod
    def get_perturbation_scale(self, iter_local: int) -> np.ndarray:
        """Return the perturbation scale c_k for this iteration."""

    @abc.abstractmethod
    def step(
        self,
        iter_local: int,
        net_wins: float,
        flip: np.ndarray,
        c_k: np.ndarray,
        batch_size_pairs: int,
    ) -> None:
        """Advance theta using the gradient estimate from one batch."""

    def get_params(self) -> np.ndarray:
        """Return the current parameter vector."""
        return self.theta

    def _clip(self, vec: np.ndarray) -> np.ndarray:
        """Clip a vector to the developer-visible parameter bounds."""
        return np.clip(vec, self.config.theta_min, self.config.theta_max)

    def _clip_theta(self) -> None:
        """Clip the internal theta state to the configured bounds."""
        self.theta = self._clip(self.theta)


class SPSA(Optimizer):
    """Classic SPSA with power-law a_k and c_k schedules."""

    def __init__(self, config: SPSAConfig) -> None:
        """Initialize SPSA and precompute schedule constants."""
        super().__init__(config)
        self._calculate_schedule_constants()

    def _calculate_schedule_constants(self) -> None:
        """Precompute per-parameter bases for a_k and c_k schedules."""
        # Use developer-level c_dev as the canonical end-of-run perturbation
        # scale; if missing, fall back to a simple fraction of the range.
        c_end = _default_c_constant(self.config)

        sched = self.config.spsa

        # c_k = c_base / k^gamma, with c_k(num_pairs) = c_end
        self.c_base = c_end * (self.config.num_pairs**sched.gamma)

        # a_k = a_base / (A + k)^alpha, with a_k(num_pairs) = r_end * c_end^2
        a_end = sched.r_end * (c_end**2)
        self.a_base = a_end * ((sched.A + self.config.num_pairs) ** sched.alpha)

    def get_perturbation_scale(self, iter_local: int) -> np.ndarray:
        """Return c_k for the given pair index (1-based)."""
        k = max(iter_local, 1)
        return self.c_base / (k**self.config.spsa.gamma)

    def step(
        self,
        iter_local: int,
        net_wins: float,
        flip: np.ndarray,
        c_k: np.ndarray,
        batch_size_pairs: int,  # noqa: ARG002 - kept for unified API
    ) -> None:
        """Update theta using classic SPSA rule."""
        k = max(iter_local, 1)
        sched = self.config.spsa
        a_k = self.a_base / ((sched.A + k) ** sched.alpha)

        # step = (a_k / c_k) * net_wins * flip (maximize Elo)
        step_vec = (a_k / c_k) * float(net_wins) * flip
        self.theta += step_vec
        self._clip_theta()


class SPSABlock(SPSA):
    """Block-corrected SPSA using mean gain over a batch of pairs."""

    def step(
        self,
        iter_local: int,
        net_wins: float,
        flip: np.ndarray,
        c_k: np.ndarray,  # noqa: ARG002 - kept for API symmetry
        batch_size_pairs: int,
    ) -> None:
        """Perform one block SPSA update using mean gain over the batch."""
        n = int(batch_size_pairs)
        if n <= 0:
            return

        # Micro-steps inside this batch: k_start .. k_start + n - 1
        ks = np.arange(iter_local, iter_local + n, dtype=float).reshape(-1, 1)

        sched = self.config.spsa
        a = self.a_base[None, :] / ((sched.A + ks) ** sched.alpha)
        c = self.c_base[None, :] / (ks**sched.gamma)
        gains = a / c

        mean_gain = gains.mean(axis=0)
        step_vec = mean_gain * float(net_wins) * flip
        self.theta += step_vec
        self._clip_theta()


class ScheduleFreeCore(Optimizer):
    """Shared core for schedule-free optimizers.

    Provides the shared state (z, x, weight_sum) and helper methods for:
    - reconstructing x from (theta, z) given an export coefficient beta
    - maintaining Polyak averaging weights (weight_sum/report_weight/a_k)
    """

    z: np.ndarray
    x: np.ndarray
    weight_sum: float

    def _init_schedule_free_state(self) -> None:
        """Initialize shared schedule-free state."""
        self.z = self.theta.copy()
        self.x = self.theta.copy()
        self.weight_sum = 0.0

    def _reconstruct_x_from_theta_z(self, beta: float) -> np.ndarray:
        """Reconstruct x from the exported parameters theta and fast iterate z."""
        x_rec = (self.theta - (1.0 - beta) * self.z) / beta
        return self._clip(x_rec)

    def _assert_reconstruction(self, beta: float, *, atol: float = 1e-5) -> None:
        """Assert that x matches reconstruction from (theta, z) for beta > 0."""
        if beta <= 0.0:
            return

        x_rec = self._reconstruct_x_from_theta_z(beta)
        if not np.allclose(self.x, x_rec, atol=atol):
            max_diff = float(np.max(np.abs(self.x - x_rec)))
            msg = f"Reconstruction failed. Max diff: {max_diff}"
            raise AssertionError(msg)

    def _sync_x_from_theta_z(self, beta: float) -> None:
        """Set x to match reconstruction from (theta, z) for beta > 0."""
        if beta <= 0.0:
            return
        self.x = self._reconstruct_x_from_theta_z(beta)

    def _update_weight_sum(
        self,
        *,
        weight: float,
        batch_size_pairs: int,
    ) -> tuple[float, float, float]:
        """Update weight_sum and return (report_weight, prev, curr)."""
        report_weight = float(weight) * float(batch_size_pairs)
        weight_sum_prev = float(self.weight_sum)
        weight_sum_curr = weight_sum_prev + report_weight
        self.weight_sum = weight_sum_curr
        return report_weight, weight_sum_prev, weight_sum_curr

    @staticmethod
    def _a_k(*, report_weight: float, weight_sum_curr: float) -> float:
        """Compute Polyak blend factor a_k from report-weighted totals."""
        if weight_sum_curr <= 0.0:
            return 1.0
        return float(report_weight) / float(weight_sum_curr)

    def _polyak_update_x_simple(
        self,
        *,
        x_prev: np.ndarray,
        z: np.ndarray,
        report_weight: float,
        weight_sum_curr: float,
    ) -> np.ndarray:
        """Update x by (1-a_k) * x_prev + a_k * z with clipping."""
        a_k = self._a_k(report_weight=report_weight, weight_sum_curr=weight_sum_curr)
        x_new = (1.0 - a_k) * x_prev + a_k * z
        return self._clip(x_new)

    def _polyak_update_x_triangular(  # noqa: PLR0913
        self,
        *,
        x_prev: np.ndarray,
        z_prev: np.ndarray,
        delta_total_step: np.ndarray,
        weight: float,
        batch_size_pairs: int,
        report_weight: float,
        weight_sum_prev: float,
        weight_sum_curr: float,
    ) -> np.ndarray:
        """Block-corrected triangular Polyak averaging update for x."""
        batch_size = float(batch_size_pairs)
        tri_factor = (batch_size + 1.0) / 2.0

        numerator = (
            weight_sum_prev * x_prev
            + report_weight * z_prev
            + float(weight) * delta_total_step * tri_factor
        )
        x_new = numerator / float(weight_sum_curr)
        return self._clip(x_new)


class SFSGD(ScheduleFreeCore):
    """Schedule-Free SGD without block compensation."""

    def __init__(self, config: SPSAConfig) -> None:
        """Initialize Schedule-Free SGD state and perturbation scale."""
        super().__init__(config)
        self._init_schedule_free_state()

        # Developer-level constant perturbation scale
        self.c_constant = _default_c_constant(config)

    def get_perturbation_scale(self, iter_local: int) -> np.ndarray:
        """Return the constant developer-level perturbation scale."""
        if iter_local <= 0:
            msg = "iter_local must be positive."
            raise ValueError(msg)
        return self.c_constant

    def step(
        self,
        iter_local: int,
        net_wins: float,
        flip: np.ndarray,
        c_k: np.ndarray,
        batch_size_pairs: int,
    ) -> None:
        """Perform one Schedule-Free SGD update using a batch of pairs."""
        lr = _apply_linear_warmup(
            float(self.config.sf_sgd.lr),
            iter_local=iter_local,
            num_pairs=self.config.num_pairs,
            warmup_fraction=float(self.config.sf_sgd.warmup_fraction),
        )

        beta = self.config.sf_sgd.beta
        self._assert_reconstruction(beta)

        delta_total_step = lr * c_k * float(net_wins) * flip
        x_prev = self.x.copy()

        self.z += delta_total_step

        weight = lr
        report_weight, _weight_sum_prev, weight_sum_curr = self._update_weight_sum(
            weight=weight,
            batch_size_pairs=batch_size_pairs,
        )

        if beta == 0.0:
            self.theta = self.z.copy()
            self._clip_theta()
            return

        self.x = self._polyak_update_x_simple(
            x_prev=x_prev,
            z=self.z,
            report_weight=report_weight,
            weight_sum_curr=weight_sum_curr,
        )

        self.theta = (1.0 - beta) * self.z + beta * self.x
        self._clip_theta()


class SFSGDBlock(ScheduleFreeCore):
    """Block-corrected Schedule-Free SGD with triangular weighting."""

    def __init__(self, config: SPSAConfig) -> None:
        """Initialize block Schedule-Free SGD state and scales."""
        super().__init__(config)
        self._init_schedule_free_state()

        self.c_constant = _default_c_constant(config)

    def get_perturbation_scale(self, iter_local: int) -> np.ndarray:
        """Return the constant perturbation scale used for all blocks."""
        if iter_local <= 0:
            msg = "iter_local must be positive."
            raise ValueError(msg)
        return self.c_constant

    def step(
        self,
        iter_local: int,
        net_wins: float,
        flip: np.ndarray,
        c_k: np.ndarray,
        batch_size_pairs: int,
    ) -> None:
        """Perform one block Schedule-Free SGD update."""
        lr = _apply_linear_warmup(
            float(self.config.sf_sgd.lr),
            iter_local=iter_local,
            num_pairs=self.config.num_pairs,
            warmup_fraction=float(self.config.sf_sgd.warmup_fraction),
        )

        beta = self.config.sf_sgd.beta
        self._assert_reconstruction(beta)

        delta_total_step = lr * c_k * float(net_wins) * flip

        z_prev = self.z.copy()
        x_prev = self.x.copy()

        self.z += delta_total_step

        weight = lr
        report_weight, weight_sum_prev, weight_sum_curr = self._update_weight_sum(
            weight=weight,
            batch_size_pairs=batch_size_pairs,
        )

        if beta == 0.0:
            self.theta = self.z.copy()
            self._clip_theta()
            return

        self.x = self._polyak_update_x_triangular(
            x_prev=x_prev,
            z_prev=z_prev,
            delta_total_step=delta_total_step,
            weight=weight,
            batch_size_pairs=batch_size_pairs,
            report_weight=report_weight,
            weight_sum_prev=weight_sum_prev,
            weight_sum_curr=weight_sum_curr,
        )

        self.theta = (1.0 - beta) * self.z + beta * self.x
        self._clip_theta()

        self._sync_x_from_theta_z(beta)


class SFAdam(ScheduleFreeCore):
    """Schedule-Free Adam without k(N, beta2) damping."""

    def __init__(self, config: SPSAConfig) -> None:
        """Initialize Schedule-Free Adam state and perturbation scale."""
        super().__init__(config)
        self._init_schedule_free_state()
        self.v = np.zeros_like(self.theta)
        self.t = 0

        self.c_constant = _default_c_constant(config)

    def get_perturbation_scale(self, iter_local: int) -> np.ndarray:
        """Return the constant perturbation scale for SFAdam."""
        if iter_local <= 0:
            msg = "iter_local must be positive."
            raise ValueError(msg)
        return self.c_constant

    def step(
        self,
        iter_local: int,
        net_wins: float,
        flip: np.ndarray,
        c_k: np.ndarray,
        batch_size_pairs: int,
    ) -> None:
        """Perform one Schedule-Free Adam update over a batch."""
        lr = _apply_linear_warmup(
            float(self.config.sf_adam.lr),
            iter_local=iter_local,
            num_pairs=self.config.num_pairs,
            warmup_fraction=float(self.config.sf_adam.warmup_fraction),
        )

        beta1 = self.config.sf_adam.beta1
        beta2 = self.config.sf_adam.beta2
        eps = self.config.sf_adam.eps
        batch_size = int(batch_size_pairs)
        if batch_size <= 0:
            return

        self._assert_reconstruction(beta1)

        micro_steps = iter_local + batch_size - 1
        g_phi_mean = (float(net_wins) / float(batch_size)) * flip

        if beta2 < 1.0:
            beta2_pow_n = beta2**batch_size
            self.v = beta2_pow_n * self.v + (1.0 - beta2_pow_n) * (g_phi_mean**2)

            bc_denom = 1.0 - (beta2**micro_steps)
            v_hat = self.v / bc_denom if bc_denom > TINY_EPSILON else self.v
        else:
            v_hat = self.v

        denom = np.sqrt(v_hat) + eps
        step_phi = (lr * float(net_wins) * flip) / denom
        delta_total_step = step_phi * c_k

        self.z += delta_total_step

        weight = lr
        report_weight, _weight_sum_prev, weight_sum_curr = self._update_weight_sum(
            weight=weight,
            batch_size_pairs=batch_size,
        )

        self.x = self._polyak_update_x_simple(
            x_prev=self.x,
            z=self.z,
            report_weight=report_weight,
            weight_sum_curr=weight_sum_curr,
        )

        # Export blend uses beta1 (Polyak/export coefficient).
        self.theta = (1.0 - beta1) * self.z + beta1 * self.x
        self._clip_theta()

        self._sync_x_from_theta_z(beta1)


class SFAdamBlock(ScheduleFreeCore):
    """Block-corrected Schedule-Free Adam with k(N, beta2) damping."""

    def __init__(self, config: SPSAConfig) -> None:
        """Initialize block Schedule-Free Adam state and μ2 statistics."""
        super().__init__(config)
        self._init_schedule_free_state()
        self.v = np.zeros_like(self.theta)  # Exp. avg of squared gradients
        self.iter_pairs = 0

        # Online μ2 estimator aggregates (global, report-level).
        mu2 = self.config.sf_adam.mu2
        self.reports = float(mu2.reports)
        self.sum_n = float(mu2.sum_n)
        self.sum_s = float(mu2.sum_s)
        self.sum_s2_over_n = float(mu2.sum_s2_over_n)
        self.mu2_init = float(mu2.init)

        self.c_constant = _default_c_constant(config)

    def get_perturbation_scale(self, iter_local: int) -> np.ndarray:
        """Return the constant perturbation scale for SFAdamBlock."""
        if iter_local <= 0:
            msg = "iter_local must be positive."
            raise ValueError(msg)
        return self.c_constant

    def _mu2_hat(self) -> float:
        if self.reports <= 0.0:
            return self.mu2_init

        mu = (self.sum_s / self.sum_n) if self.sum_n > 0.0 else 0.0
        e_s2_over_n = self.sum_s2_over_n / self.reports
        e_n = self.sum_n / self.reports
        sigma2 = e_s2_over_n - (mu * mu) * e_n
        sigma2 = max(sigma2, 0.0)
        mu2 = mu * mu + sigma2
        return float(min(max(mu2, 1e-12), 4.0))

    def _update_mu2_stats(self, n: int, s: float) -> None:
        self.reports += 1.0
        self.sum_n += float(n)
        self.sum_s += float(s)
        self.sum_s2_over_n += (float(s) * float(s)) / max(float(n), 1.0)

    def step(
        self,
        iter_local: int,
        net_wins: float,
        flip: np.ndarray,
        c_k: np.ndarray,
        batch_size_pairs: int,
    ) -> None:
        """Perform one block Schedule-Free Adam update with damping."""
        lr = _apply_linear_warmup(
            float(self.config.sf_adam.lr),
            iter_local=iter_local,
            num_pairs=self.config.num_pairs,
            warmup_fraction=float(self.config.sf_adam.warmup_fraction),
        )

        beta1 = self.config.sf_adam.beta1
        beta2 = self.config.sf_adam.beta2
        eps = self.config.sf_adam.eps
        batch_size = int(batch_size_pairs)
        if batch_size <= 0:
            return

        self._assert_reconstruction(beta1)

        # --- Adam-specific math for batched / out-of-order updates ---
        self.iter_pairs += batch_size

        g_sq_mean = self._mu2_hat()

        if beta2 < 1.0:
            beta2_pow_n = beta2**batch_size
            self.v = beta2_pow_n * self.v + (1.0 - beta2_pow_n) * g_sq_mean

            bc_denom = 1.0 - (beta2**self.iter_pairs)
            v_hat = self.v / bc_denom if bc_denom > TINY_EPSILON else self.v
        else:
            v_hat = self.v

        denom = np.sqrt(v_hat) + eps

        step_phi = (lr * float(net_wins) * flip) / denom

        if batch_size > 1 and 0.0 < beta2 < 1.0:
            sqrt_b2 = np.sqrt(beta2)
            if abs(1.0 - sqrt_b2) > SQRT_EPSILON:
                num = 1.0 - (beta2 ** (0.5 * batch_size))
                den = batch_size * (1.0 - sqrt_b2)
                k_damping = num / den if den != 0.0 else 1.0
            else:
                k_damping = 1.0 - ((batch_size - 1) * 0.25) * (1.0 - beta2)

            if not (0.0 < k_damping <= 1.0):
                k_damping = 1.0 if k_damping > 1.0 else 1e-6

            step_phi *= k_damping

        delta_total_step = step_phi * c_k

        self.z += delta_total_step

        weight = lr
        report_weight, _weight_sum_prev, weight_sum_curr = self._update_weight_sum(
            weight=weight,
            batch_size_pairs=batch_size,
        )

        self.x = self._polyak_update_x_simple(
            x_prev=self.x,
            z=self.z,
            report_weight=report_weight,
            weight_sum_curr=weight_sum_curr,
        )

        self.theta = (1.0 - beta1) * self.z + beta1 * self.x
        self._clip_theta()

        self._sync_x_from_theta_z(beta1)

        self._update_mu2_stats(batch_size, float(net_wins))


class Adam(Optimizer):
    """Textbook Adam on block-mean SPSA signals."""

    def __init__(self, config: SPSAConfig) -> None:
        """Initialize Adam optimizer state and perturbation scale."""
        super().__init__(config)
        self.m = np.zeros_like(self.theta)
        self.v = np.zeros_like(self.theta)
        self.t = 0

        self.c_constant = _default_c_constant(config)

    def get_perturbation_scale(self, iter_local: int) -> np.ndarray:
        """Return the constant perturbation scale for Adam."""
        if iter_local <= 0:
            msg = "iter_local must be positive."
            raise ValueError(msg)
        return self.c_constant

    def _adam_step_vector(
        self,
        grad: np.ndarray,
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
    ) -> None:
        self.m = beta1 * self.m + (1.0 - beta1) * grad
        self.v = beta2 * self.v + (1.0 - beta2) * (grad**2)

        self.t += 1
        t1 = float(self.t)
        m_hat = self.m / (1.0 - beta1**t1) if 0.0 <= beta1 < 1.0 else self.m
        v_hat = self.v / (1.0 - beta2**t1) if 0.0 <= beta2 < 1.0 else self.v

        denom = np.sqrt(v_hat) + eps
        step_vec = np.where(denom > 0.0, m_hat / denom, 0.0)
        self.theta -= lr * step_vec

    def step(
        self,
        iter_local: int,
        net_wins: float,
        flip: np.ndarray,
        c_k: np.ndarray,  # noqa: ARG002 - kept for API symmetry
        batch_size_pairs: int,
    ) -> None:
        """Perform one Adam update using block-mean SPSA gradients."""
        lr = _apply_linear_warmup(
            float(self.config.adam.lr),
            iter_local=iter_local,
            num_pairs=self.config.num_pairs,
            warmup_fraction=float(self.config.adam.warmup_fraction),
        )

        beta1 = self.config.adam.beta1
        beta2 = self.config.adam.beta2
        eps = self.config.adam.eps

        n = int(batch_size_pairs)
        if n <= 0:
            return

        g_scalar = -float(net_wins) / float(n)
        grad = g_scalar * flip

        for _ in range(n):
            self._adam_step_vector(grad, lr, beta1, beta2, eps)

        self._clip_theta()


class AdamBlock(Optimizer):
    """Block-Adam approximation using closed-form EMAs."""

    def __init__(self, config: SPSAConfig) -> None:
        """Initialize block-Adam optimizer state and perturbation scale."""
        super().__init__(config)
        self.m = np.zeros_like(self.theta)
        self.v = np.zeros_like(self.theta)

        self.c_constant = _default_c_constant(config)

    def get_perturbation_scale(self, iter_local: int) -> np.ndarray:
        """Return the constant perturbation scale for AdamBlock."""
        if iter_local <= 0:
            msg = "iter_local must be positive."
            raise ValueError(msg)
        return self.c_constant

    def step(
        self,
        iter_local: int,
        net_wins: float,
        flip: np.ndarray,
        c_k: np.ndarray,  # noqa: ARG002 - kept for API symmetry
        batch_size_pairs: int,
    ) -> None:
        """Perform one block-Adam update using closed-form EMAs."""
        lr = _apply_linear_warmup(
            float(self.config.adam.lr),
            iter_local=iter_local,
            num_pairs=self.config.num_pairs,
            warmup_fraction=float(self.config.adam.warmup_fraction),
        )

        beta1 = self.config.adam.beta1
        beta2 = self.config.adam.beta2
        eps = self.config.adam.eps

        n = int(batch_size_pairs)
        if n <= 0:
            return

        theta0 = self.theta
        m0 = self.m
        v0 = self.v

        g_scalar = -float(net_wins) / float(n)
        grad = g_scalar * flip

        beta1_n = beta1**n
        beta2_n = beta2**n

        m_n = beta1_n * m0 + (1.0 - beta1_n) * grad
        v_n = beta2_n * v0 + (1.0 - beta2_n) * (grad**2)

        s_beta1 = beta1 * (1.0 - beta1_n) / (1.0 - beta1) if beta1 != 1.0 else float(n)

        sum_m = m0 * s_beta1 + grad * (n - s_beta1)

        denom = np.sqrt(v_n) + eps
        step_vec = np.where(denom > 0.0, sum_m / denom, 0.0)

        theta_n = theta0 - lr * step_vec

        self.theta = self._clip(theta_n)
        self.m = m_n
        self.v = v_n


# --- Extensions below this line (new optimizers) ---


def _default_c_constant(config: SPSAConfig) -> np.ndarray:
    if config.c_dev is not None:
        return np.asarray(config.c_dev, dtype=float)
    param_range = config.theta_max - config.theta_min
    return 0.05 * param_range


_PENTA_PROBS_SIZE = 5
_PENTA_DEBIAS_EPS = 1e-6


class SPSACWD(SPSA):
    """SPSA with cautious weight decay (CWD)."""

    def __init__(self, config: SPSAConfig) -> None:
        """Initialize cautious weight decay state."""
        super().__init__(config)
        # Decay is applied to deviation from the start, making it symmetric
        # w.r.t. the peak position relative to the starting point.
        self.theta_center = np.asarray(config.theta_start, dtype=float).copy()

    def step(
        self,
        iter_local: int,
        net_wins: float,
        flip: np.ndarray,
        c_k: np.ndarray,
        batch_size_pairs: int,  # noqa: ARG002 - kept for unified API
    ) -> None:
        """Perform one SPSA step with optional cautious weight decay."""
        k = max(iter_local, 1)
        sched = self.config.spsa
        a_k = self.a_base / ((sched.A + k) ** sched.alpha)

        theta_prev = self.theta
        step_vec = (a_k / c_k) * float(net_wins) * flip

        lambda_cwd = float(self.config.spsa_cwd.lambda_)
        if lambda_cwd > 0.0:
            # CWD rule (asymmetric): apply weight decay only when it reinforces
            # the current SPSA gradient direction.
            #
            # Using u_t (gradient-like direction) and x_t (current params):
            #   apply on coords where u_t ⊙ x_t >= 0
            # Here, step_vec is proportional to -u_t, so u_t = -step_vec.
            deviation = theta_prev - self.theta_center
            u_t = -step_vec
            mask = (u_t * deviation) >= 0.0
            r_k = a_k / (c_k * c_k)
            decay = (lambda_cwd * r_k) * np.where(mask, deviation, 0.0)
            theta_new = theta_prev + step_vec - decay
        else:
            theta_new = theta_prev + step_vec

        self.theta = theta_new
        self._clip_theta()


class PentaStatsMixin(abc.ABC):
    """Explicit interface for optimizers that consume pentanomial counts."""

    asym_history: list[float]
    mu_history: list[float]
    asym_cum_history: list[float]
    mu_cum_history: list[float]
    penta_coeff_history: list[float]
    gain_scale_history: list[float]

    @abc.abstractmethod
    def update_penta_stats(self, counts: np.ndarray) -> None:
        """Consume a pentanomial counts vector (LL, DL, DD, WD, WW)."""


class SPSAPenta(PentaStatsMixin, SPSA):
    """SPSA variant that tracks pentanomial asymmetry and mean."""

    def __init__(self, config: SPSAConfig) -> None:
        """Initialize pentanomial tracking state."""
        super().__init__(config)

        self.p_ema: np.ndarray = np.zeros(_PENTA_PROBS_SIZE, dtype=float)
        self.p_beta_prod: float = 1.0

        self.asym_history: list[float] = []
        self.mu_history: list[float] = []
        self.last_asym: float = 0.0
        self.last_mu: float = 0.0

        self.penta_coeff_history: list[float] = []
        self.gain_scale_history: list[float] = []

        self.cum_counts: np.ndarray | None = None
        self.asym_cum_history: list[float] = []
        self.mu_cum_history: list[float] = []

    def update_penta_stats(self, counts: np.ndarray) -> None:
        """Update EMA and cumulative pentanomial-derived statistics."""
        if counts.size != _PENTA_PROBS_SIZE:
            return
        n = int(np.sum(counts))
        if n <= 0:
            return

        p_batch = counts.astype(float) / float(n)

        beta_pg = float(self.config.spsa_penta.beta_pg)
        beta_eff = float(beta_pg**n)

        self.p_ema = beta_eff * self.p_ema + (1.0 - beta_eff) * p_batch
        self.p_beta_prod *= beta_eff

        debias_denom = 1.0 - self.p_beta_prod
        if debias_denom <= _PENTA_DEBIAS_EPS:
            p_hat = self.p_ema.copy()
        else:
            p_hat = self.p_ema / debias_denom

        outcomes = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=float)
        p = p_hat

        p_ll, p_dl, _p_dd, p_wd, p_ww = p
        asym = abs(p_ww - p_ll) + 0.5 * abs(p_wd - p_dl)
        mu = float(np.dot(p, outcomes))

        self.last_asym = float(asym)
        self.last_mu = float(mu)
        self.asym_history.append(self.last_asym)
        self.mu_history.append(self.last_mu)

        if self.cum_counts is None:
            self.cum_counts = counts.astype(float)
        else:
            self.cum_counts += counts.astype(float)

        total_cum = float(np.sum(self.cum_counts))
        if total_cum > 0.0:
            p_cum = self.cum_counts / total_cum
            p_ll_c, p_dl_c, _p_dd_c, p_wd_c, p_ww_c = p_cum
            asym_c = abs(p_ww_c - p_ll_c) + 0.5 * abs(p_wd_c - p_dl_c)
            mu_c = float(np.dot(p_cum, outcomes))
            self.asym_cum_history.append(float(asym_c))
            self.mu_cum_history.append(float(mu_c))

            mu_weight = float(self.config.spsa_penta.mu_weight)
            r_cum = abs(asym_c) + mu_weight * abs(mu_c)
            self.penta_coeff_history.append(float(r_cum))

    def step(
        self,
        iter_local: int,
        net_wins: float,
        flip: np.ndarray,
        c_k: np.ndarray,
        batch_size_pairs: int,  # noqa: ARG002 - kept for unified API
    ) -> None:
        """Perform one SPSA step with a gain scale derived from penta stats."""
        k = max(iter_local, 1)
        sched = self.config.spsa
        a_k = self.a_base / ((sched.A + k) ** sched.alpha)

        if not self.asym_cum_history:
            scale = 1.0
        else:
            asym_c = abs(self.asym_cum_history[-1])
            mu_c = abs(self.mu_cum_history[-1])
            mu_weight = float(self.config.spsa_penta.mu_weight)
            r = float(asym_c + mu_weight * mu_c)

            r_small = float(self.config.spsa_penta.r_small)
            r_large = float(self.config.spsa_penta.r_large)

            min_scale = float(self.config.spsa_penta.min_scale)
            max_scale = float(self.config.spsa_penta.max_scale)

            if r <= r_small:
                scale = min_scale
            elif r >= r_large:
                scale = max_scale
            else:
                t = (r - r_small) / (r_large - r_small)
                scale = min_scale + (max_scale - min_scale) * float(t)

        self.gain_scale_history.append(float(scale))

        step_vec = (a_k / c_k) * float(net_wins) * flip * float(scale)
        self.theta += step_vec
        self._clip_theta()


class AcceleratedSPSA(SPSA):
    """Accelerated SPSA in the accelerated-SGD framework (Eq. (1))."""

    def __init__(self, config: SPSAConfig) -> None:
        """Initialize accelerated-SPSA momentum state."""
        super().__init__(config)
        self.m = np.zeros_like(self.theta, dtype=float)
        acc = self.config.accelerated_spsa
        self.beta_a_const = float(acc.beta)
        self.beta_mode = acc.beta_mode
        self.beta_k = float(acc.beta_k)
        self.eta_scale = float(acc.eta_scale)
        self.alpha_scale = float(acc.alpha_scale)

    def _beta_a(self, k: int) -> float:
        if self.beta_mode == "inv_time":
            t = float(max(k, 1))
            val = 1.0 - self.beta_k / t
            return float(max(0.0, min(0.9999, val)))
        return self.beta_a_const

    def step(
        self,
        iter_local: int,
        net_wins: float,
        flip: np.ndarray,
        c_k: np.ndarray,
        batch_size_pairs: int,  # noqa: ARG002 - kept for unified API
    ) -> None:
        """Perform one accelerated SPSA step."""
        k = max(iter_local, 1)
        sched = self.config.spsa
        a_k = self.a_base / ((sched.A + k) ** sched.alpha)

        g_k = (float(net_wins) / c_k) * flip
        beta_a = self._beta_a(k)
        self.m = beta_a * self.m + g_k

        if self.beta_mode == "inv_time":
            beta_ref = self.beta_a_const
            one_minus_ref = max(1.0 - beta_ref, 1e-4)
            one_minus_beta = max(1.0 - beta_a, 1e-6)
            eta_scale_effective = self.eta_scale * (one_minus_beta / one_minus_ref)
        else:
            eta_scale_effective = self.eta_scale

        eta_a = eta_scale_effective * a_k
        alpha_a = self.alpha_scale * a_k
        step_vec = eta_a * self.m + alpha_a * g_k
        self.theta += step_vec
        self._clip_theta()


class AdEMAMix(Optimizer):
    """Full AdEMAMix optimizer on top of SPSA gradients."""

    def __init__(self, config: SPSAConfig) -> None:
        """Initialize AdEMAMix state."""
        super().__init__(config)
        self.m1 = np.zeros_like(self.theta)
        self.m2 = np.zeros_like(self.theta)
        self.nu = np.zeros_like(self.theta)
        self.t = 0
        self.c_constant = _default_c_constant(config)

    def get_perturbation_scale(self, iter_local: int) -> np.ndarray:
        """Return the developer-level constant perturbation scale."""
        if iter_local <= 0:
            msg = "iter_local must be positive."
            raise ValueError(msg)
        return self.c_constant

    def _step_vector(self, grad: np.ndarray, lr: float) -> None:
        cfg = self.config.ademamix
        b1 = float(cfg.beta1)
        b2 = float(cfg.beta2)
        b3 = float(cfg.beta3)
        alpha = float(cfg.alpha)
        eps = float(cfg.eps)
        eps_root = float(cfg.eps_root)

        self.t += 1
        t = float(self.t)

        self.m1 = b1 * self.m1 + (1.0 - b1) * grad
        self.m2 = b3 * self.m2 + (1.0 - b3) * grad
        self.nu = b2 * self.nu + (1.0 - b2) * (grad * grad)

        m1_hat = self.m1 / (1.0 - b1**t) if 0.0 <= b1 < 1.0 else self.m1
        nu_hat = self.nu / (1.0 - b2**t) if 0.0 <= b2 < 1.0 else self.nu

        denom = np.sqrt(nu_hat + eps_root) + eps
        num = m1_hat + alpha * self.m2
        step_vec = lr * np.where(denom > 0.0, num / denom, 0.0)
        self.theta -= step_vec

    def step(
        self,
        iter_local: int,
        net_wins: float,
        flip: np.ndarray,
        c_k: np.ndarray,  # noqa: ARG002 - kept for API symmetry
        batch_size_pairs: int,
    ) -> None:
        """Apply AdEMAMix updates for a whole report block."""
        lr = _apply_linear_warmup(
            float(self.config.ademamix.lr),
            iter_local=iter_local,
            num_pairs=self.config.num_pairs,
            warmup_fraction=float(self.config.ademamix.warmup_fraction),
        )

        n = int(batch_size_pairs)
        if n <= 0:
            return

        g_scalar = -float(net_wins) / float(n)
        grad = g_scalar * flip

        for _ in range(n):
            self._step_vector(grad, lr)

        self._clip_theta()


__all__ = [
    "OPTIMIZER_REGISTRY",
    "SFSGD",
    "SPSA",
    "SPSACWD",
    "AcceleratedSPSA",
    "AdEMAMix",
    "Adam",
    "AdamBlock",
    "Optimizer",
    "PentaStatsMixin",
    "SFAdam",
    "SFAdamBlock",
    "SFSGDBlock",
    "SPSABlock",
    "SPSAPenta",
    "ScheduleFreeCore",
]


OPTIMIZER_REGISTRY: dict[str, type[Optimizer]] = {
    "spsa": SPSA,
    "spsa-block": SPSABlock,
    "spsa-penta": SPSAPenta,
    "spsa-cwd": SPSACWD,
    "accelerated-spsa": AcceleratedSPSA,
    "sf-sgd": SFSGD,
    "sf-sgd-block": SFSGDBlock,
    "sf-adam": SFAdam,
    "sf-adam-block": SFAdamBlock,
    "adam": Adam,
    "adam-block": AdamBlock,
    "ademamix": AdEMAMix,
}
