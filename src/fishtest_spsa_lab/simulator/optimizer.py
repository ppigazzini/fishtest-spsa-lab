"""Optimizer implementations for SPSA simulation."""

from __future__ import annotations

import abc

import numpy as np

from fishtest_spsa_lab.simulator.config import (
    SQRT_EPSILON,
    TINY_EPSILON,
    SPSAConfig,
)


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
        net_wins: int,
        flip: np.ndarray,
        c_k: np.ndarray,
        batch_size_pairs: int,
    ) -> None:
        """Advance theta using the gradient estimate from one batch."""

    def get_params(self) -> np.ndarray:
        """Return the current parameter vector."""
        return self.theta


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
        if self.config.c_dev is not None:
            c_end = np.asarray(self.config.c_dev, dtype=float)
        else:
            param_range = self.config.theta_max - self.config.theta_min
            c_end = 0.05 * param_range

        # c_k = c_base / k^gamma, with c_k(num_pairs) = c_end
        self.c_base = c_end * (self.config.num_pairs**self.config.gamma)

        # a_k = a_base / (A + k)^alpha, with a_k(num_pairs) = r_end * c_end^2
        a_end = self.config.r_end * (c_end**2)
        self.a_base = a_end * (
            (self.config.A + self.config.num_pairs) ** self.config.alpha
        )

    def get_perturbation_scale(self, iter_local: int) -> np.ndarray:
        """Return c_k for the given pair index (1-based)."""
        k = max(iter_local, 1)
        return self.c_base / (k**self.config.gamma)

    def step(
        self,
        iter_local: int,
        net_wins: int,
        flip: np.ndarray,
        c_k: np.ndarray,
        batch_size_pairs: int,  # noqa: ARG002 - kept for unified API
    ) -> None:
        """Update theta using classic SPSA rule."""
        k = max(iter_local, 1)
        a_k = self.a_base / ((self.config.A + k) ** self.config.alpha)

        # step = (a_k / c_k) * net_wins * flip (maximize Elo)
        step_vec = (a_k / c_k) * float(net_wins) * flip
        self.theta += step_vec
        self.theta = np.clip(self.theta, self.config.theta_min, self.config.theta_max)


class SPSABlock(SPSA):
    """Block-corrected SPSA using mean gain over a batch of pairs."""

    def step(
        self,
        iter_local: int,
        net_wins: int,
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

        a = self.a_base[None, :] / ((self.config.A + ks) ** self.config.alpha)
        c = self.c_base[None, :] / (ks**self.config.gamma)
        gains = a / c

        mean_gain = gains.mean(axis=0)
        step_vec = mean_gain * float(net_wins) * flip
        self.theta += step_vec
        self.theta = np.clip(self.theta, self.config.theta_min, self.config.theta_max)


class SFSGD(Optimizer):
    """Schedule-Free SGD without block compensation."""

    def __init__(self, config: SPSAConfig) -> None:
        """Initialize Schedule-Free SGD state and perturbation scale."""
        super().__init__(config)
        self.z = self.theta.copy()  # Fast iterate
        self.x = self.theta.copy()  # Polyak average
        self.weight_sum = 0.0

        # Developer-level constant perturbation scale
        if self.config.c_dev is not None:
            self.c_constant = np.asarray(self.config.c_dev, dtype=float)
        else:
            param_range = self.config.theta_max - self.config.theta_min
            self.c_constant = 0.05 * param_range

    def get_perturbation_scale(self, _iter_local: int) -> np.ndarray:
        """Return the constant developer-level perturbation scale."""
        return self.c_constant

    def step(
        self,
        iter_local: int,
        net_wins: int,
        flip: np.ndarray,
        c_k: np.ndarray,
        batch_size_pairs: int,
    ) -> None:
        """Perform one Schedule-Free SGD update using a batch of pairs."""
        lr = self.config.sf_sgd_lr

        # Linear warmup over a fraction of total pairs.
        warmup_end = self.config.num_pairs * self.config.sf_sgd_warmup_fraction
        if warmup_end > 0 and iter_local < warmup_end:
            lr *= iter_local / warmup_end

        beta1 = self.config.sf_sgd_beta1
        batch_size = float(batch_size_pairs)

        # Validate x reconstruction from (theta, z).
        if beta1 > 0.0:
            x_rec = (self.theta - (1.0 - beta1) * self.z) / beta1
            x_rec = np.clip(x_rec, self.config.theta_min, self.config.theta_max)
            if not np.allclose(self.x, x_rec, atol=1e-5):
                max_diff = float(np.max(np.abs(self.x - x_rec)))
                msg = f"Reconstruction failed. Max diff: {max_diff}"
                raise AssertionError(msg)

        delta_total_step = lr * c_k * float(net_wins) * flip
        x_prev = self.x.copy()

        self.z += delta_total_step

        weight = lr
        report_weight = weight * batch_size
        weight_sum_prev = self.weight_sum
        weight_sum_curr = weight_sum_prev + report_weight
        self.weight_sum = weight_sum_curr

        if beta1 == 0.0:
            self.theta = self.z.copy()
            self.theta = np.clip(
                self.theta,
                self.config.theta_min,
                self.config.theta_max,
            )
            return

        a_k = report_weight / weight_sum_curr if weight_sum_curr > 0.0 else 1.0

        self.x = (1.0 - a_k) * x_prev + a_k * self.z
        self.x = np.clip(self.x, self.config.theta_min, self.config.theta_max)

        self.theta = (1.0 - beta1) * self.z + beta1 * self.x
        self.theta = np.clip(self.theta, self.config.theta_min, self.config.theta_max)


class SFSGDBlock(Optimizer):
    """Block-corrected Schedule-Free SGD with triangular weighting."""

    def __init__(self, config: SPSAConfig) -> None:
        """Initialize block Schedule-Free SGD state and scales."""
        super().__init__(config)
        self.z = self.theta.copy()  # Fast iterate
        self.x = self.theta.copy()  # Polyak average
        self.weight_sum = 0.0

        if self.config.c_dev is not None:
            self.c_constant = np.asarray(self.config.c_dev, dtype=float)
        else:
            param_range = self.config.theta_max - self.config.theta_min
            self.c_constant = 0.05 * param_range

    def get_perturbation_scale(self, _iter_local: int) -> np.ndarray:
        """Return the constant perturbation scale used for all blocks."""
        return self.c_constant

    def step(
        self,
        iter_local: int,
        net_wins: int,
        flip: np.ndarray,
        c_k: np.ndarray,
        batch_size_pairs: int,
    ) -> None:
        """Perform one block Schedule-Free SGD update."""
        lr = self.config.sf_sgd_lr

        warmup_end = self.config.num_pairs * self.config.sf_sgd_warmup_fraction
        if warmup_end > 0 and iter_local < warmup_end:
            lr *= iter_local / warmup_end

        beta1 = self.config.sf_sgd_beta1
        batch_size = float(batch_size_pairs)

        # Validate x reconstruction
        if beta1 > 0.0:
            x_rec = (self.theta - (1.0 - beta1) * self.z) / beta1
            x_rec = np.clip(x_rec, self.config.theta_min, self.config.theta_max)
            if not np.allclose(self.x, x_rec, atol=1e-5):
                max_diff = float(np.max(np.abs(self.x - x_rec)))
                msg = f"Reconstruction failed. Max diff: {max_diff}"
                raise AssertionError(msg)

        delta_total_step = lr * c_k * float(net_wins) * flip

        z_prev = self.z.copy()
        x_prev = self.x.copy()

        self.z += delta_total_step

        weight = lr
        report_weight = weight * batch_size
        weight_sum_prev = self.weight_sum
        weight_sum_curr = weight_sum_prev + report_weight
        self.weight_sum = weight_sum_curr

        if beta1 == 0.0:
            self.theta = self.z.copy()
            self.theta = np.clip(
                self.theta,
                self.config.theta_min,
                self.config.theta_max,
            )
            return

        tri_factor = (batch_size + 1.0) / 2.0

        numerator = (
            weight_sum_prev * x_prev
            + report_weight * z_prev
            + weight * delta_total_step * tri_factor
        )
        self.x = numerator / weight_sum_curr
        self.x = np.clip(self.x, self.config.theta_min, self.config.theta_max)

        self.theta = (1.0 - beta1) * self.z + beta1 * self.x
        self.theta = np.clip(self.theta, self.config.theta_min, self.config.theta_max)

        if beta1 > 0.0:
            self.x = (self.theta - (1.0 - beta1) * self.z) / beta1
            self.x = np.clip(self.x, self.config.theta_min, self.config.theta_max)


class SFAdam(Optimizer):
    """Schedule-Free Adam without k(N, beta2) damping."""

    def __init__(self, config: SPSAConfig) -> None:
        """Initialize Schedule-Free Adam state and perturbation scale."""
        super().__init__(config)
        self.z = self.theta.copy()
        self.x = self.theta.copy()
        self.v = np.zeros_like(self.theta)
        self.weight_sum = 0.0
        self.t = 0

        if self.config.c_dev is not None:
            self.c_constant = np.asarray(self.config.c_dev, dtype=float)
        else:
            param_range = self.config.theta_max - self.config.theta_min
            self.c_constant = 0.05 * param_range

    def get_perturbation_scale(self, _iter_local: int) -> np.ndarray:
        """Return the constant perturbation scale for SFAdam."""
        return self.c_constant

    def step(
        self,
        iter_local: int,
        net_wins: int,
        flip: np.ndarray,
        c_k: np.ndarray,
        batch_size_pairs: int,
    ) -> None:
        """Perform one Schedule-Free Adam update over a batch."""
        lr = self.config.sf_adam_lr

        warmup_end = self.config.num_pairs * self.config.sf_adam_warmup_fraction
        if warmup_end > 0 and iter_local < warmup_end:
            lr *= iter_local / warmup_end

        beta1 = self.config.sf_adam_beta1
        beta2 = self.config.sf_adam_beta2
        eps = self.config.sf_adam_eps
        batch_size = int(batch_size_pairs)
        if batch_size <= 0:
            return

        if beta1 > 0.0:
            x_rec = (self.theta - (1.0 - beta1) * self.z) / beta1
            x_rec = np.clip(x_rec, self.config.theta_min, self.config.theta_max)
            if not np.allclose(self.x, x_rec, atol=1e-5):
                max_diff = float(np.max(np.abs(self.x - x_rec)))
                msg = f"Reconstruction failed. Max diff: {max_diff}"
                raise AssertionError(msg)

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
        report_weight = weight * float(batch_size)
        weight_sum_prev = self.weight_sum
        weight_sum_curr = weight_sum_prev + report_weight
        self.weight_sum = weight_sum_curr

        a_k = report_weight / weight_sum_curr if weight_sum_curr > 0.0 else 1.0

        x_prev = self.x
        self.x = (1.0 - a_k) * x_prev + a_k * self.z
        self.x = np.clip(self.x, self.config.theta_min, self.config.theta_max)

        self.theta = (1.0 - beta1) * self.z + beta1 * self.x
        self.theta = np.clip(self.theta, self.config.theta_min, self.config.theta_max)

        if beta1 > 0.0:
            self.x = (self.theta - (1.0 - beta1) * self.z) / beta1
            self.x = np.clip(self.x, self.config.theta_min, self.config.theta_max)


class SFAdamBlock(Optimizer):
    """Block-corrected Schedule-Free Adam with k(N, beta2) damping."""

    def __init__(self, config: SPSAConfig) -> None:
        """Initialize block Schedule-Free Adam state and μ2 statistics."""
        super().__init__(config)
        self.z = self.theta.copy()  # Fast iterate
        self.x = self.theta.copy()  # Polyak average
        self.v = np.zeros_like(self.theta)  # Exp. avg of squared gradients
        self.weight_sum = 0.0
        self.iter_pairs = 0

        # Online μ2 estimator aggregates (global, report-level).
        self.reports = float(self.config.sf_adam_mu2_reports)
        self.sum_n = float(self.config.sf_adam_mu2_sum_n)
        self.sum_s = float(self.config.sf_adam_mu2_sum_s)
        self.sum_s2_over_n = float(self.config.sf_adam_mu2_sum_s2_over_n)
        self.mu2_init = float(self.config.sf_adam_mu2_init)

        if self.config.c_dev is not None:
            self.c_constant = np.asarray(self.config.c_dev, dtype=float)
        else:
            param_range = self.config.theta_max - self.config.theta_min
            self.c_constant = 0.05 * param_range

    def get_perturbation_scale(self, _iter_local: int) -> np.ndarray:
        """Return the constant perturbation scale for SFAdamBlock."""
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

    def step(  # noqa: PLR0915
        self,
        iter_local: int,
        net_wins: int,
        flip: np.ndarray,
        c_k: np.ndarray,
        batch_size_pairs: int,
    ) -> None:
        """Perform one block Schedule-Free Adam update with damping."""
        lr = self.config.sf_adam_lr

        warmup_end = self.config.num_pairs * self.config.sf_adam_warmup_fraction
        if warmup_end > 0 and iter_local < warmup_end:
            lr *= iter_local / warmup_end

        beta1 = self.config.sf_adam_beta1
        beta2 = self.config.sf_adam_beta2
        eps = self.config.sf_adam_eps
        batch_size = int(batch_size_pairs)
        if batch_size <= 0:
            return

        if beta1 > 0.0:
            x_rec = (self.theta - (1.0 - beta1) * self.z) / beta1
            x_rec = np.clip(x_rec, self.config.theta_min, self.config.theta_max)
            if not np.allclose(self.x, x_rec, atol=1e-5):
                max_diff = float(np.max(np.abs(self.x - x_rec)))
                msg = f"Reconstruction failed. Max diff: {max_diff}"
                raise AssertionError(msg)

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
        report_weight = weight * float(batch_size)
        weight_sum_prev = self.weight_sum
        weight_sum_curr = weight_sum_prev + report_weight
        self.weight_sum = weight_sum_curr

        a_k = report_weight / weight_sum_curr if weight_sum_curr > 0.0 else 1.0

        x_prev = self.x
        self.x = (1.0 - a_k) * x_prev + a_k * self.z
        self.x = np.clip(self.x, self.config.theta_min, self.config.theta_max)

        self.theta = (1.0 - beta1) * self.z + beta1 * self.x
        self.theta = np.clip(self.theta, self.config.theta_min, self.config.theta_max)

        if beta1 > 0.0:
            self.x = (self.theta - (1.0 - beta1) * self.z) / beta1
            self.x = np.clip(self.x, self.config.theta_min, self.config.theta_max)

        self._update_mu2_stats(batch_size, float(net_wins))


class Adam(Optimizer):
    """Textbook Adam on block-mean SPSA signals."""

    def __init__(self, config: SPSAConfig) -> None:
        """Initialize Adam optimizer state and perturbation scale."""
        super().__init__(config)
        self.m = np.zeros_like(self.theta)
        self.v = np.zeros_like(self.theta)
        self.t = 0

        if self.config.c_dev is not None:
            self.c_constant = np.asarray(self.config.c_dev, dtype=float)
        else:
            param_range = self.config.theta_max - self.config.theta_min
            self.c_constant = 0.05 * param_range

    def get_perturbation_scale(self, _iter_local: int) -> np.ndarray:
        """Return the constant perturbation scale for Adam."""
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
        net_wins: int,
        flip: np.ndarray,
        c_k: np.ndarray,  # noqa: ARG002 - kept for API symmetry
        batch_size_pairs: int,
    ) -> None:
        """Perform one Adam update using block-mean SPSA gradients."""
        lr = self.config.adam_lr

        warmup_end = self.config.num_pairs * self.config.adam_warmup_fraction
        if warmup_end > 0 and iter_local < warmup_end:
            lr *= iter_local / warmup_end

        beta1 = self.config.adam_beta1
        beta2 = self.config.adam_beta2
        eps = self.config.adam_eps

        n = int(batch_size_pairs)
        if n <= 0:
            return

        g_scalar = -float(net_wins) / float(n)
        grad = g_scalar * flip

        for _ in range(n):
            self._adam_step_vector(grad, lr, beta1, beta2, eps)

        self.theta = np.clip(self.theta, self.config.theta_min, self.config.theta_max)


class AdamBlock(Optimizer):
    """Block-Adam approximation using closed-form EMAs."""

    def __init__(self, config: SPSAConfig) -> None:
        """Initialize block-Adam optimizer state and perturbation scale."""
        super().__init__(config)
        self.m = np.zeros_like(self.theta)
        self.v = np.zeros_like(self.theta)

        if self.config.c_dev is not None:
            self.c_constant = np.asarray(self.config.c_dev, dtype=float)
        else:
            param_range = self.config.theta_max - self.config.theta_min
            self.c_constant = 0.05 * param_range

    def get_perturbation_scale(self, _iter_local: int) -> np.ndarray:
        """Return the constant perturbation scale for AdamBlock."""
        return self.c_constant

    def step(
        self,
        iter_local: int,
        net_wins: int,
        flip: np.ndarray,
        c_k: np.ndarray,  # noqa: ARG002 - kept for API symmetry
        batch_size_pairs: int,
    ) -> None:
        """Perform one block-Adam update using closed-form EMAs."""
        lr = self.config.adam_lr

        warmup_end = self.config.num_pairs * self.config.adam_warmup_fraction
        if warmup_end > 0 and iter_local < warmup_end:
            lr *= iter_local / warmup_end

        beta1 = self.config.adam_beta1
        beta2 = self.config.adam_beta2
        eps = self.config.adam_eps

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

        s_beta1 = (
            beta1 * (1.0 - beta1_n) / (1.0 - beta1) if beta1 != 1.0 else float(n)
        )

        sum_m = m0 * s_beta1 + grad * (n - s_beta1)

        denom = np.sqrt(v_n) + eps
        step_vec = np.where(denom > 0.0, sum_m / denom, 0.0)

        theta_n = theta0 - lr * step_vec

        self.theta = np.clip(theta_n, self.config.theta_min, self.config.theta_max)
        self.m = m_n
        self.v = v_n
