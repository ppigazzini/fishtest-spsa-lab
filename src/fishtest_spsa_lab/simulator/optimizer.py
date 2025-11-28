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
    """Abstract base class for optimizers."""

    def __init__(self, config: SPSAConfig) -> None:
        """Initialize the optimizer."""
        self.config = config
        self.theta = config.param_start.copy()
        self.num_params = len(self.theta)

    @abc.abstractmethod
    def get_perturbation_scale(self, iter_local: int) -> np.ndarray:
        """Return the perturbation scale (c_k) for the given iteration."""

    @abc.abstractmethod
    def step(
        self,
        iter_local: int,
        net_wins: int,
        flip: np.ndarray,
        c_k: np.ndarray,
    ) -> None:
        """Update parameters based on the gradient estimate."""

    def get_params(self) -> np.ndarray:
        """Return the current parameter vector."""
        return self.theta


class ClassicSPSA(Optimizer):
    """Classic SPSA implementation matching Fishtest logic."""

    def __init__(self, config: SPSAConfig) -> None:
        """Initialize the Classic SPSA optimizer."""
        super().__init__(config)
        self._calculate_schedule_constants()

    def _calculate_schedule_constants(self) -> None:
        """Calculate the constants for the SPSA schedule."""
        # Range for c_end calculation
        param_range = self.config.param_max - self.config.param_min
        c_end = param_range * self.config.c_end_fraction

        # c schedule: c_k = c_base / k^gamma
        self.c_base = c_end * (self.config.num_pairs**self.config.gamma)

        # a schedule: a_k = a_base / (A + k)^alpha
        a_end = self.config.r_end * (c_end**2)
        self.a_base = a_end * (
            (self.config.A + self.config.num_pairs) ** self.config.alpha
        )

    def get_perturbation_scale(self, iter_local: int) -> np.ndarray:
        """Return the perturbation scale (c_k) for the given iteration."""
        # Decay c_k: c_k = c_base / k^gamma
        return self.c_base / (iter_local**self.config.gamma)

    def step(
        self,
        iter_local: int,
        net_wins: int,
        flip: np.ndarray,
        c_k: np.ndarray,
    ) -> None:
        """Update parameters based on the gradient estimate."""
        # Decay learning rate a_k: a_k = a_base / (A + k)^alpha
        a_k = self.a_base / ((self.config.A + iter_local) ** self.config.alpha)

        # Update Rule: step = (a_k / c_k) * net_wins * flip
        step = (a_k / c_k) * net_wins * flip
        self.theta += step
        self.theta = np.clip(self.theta, self.config.param_min, self.config.param_max)


class SFSGD(Optimizer):
    """Schedule-Free SGD implementation."""

    def __init__(self, config: SPSAConfig) -> None:
        """Initialize the Schedule-Free SGD optimizer."""
        super().__init__(config)
        self.z = self.theta.copy()  # Fast iterate
        self.x = self.theta.copy()  # Polyak average
        self.weight_sum = 0.0

        # Constant perturbation scale
        param_range = self.config.param_max - self.config.param_min
        self.c_constant = param_range * self.config.sf_sgd_c_fraction

    def get_perturbation_scale(self, _iter_local: int) -> np.ndarray:
        """Return the perturbation scale (c_k) for the given iteration."""
        return self.c_constant

    def step(
        self,
        iter_local: int,
        net_wins: int,
        flip: np.ndarray,
        c_k: np.ndarray,
    ) -> None:
        """Update parameters using Schedule-Free SGD logic."""
        lr = self.config.sf_sgd_lr

        # Warmup logic
        warmup_end = self.config.num_pairs * self.config.sf_sgd_warmup_fraction
        if warmup_end > 0 and iter_local < warmup_end:
            lr *= iter_local / warmup_end

        beta1 = self.config.sf_sgd_beta1
        batch_size = self.config.batch_size

        # Validation: Reconstruct x from theta and z
        if beta1 > 0:
            x_rec = (self.theta - (1.0 - beta1) * self.z) / beta1
            # We must clamp the reconstruction to match the stored x which is clamped
            x_rec = np.clip(x_rec, self.config.param_min, self.config.param_max)

            # Assert equality
            if not np.allclose(self.x, x_rec, atol=1e-5):
                max_diff = np.max(np.abs(self.x - x_rec))
                msg = f"Reconstruction failed. Max diff: {max_diff}"
                raise AssertionError(msg)

        # Update logic
        delta_total_step = lr * c_k * net_wins * flip

        z_prev = self.z.copy()
        x_prev = self.x.copy()

        self.z += delta_total_step
        # z is NOT clamped

        # Weighting
        weight = lr
        report_weight = weight * batch_size
        weight_sum_prev = self.weight_sum
        weight_sum_curr = weight_sum_prev + report_weight
        self.weight_sum = weight_sum_curr

        if beta1 == 0.0:
            self.theta = self.z.copy()
            # Clamp theta
            self.theta = np.clip(
                self.theta,
                self.config.param_min,
                self.config.param_max,
            )
            return

        # x_new
        tri_factor = (batch_size + 1) / 2.0

        numerator = (
            weight_sum_prev * x_prev
            + report_weight * z_prev
            + weight * delta_total_step * tri_factor
        )
        self.x = numerator / weight_sum_curr

        # Clamp x
        self.x = np.clip(self.x, self.config.param_min, self.config.param_max)

        self.theta = (1.0 - beta1) * self.z + beta1 * self.x

        # Clamp theta
        self.theta = np.clip(self.theta, self.config.param_min, self.config.param_max)

        # Re-sync x with theta to match sf-sgd behavior
        if beta1 > 0:
            self.x = (self.theta - (1.0 - beta1) * self.z) / beta1
            self.x = np.clip(self.x, self.config.param_min, self.config.param_max)


class SFAdam(Optimizer):
    """Schedule-Free Adam implementation."""

    def __init__(self, config: SPSAConfig) -> None:
        """Initialize the Schedule-Free Adam optimizer."""
        super().__init__(config)
        self.z = self.theta.copy()  # Fast iterate
        self.x = self.theta.copy()  # Polyak average
        self.v = np.zeros_like(self.theta)  # Exp. avg of squared gradients
        self.weight_sum = 0.0
        self.t = 0  # Step counter

        # Constant perturbation scale
        param_range = self.config.param_max - self.config.param_min
        self.c_constant = param_range * self.config.sf_adam_c_fraction

    def get_perturbation_scale(self, _iter_local: int) -> np.ndarray:
        """Return the perturbation scale (c_k) for the given iteration."""
        return self.c_constant

    def step(  # noqa: PLR0915
        self,
        iter_local: int,
        net_wins: int,
        flip: np.ndarray,
        c_k: np.ndarray,
    ) -> None:
        """Update parameters using Schedule-Free Adam logic."""
        lr = self.config.sf_adam_lr

        # Warmup logic
        warmup_end = self.config.num_pairs * self.config.sf_adam_warmup_fraction
        if warmup_end > 0 and iter_local < warmup_end:
            lr *= iter_local / warmup_end

        beta1 = self.config.sf_adam_beta1
        beta2 = self.config.sf_adam_beta2
        eps = self.config.sf_adam_eps
        batch_size = self.config.batch_size

        # Validation: Reconstruct x from theta and z
        if beta1 > 0:
            x_rec = (self.theta - (1.0 - beta1) * self.z) / beta1
            x_rec = np.clip(x_rec, self.config.param_min, self.config.param_max)
            if not np.allclose(self.x, x_rec, atol=1e-5):
                max_diff = np.max(np.abs(self.x - x_rec))
                msg = f"Reconstruction failed. Max diff: {max_diff}"
                raise AssertionError(msg)

        # --- Adam-specific complex math for out-of-order/batching ---

        batch_size_val = batch_size
        # micro_steps is the total number of samples processed including this batch
        micro_steps = iter_local + batch_size_val - 1

        # 1. Update Second Moment (v)
        # g_phi_mean is the mean gradient per pair in the batch
        g_phi_mean = (net_wins / batch_size_val) * flip

        if beta2 < 1.0:
            beta2_pow_n = beta2**batch_size_val
            self.v = beta2_pow_n * self.v + (1.0 - beta2_pow_n) * (g_phi_mean**2)

            bc_denom = 1.0 - (beta2**micro_steps)
            v_hat = self.v / bc_denom if bc_denom > TINY_EPSILON else self.v
        else:
            v_hat = self.v

        denom = np.sqrt(v_hat) + eps

        # 2. Calculate Step (delta_total_step)
        # step_phi is the directional step in phi space
        step_phi = (lr * net_wins * flip) / denom

        # Apply damping factor k(N, beta2)
        if batch_size_val > 1 and 0.0 < beta2 < 1.0:
            sqrt_b2 = np.sqrt(beta2)
            if abs(1.0 - sqrt_b2) > SQRT_EPSILON:
                num = 1.0 - (beta2 ** (0.5 * batch_size_val))
                den = batch_size_val * (1.0 - sqrt_b2)
                k_damping = num / den if den != 0.0 else 1.0
            else:
                # Series expansion near beta2 -> 1
                k_damping = 1.0 - ((batch_size_val - 1) * 0.25) * (1.0 - beta2)

            if not (0.0 < k_damping <= 1.0):
                k_damping = 1.0 if k_damping > 1.0 else 1e-6

            step_phi *= k_damping

        delta_total_step = step_phi * c_k

        # 3. Update Fast Iterate (z)
        self.z += delta_total_step
        # z is NOT clamped

        # 4. Update Polyak Average (x)
        # Use simple mass-weighted average for Adam path (no triangular surrogate)
        weight = lr
        report_weight = weight * batch_size_val

        weight_sum_prev = self.weight_sum
        weight_sum_curr = weight_sum_prev + report_weight
        self.weight_sum = weight_sum_curr

        a_k = (report_weight / weight_sum_curr) if weight_sum_curr > 0 else 1.0

        x_prev = self.x
        # Simple Polyak average: x_new = (1 - a_k) * x_prev + a_k * z_new
        # Note: z has already been updated to z_new
        self.x = (1.0 - a_k) * x_prev + a_k * self.z

        # Clamp x
        self.x = np.clip(self.x, self.config.param_min, self.config.param_max)

        self.theta = (1.0 - beta1) * self.z + beta1 * self.x

        # Clamp theta
        self.theta = np.clip(self.theta, self.config.param_min, self.config.param_max)

        # Re-sync x
        if beta1 > 0:
            self.x = (self.theta - (1.0 - beta1) * self.z) / beta1
            self.x = np.clip(self.x, self.config.param_min, self.config.param_max)
