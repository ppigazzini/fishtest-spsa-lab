"""Shared fixtures for the fishtest-spsa-lab test suite."""

import random

import pytest


@pytest.fixture
def base_seed() -> int:
    """Deterministic seed shared across validation tests."""
    return 424242


@pytest.fixture
def schedule_params() -> dict:
    """Standard pentanomial schedule parameters."""
    return {
        "num_reports": 100,
        "n_min": 1,
        "n_max": 32,
        "p5": (0.025, 0.20, 0.55, 0.20, 0.025),
    }


@pytest.fixture
def swap_rng(base_seed: int) -> random.Random:
    """RNG for end-adjacent shuffling."""
    return random.Random(base_seed + 1337)  # noqa: S311
