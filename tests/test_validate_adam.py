"""Test Adam macro vs micro correctness (harvested from validate_adam.py)."""

from fishtest_spsa_lab.analysis.common import make_schedule, series_allclose
from fishtest_spsa_lab.analysis.validate_adam import (
    run_macro_const_mean_adam,
    run_micro_const_mean_adam,
    run_micro_real_adam,
)

# Hyper-parameters matching the main() defaults
LR = 0.01
BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-8


def test_macro_const_mean_equals_micro_const_mean(
    base_seed: int, schedule_params: dict
) -> None:
    _, outcomes = make_schedule(
        schedule_params["num_reports"],
        schedule_params["n_min"],
        schedule_params["n_max"],
        schedule_params["p5"],
        base_seed,
    )

    micro_mean = run_micro_const_mean_adam(
        outcomes, lr=LR, beta1=BETA1, beta2=BETA2, eps=EPS
    )
    macro_const = run_macro_const_mean_adam(
        outcomes, lr=LR, beta1=BETA1, beta2=BETA2, eps=EPS
    )

    assert micro_mean.t_pairs == macro_const.t_pairs, "time axes differ"
    assert series_allclose(micro_mean.theta, macro_const.theta), (
        "macro const-mean != micro const-mean"
    )


def test_time_axes_consistent(base_seed: int, schedule_params: dict) -> None:
    _, outcomes = make_schedule(
        schedule_params["num_reports"],
        schedule_params["n_min"],
        schedule_params["n_max"],
        schedule_params["p5"],
        base_seed,
    )

    micro_real = run_micro_real_adam(outcomes, lr=LR, beta1=BETA1, beta2=BETA2, eps=EPS)
    micro_mean = run_micro_const_mean_adam(
        outcomes, lr=LR, beta1=BETA1, beta2=BETA2, eps=EPS
    )

    assert micro_real.t_pairs == micro_mean.t_pairs, "time axes differ"
