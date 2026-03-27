"""Test schedule-free SGD macro vs micro correctness (harvested from validate_sf_sgd.py)."""

from fishtest_spsa_lab.analysis.common import (
    build_sequence,
    make_schedule,
    series_allclose,
)
from fishtest_spsa_lab.analysis.validate_sf_sgd import run_macro, run_micro

# Hyper-parameters matching the main() defaults
LR = 0.1
BETA = 0.9
C = 0.5


def test_macro_equals_micro_mean(base_seed: int, schedule_params: dict) -> None:
    _, outcomes = make_schedule(
        schedule_params["num_reports"],
        schedule_params["n_min"],
        schedule_params["n_max"],
        schedule_params["p5"],
        base_seed,
    )

    macro = run_macro(outcomes, lr=LR, beta=BETA, c=C)
    seqs_mean = [build_sequence(outs, "const_mean") for outs in outcomes]
    micro_mean = run_micro(seqs_mean, lr=LR, beta=BETA, c=C)

    assert macro.t_pairs == micro_mean.t_pairs, "time axes differ"
    assert series_allclose(macro.x, micro_mean.x), "macro x != micro const-mean x"
    assert series_allclose(macro.z, micro_mean.z), "macro z != micro const-mean z"
    assert series_allclose(macro.theta, micro_mean.theta), (
        "macro theta != micro const-mean theta"
    )


def test_time_axes_consistent(base_seed: int, schedule_params: dict) -> None:
    _, outcomes = make_schedule(
        schedule_params["num_reports"],
        schedule_params["n_min"],
        schedule_params["n_max"],
        schedule_params["p5"],
        base_seed,
    )

    macro = run_macro(outcomes, lr=LR, beta=BETA, c=C)
    seqs_mean = [build_sequence(outs, "const_mean") for outs in outcomes]
    seqs_real = [build_sequence(outs, "outcomes") for outs in outcomes]
    micro_mean = run_micro(seqs_mean, lr=LR, beta=BETA, c=C)
    micro_real = run_micro(seqs_real, lr=LR, beta=BETA, c=C)

    assert macro.t_pairs == micro_mean.t_pairs == micro_real.t_pairs, "time axes differ"
