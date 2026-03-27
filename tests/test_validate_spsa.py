"""Test SPSA macro vs micro correctness (harvested from validate_spsa.py)."""

from fishtest_spsa_lab.analysis.common import (
    build_sequence,
    compute_a_from_outcomes,
    end_adjacent_shuffle,
    make_schedule,
    series_allclose,
)
from fishtest_spsa_lab.analysis.validate_spsa import (
    SpsaSchedule,
    run_macro_corrected,
    run_macro_uncorrected,
    run_micro,
)


def _make_sched(outcomes_by_report: list[list[int]]) -> SpsaSchedule:
    a_val = compute_a_from_outcomes(outcomes_by_report)
    return SpsaSchedule(a=0.1, a_stability=a_val, alpha=0.602, c=1.0, gamma=0.101)


def test_corrected_macro_equals_micro_mean(
    base_seed: int, schedule_params: dict
) -> None:
    _, outcomes = make_schedule(
        schedule_params["num_reports"],
        schedule_params["n_min"],
        schedule_params["n_max"],
        schedule_params["p5"],
        base_seed,
    )
    sched = _make_sched(outcomes)

    macro_cor = run_macro_corrected(outcomes, sched=sched)
    seqs_mean = [build_sequence(outs, "const_mean") for outs in outcomes]
    micro_mean = run_micro(seqs_mean, sched=sched)

    assert macro_cor.t_pairs == micro_mean.t_pairs, "time axes differ"
    assert series_allclose(macro_cor.theta, micro_mean.theta), (
        "corrected macro != micro const-mean"
    )


def test_corrected_macro_equals_micro_mean_shuffled(
    base_seed: int, schedule_params: dict, swap_rng
) -> None:
    _, outcomes = make_schedule(
        schedule_params["num_reports"],
        schedule_params["n_min"],
        schedule_params["n_max"],
        schedule_params["p5"],
        base_seed,
    )

    idx = end_adjacent_shuffle(
        list(range(schedule_params["num_reports"])),
        p=4.0 / 5.0,
        rng=swap_rng,
    )
    outcomes_shuf = [outcomes[i] for i in idx]
    sched = _make_sched(outcomes_shuf)

    macro_cor = run_macro_corrected(outcomes_shuf, sched=sched)
    seqs_mean = [build_sequence(outs, "const_mean") for outs in outcomes_shuf]
    micro_mean = run_micro(seqs_mean, sched=sched)

    assert macro_cor.t_pairs == micro_mean.t_pairs, "time axes differ (shuffled)"
    assert series_allclose(macro_cor.theta, micro_mean.theta), (
        "corrected macro != micro const-mean (shuffled)"
    )


def test_time_axes_consistent(base_seed: int, schedule_params: dict) -> None:
    _, outcomes = make_schedule(
        schedule_params["num_reports"],
        schedule_params["n_min"],
        schedule_params["n_max"],
        schedule_params["p5"],
        base_seed,
    )
    sched = _make_sched(outcomes)

    macro_cor = run_macro_corrected(outcomes, sched=sched)
    macro_unc = run_macro_uncorrected(outcomes, sched=sched)
    seqs_mean = [build_sequence(outs, "const_mean") for outs in outcomes]
    seqs_real = [build_sequence(outs, "outcomes") for outs in outcomes]
    micro_mean = run_micro(seqs_mean, sched=sched)
    micro_real = run_micro(seqs_real, sched=sched)

    assert (
        macro_cor.t_pairs
        == micro_mean.t_pairs
        == micro_real.t_pairs
        == macro_unc.t_pairs
    ), "time axes differ"
