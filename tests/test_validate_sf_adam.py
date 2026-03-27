"""Test schedule-free Adam macro vs micro correctness (harvested from validate_sf_adam.py)."""

from fishtest_spsa_lab.analysis.common import (
    compute_pentanomial_moments,
    end_adjacent_shuffle,
    make_schedule,
)
from fishtest_spsa_lab.analysis.validate_sf_adam import (
    InitStats,
    build_const_mean_online_sequences,
    build_sequence,
    run_macro,
    run_micro,
)

# Hyper-parameters matching the main() defaults
LR = 0.1
BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-8
C = 0.5
MU2_INIT = 1.0
PRIOR_P5: tuple[float, float, float, float, float] = (0.05, 0.20, 0.50, 0.20, 0.05)
PRIOR_REPORTS = 5.0


def _make_init_stats(n_min: int, n_max: int) -> InitStats:
    prior_mean_n = (n_min + n_max) / 2.0
    mu_p, _mu2_p, var_p = compute_pentanomial_moments(PRIOR_P5)
    return InitStats(
        reports=PRIOR_REPORTS,
        sum_n=PRIOR_REPORTS * prior_mean_n,
        sum_s=PRIOR_REPORTS * prior_mean_n * mu_p,
        sum_s2_over_n=PRIOR_REPORTS * (var_p + prior_mean_n * (mu_p * mu_p)),
    )


def test_time_axes_shuffled(base_seed: int, schedule_params: dict, swap_rng) -> None:
    _, outcomes = make_schedule(
        schedule_params["num_reports"],
        schedule_params["n_min"],
        schedule_params["n_max"],
        schedule_params["p5"],
        base_seed,
    )
    init_stats = _make_init_stats(schedule_params["n_min"], schedule_params["n_max"])

    idx = end_adjacent_shuffle(
        list(range(schedule_params["num_reports"])),
        p=4.0 / 5.0,
        rng=swap_rng,
    )
    outcomes_shuf = [outcomes[i] for i in idx]

    macro = run_macro(
        outcomes_shuf,
        lr=LR,
        beta1=BETA1,
        beta2=BETA2,
        eps=EPS,
        c=C,
        mu2_init=MU2_INIT,
        init_stats=init_stats,
    )
    seqs_mean = build_const_mean_online_sequences(
        outcomes_shuf, MU2_INIT, init_stats=init_stats
    )
    seqs_real = [build_sequence(outs, "outcomes") for outs in outcomes_shuf]
    micro_mean = run_micro(seqs_mean, lr=LR, beta1=BETA1, beta2=BETA2, eps=EPS, c=C)
    micro_real = run_micro(seqs_real, lr=LR, beta1=BETA1, beta2=BETA2, eps=EPS, c=C)

    assert macro.t_pairs == micro_mean.t_pairs == micro_real.t_pairs, (
        "time axes differ (shuffled)"
    )
