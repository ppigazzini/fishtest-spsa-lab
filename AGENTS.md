# AGENTS.md

This repository is a laboratory for the SPSA tuner used by
[fishtest](https://tests.stockfishchess.org). Its output is *claims about SPSA*,
not shipped software. A claim lands only with the seeds, the budget and the
uncertainty behind it; a number without them is not a result.

**Read [README.md](README.md) for what the tools are, and [docs/](docs/) for the
derivations.** This file is only what an agent gets wrong before it has read
either.

Two references sit on disk next to this repo and settle every factual question
about the real system. Read them rather than recalling.

| clone | what it settles |
|---|---|
| `__fishtest` -> `../fishtest` | the real update rule (`server/fishtest/spsa_workflow.py`, `spsa_handler.py`), the real batch sizing (`worker/games.py`), task sizing (`rundb.py :: worker_cap`) |
| `../spsa_simul` | M. Van den Bergh's simulator and `doc/theoretical_basis.pdf` -- the closest prior art for the noise-ball theory |
| `../Stockfish` | the engine whose parameters real tunes target |

They are read-only. Never commit into them.

## Setup

Dev tools are **not** installed by default. `pyproject.toml` sets
`[tool.uv] default-groups = []`, so a bare `uv run pytest` fails with a
resolution error that looks like a broken environment and is not.

```sh
uv sync --group dev
uv run --group dev pytest -q          # 17 tests
uv run --group dev ruff check .
uv run --group dev ruff format --check .
uv run --group dev ty check
uv run --group dev pre-commit run --all-files
```

The lab's tools are the thirteen console scripts in `[project.scripts]`. Five
parse arguments -- `noise-ball`, `optimize-spsa-toy`, `validate-penta`,
`rademacher`, `spsa-gradient`. The other eight ignore `--help` and run instead.

```sh
uv run run-simulation        # 12-optimizer x 8-seed sweep, paired table, ~45 s
uv run noise-ball            # the stationary Elo floor
uv run validate-spsa         # macro-vs-micro equivalence, classic SPSA
uv run optimize-spsa-toy     # toy optimizer and spectrum tools
uv run plot-spsa-schedule    # c_k, a_k, r_k
```

The seven `validate-*` entry points print a result table and **return a real
exit code**. `SPSA_LAB_NO_PLOT=1` suppresses figures, as does any
non-interactive matplotlib backend, so they are usable as gates; `pytest` runs
all seven. `run-simulation` draws no plots -- the rest do.

## The anchors

Three invariants hold the repo up. A change that moves one is a finding, not a
detail.

**1. Macro equals micro.** One batched update must equal N sequential
micro-updates wherever the dynamics are linear. This is the property that makes
a block-reporting distributed tuner equivalent to the textbook algorithm, and
`analysis/validate_*.py` exists to pin it. These are the real regression suite;
`tests/` is thinner than it looks.

**2. The noise ball.** `docs/Noise_ball.md` derives a stationary Elo floor from
a rank-1 second-moment recursion, ending at `D_inf = k * S_*`; `analysis/
noise_ball.py :: estimate_noise_ball_isotropic_end` computes it. Taking the
small-gain limit, `k` and `c` cancel and it reduces to

```text
E[Elo drop] = -(r_end / 16) * C * n_active * sigma2,    C = 800/ln(10)
```

which is what `../spsa_simul` proves from an SDE, and the two agree to 1 part in
10,000 across N = 4..64. **Neither the page nor the module states that reduced
form or the `n_active` distinction** -- both are results from
`__DEV/260809-0-REPORT.md`, not from the code. Any change to
`analysis/noise_ball.py` that breaks the agreement is wrong until proven
otherwise.

**3. The Fishtest protocol.** Dispatch snapshots `theta` and `flip`; the result
applies to whatever `theta` is current at arrival; `iter` counts pairs and
advances **only on arrival**; probes and updates are both clipped to the
developer bounds.

The lab deviates from Fishtest's *gain* on purpose -- `runner.py` halves the
signal and scales by `1/sqrt(N)`, together `1/(2*sqrt(n))` of the server's step.
That is a proposal, derived in `docs/Algorithms.md` ch. 4 and
`docs/Rademacher.md`, not a bug. **Deviating from the protocol is a bug.** Keep
the two straight in code and in the commit message.

## Gates: which check answers which claim

`pytest` proves the code runs. It does not prove the math is right or the
experiment means anything.

| the change claims | gate |
|---|---|
| "the update rule is unchanged" | the matching `validate-*` entry point AND `tests/test_optimizer_parity.py`, which is the only thing comparing the shipped optimizer to its clean-room twin |
| "this optimizer is better" | multiple seeds, shared worker pool, paired difference, stated CI |
| "this is the stationary loss" | `noise-ball`, cross-checked against the closed form above |
| "this matches Fishtest" | drive `__fishtest/server/fishtest/spsa_workflow.py` over the same scripted sequence |
| "this is a pure refactor" | every `validate-*` produces byte-identical **stdout** and exits 0 |

**Check the gate's EXIT CODE, never a piped fragment.** `cmd | tail -1` reads 0
from `tail` while the gate is red. A check that was skipped proves nothing;
never report it as a pass.

## Traps that cost real time

| trap | detail |
|---|---|
| **`run-simulation` is unseeded and runs each optimizer once.** `SPSAConfig.seed` defaults to `None` and `main()` never sets it. Measured: between-optimizer spread 0.058 Elo against a per-seed std of 0.034-0.055. At seed 7 `sf-adam-block` wins and `sf-sgd` loses; over 8 seeds the ranking **inverts**. It does not have wide error bars, it gives the wrong answer. | never quote it as evidence |
| **The default geometry is internally inconsistent.** `start_elo = -0.5` says the tune is worth half an Elo; `c_elo_gap = 2.0` asks each probe to cost two. The derived `c_dev = 748` against a distance-to-optimum of 100, with bounds `[-6583, 8383]`. Nothing crashes, clipping never engages, the numbers just do not describe a tune. | `simulator/config.py :: __post_init__` |
| **The defaults are 0.10x the gain their own theory prescribes**, on a budget of 60,000 games where `6*max_j lambda_j` is 305k-1.22M. Nothing converges, and every optimizer looks alike. That is the predicted outcome, not a finding about optimizers. | size the run before running it |
| **The noise ball counts Elo-active parameters, not perturbed ones.** An inactive parameter (`w_true == 0`) random-walks but contributes nothing. Measured: adding 28 dead parameters changes the stationary loss not at all, while an `n_total` prediction is wrong by 11x. | `n_active`, always |
| **The lab advances `iter` at dispatch; Fishtest advances it at arrival.** With 20 workers the schedule leads by the in-flight window where the server lags, and the `out_of_order_stats` telemetry measures a quantity the server does not have. | `runner.py :: _schedule_job` |
| **`../spsa_simul` is prior art, not an oracle.** Its asymptotic covariance is printed as `(r^2*sigma2/4)*A^-1`; solving the Lyapunov equation gives `(r*sigma2/4)*E^-1`, which is what its own diagonal section uses. The "if E and A commute" hedge is both insufficient (it is wrong in the commuting case too) and unnecessary (the correct form satisfies the equation identically). Both theorems survive. | verify before adopting |
| **Decay is not useless.** `spsa_simul` argues that since convergence is unreachable, constant `r` is enough. The noise floor is proportional to the *current* gain, so a decaying gain shrinks the ball with the iterations. Measured from the optimum: decay is worse early, crosses the constant arm near 90k pairs, ends below it and is still falling. Constant wins on short budgets, decay on long ones. | do not repeat the claim |
| **`ruff`'s defaults move under you.** 0.15.9 -> 0.16.2 took the default rule set from 59 rules to 413, turning a clean tree into 80 findings with no source change. The selection is pinned in `pyproject.toml` for that reason. Widening it is a decision, not a bump. | `[tool.ruff.lint] select` |
| **Do not delete `noqa` comments to satisfy `RUF100`.** They document intent (`ARG002 - kept for unified API`) and are needed the moment `ARG`/`PLR` are enabled. | ask first |
| **`ruff format` rewrites Python inside Markdown.** Only ` ```python ` fences; ` ```text `, ` ```bash `, ` ```c ` are untouched. Keep pseudo-math in ` ```text ` or the formatter will try to parse `theta_{k+1}[i]`. | `docs/`, `__DEV/` are in scope |
| **Pre-commit cannot fail on lint.** `ruff-check` runs with `--fix --exit-zero`. Green hooks are not evidence that lint is clean. There is no CI. | run the checks yourself |
| **`analysis/common.py` imports from `analysis/validate_variance.py`** -- the shared module depends on a leaf script. Do not "fix" it casually; it is load-bearing for five validators. | tracked as M6 |
| **`mu2_hat`/`update_mu2_stats` exist three times** (`analysis/common.py`, `simulator/optimizer.py::SFAdamBlock`, `validate_sf_adam.py`), identical down to the `min(max(mu2, 1e-12), 4.0)` clamp. Change one and the others silently diverge. | one edit is never enough |
| **The `validate-*` scripts could not fail.** Until 2026-08-16 five of seven printed nothing at all and none could return a non-zero exit, so "byte-identical output" compared empty against empty. Fixed; the lesson is that a gate is worth nothing until it has been *observed* to fail, so change the math on purpose and watch it go red. | `analysis/gate.py` |
| **A gate can drift from the code it gates.** `analysis/validate_*.py` reimplements each rule independently, so editing `simulator/optimizer.py` alone cannot fail it. Two divergences shipped that way in one commit. `tests/test_optimizer_parity.py` now compares the two directly -- run it after touching either side. | one edit is never enough |
| **The Adam family has no per-parameter adaptivity.** `grad = scalar * flip` with `flip` in `{-1,+1}` makes `grad**2` coordinate-identical, so `v.max() - v.min()` is exactly 0 for `adam`, `adam-block`, `sf-adam`, `sf-adam-block` and `ademamix`. All five are normalized-momentum SGD. Never attribute a result to adaptivity. | `docs/Simulator.md` 2.8 |
| **An optimizer must never read ground truth.** `w_true`, `theta_peak` and `k_elo` are the simulator's, not the tuner's. `w_dev` and `c_dev` are the developer's belief. Leaking the first set makes every result meaningless and nothing will complain. | `simulator/config.py` |
| **Verify every citation against the source.** ID, full author list, year, and that the live page still shows the cited title. Five entries in `docs/Algorithms.md` were wrong at once: Defazio et al. and Ahn/Magakyan/Cutkosky both carried the same fabricated five-author list; two more had wrong author order and one a wrong title; Pun/Buchholz/Gower's link had come to resolve to a different paper. A prior prompt also supplied three wrong IDs, and a whole entry was once admitted on the strength of its title matching the problem rather than the oracle. Cite by name here, not by number -- the list is chronological and renumbers on insert. | never cite from memory |
| **The "ASCII-only" rule is dead as written.** `__DEV/DOCS-BEST-PRACTICES.md:26` states it; all nine `docs/` pages break it, using theta/phi/sigma/Delta and the usual math operators. Prose and this file stay ASCII; mathematical notation in `docs/` does not. The standard needs amending, not the pages. | unresolved |

## Experiments

- Fix the seed. State it. Two runs with the same seed must be bit-identical.
- Share the worker pool across arms; compare paired differences, not two means.
- Size the budget from `lambda_j = C / (8*r*c_j^2*eps_j)` before running, and say
  so when the budget cannot separate the arms.
- Scratch scripts live outside the package. If an experiment earns a permanent
  place it becomes a test or an `analysis/` entry point with a console script.
- Long sweeps run in the background; a full sweep is minutes to tens of minutes.
- Findings land in a dated `__DEV/` report. **Never edit a historical one.**
  `__DEV/` has never been tracked by git -- it is local by design.

## Commits

**One logical change per commit.** Conventional Commits v1.0.0: the subject is
`type(scope): description`, 72 characters or fewer, lower case after the colon,
no full stop.

| type | use for |
|---|---|
| `feat` | a capability the lab did not have -- an optimizer, an entry point |
| `fix` | a defect: wrong math, a broken protocol match, a wrong citation |
| `refactor` | behaviour-preserving structure; every `validate-*` output identical |
| `perf` | a change whose claim is speed -- carries its measurement |
| `test` | anything under `tests/` |
| `docs` | `docs/`, `README.md`, `AGENTS.md`, comment-only source changes |
| `build` | `pyproject.toml`, `uv.lock`, the pre-commit config |
| `chore` | dependency bumps |

Scope is the package -- `simulator`, `analysis` -- or a module, omitted when the
change is not confined to one.

Body wrapped at 80, authoritative mood, carrying the evidence: the commands run
and their exit codes, not "should work". A change to an update rule states what
moved and what pinned it. No meta-commentary about the process that produced the
change.

**No footer names a non-author.** Never a `Co-Authored-By:` for a tool or an
assistant, and never a generated-by advertisement of any kind: a footer naming a
non-author is a false claim about who wrote the change, and every blame view
repeats it forever. Configure tooling that appends one by default not to, rather
than stripping it in a later rewrite.

**Don't** `git push` -- commit locally and stop unless asked. Never edit
`src/fishtest_spsa_lab/vendor/`. Never run destructive commands unless asked.
