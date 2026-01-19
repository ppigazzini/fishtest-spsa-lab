#!/usr/bin/env python3
"""Rademacher Monte Carlo: sqrt(N) step size and 1/sqrt(N) alignment.

This is a small, self-contained simulation to sanity-check the two
high-dimensional scalings used throughout the toy SPSA discussion:

1) If delta is in {+1,-1}^N and the per-coordinate step is proportional to delta,
   then the L2 step size grows like sqrt(N).

2) If g is a fixed direction (think: true gradient) and delta is a random
   Rademacher direction, then the *typical* cosine alignment shrinks like
   1/sqrt(N):

   - cos_raw = <g,delta> / (||g|| * ||delta||) has mean 0 and RMS ~ 1/sqrt(N)
   - cos_update = |<g,delta>| / (||g|| * ||delta||) has positive mean
     ~ sqrt(2/pi)/sqrt(N)

Run:
  uv run python -m fishtest_spsa_lab.analysis.rademacher

Examples:
  uv run python -m fishtest_spsa_lab.analysis.rademacher --samples 200000
  uv run python -m fishtest_spsa_lab.analysis.rademacher --n 8 16 32 64 128 --seed 1
  uv run python -m fishtest_spsa_lab.analysis.rademacher --plot
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Stats:
    n: int
    samples: int
    mean_step_norm: float
    mean_step_norm_normalized: float
    rms_cos_raw: float
    mean_abs_cos_update: float
    se_rms_cos_raw: float
    se_mean_abs_cos_update: float


def _make_unit_vector(*, n: int, rng: np.random.Generator, kind: str) -> np.ndarray:
    if n <= 0:
        raise ValueError("n must be positive")

    if kind == "ones":
        g = np.ones(n, dtype=np.float64)
    elif kind == "gaussian":
        g = rng.normal(loc=0.0, scale=1.0, size=n).astype(np.float64)
    else:
        raise ValueError(f"unknown g kind: {kind!r}")

    norm = float(np.linalg.norm(g))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("failed to build a finite non-zero g")
    return g / norm


def _simulate_for_n(
    *,
    n: int,
    samples: int,
    rng: np.random.Generator,
    g_kind: str,
    c: float,
    scalar_m: float,
    chunk_size: int,
) -> Stats:
    if samples <= 0:
        raise ValueError("samples must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    g = _make_unit_vector(n=n, rng=rng, kind=g_kind)

    # Step vector (noiseless, constant scalar multiplier):
    #   step = (scalar_m * c) * delta
    # so ||step||_2 is deterministic = |scalar_m*c| * sqrt(N).
    sqrt_n = math.sqrt(float(n))
    step_norm = abs(float(scalar_m) * float(c)) * sqrt_n
    step_norm_normalized = abs(float(scalar_m) * float(c))

    # Monte Carlo for dot products (and derived cosines).
    # We batch to avoid allocating samples x n for large settings.
    sum_cos2 = 0.0
    sum_abs_cos = 0.0
    sum_cos4 = 0.0
    sum_abs_cos2 = 0.0

    remaining = int(samples)
    while remaining > 0:
        b = min(int(chunk_size), remaining)
        # delta: shape [b, n], entries in {-1, +1}
        # Use int8 to keep memory down; promote to float for dot.
        delta = rng.integers(low=0, high=2, size=(b, n), dtype=np.int8)
        delta = (2 * delta - 1).astype(np.float64, copy=False)

        dots = delta @ g  # shape [b]
        # Since ||g||=1 and ||delta||=sqrt(N):
        cos_raw = dots / sqrt_n
        abs_cos_update = np.abs(cos_raw)

        cos2 = cos_raw * cos_raw
        sum_cos2 += float(np.sum(cos2))
        sum_cos4 += float(np.sum(cos2 * cos2))

        sum_abs_cos += float(np.sum(abs_cos_update))
        sum_abs_cos2 += float(np.sum(abs_cos_update * abs_cos_update))

        remaining -= b

    s = float(samples)
    mean_cos2 = sum_cos2 / s
    rms_cos_raw = math.sqrt(max(0.0, mean_cos2))

    mean_abs_cos_update = sum_abs_cos / s

    # Standard errors via delta method on second moments.
    # For RMS: sqrt(E[X]) with X=cos^2.
    var_cos2 = max(0.0, sum_cos4 / s - mean_cos2 * mean_cos2)
    se_mean_cos2 = math.sqrt(var_cos2 / s)
    se_rms_cos_raw = 0.0 if rms_cos_raw == 0.0 else 0.5 * se_mean_cos2 / rms_cos_raw

    var_abs_cos = max(0.0, sum_abs_cos2 / s - mean_abs_cos_update * mean_abs_cos_update)
    se_mean_abs_cos_update = math.sqrt(var_abs_cos / s)

    return Stats(
        n=n,
        samples=samples,
        mean_step_norm=step_norm,
        mean_step_norm_normalized=step_norm_normalized,
        rms_cos_raw=rms_cos_raw,
        mean_abs_cos_update=mean_abs_cos_update,
        se_rms_cos_raw=se_rms_cos_raw,
        se_mean_abs_cos_update=se_mean_abs_cos_update,
    )


def _fmt_pm(x: float, se: float) -> str:
    if not math.isfinite(x) or not math.isfinite(se):
        return str(x)
    return f"{x:.6g} ± {se:.2g}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Monte Carlo demo of Rademacher scaling: ||delta|| ~ sqrt(N) and "
            "typical cosine alignment ~ 1/sqrt(N)."
        )
    )
    parser.add_argument(
        "--n",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32, 64, 128, 256, 512],
        help="Dimensions N to simulate.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=200_000,
        help="Monte Carlo samples per N.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--g",
        dest="g_kind",
        choices=["ones", "gaussian"],
        default="gaussian",
        help="How to choose the fixed direction g (always normalized to ||g||=1).",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=1.0,
        help="Per-coordinate probe radius multiplier used in the step demo.",
    )
    parser.add_argument(
        "--m",
        type=float,
        default=1.0,
        help="Scalar multiplier used in the step demo.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50_000,
        help="Batch size to limit memory.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot log-log scaling curves (requires matplotlib).",
    )

    args = parser.parse_args(argv)

    rng = np.random.default_rng(int(args.seed))

    ns = [int(x) for x in args.n]
    if any(x <= 0 for x in ns):
        raise ValueError("all N must be positive")

    rows: list[Stats] = []
    for n in ns:
        rows.append(
            _simulate_for_n(
                n=n,
                samples=int(args.samples),
                rng=rng,
                g_kind=str(args.g_kind),
                c=float(args.c),
                scalar_m=float(args.m),
                chunk_size=int(args.chunk_size),
            )
        )

    # Print a compact table.
    print("Rademacher scaling Monte Carlo")
    print(f"  samples per N: {int(args.samples)}")
    print(f"  g: {args.g_kind} (normalized to ||g||=1)")
    print(
        f"  step demo: step = (m*c)*delta with m={float(args.m):g}, c={float(args.c):g}"
    )
    print("")

    header = (
        "N",
        "E||step||",
        "theory |m*c|*sqrt(N)",
        "E||step|| (scalar*1/sqrt(N))",
        "RMS(cos_raw)",
        "theory 1/sqrt(N)",
        "E[cos_update]",
        "theory sqrt(2/pi)/sqrt(N)",
    )
    print(
        f"{header[0]:>6}  {header[1]:>12}  {header[2]:>20}  {header[3]:>30}  "
        f"{header[4]:>14}  {header[5]:>16}  {header[6]:>16}  {header[7]:>26}"
    )

    for st in rows:
        n = int(st.n)
        sqrt_n = math.sqrt(float(n))
        theory_step = abs(float(args.m) * float(args.c)) * sqrt_n
        theory_rms = 1.0 / sqrt_n
        theory_mean_abs = math.sqrt(2.0 / math.pi) / sqrt_n

        print(
            f"{n:6d}  "
            f"{st.mean_step_norm:12.6g}  "
            f"{theory_step:20.6g}  "
            f"{st.mean_step_norm_normalized:30.6g}  "
            f"{_fmt_pm(st.rms_cos_raw, st.se_rms_cos_raw):>14}  "
            f"{theory_rms:16.6g}  "
            f"{_fmt_pm(st.mean_abs_cos_update, st.se_mean_abs_cos_update):>16}  "
            f"{theory_mean_abs:26.6g}"
        )

    if args.plot:
        import matplotlib.pyplot as plt

        ns_f = np.asarray([float(st.n) for st in rows], dtype=np.float64)
        sqrt_ns = np.sqrt(ns_f)
        rms = np.asarray([float(st.rms_cos_raw) for st in rows], dtype=np.float64)
        mean_abs = np.asarray(
            [float(st.mean_abs_cos_update) for st in rows], dtype=np.float64
        )

        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))

        axes[0].set_title("Step L2 norm scaling")
        axes[0].plot(
            ns_f,
            abs(float(args.m) * float(args.c)) * sqrt_ns,
            label="theory |m*c|*sqrt(N)",
        )
        axes[0].set_xscale("log")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("N")
        axes[0].set_ylabel("||step||")
        axes[0].grid(True, which="both", alpha=0.25)
        axes[0].legend()

        axes[1].set_title("Cosine scaling")
        axes[1].plot(ns_f, rms, marker="o", label="MC RMS(cos_raw)")
        axes[1].plot(ns_f, 1.0 / sqrt_ns, linestyle="--", label="theory 1/sqrt(N)")
        axes[1].plot(ns_f, mean_abs, marker="o", label="MC E[cos_update]")
        axes[1].plot(
            ns_f,
            math.sqrt(2.0 / math.pi) / sqrt_ns,
            linestyle="--",
            label="theory sqrt(2/pi)/sqrt(N)",
        )
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("N")
        axes[1].set_ylabel("cosine")
        axes[1].grid(True, which="both", alpha=0.25)
        axes[1].legend()

        fig.tight_layout()
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
