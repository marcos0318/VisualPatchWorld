#!/usr/bin/env python3
"""Aggregate VisualPatchWorld seed-suffixed planning reports.

The evaluator JSON files contain an ``episode_successes`` list of Bernoulli
outcomes. This script reports per-seed Wilson intervals, an across-seed
mean/std, and a pooled Wilson interval over all completed episodes.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path


DEFAULT_REPORT_DIR = (
    Path(__file__).resolve().parents[1]
    / "artifacts" / "visualpatchworld" / "reports"
)

DEFAULT_PATTERNS = {
    "Two-room Tool+Induced": "cross_env_tworoom_tool_live_n50_seed{seed}.json",
    "Two-room Tool+Sim-verify": "cross_env_tworoom_tool_live_hybrid_sv03_n50_seed{seed}.json",
    "Two-room Oracle+Induced": "cross_env_tworoom_n50_seed{seed}.json",
    "Two-room Oracle+Sim-verify": "cross_env_tworoom_hybrid_oracle_sv03_n50_seed{seed}.json",
    "Two-room MuJoCo+CEM": "cross_env_tworoom_sim_cem_fair_n50_seed{seed}.json",
    "Cube Tool+Induced": "cross_env_cube_tool_live_n50_seed{seed}.json",
    "Cube Tool+Sim-verify": "cross_env_cube_tool_live_hybrid_sv03_n50_seed{seed}.json",
    "Cube Oracle+Induced": "cross_env_cube_n50_seed{seed}.json",
    "Cube Oracle+Sim-verify": "cross_env_cube_hybrid_oracle_sv03_n50_seed{seed}.json",
    "Cube MuJoCo+CEM": "cross_env_cube_sim_cem_fair_n50_seed{seed}.json",
    "Reacher Tool+Induced": "cross_env_reacher_tool_live_n50_seed{seed}.json",
    "Reacher Tool+Sim-verify": "cross_env_reacher_tool_live_hybrid_sv03_n50_seed{seed}.json",
    "Reacher Oracle+Induced": "cross_env_reacher_n50_seed{seed}.json",
    "Reacher Oracle+Sim-verify": "cross_env_reacher_hybrid_oracle_sv03_n50_seed{seed}.json",
    "Reacher MuJoCo+CEM": "cross_env_reacher_sim_cem_fair_n50_seed{seed}.json",
    "PushT Tool+Induced": "pusht_tool_live_program_n50_seed{seed}.json",
    "PushT Tool+Sim-verify": "pusht_tool_live_program_hybrid_sv03_n50_seed{seed}.json",
    "PushT Oracle+Induced": "pusht_oracle_program_n50_seed{seed}.json",
    "PushT Oracle+Sim-verify": "pusht_oracle_program_hybrid_sv03_n50_seed{seed}.json",
    "PushT MuJoCo+CEM": "pusht_planning_simulator_horizon1_fair_n50_seed{seed}.json",
}


@dataclass
class SeedResult:
    seed: int
    path: Path
    successes: int
    trials: int

    @property
    def rate(self) -> float:
        return 100.0 * self.successes / self.trials if self.trials else float("nan")


def wilson_ci(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials == 0:
        return float("nan"), float("nan")
    p = successes / trials
    denom = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denom
    margin = z * math.sqrt((p * (1.0 - p) / trials) + (z * z / (4.0 * trials * trials))) / denom
    return 100.0 * (center - margin), 100.0 * (center + margin)


def read_result(path: Path, seed: int) -> SeedResult:
    data = json.loads(path.read_text())
    if "episode_successes" not in data:
        raise KeyError(f"{path} does not contain episode_successes")
    outcomes = [float(x) for x in data["episode_successes"]]
    successes = int(round(sum(outcomes)))
    return SeedResult(seed=seed, path=path, successes=successes, trials=len(outcomes))


def format_pct(value: float) -> str:
    if math.isnan(value):
        return "--"
    return f"{value:.1f}"


def aggregate_condition(
    name: str,
    pattern: str,
    seeds: list[int],
    report_dir: Path,
) -> tuple[str, list[SeedResult]]:
    results = []
    for seed in seeds:
        path = report_dir / pattern.format(seed=seed)
        if path.exists():
            results.append(read_result(path, seed))
    return name, results


def latex_row(name: str, results: list[SeedResult]) -> str:
    if not results:
        return f"{name} & pending & -- & -- \\\\"
    rates = [result.rate for result in results]
    total_successes = sum(result.successes for result in results)
    total_trials = sum(result.trials for result in results)
    lo, hi = wilson_ci(total_successes, total_trials)
    sd = statistics.stdev(rates) if len(rates) > 1 else 0.0
    seeds_done = ",".join(str(result.seed) for result in results)
    return (
        f"{name} & {seeds_done} & "
        f"{statistics.mean(rates):.1f} $\\pm$ {sd:.1f} & "
        f"[{lo:.1f}, {hi:.1f}] \\\\"
    )


def text_block(name: str, results: list[SeedResult]) -> str:
    lines = [name]
    if not results:
        lines.append("  pending")
        return "\n".join(lines)
    for result in results:
        lo, hi = wilson_ci(result.successes, result.trials)
        lines.append(
            "  "
            f"seed {result.seed}: {result.successes}/{result.trials} "
            f"= {result.rate:.1f}% [{lo:.1f}, {hi:.1f}] "
            f"({result.path.name})"
        )
    rates = [result.rate for result in results]
    total_successes = sum(result.successes for result in results)
    total_trials = sum(result.trials for result in results)
    lo, hi = wilson_ci(total_successes, total_trials)
    sd = statistics.stdev(rates) if len(rates) > 1 else 0.0
    lines.append(
        f"  across seeds: mean={statistics.mean(rates):.1f}%, sd={sd:.1f} pp; "
        f"pooled={100.0 * total_successes / total_trials:.1f}% [{lo:.1f}, {hi:.1f}]"
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report_dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument(
        "--format",
        choices=["text", "latex"],
        default="text",
        help="Output text summary or LaTeX table rows.",
    )
    parser.add_argument(
        "--condition",
        action="append",
        default=[],
        metavar="NAME=PATTERN",
        help="Override/add a condition pattern; pattern may contain {seed}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    patterns = dict(DEFAULT_PATTERNS)
    for spec in args.condition:
        if "=" not in spec:
            raise ValueError(f"condition must be NAME=PATTERN, got {spec!r}")
        name, pattern = spec.split("=", 1)
        patterns[name] = pattern

    aggregated = [
        aggregate_condition(name, pattern, seeds, args.report_dir)
        for name, pattern in patterns.items()
    ]
    if args.format == "latex":
        for name, results in aggregated:
            print(latex_row(name, results))
    else:
        for idx, (name, results) in enumerate(aggregated):
            if idx:
                print()
            print(text_block(name, results))


if __name__ == "__main__":
    main()
