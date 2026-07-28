#!/usr/bin/env python3
"""Aggregate fair-protocol multi-seed reports under fair_wm_ranking_gt90/all_methods."""

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
    / "fair_wm_ranking_gt90" / "all_methods"
)

# Paper-facing names -> fair JSON patterns (selected refresh + seed-42 priors).
DEFAULT_PATTERNS = {
    "Two-room Tool+Hybrid": "vpw_tool_hybrid_tworoom_n50_seed{seed}.json",
    "Two-room Oracle+Hybrid": "vpw_oracle_hybrid_tworoom_n50_seed{seed}.json",
    "Two-room MuJoCo+CEM": "mujoco_cem_tworoom_n50_seed{seed}.json",
    "PushT Tool+Induced": "vpw_tool_induced_pusht_n50_seed{seed}.json",
    "PushT Tool+Hybrid": "vpw_tool_hybrid_pusht_n50_seed{seed}.json",
    "Cube Tool+Induced": "vpw_tool_induced_cube_n50_seed{seed}.json",
    "Cube Tool+Hybrid": "vpw_tool_hybrid_cube_n50_seed{seed}.json",
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
            f"  seed {result.seed}: {result.rate:.1f}% "
            f"({result.successes}/{result.trials}) [{lo:.1f}, {hi:.1f}]"
        )
    rates = [r.rate for r in results]
    tot_s = sum(r.successes for r in results)
    tot_n = sum(r.trials for r in results)
    lo, hi = wilson_ci(tot_s, tot_n)
    sd = statistics.stdev(rates) if len(rates) > 1 else 0.0
    lines.append(
        f"  pooled: {statistics.mean(rates):.1f} ± {sd:.1f}  "
        f"Wilson [{lo:.1f}, {hi:.1f}] over {tot_n} trials"
    )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report_dir", type=Path, default=DEFAULT_REPORT_DIR)
    p.add_argument("--seeds", default="42,43,44,45,46")
    p.add_argument("--format", choices=("text", "latex"), default="text")
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    blocks = []
    for name, pattern in DEFAULT_PATTERNS.items():
        results = []
        for seed in seeds:
            path = args.report_dir / pattern.format(seed=seed)
            if path.exists():
                try:
                    results.append(read_result(path, seed))
                except KeyError as exc:
                    print(f"warn: {exc}")
        if args.format == "latex":
            blocks.append(latex_row(name, results))
        else:
            blocks.append(text_block(name, results))

    print("\n".join(blocks))


if __name__ == "__main__":
    main()
