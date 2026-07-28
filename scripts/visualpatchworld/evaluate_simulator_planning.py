#!/usr/bin/env python
"""Evaluate PushT planning with the ground-truth simulator as the dynamics model."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import hdf5plugin  # noqa: F401
except ModuleNotFoundError:
    pass

from visualpatchworld.planning_eval import evaluate_simulator_planning


def _log(message: str) -> None:
    print(f"[visualpatchworld][{datetime.now().isoformat(timespec='seconds')}][sim-planning] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="pusht_expert_train.h5")
    parser.add_argument("--stablewm_home", default=os.environ.get("STABLEWM_HOME"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--num_eval", type=int, default=50)
    parser.add_argument("--goal_offset_steps", type=int, default=25)
    parser.add_argument("--eval_budget", type=int, default=50)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument(
        "--planner",
        choices=["random_shooting", "cem", "cem_mpc", "beam_mpc"],
        default="random_shooting",
        help="cem_mpc=per-step CEM-MPC; beam_mpc=macro beam-MPC (fair vs induced beam_mpc)",
    )
    parser.add_argument("--cem_iters", type=int, default=5)
    parser.add_argument("--elite_frac", type=float, default=0.1)
    parser.add_argument("--action_block", type=int, default=5)
    parser.add_argument("--receding_horizon", type=int, default=5)
    parser.add_argument("--no_warm_start", action="store_true")
    parser.add_argument("--action_low", type=float, default=-1.0)
    parser.add_argument("--action_high", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate_simulator_planning(
        dataset=args.dataset,
        stablewm_home=args.stablewm_home,
        num_eval=args.num_eval,
        goal_offset_steps=args.goal_offset_steps,
        eval_budget=args.eval_budget,
        frameskip=args.frameskip,
        horizon=args.horizon,
        num_samples=args.num_samples,
        planner=args.planner,
        cem_iters=args.cem_iters,
        elite_frac=args.elite_frac,
        action_block=args.action_block,
        receding_horizon=args.receding_horizon,
        warm_start=not args.no_warm_start,
        action_low=args.action_low,
        action_high=args.action_high,
        seed=args.seed,
    )

    output = (
        Path(args.output)
        if args.output
        else ROOT / "artifacts" / "visualpatchworld" / "reports" / "pusht_planning_simulator_ground_truth.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _log(f"Wrote simulator planning report to {output}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
