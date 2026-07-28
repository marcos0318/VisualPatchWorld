#!/usr/bin/env python
"""Evaluate induced PatchWorld models with LeWM-style PushT planning."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PATCHWORLD_ROOT = ROOT  # vendored package at repo root
pythonpath_parts = [str(PATCHWORLD_ROOT), str(SRC)]
if os.environ.get("PYTHONPATH"):
    pythonpath_parts.append(os.environ["PYTHONPATH"])
os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(PATCHWORLD_ROOT) not in sys.path:
    sys.path.insert(0, str(PATCHWORLD_ROOT))

try:
    import hdf5plugin  # noqa: F401
except ModuleNotFoundError:
    pass

from visualpatchworld.paths import resolve_vlm_jsonl
from visualpatchworld.planning_eval import evaluate_planning
from visualpatchworld.sim_eval import load_induced_model, resolve_generated_model_path


def _log(message: str) -> None:
    print(f"[visualpatchworld][{datetime.now().isoformat(timespec='seconds')}][planning] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["oracle", "vlm", "tool"], default="vlm")
    parser.add_argument("--dataset", default="pusht_expert_train.h5")
    parser.add_argument("--stablewm_home", default=os.environ.get("STABLEWM_HOME"))
    parser.add_argument("--results_dir", default=str(ROOT / "artifacts" / "visualpatchworld" / "lewm_fair_results"))
    parser.add_argument("--data_dir", default=str(ROOT / "artifacts" / "visualpatchworld" / "lewm_fair_data"))
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--trajectories_jsonl", default=None)
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
        help="cem_mpc=continuous CEM-MPC; beam_mpc=discrete contact-aware beam search MPC",
    )
    parser.add_argument("--cem_iters", type=int, default=5)
    parser.add_argument("--elite_frac", type=float, default=0.1)
    parser.add_argument("--receding_horizon", type=int, default=1)
    parser.add_argument(
        "--model_rollout_horizon",
        type=int,
        default=1,
        help="For cem_mpc, cap how many macro-steps the induced model rolls out during CEM scoring. "
        "Induced models are reliable for about one macro-step, so values above 1 usually hurt planning.",
    )
    parser.add_argument("--no_warm_start", action="store_true")
    parser.add_argument(
        "--sim_verify_frac",
        type=float,
        default=0.0,
        help="Re-score the top fraction of CEM candidates in the true simulator.",
    )
    parser.add_argument("--action_low", type=float, default=-1.0)
    parser.add_argument("--action_high", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--observation_mode",
        choices=["oracle", "vlm_traj", "tool_traj", "tool_live"],
        default="oracle",
        help="oracle uses privileged sim state; vlm_traj/tool_traj use cached traj graphs at t=0 "
        "and live tool re-observation at replan; tool_live uses CV tool extraction on every replan.",
    )
    parser.add_argument(
        "--expose_block_velocity",
        action="store_true",
        help="Inject finite-difference block velocity (vx, vy, w) into oracle observations at re-observation. "
        "Lets second-order dynamics models (pushing-mechanics) condition on incoming block momentum.",
    )
    parser.add_argument(
        "--model_action_repeat",
        type=int,
        default=1,
        help="Step the (per-control-step) induced model this many times per macro action during CEM scoring. "
        "Set equal to --frameskip for a per-frame model so its rollout matches the env's macro semantics "
        "(enables the efficient macro protocol without the per-frame short-horizon penalty).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (PATCHWORLD_ROOT / "patchworld").exists():
        raise SystemExit(f"PatchWorld dependency not found: {PATCHWORLD_ROOT}")

    model_path = resolve_generated_model_path(Path(args.results_dir), args.source, args.model_path)
    _log(f"Loading induced model from {model_path}")
    model = load_induced_model(model_path)

    traj_path = Path(args.trajectories_jsonl) if args.trajectories_jsonl else None
    if traj_path is None and args.observation_mode in ("vlm_traj", "tool_traj"):
        if args.observation_mode == "tool_traj":
            tool_data = Path(args.data_dir) if args.data_dir else ROOT / "artifacts" / "visualpatchworld" / "lewm_tool_data"
            traj_path = tool_data / "pusht_tool_traj_val.jsonl"
        else:
            traj_path = resolve_vlm_jsonl(Path(args.data_dir), "val")

    summary = evaluate_planning(
        model=model,
        dataset=args.dataset,
        stablewm_home=args.stablewm_home,
        trajectories_jsonl=traj_path,
        num_eval=args.num_eval,
        goal_offset_steps=args.goal_offset_steps,
        eval_budget=args.eval_budget,
        frameskip=args.frameskip,
        horizon=args.horizon,
        num_samples=args.num_samples,
        planner=args.planner,
        cem_iters=args.cem_iters,
        elite_frac=args.elite_frac,
        action_low=args.action_low,
        action_high=args.action_high,
        receding_horizon=args.receding_horizon,
        model_rollout_horizon=args.model_rollout_horizon,
        sim_verify_frac=args.sim_verify_frac,
        warm_start=not args.no_warm_start,
        seed=args.seed,
        observation_mode=args.observation_mode,
        expose_block_velocity=args.expose_block_velocity,
        model_action_repeat=args.model_action_repeat,
    )
    summary["model_path"] = str(model_path)
    if traj_path is not None:
        summary["trajectories_jsonl"] = str(traj_path)

    output = (
        Path(args.output)
        if args.output
        else ROOT / "artifacts" / "visualpatchworld" / "reports" / f"pusht_planning_{args.source}_fair.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _log(f"Wrote planning report to {output}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
