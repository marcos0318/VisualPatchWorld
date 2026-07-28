#!/usr/bin/env python
"""Evaluate cross-environment planning for induced general scaffold-fit models."""

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
for part in (str(PATCHWORLD_ROOT), str(SRC)):
    if part not in sys.path:
        sys.path.insert(0, part)
os.environ["PYTHONPATH"] = os.pathsep.join([str(PATCHWORLD_ROOT), str(SRC), os.environ.get("PYTHONPATH", "")])

try:
    import hdf5plugin  # noqa: F401
except ModuleNotFoundError:
    pass

from visualpatchworld.general_planning_eval import evaluate_general_planning, evaluate_simulator_general_planning
from visualpatchworld.lewm_env_specs import get_env_spec
from visualpatchworld.sim_eval import load_induced_model, resolve_generated_model_path


def _log(msg: str) -> None:
    print(f"[visualpatchworld][{datetime.now().isoformat(timespec='seconds')}][cross-env] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", required=True, choices=["tworoom", "cube", "reacher", "pusht"])
    p.add_argument("--results_dir", required=True)
    p.add_argument("--source", default="oracle")
    p.add_argument("--model_path", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--num_eval", type=int, default=50)
    p.add_argument("--num_samples", type=int, default=300)
    p.add_argument("--cem_iters", type=int, default=10)
    p.add_argument("--sim_verify_frac", type=float, default=0.0)
    p.add_argument(
        "--planner",
        choices=[
            "induced",
            "sim_cem",
            "sim_mppi",
            "sim_discrete_beam",
            "sim_library_shoot",
            "induced_discrete_beam",
            "induced_library_shoot",
        ],
        default="induced",
    )
    p.add_argument("--observation_mode", choices=["oracle", "tool_live"], default="oracle")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warm_start", action="store_true", help="MPC warm-start for CEM planners")
    p.add_argument("--beam_width", type=int, default=16, help="Beam width for discrete_beam planner")
    p.add_argument(
        "--goal_bias_frac",
        type=float,
        default=0.0,
        help="Fraction of CEM samples replaced by noisy goal-directional macros (sim_cem)",
    )
    p.add_argument(
        "--success_metric",
        choices=["default", "finger_match"],
        default="default",
        help="Reacher diagnostic: finger_match scores fingertip-to-target distance",
    )
    p.add_argument("--mppi_temperature", type=float, default=1.0)
    p.add_argument("--mppi_noise_sigma", type=float, default=0.35)
    p.add_argument("--ladder_tag", default="", help="Label for sim-ceiling ladder reports")
    p.add_argument("--frameskip", type=int, default=None, help="Override env default frameskip")
    p.add_argument("--horizon", type=int, default=None, help="Override env default CEM horizon")
    p.add_argument("--receding_horizon", type=int, default=None, help="Override env default receding horizon")
    p.add_argument(
        "--sim_rollout_batch",
        type=int,
        default=50,
        help="Score sim CEM/MPPI candidates in batches (reduces peak memory; 0=all at once)",
    )
    p.add_argument(
        "--sim_env_recycle_every",
        type=int,
        default=200,
        help="Recreate MuJoCo env every N sim rollouts during CEM (0=disable)",
    )
    return p.parse_args()


def _mpc_settings(args: argparse.Namespace, spec) -> tuple[int, int, int]:
    frameskip = spec.frameskip if args.frameskip is None else args.frameskip
    horizon = spec.planning_horizon if args.horizon is None else args.horizon
    receding = spec.planning_receding if args.receding_horizon is None else args.receding_horizon
    return frameskip, horizon, receding


def main() -> None:
    args = parse_args()
    spec = get_env_spec(args.env)
    frameskip, horizon, receding_horizon = _mpc_settings(args, spec)
    results_dir = Path(args.results_dir)
    if args.planner in ("sim_cem", "sim_mppi", "sim_discrete_beam", "sim_library_shoot"):
        if args.planner == "sim_mppi":
            sim_planner = "mppi"
        elif args.planner == "sim_discrete_beam":
            sim_planner = "discrete_beam"
        elif args.planner == "sim_library_shoot":
            sim_planner = "library_shoot"
        else:
            sim_planner = "cem"
        report = evaluate_simulator_general_planning(
            env_name=args.env,
            num_eval=args.num_eval,
            goal_offset_steps=spec.goal_offset_steps,
            eval_budget=spec.eval_budget,
            frameskip=frameskip,
            horizon=horizon,
            receding_horizon=receding_horizon,
            num_samples=args.num_samples,
            cem_iters=args.cem_iters,
            seed=args.seed,
            warm_start=args.warm_start,
            sim_planner=sim_planner,
            success_metric=args.success_metric,
            mppi_temperature=args.mppi_temperature,
            mppi_noise_sigma=args.mppi_noise_sigma,
            ladder_tag=args.ladder_tag,
            rollout_batch_size=args.sim_rollout_batch,
            env_recycle_every=args.sim_env_recycle_every,
            beam_width=args.beam_width,
            goal_bias_frac=args.goal_bias_frac,
        )
    else:
        if args.model_path:
            model_path = Path(args.model_path)
        else:
            candidates = sorted((results_dir / "generated_models").glob("*.py"), key=lambda p: p.stat().st_mtime)
            if not candidates:
                model_path = resolve_generated_model_path(results_dir, args.source, None)
            else:
                model_path = candidates[-1]
        _log(f"Loading model {model_path}")
        model = load_induced_model(model_path)
        report = evaluate_general_planning(
            model=model,
            env_name=args.env,
            num_eval=args.num_eval,
            goal_offset_steps=spec.goal_offset_steps,
            eval_budget=spec.eval_budget,
            frameskip=frameskip,
            horizon=horizon,
            receding_horizon=receding_horizon,
            num_samples=args.num_samples,
            cem_iters=args.cem_iters,
            sim_verify_frac=args.sim_verify_frac,
            seed=args.seed,
            observation_mode=args.observation_mode,
            success_metric=args.success_metric,
            warm_start=args.warm_start,
            induced_planner=(
                "discrete_beam"
                if args.planner == "induced_discrete_beam"
                else "library_shoot"
                if args.planner == "induced_library_shoot"
                else "cem"
            ),
            beam_width=args.beam_width,
        )
    out = Path(args.output) if args.output else results_dir / f"plan_{args.env}_n{args.num_eval}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _log(f"Wrote {out}")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
