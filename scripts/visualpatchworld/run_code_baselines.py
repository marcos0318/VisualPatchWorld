#!/usr/bin/env python
"""Induce and evaluate programmatic code WM baselines on LeWM oracle graphs."""

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

from visualpatchworld.code_baselines.runner import (
    DEFAULT_MODEL,
    METHODS,
    apply_code_baseline_llm_env,
    induce_code_baseline,
)
from visualpatchworld.general_planning_eval import evaluate_general_planning
from visualpatchworld.lewm_env_specs import get_env_spec
from visualpatchworld.sim_eval import load_induced_model


def _log(msg: str) -> None:
    print(f"[visualpatchworld][{datetime.now().isoformat(timespec='seconds')}][code-baselines] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", required=True, choices=list(METHODS))
    p.add_argument("--env", required=True, choices=["tworoom", "cube", "reacher", "pusht"])
    p.add_argument("--data_dir", default="artifacts/visualpatchworld/data")
    p.add_argument("--output_dir", default=None, help="Defaults to artifacts/visualpatchworld/code_baselines/{method}/{env}")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--llm_request_timeout_s", type=int, default=None)
    p.add_argument("--llm_max_retries", type=int, default=None)
    p.add_argument("--max_train", type=int, default=24)
    p.add_argument("--max_eval", type=int, default=12)
    p.add_argument("--max_refine_rounds", type=int, default=3)
    p.add_argument("--worldcoder_programs", type=int, default=2)
    p.add_argument("--poeworld_batches", type=int, default=3)
    p.add_argument("--poeworld_experts_per_batch", type=int, default=4)
    p.add_argument("--gif_mcts_iterations", type=int, default=50)
    p.add_argument("--pomdp_coder_candidates", type=int, default=3)
    p.add_argument("--cwm_retries", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_planning", action="store_true")
    p.add_argument("--num_eval", type=int, default=50)
    p.add_argument("--num_samples", type=int, default=300)
    p.add_argument("--cem_iters", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    train_glob = str(data_dir / f"{args.env}_oracle_traj_train.jsonl")
    eval_glob = str(data_dir / f"{args.env}_oracle_traj_val.jsonl")
    output_dir = Path(args.output_dir or f"artifacts/visualpatchworld/code_baselines/{args.method}/{args.env}")

    llm_env = apply_code_baseline_llm_env(
        llm_request_timeout_s=args.llm_request_timeout_s,
        llm_max_retries=args.llm_max_retries,
    )
    _log(
        f"LLM model={args.model} timeout={llm_env['PATCHWORLD_LLM_REQUEST_TIMEOUT_S']}s "
        f"retries={llm_env['PATCHWORLD_LLM_MAX_RETRIES']} cache={llm_env['PATCHWORLD_LLM_CACHE']}"
    )
    _log(f"Inducing method={args.method} env={args.env} -> {output_dir}")
    report = induce_code_baseline(
        method=args.method,
        env_name=args.env,
        train_glob=train_glob,
        eval_glob=eval_glob,
        output_dir=output_dir,
        model=args.model,
        max_train=args.max_train,
        max_eval=args.max_eval,
        seed=args.seed,
        max_refine_rounds=args.max_refine_rounds,
        worldcoder_programs=args.worldcoder_programs,
        poeworld_batches=args.poeworld_batches,
        poeworld_experts_per_batch=args.poeworld_experts_per_batch,
        gif_mcts_iterations=args.gif_mcts_iterations,
        pomdp_coder_candidates=args.pomdp_coder_candidates,
        cwm_retries=args.cwm_retries,
        llm_request_timeout_s=args.llm_request_timeout_s,
        llm_max_retries=args.llm_max_retries,
    )
    model_path = Path(report["symbolic_model_path"])
    _log(f"Induction complete: {model_path} val_errors={report['validation']['error_count']}")

    if args.skip_planning:
        print(json.dumps(report, indent=2))
        return

    spec = get_env_spec(args.env)
    model = load_induced_model(model_path)
    planning = evaluate_general_planning(
        env_name=args.env,
        model=model,
        num_eval=args.num_eval,
        goal_offset_steps=spec.goal_offset_steps,
        eval_budget=spec.eval_budget,
        frameskip=spec.frameskip,
        horizon=spec.planning_horizon,
        receding_horizon=spec.planning_receding,
        num_samples=args.num_samples,
        cem_iters=args.cem_iters,
        seed=args.seed,
        observation_mode="oracle",
    )
    report["planning"] = planning
    planning_path = output_dir / f"planning_{args.env}_{args.method}_n{args.num_eval}.json"
    planning_path.write_text(json.dumps(planning, indent=2), encoding="utf-8")
    _log(f"Planning success={planning.get('success_rate')} saved={planning_path}")
    print(json.dumps({"induction": report, "planning_success_rate": planning.get("success_rate")}, indent=2))


if __name__ == "__main__":
    main()
