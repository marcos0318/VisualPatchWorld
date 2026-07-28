#!/usr/bin/env python
"""Fair-protocol Oracle+Induced eval for no-probe / wrong-sketch ablation models."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PATCHWORLD = ROOT  # vendored package at repo root
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (str(PATCHWORLD), str(SRC), str(SCRIPT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("STABLEWM_HOME", str(Path.home() / ".stable-wm"))

from visualpatchworld.general_planning_eval import evaluate_general_planning  # noqa: E402
from visualpatchworld.lewm_env_specs import get_env_spec  # noqa: E402
from visualpatchworld.parallel_planning_eval import (  # noqa: E402
    PlannerConfig,
    collect_trial_specs,
    evaluate_parallel,
)
from visualpatchworld.sim_eval import load_induced_model  # noqa: E402
from run_fair_wm_ranking_gt90 import PROTOCOLS  # noqa: E402

DEFAULT_JOBS = [
    (
        "pusht_noprobe_pd",
        "pusht",
        ROOT
        / "artifacts/visualpatchworld/lewm_pusht_noprobe_pd_results/generated_models",
    ),
    (
        "pusht_wrong_scaffold",
        "pusht",
        ROOT
        / "artifacts/visualpatchworld/lewm_pusht_wrong_scaffold_results/generated_models",
    ),
    (
        "reacher_noprobe_joint",
        "reacher",
        ROOT
        / "artifacts/visualpatchworld/lewm_reacher_noprobe_joint_results/generated_models",
    ),
    (
        "reacher_wrong_linear",
        "reacher",
        ROOT
        / "artifacts/visualpatchworld/lewm_reacher_wrong_linear_results/generated_models",
    ),
    (
        "pusht_paper_l1l2",
        "pusht",
        ROOT
        / "artifacts/visualpatchworld/lewm_pusht_program_results/generated_models/pusht_patchworld_program_induced_best.py",
    ),
    (
        "reacher_paper_l1l2",
        "reacher",
        ROOT
        / "artifacts/visualpatchworld/lewm_reacher_program_results/generated_models/reacher_patchworld_20260722_165027.py",
    ),
]


def _latest_model(path: Path) -> Path:
    if path.is_file():
        return path
    cands = sorted(path.glob("*_patchworld_*.py"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise FileNotFoundError(f"No model under {path}")
    return cands[-1]


def _eval_pusht(model_path: Path, num_eval: int, seed: int, workers: int) -> Dict[str, Any]:
    proto = PROTOCOLS["pusht"]
    _, trials = collect_trial_specs(
        dataset="pusht_expert_train.h5",
        stablewm_home=None,
        num_eval=num_eval,
        goal_offset_steps=25,
        seed=seed,
    )
    cfg = PlannerConfig(
        planner="cem_mpc",
        dynamics="vpw",
        model_path=str(model_path),
        frameskip=int(proto["frameskip"]),
        horizon=int(proto["horizon"]),
        receding_horizon=int(proto["receding_horizon"]),
        model_rollout_horizon=int(proto["horizon"]),
        action_block=int(proto["action_block"]),
        num_samples=int(proto["num_samples"]),
        cem_iters=int(proto["cem_iters"]),
        warm_start=bool(proto.get("warm_start", True)),
        seed=seed,
    )
    t0 = time.time()
    report = evaluate_parallel(trials=trials, cfg=cfg, num_workers=workers)
    report["elapsed_s"] = time.time() - t0
    report["tag"] = proto["tag"]
    return report


def _eval_general(env: str, model_path: Path, num_eval: int, seed: int) -> Dict[str, Any]:
    proto = PROTOCOLS[env]
    spec = get_env_spec(env)
    eval_budget = int(proto.get("eval_budget", spec.eval_budget))
    model = load_induced_model(model_path)
    t0 = time.time()
    report = evaluate_general_planning(
        model=model,
        env_name=env,
        num_eval=num_eval,
        goal_offset_steps=spec.goal_offset_steps,
        eval_budget=eval_budget,
        frameskip=int(proto["frameskip"]),
        horizon=int(proto["horizon"]),
        receding_horizon=int(proto["receding_horizon"]),
        num_samples=int(proto["num_samples"]),
        cem_iters=int(proto.get("cem_iters", 10)),
        seed=seed,
        observation_mode="oracle",
        success_metric=str(proto.get("success_metric", "default")),
        warm_start=bool(proto.get("warm_start", False)),
        induced_planner=str(proto.get("induced_planner", "cem")),
    )
    report["elapsed_s"] = time.time() - t0
    report["tag"] = proto["tag"]
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num_eval", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episode_workers", type=int, default=8)
    p.add_argument(
        "--output_dir",
        default=str(ROOT / "artifacts/visualpatchworld/reports/noprobe_ablation"),
    )
    p.add_argument("--jobs", default="all", help="Comma tags or all")
    args = p.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted = {j.strip() for j in args.jobs.split(",") if j.strip()}
    jobs = DEFAULT_JOBS if "all" in wanted else [j for j in DEFAULT_JOBS if j[0] in wanted]

    summary: Dict[str, Any] = {"seed": args.seed, "num_eval": args.num_eval, "cells": {}}
    for tag, env, model_src in jobs:
        model_path = _latest_model(Path(model_src))
        out_path = out_dir / f"{tag}_{env}_n{args.num_eval}_seed{args.seed}.json"
        print(f"[noprobe] {tag}/{env} model={model_path.name}", flush=True)
        try:
            if env == "pusht":
                report = _eval_pusht(model_path, args.num_eval, args.seed, args.episode_workers)
            else:
                report = _eval_general(env, model_path, args.num_eval, args.seed)
            report["tag_name"] = tag
            report["env"] = env
            report["model_path"] = str(model_path)
            report["success"] = True
        except Exception as exc:  # noqa: BLE001
            report = {
                "tag_name": tag,
                "env": env,
                "model_path": str(model_path),
                "success": False,
                "error": f"{type(exc).__name__}: {exc}",
                "success_rate": None,
            }
            print(f"[noprobe] FAIL {tag}: {report['error']}", flush=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        summary["cells"][tag] = {
            "env": env,
            "success_rate": report.get("success_rate"),
            "elapsed_s": report.get("elapsed_s"),
            "out": str(out_path),
            "model_path": str(model_path),
        }
        print(f"[noprobe] RESULT {tag} sr={report.get('success_rate')}", flush=True)

    summary_path = out_dir / f"summary_n{args.num_eval}_seed{args.seed}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary["cells"], indent=2), flush=True)
    print(f"[noprobe] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
