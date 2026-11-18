#!/usr/bin/env python
"""Re-plan code baselines + VPW Tool/Hybrid under GT>=90 frozen protocols.

Reuses existing induced models (no LLM induction). Same per-env planner/budget
as MuJoCo GT / Oracle+Induced in run_fair_wm_ranking_gt90.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from visualpatchworld.planning_eval import evaluate_planning  # noqa: E402
from visualpatchworld.sim_eval import load_induced_model  # noqa: E402

# Reuse frozen protocols from the GT90 ranking script.
from run_fair_wm_ranking_gt90 import PROTOCOLS  # noqa: E402
from run_fair_wm_ranking_gt90 import resolve_model  # noqa: E402

BASELINE_ROOT = ROOT / "artifacts/visualpatchworld/code_baselines"
BASELINE_METHODS = (
    "tracepatch",
    "worldcoder",
    "pomdp_coder",
    "poeworld",
    "gif_mcts",
    "cwm_game",
)
VPW_VARIANTS = (
    "vpw_tool_induced",
    "vpw_tool_hybrid",
    "vpw_oracle_induced",
    "vpw_oracle_hybrid",
)
ENVS = ("tworoom", "pusht", "cube", "reacher")


def _latest_baseline_model(method: str, env: str) -> Path:
    d = BASELINE_ROOT / method / env / "generated_models"
    files = sorted(d.glob(f"{env}_{method}_*.py"))
    if not files:
        raise FileNotFoundError(f"No baseline model for {method}/{env} under {d}")
    return files[-1]


def _method_spec(method: str, env: str) -> Tuple[Path, str, float]:
    """Return (model_path, observation_mode, sim_verify_frac)."""
    if method in BASELINE_METHODS:
        return _latest_baseline_model(method, env), "oracle", 0.0
    if method == "vpw_tool_induced":
        return resolve_model(env), "tool_live", 0.0
    if method == "vpw_tool_hybrid":
        return resolve_model(env), "tool_live", 0.3
    if method == "vpw_oracle_induced":
        return resolve_model(env), "oracle", 0.0
    if method == "vpw_oracle_hybrid":
        return resolve_model(env), "oracle", 0.3
    raise ValueError(f"unknown method {method}")


def _run_general(
    *,
    env: str,
    proto: Dict[str, Any],
    model_path: Path,
    observation_mode: str,
    sim_verify_frac: float,
    num_eval: int,
    seed: int,
) -> Dict[str, Any]:
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
        observation_mode=observation_mode,
        success_metric=str(proto.get("success_metric", "default")),
        warm_start=bool(proto.get("warm_start", False)),
        induced_planner=str(proto.get("induced_planner", "cem")),
        sim_verify_frac=float(sim_verify_frac),
    )
    report["elapsed_s"] = time.time() - t0
    report["model_path"] = str(model_path)
    report["tag"] = proto["tag"]
    report["observation_mode"] = observation_mode
    report["sim_verify_frac"] = sim_verify_frac
    return report


def _run_pusht_induced(
    *,
    proto: Dict[str, Any],
    model_path: Path,
    num_eval: int,
    seed: int,
    workers: int,
) -> Dict[str, Any]:
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
    report["env"] = "pusht"
    report["model_path"] = str(model_path)
    report["observation_mode"] = "oracle"
    report["sim_verify_frac"] = 0.0
    return report


def _run_pusht_hybrid(
    *,
    proto: Dict[str, Any],
    model_path: Path,
    observation_mode: str,
    sim_verify_frac: float,
    num_eval: int,
    seed: int,
) -> Dict[str, Any]:
    """Match ab5 semantics via cem_mpc with horizon*ab per-step actions (fs=1)."""
    ab = int(proto["action_block"])
    h = int(proto["horizon"]) * ab
    r = int(proto["receding_horizon"]) * ab
    model = load_induced_model(model_path)
    t0 = time.time()
    report = evaluate_planning(
        model=model,
        dataset="pusht_expert_train.h5",
        stablewm_home=None,
        num_eval=num_eval,
        goal_offset_steps=25,
        eval_budget=50,
        frameskip=1,
        horizon=h,
        receding_horizon=r,
        model_rollout_horizon=h,
        num_samples=int(proto["num_samples"]),
        planner="cem_mpc",
        cem_iters=int(proto["cem_iters"]),
        warm_start=bool(proto.get("warm_start", True)),
        seed=seed,
        observation_mode=observation_mode,
        expose_block_velocity=True,
        model_action_repeat=1,
        sim_verify_frac=float(sim_verify_frac),
    )
    report["elapsed_s"] = time.time() - t0
    report["tag"] = proto["tag"]
    report["model_path"] = str(model_path)
    report["observation_mode"] = observation_mode
    report["sim_verify_frac"] = sim_verify_frac
    report["action_block_matched"] = ab
    return report


def run_one(
    method: str,
    env: str,
    *,
    num_eval: int,
    seed: int,
    workers: int,
    out_dir: Path,
) -> Dict[str, Any]:
    proto = PROTOCOLS[env]
    model_path, obs_mode, sv = _method_spec(method, env)
    out_path = out_dir / f"{method}_{env}_n{num_eval}_seed{seed}.json"
    print(
        f"[fair-replan] {method}/{env} proto={proto['tag']} obs={obs_mode} sv={sv} "
        f"model={model_path.name}",
        flush=True,
    )
    t0 = time.time()
    try:
        if env == "pusht":
            if sv > 0.0 or obs_mode == "tool_live":
                # tool_live also needs evaluate_planning (parallel is oracle-only).
                report = _run_pusht_hybrid(
                    proto=proto,
                    model_path=model_path,
                    observation_mode=obs_mode,
                    sim_verify_frac=sv,
                    num_eval=num_eval,
                    seed=seed,
                )
            else:
                report = _run_pusht_induced(
                    proto=proto,
                    model_path=model_path,
                    num_eval=num_eval,
                    seed=seed,
                    workers=workers,
                )
        else:
            report = _run_general(
                env=env,
                proto=proto,
                model_path=model_path,
                observation_mode=obs_mode,
                sim_verify_frac=sv,
                num_eval=num_eval,
                seed=seed,
            )
        report["method"] = method
        report["env"] = env
        report["success"] = True
        report["suite_elapsed_s"] = time.time() - t0
    except Exception as exc:  # noqa: BLE001 — surface per-cell failures in summary
        report = {
            "method": method,
            "env": env,
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
            "success_rate": None,
            "model_path": str(model_path),
            "tag": proto["tag"],
            "observation_mode": obs_mode,
            "sim_verify_frac": sv,
            "suite_elapsed_s": time.time() - t0,
        }
        print(f"[fair-replan] FAIL {method}/{env}: {report['error']}", flush=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    sr = report.get("success_rate")
    print(f"[fair-replan] RESULT {method}/{env} success_rate={sr}", flush=True)
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--methods", default="all", help="Comma list or 'all'")
    p.add_argument("--envs", default="tworoom,pusht,cube,reacher")
    p.add_argument("--num_eval", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episode_workers", type=int, default=8)
    p.add_argument(
        "--output_dir",
        default=str(ROOT / "artifacts/visualpatchworld/reports/fair_wm_ranking_gt90/all_methods"),
    )
    p.add_argument("--baseline_root", default=str(BASELINE_ROOT))
    return p.parse_args()


def main() -> None:
    global BASELINE_ROOT
    args = parse_args()
    BASELINE_ROOT = Path(args.baseline_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.methods.strip().lower() == "all":
        methods: List[str] = list(BASELINE_METHODS) + list(VPW_VARIANTS)
    else:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    envs = [e.strip() for e in args.envs.split(",") if e.strip()]

    t0 = time.time()
    cells: Dict[str, Dict[str, Any]] = {}
    for method in methods:
        for env in envs:
            key = f"{method}/{env}"
            cells[key] = run_one(
                method,
                env,
                num_eval=args.num_eval,
                seed=args.seed,
                workers=args.episode_workers,
                out_dir=out_dir,
            )

    # Merge prior Oracle+Induced / MuJoCo if present.
    prior_root = out_dir.parent
    prior: Dict[str, Any] = {}
    for env in ENVS:
        p = prior_root / env / "summary.json"
        if p.is_file():
            prior[env] = json.loads(p.read_text(encoding="utf-8"))

    table_rows = []
    for method in methods:
        row = {"method": method}
        vals = []
        for env in ENVS:
            cell = cells.get(f"{method}/{env}", {})
            sr = cell.get("success_rate")
            row[env] = sr
            if isinstance(sr, (int, float)):
                vals.append(float(sr))
        row["mean"] = (sum(vals) / len(vals)) if len(vals) == len(ENVS) else None
        table_rows.append(row)

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "num_eval": args.num_eval,
        "seed": args.seed,
        "protocols": {e: {k: (str(v) if isinstance(v, Path) else v) for k, v in PROTOCOLS[e].items()} for e in ENVS},
        "suite_elapsed_s": time.time() - t0,
        "table": table_rows,
        "cells": cells,
        "prior_oracle_mujoco": {
            env: {
                "mujoco_gt": prior[env].get("mujoco_gt", {}).get("success_rate"),
                "oracle_induced": prior[env].get("model", {}).get("success_rate"),
            }
            for env in prior
        },
        "notes": [
            "Planning-only re-eval under GT>=90 frozen planners; no LLM re-induction.",
            "PushT induced-only uses parallel cem_mpc action_block=5.",
            "PushT tool/hybrid uses evaluate_planning cem_mpc with h=r=25, fs=1 to match ab5.",
            "Cube uses library_shoot with optional sim_verify for hybrid rows.",
        ],
    }
    out_path = out_dir / f"summary_n{args.num_eval}_seed{args.seed}.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"wrote": str(out_path), "table": table_rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
