#!/usr/bin/env python
"""GT-first fair ranking with GT success target >=90% (prefer 95%) on all 4 LeWM envs.

Frozen protocols (selected so MuJoCo GT is strong):
  - tworoom:  sim_cem fs=5 H=5 R=5 300x10                 (GT 100% after env-reset fix)
  - pusht:    cem_mpc ab=5 H=5 R=5 600x15                 (GT 96%; VPW matches action_block)
  - cube:     sim_library_shoot fs=5 H=1 R=1 400 budget100 (GT 94%)
  - reacher:  sim_cem fs=1 H=1 R=1 finger_match            (GT 100%)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PATCHWORLD = ROOT  # vendored package at repo root
for path in (str(PATCHWORLD), str(SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from visualpatchworld.general_planning_eval import (  # noqa: E402
    evaluate_general_planning,
    evaluate_simulator_general_planning,
)
from visualpatchworld.lewm_env_specs import get_env_spec  # noqa: E402
from visualpatchworld.parallel_planning_eval import (  # noqa: E402
    PlannerConfig,
    collect_trial_specs,
    evaluate_parallel,
)
from visualpatchworld.sim_eval import load_induced_model  # noqa: E402


def _latest_model(directory: Path, patterns: tuple[str, ...], fallback: Path) -> Path:
    candidates: List[Path] = []
    for pattern in patterns:
        candidates.extend(directory.glob(pattern))
    # Prefer newest mtime; ignore empty stubs.
    usable = [p for p in candidates if p.is_file() and p.stat().st_size > 100]
    if not usable:
        return fallback
    return max(usable, key=lambda p: p.stat().st_mtime)


_LEGACY_MODELS = {
    "pusht": ROOT / "artifacts/visualpatchworld/induced/program_induced_best.py",
    "tworoom": ROOT
    / "artifacts/visualpatchworld/lewm_tworoom_general_results/generated_models/tworoom_patchworld_20260622_165539.py",
    "cube": ROOT
    / "artifacts/visualpatchworld/lewm_cube_general_results/generated_models/cube_contact_scaffold_fixed.py",
    "reacher": ROOT
    / "artifacts/visualpatchworld/lewm_reacher_joint_results/generated_models/reacher_patchworld_20260623_144440.py",
}


def resolve_model(env: str) -> Path:
    """Prefer newest L1+L2 program_induction model; fall back to legacy path."""
    prog_dir = ROOT / f"artifacts/visualpatchworld/lewm_{env}_program_results/generated_models"
    return _latest_model(prog_dir, (f"{env}_patchworld_*.py",), _LEGACY_MODELS[env])

PROTOCOLS: Dict[str, Dict[str, Any]] = {
    "tworoom": {
        "tag": "cem_fs5_h5_r5_300x10",
        "backend": "general",
        "frameskip": 5,
        "horizon": 5,
        "receding_horizon": 5,
        "num_samples": 300,
        "cem_iters": 10,
        "warm_start": False,
        "success_metric": "default",
        "sim_planner": "cem",
        "induced_planner": "cem",
        "gt_cache": ROOT
        / "artifacts/visualpatchworld/reports/gt90_sweep/tworoom_cem_fair_fixed_n50.json",
    },
    "pusht": {
        # Matched ab5 per-step CEM-MPC (MuJoCo GT 96%; VPW honors action_block).
        "tag": "cem_ab5_h5_r5_600x15",
        "backend": "pusht_parallel",
        "frameskip": 5,
        "horizon": 5,
        "receding_horizon": 5,
        "action_block": 5,
        "num_samples": 600,
        "cem_iters": 15,
        "warm_start": True,
        "gt_cache": ROOT
        / "artifacts/visualpatchworld/reports/gt90_sweep/pusht_cem_ab5_600x15_n50_proven.json",
    },
    "cube": {
        "tag": "library_shoot_fs5_h1_r1_400_budget100",
        "backend": "general",
        "frameskip": 5,
        "horizon": 1,
        "receding_horizon": 1,
        "num_samples": 400,
        "cem_iters": 1,
        "warm_start": False,
        "success_metric": "default",
        "sim_planner": "library_shoot",
        "induced_planner": "library_shoot",
        "eval_budget": 100,
        "gt_cache": ROOT
        / "artifacts/visualpatchworld/reports/gt90_sweep/cube_library_shoot_budget100_n50.json",
    },
    "reacher": {
        "tag": "cem_fs1_h1_r1_finger_300x10",
        "backend": "general",
        "frameskip": 1,
        "horizon": 1,
        "receding_horizon": 1,
        "num_samples": 300,
        "cem_iters": 10,
        "warm_start": False,
        "success_metric": "finger_match",
        "sim_planner": "cem",
        "induced_planner": "cem",
        "gt_cache": ROOT
        / "artifacts/visualpatchworld/reports/reacher_improve_sim_cem_finger_frame_nowarm_n50.json",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--envs", default="tworoom,pusht,cube,reacher")
    p.add_argument("--num_eval", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gt_target", type=float, default=90.0)
    p.add_argument(
        "--output_dir",
        default=str(ROOT / "artifacts/visualpatchworld/reports/fair_wm_ranking_gt90"),
    )
    p.add_argument("--skip_gt_eval", action="store_true", help="Reuse gt_cache when present")
    p.add_argument("--episode_workers", type=int, default=8, help="PushT parallel workers")
    return p.parse_args()


def _run_gt_general(env: str, proto: Dict[str, Any], num_eval: int, seed: int) -> Dict[str, Any]:
    spec = get_env_spec(env)
    eval_budget = int(proto.get("eval_budget", spec.eval_budget))
    t0 = time.time()
    report = evaluate_simulator_general_planning(
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
        warm_start=bool(proto.get("warm_start", False)),
        sim_planner=str(proto["sim_planner"]),
        success_metric=str(proto.get("success_metric", "default")),
        rollout_batch_size=50,
        env_recycle_every=300,
    )
    report["elapsed_s"] = time.time() - t0
    report["tag"] = proto["tag"]
    return report


def _run_model_general(env: str, proto: Dict[str, Any], model_path: Path, num_eval: int, seed: int) -> Dict[str, Any]:
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
    report["model_path"] = str(model_path)
    report["tag"] = proto["tag"]
    return report


def _run_pusht(proto: Dict[str, Any], model_path: Optional[Path], num_eval: int, seed: int, workers: int) -> Dict[str, Any]:
    _, trials = collect_trial_specs(
        dataset="pusht_expert_train.h5",
        stablewm_home=None,
        num_eval=num_eval,
        goal_offset_steps=25,
        seed=seed,
    )
    cfg = PlannerConfig(
        planner="cem_mpc",
        dynamics="mujoco" if model_path is None else "vpw",
        model_path=str(model_path) if model_path else "",
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
    return report


def run_env(env: str, args: argparse.Namespace, out_root: Path) -> Dict[str, Any]:
    proto = PROTOCOLS[env]
    out_dir = out_root / env
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = resolve_model(env)
    print(f"[gt90] {env}: model {model_path}", flush=True)

    print(f"[gt90] {env}: freeze {proto['tag']}", flush=True)
    gt: Optional[Dict[str, Any]] = None
    cache = Path(proto["gt_cache"]) if proto.get("gt_cache") else None
    if args.skip_gt_eval and cache and cache.is_file():
        gt = json.loads(cache.read_text(encoding="utf-8"))
        # normalize env field for PushT artifact
        gt.setdefault("env", env)
        gt["reused_from"] = str(cache)
        print(f"[gt90] {env}: reused GT {gt.get('success_rate')}% from {cache}", flush=True)
    if gt is None:
        if proto["backend"] == "pusht_parallel":
            gt = _run_pusht(proto, None, args.num_eval, args.seed, args.episode_workers)
        else:
            gt = _run_gt_general(env, proto, args.num_eval, args.seed)
        (out_dir / f"mujoco_gt_n{args.num_eval}.json").write_text(
            json.dumps(gt, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"[gt90] {env}: GT {gt['success_rate']:.1f}%", flush=True)

    gt_sr = float(gt["success_rate"])
    if gt_sr + 1e-9 < args.gt_target:
        print(f"[gt90] WARNING {env}: GT {gt_sr}% < target {args.gt_target}%", flush=True)

    if proto["backend"] == "pusht_parallel":
        model_rep = _run_pusht(proto, model_path, args.num_eval, args.seed, args.episode_workers)
    else:
        model_rep = _run_model_general(env, proto, model_path, args.num_eval, args.seed)
    (out_dir / f"model_n{args.num_eval}.json").write_text(
        json.dumps(model_rep, indent=2, sort_keys=True), encoding="utf-8"
    )
    vs = (model_rep["success_rate"] / gt_sr) if gt_sr > 0 else None
    print(f"[gt90] {env}: model {model_rep['success_rate']:.1f}% vs_gt={vs}", flush=True)

    proto_json = {k: (str(v) if isinstance(v, Path) else v) for k, v in proto.items()}
    summary = {
        "env": env,
        "gt_target": args.gt_target,
        "frozen_protocol": proto_json,
        "mujoco_gt": {
            "success_rate": gt_sr,
            "mean_goal_distance": gt.get("mean_goal_distance"),
            "reused_from": gt.get("reused_from"),
            "elapsed_s": gt.get("elapsed_s"),
        },
        "model": {
            "name": model_path.name,
            "model_path": str(model_path),
            "success_rate": model_rep["success_rate"],
            "mean_goal_distance": model_rep.get("mean_goal_distance"),
            "vs_gt_ratio": vs,
            "elapsed_s": model_rep.get("elapsed_s"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    envs = [e.strip() for e in args.envs.split(",") if e.strip()]
    t0 = time.time()
    per_env: Dict[str, Any] = {}
    for env in envs:
        per_env[env] = run_env(env, args, out_root)

    table = []
    for env in ("tworoom", "pusht", "cube", "reacher"):
        if env not in per_env:
            # merge from disk
            p = out_root / env / "summary.json"
            if p.is_file():
                per_env[env] = json.loads(p.read_text(encoding="utf-8"))
        if env not in per_env:
            continue
        s = per_env[env]
        table.append(
            {
                "env": env,
                "gt_success": s["mujoco_gt"]["success_rate"],
                "model_success": s["model"]["success_rate"],
                "vs_gt": s["model"].get("vs_gt_ratio"),
                "protocol": s["frozen_protocol"]["tag"],
            }
        )

    combined = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "gt_target": args.gt_target,
        "num_eval": args.num_eval,
        "seed": args.seed,
        "suite_elapsed_s": time.time() - t0,
        "table": table,
        "per_env": per_env,
        "notes": [
            "MuJoCo GT must be >=90% under the frozen controller before ranking WMs.",
            "Fixed general_planning_eval to reset env before executing CEM plans.",
            "Cube uses expert-macro library shooting (same library for GT and induced).",
            "PushT uses matched ab5 CEM-MPC 600x15 (GT 96%). VPW cem_mpc honors action_block.",
            "VPW models resolved from lewm_{env}_program_results (L1 probe + L2 fit).",
            "Two-room/Cube/Reacher re-induced 2026-07-22 via program_induction; "
            "PushT uses the planner-selected PD L1+L2 checkpoint "
            "(fresh default-geometry refit scored 0% and was discarded).",
        ],
    }
    out_path = out_root / f"summary_n{args.num_eval}_seed{args.seed}.json"
    out_path.write_text(json.dumps(combined, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Fair WM ranking — GT ≥ 90%",
        "",
        "| Env | GT success | Model success | vs GT | Frozen protocol |",
        "|-----|------------|---------------|-------|-----------------|",
    ]
    for row in table:
        lines.append(
            f"| {row['env']} | {row['gt_success']} | {row['model_success']} | "
            f"{row['vs_gt'] if row['vs_gt'] is not None else 'n/a'} | `{row['protocol']}` |"
        )
    lines.append("")
    lines.append(f"Seed {args.seed}, n_eval={args.num_eval}. Target GT ≥ {args.gt_target}%.")
    (out_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"table": table, "wrote": str(out_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
