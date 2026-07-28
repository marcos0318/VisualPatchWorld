#!/usr/bin/env python
"""Fast parallel MuJoCo vs VPW PushT planner suite.

Runs episode-parallel CEM-MPC and beam-MPC under a lean shared budget, and
launches arms concurrently. Defaults target multi-core machines (~minutes).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PATCHWORLD = ROOT  # vendored package at repo root
for path in (str(PATCHWORLD), str(SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)
os.environ["PYTHONPATH"] = os.pathsep.join(
    [str(PATCHWORLD), str(SRC)] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])
)
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from visualpatchworld.parallel_planning_eval import (  # noqa: E402
    PlannerConfig,
    TrialSpec,
    collect_trial_specs,
    evaluate_parallel,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num_eval", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--goal_offset_steps", type=int, default=25)
    p.add_argument("--eval_budget", type=int, default=50)
    # Lean but still meaningful vs paper 600x15
    p.add_argument("--cem_samples", type=int, default=100)
    p.add_argument("--cem_iters", type=int, default=5)
    p.add_argument("--beam_width", type=int, default=12)
    p.add_argument(
        "--episode_workers",
        type=int,
        default=32,
        help="process workers per arm (episode parallel)",
    )
    p.add_argument(
        "--arm_workers",
        type=int,
        default=1,
        help="arms run concurrently; keep 1 if MuJoCo EGL contends (recommended)",
    )
    p.add_argument(
        "--model_path",
        default=str(ROOT / "artifacts/visualpatchworld/induced/program_induced_best.py"),
    )
    p.add_argument(
        "--output_dir",
        default=str(ROOT / "artifacts/visualpatchworld/reports/planner_suite_fast"),
    )
    p.add_argument("--dataset", default="pusht_expert_train.h5")
    p.add_argument("--stablewm_home", default=os.environ.get("STABLEWM_HOME"))
    p.add_argument(
        "--arms",
        default="vpw_cem,vpw_beam,mujoco_cem,mujoco_beam",
        help="comma list: vpw_cem,vpw_beam,mujoco_cem,mujoco_beam",
    )
    return p.parse_args()


def _arm_config(name: str, args: argparse.Namespace) -> Tuple[str, PlannerConfig]:
    common = dict(
        eval_budget=args.eval_budget,
        frameskip=1,
        horizon=5,
        receding_horizon=1,
        model_rollout_horizon=5,
        action_block=1,
        elite_frac=0.1,
        seed=args.seed,
        warm_start=True,
    )
    if name == "vpw_cem":
        return name, PlannerConfig(
            planner="cem_mpc",
            dynamics="vpw",
            model_path=args.model_path,
            num_samples=args.cem_samples,
            cem_iters=args.cem_iters,
            **common,
        )
    if name == "vpw_beam":
        return name, PlannerConfig(
            planner="beam_mpc",
            dynamics="vpw",
            model_path=args.model_path,
            num_samples=args.beam_width,
            cem_iters=0,
            **common,
        )
    if name == "mujoco_cem":
        return name, PlannerConfig(
            planner="cem_mpc",
            dynamics="mujoco",
            num_samples=args.cem_samples,
            cem_iters=args.cem_iters,
            **common,
        )
    if name == "mujoco_beam":
        return name, PlannerConfig(
            planner="beam_mpc",
            dynamics="mujoco",
            num_samples=args.beam_width,
            cem_iters=0,
            **common,
        )
    raise ValueError(f"unknown arm {name}")


def _run_arm(
    name: str,
    trials: List[TrialSpec],
    cfg: PlannerConfig,
    episode_workers: int,
    out_path: Path,
) -> Tuple[str, Dict[str, Any], float]:
    t0 = time.time()
    report = evaluate_parallel(trials=trials, cfg=cfg, num_workers=episode_workers)
    report["elapsed_s"] = time.time() - t0
    report["arm"] = name
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return name, report, report["elapsed_s"]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[planner-suite-fast] collecting {args.num_eval} starts seed={args.seed} "
        f"cem={args.cem_samples}x{args.cem_iters} beam={args.beam_width} "
        f"episode_workers={args.episode_workers} arm_workers={args.arm_workers}",
        flush=True,
    )
    dataset_path, trials = collect_trial_specs(
        dataset=args.dataset,
        stablewm_home=args.stablewm_home,
        num_eval=args.num_eval,
        goal_offset_steps=args.goal_offset_steps,
        seed=args.seed,
    )

    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    # Threads coordinate arms; each arm uses a process pool for episodes.
    # Default 4 arms × 16 workers ≈ 64 procs on a 128-core box.
    jobs = []
    for name in arm_names:
        _, cfg = _arm_config(name, args)
        out_path = out_dir / f"{name}_n{args.num_eval}_seed{args.seed}.json"
        jobs.append((name, trials, cfg, args.episode_workers, out_path))

    arms: Dict[str, Any] = {}
    t_suite = time.time()
    with ThreadPoolExecutor(max_workers=max(1, min(args.arm_workers, len(jobs)))) as pool:
        futs = {
            pool.submit(_run_arm, name, arm_trials, cfg, workers, out_path): name
            for name, arm_trials, cfg, workers, out_path in jobs
        }
        for fut in as_completed(futs):
            name, report, elapsed = fut.result()
            arms[name] = {
                "success_rate": report["success_rate"],
                "mean_goal_distance": report["mean_goal_distance"],
                "elapsed_s": elapsed,
                "planner": report["planner"],
                "dynamics": report["dynamics"],
                "num_samples": report["num_samples"],
                "cem_iters": report["cem_iters"],
                "beam_width": report.get("beam_width"),
                "num_workers": report["num_workers"],
                "report": str(out_dir / f"{name}_n{args.num_eval}_seed{args.seed}.json"),
            }
            print(
                f"[planner-suite-fast] {name}: success={report['success_rate']:.1f}% "
                f"dist={report['mean_goal_distance']:.2f} elapsed={elapsed:.1f}s",
                flush=True,
            )

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "protocol": {
            "dataset": dataset_path,
            "num_eval": args.num_eval,
            "seed": args.seed,
            "eval_budget": args.eval_budget,
            "frameskip": 1,
            "horizon": 5,
            "receding_horizon": 1,
            "cem": {"num_samples": args.cem_samples, "cem_iters": args.cem_iters},
            "beam_width": args.beam_width,
            "episode_workers": args.episode_workers,
            "arm_workers": args.arm_workers,
            "model_path": args.model_path,
            "notes": [
                "Episode-parallel + arm-parallel suite.",
                "Lean CEM budget (default 100x5) for fair MuJoCo↔VPW speed; paper headline uses 600x15.",
                "Macro protocol: plan H=5, execute 1, replan (frameskip=1, action_block=1).",
            ],
        },
        "suite_elapsed_s": time.time() - t_suite,
        "arms": arms,
        "paper_reference": {
            "vpw_cem_600x15_n50": 66.0,
            "mujoco_cem_block5_600x15_n50": 96.0,
        },
    }
    summary_path = out_dir / f"summary_n{args.num_eval}_seed{args.seed}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[planner-suite-fast] wrote {summary_path}", flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
