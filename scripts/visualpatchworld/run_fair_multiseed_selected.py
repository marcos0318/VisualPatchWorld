#!/usr/bin/env python
"""Fair-protocol multi-seed CI for the selected paper refresh set.

Batches (seeds 42-46, n=50, GT>=90 frozen planners):
  A) tworoom: vpw_tool_hybrid, vpw_oracle_hybrid, mujoco_cem
  B) pusht:   vpw_tool_induced, vpw_tool_hybrid
  C) cube:    vpw_tool_induced, vpw_tool_hybrid
  D) all four envs: vpw_oracle_induced  (headline matched claim armor)

Skips existing JSON reports unless --force. Writes under
artifacts/visualpatchworld/reports/fair_wm_ranking_gt90/all_methods/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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

from visualpatchworld.general_planning_eval import (  # noqa: E402
    evaluate_simulator_general_planning,
)
from visualpatchworld.lewm_env_specs import get_env_spec  # noqa: E402
from visualpatchworld.parallel_planning_eval import (  # noqa: E402
    PlannerConfig,
    collect_trial_specs,
    evaluate_parallel,
)
from run_fair_baseline_replan_gt90 import run_one as run_vpw_one  # noqa: E402
from run_fair_wm_ranking_gt90 import PROTOCOLS  # noqa: E402

DEFAULT_OUT = ROOT / "artifacts/visualpatchworld/reports/fair_wm_ranking_gt90/all_methods"
LOG_DIR = ROOT / "logs" / "paper_experiments"

# Selected refresh matrix (paper cost-aware plan).
JOBS_BY_BATCH: Dict[str, List[Tuple[str, str]]] = {
    # Batch A — kill stale Two-room hybrid/MuJoCo CIs
    "A": [
        ("vpw_tool_hybrid", "tworoom"),
        ("vpw_oracle_hybrid", "tworoom"),
        ("mujoco_cem", "tworoom"),
    ],
    # Batch B — fair PushT contact claim
    "B": [
        ("vpw_tool_induced", "pusht"),
        ("vpw_tool_hybrid", "pusht"),
    ],
    # Batch C — Cube library roadmap
    "C": [
        ("vpw_tool_induced", "cube"),
        ("vpw_tool_hybrid", "cube"),
    ],
    # Batch D — headline Oracle+Induced matched claim (suite five-seed)
    "D": [
        ("vpw_oracle_induced", "tworoom"),
        ("vpw_oracle_induced", "pusht"),
        ("vpw_oracle_induced", "cube"),
        ("vpw_oracle_induced", "reacher"),
    ],
}


def _out_path(out_dir: Path, method: str, env: str, num_eval: int, seed: int) -> Path:
    return out_dir / f"{method}_{env}_n{num_eval}_seed{seed}.json"


def _run_mujoco(env: str, num_eval: int, seed: int, workers: int) -> Dict[str, Any]:
    proto = PROTOCOLS[env]
    t0 = time.time()
    if env == "pusht":
        _, trials = collect_trial_specs(
            dataset="pusht_expert_train.h5",
            stablewm_home=None,
            num_eval=num_eval,
            goal_offset_steps=25,
            seed=seed,
        )
        cfg = PlannerConfig(
            planner="cem_mpc",
            dynamics="mujoco",
            model_path="",
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
        report = evaluate_parallel(trials=trials, cfg=cfg, num_workers=workers)
    else:
        spec = get_env_spec(env)
        eval_budget = int(proto.get("eval_budget", spec.eval_budget))
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
    report["method"] = "mujoco_cem"
    report["env"] = env
    report["tag"] = proto["tag"]
    report["observation_mode"] = "simulator"
    report["sim_verify_frac"] = 1.0
    report["success"] = True
    return report


def _worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    method = payload["method"]
    env = payload["env"]
    seed = int(payload["seed"])
    num_eval = int(payload["num_eval"])
    workers = int(payload["workers"])
    out_dir = Path(payload["out_dir"])
    force = bool(payload["force"])
    out_path = _out_path(out_dir, method, env, num_eval, seed)
    label = f"{method}/{env}/seed{seed}"

    if out_path.is_file() and not force:
        data = json.loads(out_path.read_text(encoding="utf-8"))
        return {
            "label": label,
            "status": "skip",
            "out": str(out_path),
            "success_rate": data.get("success_rate"),
        }

    t0 = time.time()
    try:
        if method == "mujoco_cem":
            report = _run_mujoco(env, num_eval, seed, workers)
        else:
            report = run_vpw_one(
                method,
                env,
                num_eval=num_eval,
                seed=seed,
                workers=workers,
                out_dir=out_dir,
            )
            # run_vpw_one already wrote the file
            return {
                "label": label,
                "status": "done" if report.get("success", True) else "fail",
                "out": str(out_path),
                "success_rate": report.get("success_rate"),
                "elapsed_s": time.time() - t0,
                "error": report.get("error"),
            }
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        return {
            "label": label,
            "status": "done",
            "out": str(out_path),
            "success_rate": report.get("success_rate"),
            "elapsed_s": time.time() - t0,
        }
    except Exception as exc:  # noqa: BLE001
        err = {
            "method": method,
            "env": env,
            "seed": seed,
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
            "success_rate": None,
        }
        out_path.write_text(json.dumps(err, indent=2, sort_keys=True), encoding="utf-8")
        return {
            "label": label,
            "status": "fail",
            "out": str(out_path),
            "success_rate": None,
            "elapsed_s": time.time() - t0,
            "error": err["error"],
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", default="42,43,44,45,46")
    p.add_argument("--num_eval", type=int, default=50)
    p.add_argument("--max_parallel", type=int, default=2)
    p.add_argument("--episode_workers", type=int, default=8)
    p.add_argument("--force", action="store_true")
    p.add_argument("--output_dir", default=str(DEFAULT_OUT))
    p.add_argument(
        "--batches",
        default="A,B,C",
        help="Comma subset of A,B,C,D",
    )
    return p.parse_args()


def _selected_jobs(batches: str) -> List[Tuple[str, str]]:
    wanted = {b.strip().upper() for b in batches.split(",") if b.strip()}
    selected: List[Tuple[str, str]] = []
    for key in ("A", "B", "C", "D"):
        if key in wanted:
            selected.extend(JOBS_BY_BATCH[key])
    return selected


def main() -> None:
    import multiprocessing as mp

    # EGL/MuJoCo is unsafe under fork; require spawn for worker processes.
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    jobs = _selected_jobs(args.batches)

    payloads = []
    for method, env in jobs:
        for seed in seeds:
            payloads.append(
                {
                    "method": method,
                    "env": env,
                    "seed": seed,
                    "num_eval": args.num_eval,
                    "workers": args.episode_workers,
                    "out_dir": str(out_dir),
                    "force": args.force,
                }
            )

    queue_log = LOG_DIR / "fair_multiseed_selected_queue.log"
    stamp = datetime.now().isoformat(timespec="seconds")
    with queue_log.open("a", encoding="utf-8") as f:
        f.write(f"\n=== start {stamp} jobs={len(payloads)} parallel={args.max_parallel} ===\n")

    print(
        f"[fair-ms] launching {len(payloads)} cells "
        f"(batches={args.batches}, seeds={seeds}, parallel={args.max_parallel})",
        flush=True,
    )

    results: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, args.max_parallel)) as ex:
        futs = {ex.submit(_worker, p): p for p in payloads}
        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            line = (
                f"[fair-ms] {res['status']:4s} {res['label']} "
                f"sr={res.get('success_rate')} out={Path(res['out']).name}"
            )
            print(line, flush=True)
            with queue_log.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "batches": args.batches,
        "seeds": seeds,
        "num_eval": args.num_eval,
        "n_jobs": len(payloads),
        "n_done": sum(1 for r in results if r["status"] == "done"),
        "n_skip": sum(1 for r in results if r["status"] == "skip"),
        "n_fail": sum(1 for r in results if r["status"] == "fail"),
        "results": sorted(results, key=lambda r: r["label"]),
    }
    summary_path = out_dir / f"fair_multiseed_selected_summary_{stamp.replace(':', '')}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("n_done", "n_skip", "n_fail", "n_jobs")}, indent=2), flush=True)
    print(f"[fair-ms] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
