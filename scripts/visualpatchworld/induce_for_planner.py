#!/usr/bin/env python
"""Planner-aware (decision-aware) model induction.

Holds the general planner (CEM macro-MPC) FIXED and searches over model
*smoothness* knobs (soft contact-gate width, rollout-loss regularization,
rollout horizon), selecting the model by **planning success**, not prediction
error. This operationalizes the principle that emerged from our experiments:
prediction accuracy != planning utility, so induce the model that makes the
fixed planner succeed (cf. value-equivalent / decision-aware model learning).

Each variant is induced (pusht_pushing_mechanics_collision_rollout) and then
evaluated with the same CEM planner; variants run in parallel. The best by
planning success (tie-break: mean goal distance) is reported.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
ENV = dict(os.environ)
ENV["PYTHONPATH"] = f"{ROOT}/src::" + ENV.get("PYTHONPATH", "")
ENV.setdefault("MUJOCO_GL", "egl")


def _log(m):
    print(f"[induce-for-planner][{datetime.now().isoformat(timespec='seconds')}] {m}", flush=True)


def run_variant(cfg: dict, args) -> dict:
    name = cfg["name"]
    results_dir = Path(args.work_dir) / f"variant_{name}"   # plan reads {results_dir}/oracle/generated_models
    out_dir = results_dir / "oracle"                         # induce writes {out_dir}/generated_models
    report = Path(args.work_dir) / f"variant_{name}_plan.json"
    induce = [
        PY, "-m", "visualpatchworld.patchworld_benchmark", "--env", "pusht",
        "--train_glob", args.train_glob, "--eval_glob", args.eval_glob,
        "--induction_strategy", "pusht_pushing_mechanics_collision_rollout",
        "--geometry_json", args.geometry_json,
        "--rollout_k", str(cfg["rollout_k"]),
        "--rollout_fit_steps", str(args.fit_steps),
        "--rollout_gate_width", str(cfg["gate_width"]),
        "--rollout_reg_lambda", str(cfg["reg_lambda"]),
        "--output_dir", str(out_dir), "--max_train", "0", "--max_eval", "0",
    ]
    r1 = subprocess.run(induce, cwd=str(ROOT), env=ENV, capture_output=True, text=True)
    if r1.returncode != 0:
        return {"name": name, **cfg, "error": "induce_failed", "stderr": r1.stderr[-500:]}
    plan = [
        PY, str(ROOT / "scripts" / "visualpatchworld" / "evaluate_planning.py"),
        "--source", "oracle", "--results_dir", str(results_dir),
        "--planner", "cem_mpc", "--observation_mode", "oracle", "--expose_block_velocity",
        "--frameskip", "1", "--horizon", "5", "--model_rollout_horizon", "5", "--receding_horizon", "1",
        "--sim_verify_frac", "0.0", "--num_eval", str(args.num_eval),
        "--num_samples", str(args.num_samples), "--cem_iters", str(args.cem_iters),
        "--elite_frac", "0.1", "--seed", str(args.seed), "--output", str(report),
    ]
    r2 = subprocess.run(plan, cwd=str(ROOT), env=ENV, capture_output=True, text=True)
    if r2.returncode != 0 or not report.exists():
        return {"name": name, **cfg, "error": "plan_failed", "stderr": r2.stderr[-500:]}
    res = json.loads(report.read_text())
    out = {"name": name, **cfg, "success_rate": res["success_rate"], "mean_goal_distance": res["mean_goal_distance"]}
    _log(f"variant {name}: success={out['success_rate']:.1f} dist={out['mean_goal_distance']:.1f}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train_glob", default="artifacts/visualpatchworld/data_stride1/pusht_oracle_traj_train.jsonl")
    ap.add_argument("--eval_glob", default="artifacts/visualpatchworld/data_stride1/pusht_oracle_traj_val.jsonl")
    ap.add_argument("--geometry_json", default="artifacts/visualpatchworld/perceived_geometry/pusht_geometry.json")
    ap.add_argument("--work_dir", default="artifacts/visualpatchworld/induce_for_planner")
    ap.add_argument("--gate_widths", default="2.5,5.0,8.0,12.0")
    ap.add_argument("--reg_lambdas", default="5.0,20.0")
    ap.add_argument("--rollout_ks", default="6")
    ap.add_argument("--fit_steps", type=int, default=150)
    ap.add_argument("--num_eval", type=int, default=30)
    ap.add_argument("--num_samples", type=int, default=200)
    ap.add_argument("--cem_iters", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_parallel", type=int, default=8)
    args = ap.parse_args()

    gws = [float(x) for x in args.gate_widths.split(",")]
    rls = [float(x) for x in args.reg_lambdas.split(",")]
    ks = [int(x) for x in args.rollout_ks.split(",")]
    configs = [
        {"name": f"gw{gw}_rl{rl}_k{k}", "gate_width": gw, "reg_lambda": rl, "rollout_k": k}
        for gw, rl, k in itertools.product(gws, rls, ks)
    ]
    _log(f"searching {len(configs)} variants (fixed CEM {args.num_samples}x{args.cem_iters}, n={args.num_eval})")
    Path(args.work_dir).mkdir(parents=True, exist_ok=True)

    results = []
    with ThreadPoolExecutor(max_workers=args.max_parallel) as ex:
        futs = {ex.submit(run_variant, c, args): c for c in configs}
        for f in as_completed(futs):
            results.append(f.result())

    ok = [r for r in results if "success_rate" in r]
    ok.sort(key=lambda r: (-r["success_rate"], r["mean_goal_distance"]))
    summary = {"configs": len(configs), "results": ok,
               "failed": [r for r in results if "success_rate" not in r],
               "best": ok[0] if ok else None,
               "planner": {"num_samples": args.num_samples, "cem_iters": args.cem_iters, "num_eval": args.num_eval}}
    out = Path(args.work_dir) / "search_summary.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print("\n=== planner-aware model search (sorted by planning success) ===")
    for r in ok:
        print(f"  {r['name']:22s} success={r['success_rate']:5.1f}  goal_dist={r['mean_goal_distance']:6.1f}")
    if summary["best"]:
        b = summary["best"]
        print(f"\nBEST: {b['name']}  success={b['success_rate']:.1f}  goal_dist={b['mean_goal_distance']:.1f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
