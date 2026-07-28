#!/usr/bin/env python
"""Select GT-competent planner, freeze it, rank program world models.

Step 1 — GT sweep (n_sweep): find planner settings with success >= target.
Step 2 — Frozen n_eval ranking: MuJoCo ceiling + program WMs, same protocol.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PATCHWORLD = ROOT  # vendored package at repo root
for path in (str(PATCHWORLD), str(SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ["PYTHONPATH"] = os.pathsep.join(
    [str(PATCHWORLD), str(SRC)] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])
)

from visualpatchworld.parallel_planning_eval import (  # noqa: E402
    PlannerConfig,
    collect_trial_specs,
    evaluate_parallel,
)


DEFAULT_MODELS = [
    ("vpw_pd_best", str(ROOT / "artifacts/visualpatchworld/induced/program_induced_best.py")),
    ("vpw_modular_phase_e", str(ROOT / "artifacts/visualpatchworld/schema_harness/phase_e/pusht_modular_accepted/world_model.accepted.py")),
    ("vpw_sticky_phase_f", str(ROOT / "artifacts/visualpatchworld/schema_harness/phase_f/pusht_sticky_n50/world_model.accepted.py")),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num_eval", type=int, default=50)
    p.add_argument("--n_sweep", type=int, default=20, help="episodes for GT sweep")
    p.add_argument("--gt_target", type=float, default=85.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episode_workers", type=int, default=16)
    p.add_argument(
        "--output_dir",
        default=str(ROOT / "artifacts/visualpatchworld/reports/fair_wm_ranking"),
    )
    p.add_argument("--dataset", default="pusht_expert_train.h5")
    p.add_argument("--stablewm_home", default=os.environ.get("STABLEWM_HOME"))
    p.add_argument(
        "--skip_sweep",
        action="store_true",
        help="Use --frozen_json protocol instead of sweeping",
    )
    p.add_argument("--frozen_json", default="", help="Optional prior sweep winner json")
    return p.parse_args()


def _base_cfg(**kwargs: Any) -> PlannerConfig:
    defaults = dict(
        planner="cem_mpc",
        dynamics="mujoco",
        eval_budget=50,
        frameskip=1,
        horizon=5,
        receding_horizon=1,
        model_rollout_horizon=5,
        num_samples=100,
        cem_iters=5,
        elite_frac=0.1,
        action_block=1,
        seed=42,
        warm_start=True,
    )
    defaults.update(kwargs)
    return PlannerConfig(**defaults)


def gt_candidates() -> List[Tuple[str, PlannerConfig]]:
    """Candidate protocols; first to hit target wins (order = preference).

    Short-horizon CEM (H=5, ab=1) stays weak on GT even at 300x10 (~25%).
    Prefer longer CEM lookahead or wider beam under the same fs=1 / execute-1 interface.
    """
    out: List[Tuple[str, PlannerConfig]] = []
    # Matched to known-good GT "horizon1 fair" length: plan 5 steps, execute 5, replan
    # (paper GT horizon1_fair uses H=1,ab=5,rh=1 → 5 env steps; 88% @ 300x10).
    out.append(
        (
            "cem_ab1_h5_r5_300x10",
            _base_cfg(
                planner="cem_mpc",
                action_block=1,
                horizon=5,
                receding_horizon=5,
                model_rollout_horizon=5,
                num_samples=300,
                cem_iters=10,
            ),
        )
    )
    out.append(
        (
            "cem_ab1_h5_r5_600x15",
            _base_cfg(
                planner="cem_mpc",
                action_block=1,
                horizon=5,
                receding_horizon=5,
                model_rollout_horizon=5,
                num_samples=600,
                cem_iters=15,
            ),
        )
    )
    # Contact-aware beam — already 62% at width 12; widen first (faster than fat CEM).
    for bw in (32, 48, 64):
        out.append(
            (
                f"beam_h5_r1_bw{bw}",
                _base_cfg(
                    planner="beam_mpc",
                    horizon=5,
                    receding_horizon=1,
                    model_rollout_horizon=5,
                    num_samples=bw,
                    cem_iters=0,
                ),
            )
        )
    # Longer beam horizon
    for h, bw in ((8, 40), (10, 32)):
        out.append(
            (
                f"beam_h{h}_r1_bw{bw}",
                _base_cfg(
                    planner="beam_mpc",
                    horizon=h,
                    receding_horizon=1,
                    model_rollout_horizon=h,
                    num_samples=bw,
                    cem_iters=0,
                ),
            )
        )
    # CEM with long lookahead (≈ paper block5 length in one-step units), execute 1.
    for h, samples, iters in ((15, 200, 8), (25, 200, 8), (25, 300, 10)):
        out.append(
            (
                f"cem_ab1_h{h}_r1_{samples}x{iters}",
                _base_cfg(
                    planner="cem_mpc",
                    action_block=1,
                    horizon=h,
                    receding_horizon=1,
                    model_rollout_horizon=h,
                    num_samples=samples,
                    cem_iters=iters,
                ),
            )
        )
    return out


def run_cfg(
    trials,
    cfg: PlannerConfig,
    *,
    workers: int,
    tag: str,
) -> Dict[str, Any]:
    t0 = time.time()
    report = evaluate_parallel(trials=trials, cfg=cfg, num_workers=workers)
    report["elapsed_s"] = time.time() - t0
    report["tag"] = tag
    print(
        f"[fair-rank] {tag}: success={report['success_rate']:.1f}% "
        f"dist={report.get('mean_goal_distance'):.2f} elapsed={report['elapsed_s']:.1f}s "
        f"planner={cfg.planner} H={cfg.horizon} ab={cfg.action_block} "
        f"samples={cfg.num_samples} iters={cfg.cem_iters}",
        flush=True,
    )
    return report


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[fair-rank] collect starts n_sweep={args.n_sweep} n_eval={args.num_eval} "
        f"target_gt>={args.gt_target}% workers={args.episode_workers}",
        flush=True,
    )
    _, trials_sweep = collect_trial_specs(
        dataset=args.dataset,
        stablewm_home=args.stablewm_home,
        num_eval=args.n_sweep,
        goal_offset_steps=25,
        seed=args.seed,
    )
    _, trials_eval = collect_trial_specs(
        dataset=args.dataset,
        stablewm_home=args.stablewm_home,
        num_eval=args.num_eval,
        goal_offset_steps=25,
        seed=args.seed,
    )

    sweep_path = out_dir / f"gt_sweep_n{args.n_sweep}_seed{args.seed}.json"
    winner_path = out_dir / f"frozen_protocol_seed{args.seed}.json"

    if args.skip_sweep and args.frozen_json:
        winner = json.loads(Path(args.frozen_json).read_text())
        frozen_cfg = PlannerConfig(**winner["planner_config"])
        print(f"[fair-rank] loaded frozen protocol from {args.frozen_json}", flush=True)
    else:
        sweep_rows: List[Dict[str, Any]] = []
        winner = None
        for tag, cfg in gt_candidates():
            # Cap workers for heavy CEM to limit EGL contention.
            workers = args.episode_workers
            if cfg.planner == "cem_mpc" and cfg.num_samples >= 400:
                workers = min(workers, 12)
            if cfg.planner == "beam_mpc" and cfg.num_samples >= 40:
                workers = min(workers, 12)
            report = run_cfg(trials_sweep, cfg, workers=workers, tag=tag)
            row = {
                "tag": tag,
                "success_rate": report["success_rate"],
                "mean_goal_distance": report["mean_goal_distance"],
                "elapsed_s": report["elapsed_s"],
                "planner_config": asdict(cfg),
                "meets_target": report["success_rate"] + 1e-9 >= args.gt_target,
            }
            sweep_rows.append(row)
            (out_dir / f"sweep_{tag}.json").write_text(
                json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
            )
            if row["meets_target"] and winner is None:
                winner = row
                print(f"[fair-rank] SELECTED {tag} (GT {report['success_rate']:.1f}% >= {args.gt_target})", flush=True)
                # Continue a couple more cheap beams? No — freeze first hit (preference order).
                break

        if winner is None:
            # Pick best available and warn.
            sweep_rows.sort(key=lambda r: (-float(r["success_rate"]), float(r["elapsed_s"])))
            winner = sweep_rows[0]
            print(
                f"[fair-rank] WARNING: no protocol hit {args.gt_target}%; "
                f"using best {winner['tag']} at {winner['success_rate']:.1f}%",
                flush=True,
            )

        sweep_path.write_text(
            json.dumps(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "n_sweep": args.n_sweep,
                    "gt_target": args.gt_target,
                    "seed": args.seed,
                    "rows": sweep_rows,
                    "selected": winner,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        winner_path.write_text(json.dumps(winner, indent=2, sort_keys=True), encoding="utf-8")
        frozen_cfg = PlannerConfig(**winner["planner_config"])

    # --- Frozen ranking ---
    ranking: Dict[str, Any] = {}
    t_suite = time.time()

    # GT ceiling under frozen protocol
    gt_cfg = replace(frozen_cfg, dynamics="mujoco", model_path="")
    workers = min(args.episode_workers, 12 if gt_cfg.num_samples >= 300 or gt_cfg.planner == "beam_mpc" else args.episode_workers)
    gt_report = run_cfg(trials_eval, gt_cfg, workers=workers, tag="mujoco_gt_frozen")
    (out_dir / f"mujoco_gt_n{args.num_eval}_seed{args.seed}.json").write_text(
        json.dumps(gt_report, indent=2, sort_keys=True), encoding="utf-8"
    )
    ranking["mujoco_gt"] = {
        "success_rate": gt_report["success_rate"],
        "mean_goal_distance": gt_report["mean_goal_distance"],
        "elapsed_s": gt_report["elapsed_s"],
        "dynamics": "mujoco",
    }

    model_rows = []
    for name, path in DEFAULT_MODELS:
        if not Path(path).is_file():
            print(f"[fair-rank] skip missing model {name}: {path}", flush=True)
            continue
        cfg = replace(
            frozen_cfg,
            dynamics="vpw",
            model_path=path,
            # VPW uses model_rollout_horizon; keep aligned with horizon.
            model_rollout_horizon=frozen_cfg.horizon,
            action_block=1,  # program path ignores action_block; keep 1
        )
        # For beam/cem on VPW, workers can be higher (less EGL during planning).
        w = args.episode_workers if cfg.planner != "beam_mpc" else min(args.episode_workers, 20)
        report = run_cfg(trials_eval, cfg, workers=w, tag=name)
        out_p = out_dir / f"{name}_n{args.num_eval}_seed{args.seed}.json"
        out_p.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        row = {
            "name": name,
            "model_path": path,
            "success_rate": report["success_rate"],
            "mean_goal_distance": report["mean_goal_distance"],
            "elapsed_s": report["elapsed_s"],
            "vs_gt_ratio": (report["success_rate"] / gt_report["success_rate"])
            if gt_report["success_rate"] > 0
            else None,
            "report": str(out_p),
        }
        model_rows.append(row)
        ranking[name] = row

    model_rows.sort(key=lambda r: (-float(r["success_rate"]), float(r["mean_goal_distance"])))

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "selection_rule": (
            f"Choose first GT protocol with success>={args.gt_target}% on n={args.n_sweep}; "
            "freeze; rank program WMs on n_eval with identical planner settings."
        ),
        "frozen_protocol": {
            "tag": winner.get("tag") if isinstance(winner, dict) else None,
            "planner_config": asdict(frozen_cfg),
            "gt_sweep_success": winner.get("success_rate") if isinstance(winner, dict) else None,
        },
        "num_eval": args.num_eval,
        "seed": args.seed,
        "suite_elapsed_s": time.time() - t_suite,
        "mujoco_gt": ranking["mujoco_gt"],
        "model_ranking": model_rows,
        "paper_reference": {
            "vpw_pd_cem_600x15_fs1": 66.0,
            "mujoco_cem_ab5_600x15_fs5": 96.0,
        },
        "artifacts": {
            "sweep": str(sweep_path) if sweep_path.exists() else None,
            "frozen": str(winner_path),
            "output_dir": str(out_dir),
        },
    }
    summary_path = out_dir / f"summary_n{args.num_eval}_seed{args.seed}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[fair-rank] wrote {summary_path}", flush=True)
    print(json.dumps({"mujoco_gt": ranking["mujoco_gt"], "ranking": model_rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
