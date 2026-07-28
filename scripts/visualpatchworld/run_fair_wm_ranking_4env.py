#!/usr/bin/env python
"""GT-first fair planning ranking on all four LeWM environments.

Per env:
  1) Sweep MuJoCo planner settings until success >= target (or take best).
  2) Freeze protocol.
  3) Rank the canonical induced program/scaffold world model under that protocol.

PushT reuses artifacts/visualpatchworld/reports/fair_wm_ranking when present.
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
for path in (str(PATCHWORLD), str(SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ["PYTHONPATH"] = os.pathsep.join(
    [str(PATCHWORLD), str(SRC)] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])
)

from visualpatchworld.general_planning_eval import (  # noqa: E402
    evaluate_general_planning,
    evaluate_simulator_general_planning,
)
from visualpatchworld.lewm_env_specs import get_env_spec  # noqa: E402
from visualpatchworld.sim_eval import load_induced_model  # noqa: E402


def _latest_model(directory: Path, patterns: tuple[str, ...], fallback: Path) -> Path:
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(directory.glob(pattern))
    usable = [p for p in candidates if p.is_file() and p.stat().st_size > 100]
    if not usable:
        return fallback
    return max(usable, key=lambda p: p.stat().st_mtime)


_CUBE_MODEL_DIR = ROOT / "artifacts/visualpatchworld/lewm_cube_general_results/generated_models"
MODELS = {
    "pusht": ROOT / "artifacts/visualpatchworld/induced/program_induced_best.py",
    "tworoom": ROOT
    / "artifacts/visualpatchworld/lewm_tworoom_general_results/generated_models/tworoom_patchworld_20260622_165539.py",
    "cube": _latest_model(
        _CUBE_MODEL_DIR,
        ("cube_patchworld_*.py", "cube_contact_scaffold_*.py"),
        _CUBE_MODEL_DIR / "cube_patchworld_20260622_165657.py",
    ),
    "reacher": ROOT
    / "artifacts/visualpatchworld/lewm_reacher_joint_results/generated_models/reacher_patchworld_20260623_144440.py",
}

RESULTS_DIRS = {
    "pusht": ROOT / "artifacts/visualpatchworld/induced",
    "tworoom": ROOT / "artifacts/visualpatchworld/lewm_tworoom_general_results",
    "cube": ROOT / "artifacts/visualpatchworld/lewm_cube_general_results",
    "reacher": ROOT / "artifacts/visualpatchworld/lewm_reacher_joint_results",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--envs", default="tworoom,pusht,cube,reacher")
    p.add_argument("--num_eval", type=int, default=50)
    p.add_argument("--n_sweep", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output_dir",
        default=str(ROOT / "artifacts/visualpatchworld/reports/fair_wm_ranking_4env"),
    )
    p.add_argument("--env_workers", type=int, default=4, help="envs in parallel")
    p.add_argument("--reuse_pusht", action="store_true", default=True)
    p.add_argument("--no_reuse_pusht", action="store_true")
    p.add_argument(
        "--combine_only",
        action="store_true",
        help="Only merge per-env summary.json into the combined report (no eval).",
    )
    p.add_argument(
        "--mode",
        choices=("lean", "full"),
        default="lean",
        help=(
            "lean: one frozen known-strong protocol per env (reuse GT JSON when present); "
            "full: multi-candidate GT sweep (expensive)."
        ),
    )
    p.add_argument(
        "--skip_gt_eval",
        action="store_true",
        help="In lean mode, reuse cached GT reports and only evaluate the induced model.",
    )
    return p.parse_args()


# Cached GT reports under the frozen lean protocols (seed 42, n=50).
_GT_CACHE = {
    "reacher": ROOT
    / "artifacts/visualpatchworld/reports/reacher_improve_sim_cem_finger_frame_nowarm_n50.json",
    "cube": ROOT
    / "artifacts/visualpatchworld/reports/cross_env_cube_sim_cem_fair_n50_seed42.json",
    "tworoom": ROOT
    / "artifacts/visualpatchworld/reports/cross_env_tworoom_sim_cem_fair_n50_seed42.json",
}


def _candidates(env: str, *, mode: str = "lean") -> Tuple[float, List[Dict[str, Any]]]:
    """Return (gt_target, candidate protocol dicts). Cheap protocols first."""
    if env == "pusht":
        return 85.0, [
            {
                "tag": "cem_ab1_h5_r5_300x10",
                "frameskip": 1,
                "horizon": 5,
                "receding_horizon": 5,
                "num_samples": 300,
                "cem_iters": 10,
                "warm_start": True,
                "success_metric": "default",
                "planner": "sim_cem",
            }
        ]
    if env == "reacher":
        # Known 100% GT under frame MPC + finger_match.
        return 85.0, [
            {
                "tag": "cem_fs1_h1_r1_finger_300x10",
                "frameskip": 1,
                "horizon": 1,
                "receding_horizon": 1,
                "num_samples": 300,
                "cem_iters": 10,
                "warm_start": False,
                "success_metric": "finger_match",
                "planner": "sim_cem",
            }
        ]
    if env == "cube":
        base = [
            {
                "tag": "cem_fs5_h5_r5_300x10",
                "frameskip": 5,
                "horizon": 5,
                "receding_horizon": 5,
                "num_samples": 300,
                "cem_iters": 10,
                "warm_start": False,
                "success_metric": "default",
                "planner": "sim_cem",
            },
            {
                "tag": "cem_fs5_h5_r5_300x10_warm",
                "frameskip": 5,
                "horizon": 5,
                "receding_horizon": 5,
                "num_samples": 300,
                "cem_iters": 10,
                "warm_start": True,
                "success_metric": "default",
                "planner": "sim_cem",
            },
        ]
        if mode == "full":
            base.append(
                {
                    "tag": "cem_fs5_h5_r5_600x15_warm",
                    "frameskip": 5,
                    "horizon": 5,
                    "receding_horizon": 5,
                    "num_samples": 600,
                    "cem_iters": 15,
                    "warm_start": True,
                    "success_metric": "default",
                    "planner": "sim_cem",
                }
            )
        # Lean freezes the known fair protocol (GT ~66–80%); escalate only in full.
        return (85.0 if mode == "full" else 60.0), (base[:1] if mode == "lean" else base)
    if env == "tworoom":
        # Fair CEM historically ~32% — lean freezes that protocol; full tries warm/R=1.
        base = [
            {
                "tag": "cem_fs5_h5_r5_300x10",
                "frameskip": 5,
                "horizon": 5,
                "receding_horizon": 5,
                "num_samples": 300,
                "cem_iters": 10,
                "warm_start": False,
                "success_metric": "default",
                "planner": "sim_cem",
            },
            {
                "tag": "cem_fs5_h5_r5_300x10_warm",
                "frameskip": 5,
                "horizon": 5,
                "receding_horizon": 5,
                "num_samples": 300,
                "cem_iters": 10,
                "warm_start": True,
                "success_metric": "default",
                "planner": "sim_cem",
            },
            {
                "tag": "cem_fs5_h5_r5_600x15_warm",
                "frameskip": 5,
                "horizon": 5,
                "receding_horizon": 5,
                "num_samples": 600,
                "cem_iters": 15,
                "warm_start": True,
                "success_metric": "default",
                "planner": "sim_cem",
            },
        ]
        return (70.0 if mode == "full" else 30.0), (base[:1] if mode == "lean" else base)
    raise KeyError(env)


def _run_gt(env: str, proto: Dict[str, Any], num_eval: int, seed: int) -> Dict[str, Any]:
    spec = get_env_spec(env)
    t0 = time.time()
    report = evaluate_simulator_general_planning(
        env_name=env,
        num_eval=num_eval,
        goal_offset_steps=spec.goal_offset_steps,
        eval_budget=spec.eval_budget,
        frameskip=int(proto["frameskip"]),
        horizon=int(proto["horizon"]),
        receding_horizon=int(proto["receding_horizon"]),
        num_samples=int(proto["num_samples"]),
        cem_iters=int(proto["cem_iters"]),
        seed=seed,
        warm_start=bool(proto.get("warm_start", False)),
        sim_planner="cem",
        success_metric=str(proto.get("success_metric", "default")),
        rollout_batch_size=50,
        env_recycle_every=200,
    )
    report["elapsed_s"] = time.time() - t0
    report["tag"] = proto["tag"]
    report["protocol"] = proto
    return report


def _run_model(
    env: str,
    proto: Dict[str, Any],
    model_path: Path,
    num_eval: int,
    seed: int,
) -> Dict[str, Any]:
    spec = get_env_spec(env)
    model = load_induced_model(model_path)
    t0 = time.time()
    report = evaluate_general_planning(
        model=model,
        env_name=env,
        num_eval=num_eval,
        goal_offset_steps=spec.goal_offset_steps,
        eval_budget=spec.eval_budget,
        frameskip=int(proto["frameskip"]),
        horizon=int(proto["horizon"]),
        receding_horizon=int(proto["receding_horizon"]),
        num_samples=int(proto["num_samples"]),
        cem_iters=int(proto["cem_iters"]),
        seed=seed,
        observation_mode="oracle",
        success_metric=str(proto.get("success_metric", "default")),
        warm_start=bool(proto.get("warm_start", False)),
    )
    report["elapsed_s"] = time.time() - t0
    report["model_path"] = str(model_path)
    report["tag"] = proto["tag"]
    report["protocol"] = proto
    return report


def _reuse_pusht(out_dir: Path) -> Optional[Dict[str, Any]]:
    src = ROOT / "artifacts/visualpatchworld/reports/fair_wm_ranking/summary_n50_seed42.json"
    if not src.is_file():
        return None
    d = json.loads(src.read_text())
    cfg = d["frozen_protocol"]["planner_config"]
    proto = {
        "tag": d["frozen_protocol"]["tag"],
        "frameskip": cfg["frameskip"],
        "horizon": cfg["horizon"],
        "receding_horizon": cfg["receding_horizon"],
        "num_samples": cfg["num_samples"],
        "cem_iters": cfg["cem_iters"],
        "warm_start": cfg.get("warm_start", True),
        "success_metric": "default",
        "planner": "sim_cem",
    }
    gt = d["mujoco_gt"]
    models = d["model_ranking"]
    best = models[0] if models else None
    summary = {
        "env": "pusht",
        "reused_from": str(src),
        "gt_target": 85.0,
        "frozen_protocol": proto,
        "gt_sweep_success": d["frozen_protocol"].get("gt_sweep_success"),
        "mujoco_gt": {
            "success_rate": gt["success_rate"],
            "mean_goal_distance": gt["mean_goal_distance"],
            "elapsed_s": gt.get("elapsed_s"),
        },
        "model": {
            "name": best["name"] if best else "vpw_pd_best",
            "model_path": best["model_path"] if best else str(MODELS["pusht"]),
            "success_rate": best["success_rate"] if best else None,
            "mean_goal_distance": best["mean_goal_distance"] if best else None,
            "vs_gt_ratio": best.get("vs_gt_ratio") if best else None,
        },
        "all_models": models,
    }
    (out_dir / "pusht" / "summary.json").parent.mkdir(parents=True, exist_ok=True)
    (out_dir / "pusht" / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _load_gt_cache(env: str, proto: Dict[str, Any], num_eval: int, seed: int) -> Optional[Dict[str, Any]]:
    path = _GT_CACHE.get(env)
    if path is None or not path.is_file():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    checks = {
        "frameskip": int(proto["frameskip"]),
        "horizon": int(proto["horizon"]),
        "receding_horizon": int(proto["receding_horizon"]),
        "num_samples": int(proto["num_samples"]),
        "cem_iters": int(proto["cem_iters"]),
        "seed": seed,
        "num_eval": num_eval,
    }
    for k, want in checks.items():
        if int(d.get(k, -1)) != want:
            print(f"[4env] {env}: GT cache mismatch on {k}: {d.get(k)} != {want}", flush=True)
            return None
    if str(d.get("success_metric", "default")) != str(proto.get("success_metric", "default")):
        return None
    report = dict(d)
    report["elapsed_s"] = 0.0
    report["reused_from"] = str(path)
    report["tag"] = proto["tag"]
    report["protocol"] = proto
    return report


def run_one_env(payload: Tuple[Any, ...]) -> Dict[str, Any]:
    env, n_sweep, num_eval, seed, out_root, reuse_pusht, mode, skip_gt_eval = payload
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    out_dir = Path(out_root) / env
    out_dir.mkdir(parents=True, exist_ok=True)

    if env == "pusht" and reuse_pusht:
        reused = _reuse_pusht(Path(out_root))
        if reused is not None:
            print(f"[4env] {env}: reused PushT fair_wm_ranking", flush=True)
            return reused

    target, candidates = _candidates(env, mode=mode)
    model_path = MODELS[env]
    if not model_path.is_file():
        raise FileNotFoundError(f"missing model for {env}: {model_path}")

    sweep_rows: List[Dict[str, Any]] = []
    winner: Optional[Dict[str, Any]] = None

    if mode == "lean":
        proto = candidates[0]
        print(f"[4env] {env}: lean freeze {proto['tag']} (target>={target}%)", flush=True)
        gt_full: Optional[Dict[str, Any]] = None
        if skip_gt_eval:
            gt_full = _load_gt_cache(env, proto, num_eval=num_eval, seed=seed)
            if gt_full is not None:
                print(
                    f"[4env] {env}: reused GT cache {gt_full['success_rate']:.1f}% "
                    f"from {gt_full['reused_from']}",
                    flush=True,
                )
        if gt_full is None:
            gt_full = _run_gt(env, proto, num_eval=num_eval, seed=seed)
            print(
                f"[4env] {env} GT frozen n={num_eval}: {gt_full['success_rate']:.1f}% "
                f"elapsed={gt_full['elapsed_s']:.1f}s",
                flush=True,
            )
        winner = {
            "tag": proto["tag"],
            "success_rate": gt_full["success_rate"],
            "mean_goal_distance": gt_full.get("mean_goal_distance"),
            "elapsed_s": gt_full.get("elapsed_s"),
            "protocol": proto,
            "meets_target": gt_full["success_rate"] + 1e-9 >= target,
        }
        sweep_rows = [winner]
    else:
        print(f"[4env] {env}: full GT sweep target>={target}% n_sweep={n_sweep}", flush=True)
        for proto in candidates:
            report = _run_gt(env, proto, num_eval=n_sweep, seed=seed)
            row = {
                "tag": proto["tag"],
                "success_rate": report["success_rate"],
                "mean_goal_distance": report.get("mean_goal_distance"),
                "elapsed_s": report["elapsed_s"],
                "protocol": proto,
                "meets_target": report["success_rate"] + 1e-9 >= target,
            }
            sweep_rows.append(row)
            (out_dir / f"sweep_{proto['tag']}.json").write_text(
                json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
            )
            print(
                f"[4env] {env} GT {proto['tag']}: {report['success_rate']:.1f}% "
                f"elapsed={report['elapsed_s']:.1f}s",
                flush=True,
            )
            if row["meets_target"] and winner is None:
                winner = row
                print(f"[4env] {env}: SELECTED {proto['tag']}", flush=True)
                break
        if winner is None:
            sweep_rows.sort(key=lambda r: (-float(r["success_rate"]), float(r["elapsed_s"])))
            winner = sweep_rows[0]
            print(
                f"[4env] {env}: WARNING no protocol hit {target}%; "
                f"using best {winner['tag']} at {winner['success_rate']:.1f}%",
                flush=True,
            )
        proto = winner["protocol"]
        gt_full = _run_gt(env, proto, num_eval=num_eval, seed=seed)
        print(f"[4env] {env} GT frozen n={num_eval}: {gt_full['success_rate']:.1f}%", flush=True)

    (out_dir / "gt_sweep.json").write_text(
        json.dumps({"target": target, "mode": mode, "rows": sweep_rows, "selected": winner}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    assert gt_full is not None and winner is not None
    proto = winner["protocol"]
    (out_dir / f"mujoco_gt_n{num_eval}.json").write_text(
        json.dumps(gt_full, indent=2, sort_keys=True), encoding="utf-8"
    )

    model_rep = _run_model(env, proto, model_path, num_eval=num_eval, seed=seed)
    (out_dir / f"model_n{num_eval}.json").write_text(
        json.dumps(model_rep, indent=2, sort_keys=True), encoding="utf-8"
    )
    vs = (
        (model_rep["success_rate"] / gt_full["success_rate"])
        if gt_full["success_rate"] > 0
        else None
    )
    print(
        f"[4env] {env} model: {model_rep['success_rate']:.1f}% vs_gt={vs}",
        flush=True,
    )

    summary = {
        "env": env,
        "mode": mode,
        "gt_target": target,
        "frozen_protocol": proto,
        "gt_sweep_success": winner["success_rate"],
        "gt_reused": bool(gt_full.get("reused_from")),
        "mujoco_gt": {
            "success_rate": gt_full["success_rate"],
            "mean_goal_distance": gt_full.get("mean_goal_distance"),
            "elapsed_s": gt_full.get("elapsed_s"),
            "reused_from": gt_full.get("reused_from"),
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


def _combine_from_disk(
    out_root: Path,
    *,
    num_eval: int,
    n_sweep: int,
    seed: int,
) -> Dict[str, Any]:
    per_env: Dict[str, Any] = {}
    for env in ("tworoom", "pusht", "cube", "reacher"):
        p = out_root / env / "summary.json"
        if p.is_file():
            per_env[env] = json.loads(p.read_text(encoding="utf-8"))
    return {
        "selection_rule": (
            "Per env: choose first GT protocol with success>=target on n_sweep; "
            "freeze; evaluate canonical induced model on n_eval with identical knobs."
        ),
        "num_eval": num_eval,
        "n_sweep": n_sweep,
        "seed": seed,
        "per_env": per_env,
        "notes": [
            "general_planning_eval plans full horizon and executes receding macros only.",
            "PushT reuses fair_wm_ranking (CEM H=5 R=5 fs=1 300x10, GT 88%).",
            "Reacher freezes finger_match + frame MPC (fs=1 H=1 R=1), GT 100%.",
            "Cube/Tworoom lean freeze fair CEM fs=5 H=5 R=5 300x10 (GT ~66% / ~32%).",
            "Use --mode full to escalate planner budget when seeking higher GT ceilings.",
        ],
    }


def _write_readme(out_root: Path, table: List[Dict[str, Any]], num_eval: int, seed: int) -> None:
    lines = [
        "# Fair WM ranking — 4 LeWM environments",
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
    lines.append(f"Seed {seed}, n_eval={num_eval}. See `summary_n{num_eval}_seed{seed}.json`.")
    (out_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    envs = [e.strip() for e in args.envs.split(",") if e.strip()]
    reuse_pusht = args.reuse_pusht and not args.no_reuse_pusht
    t0 = time.time()
    summaries: Dict[str, Any] = {}

    if not args.combine_only:
        payloads = [
            (
                env,
                args.n_sweep,
                args.num_eval,
                args.seed,
                str(out_root),
                reuse_pusht,
                args.mode,
                args.skip_gt_eval,
            )
            for env in envs
        ]
        # Sequential in-process by default (spawn+script ProcessPool is brittle).
        # Parallelize across envs via multiple CLI invocations with --envs <one>.
        workers = max(1, min(int(args.env_workers), len(payloads)))
        if workers == 1 or len(payloads) == 1:
            for p in payloads:
                summaries[p[0]] = run_one_env(p)
        else:
            import multiprocessing as mp

            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
                futs = {pool.submit(run_one_env, p): p[0] for p in payloads}
                for fut in as_completed(futs):
                    env = futs[fut]
                    summaries[env] = fut.result()

    # Refresh combined summary from on-disk per-env folders so parallel
    # single-env invocations do not clobber each other.
    combined = _combine_from_disk(out_root, num_eval=args.num_eval, n_sweep=args.n_sweep, seed=args.seed)
    for env, s in summaries.items():
        combined.setdefault("per_env", {})[env] = s
    table = []
    for env in ("tworoom", "pusht", "cube", "reacher"):
        s = combined.get("per_env", {}).get(env)
        if not s:
            continue
        table.append(
            {
                "env": env,
                "gt_success": s["mujoco_gt"]["success_rate"],
                "model_success": s["model"]["success_rate"],
                "vs_gt": s["model"].get("vs_gt_ratio"),
                "protocol": s["frozen_protocol"]["tag"],
                "gt_target": s["gt_target"],
            }
        )
    combined["table"] = table
    combined["suite_elapsed_s"] = time.time() - t0
    combined["timestamp"] = datetime.now().isoformat(timespec="seconds")
    out_path = out_root / f"summary_n{args.num_eval}_seed{args.seed}.json"
    out_path.write_text(json.dumps(combined, indent=2, sort_keys=True), encoding="utf-8")
    _write_readme(out_root, table, args.num_eval, args.seed)
    print(json.dumps({"table": table, "wrote": str(out_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
