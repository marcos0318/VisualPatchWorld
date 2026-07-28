#!/usr/bin/env python
"""Build Induced vs Hybrid vs MuJoCo cost table from fair seed-42 reports."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "artifacts/visualpatchworld/reports/fair_wm_ranking_gt90"
OUT = ROOT / "artifacts/visualpatchworld/reports/hybrid_cost_table_seed42.json"

# Analytical sim queries per planning call (scoring only).
SPECS = [
    ("tworoom", "vpw_oracle_induced", 300, 10, 0.0, "cem"),
    ("tworoom", "vpw_oracle_hybrid", 300, 10, 0.3, "cem"),
    ("tworoom", "mujoco_cem", 300, 10, 1.0, "cem"),
    ("reacher", "vpw_oracle_induced", 300, 10, 0.0, "cem"),
    ("reacher", "vpw_oracle_hybrid", 300, 10, 0.3, "cem"),
    ("reacher", "mujoco_cem", 300, 10, 1.0, "cem"),
    ("pusht", "vpw_oracle_induced", 600, 15, 0.0, "cem"),
    ("pusht", "vpw_oracle_hybrid", 600, 15, 0.3, "cem"),
    ("pusht", "mujoco_cem", 600, 15, 1.0, "cem"),
    ("cube", "vpw_oracle_induced", 400, 1, 0.0, "library_shoot"),
    ("cube", "vpw_oracle_hybrid", 400, 1, 0.3, "library_shoot"),
    ("cube", "mujoco_cem", 400, 1, 1.0, "library_shoot"),
]


def _est_queries(N: int, iters: int, p: float, planner: str) -> int:
    if p <= 0:
        return 0
    if planner == "library_shoot":
        q = N if p >= 1 else math.ceil(N * p)
        if p >= 1:
            q += min(120, max(32, N // 4)) * 4  # GT polish budget heuristic
        return q
    return iters * (N if p >= 1 else math.ceil(N * p))


def _load_vpw(env: str, method: str) -> Dict[str, Any]:
    p = R / "all_methods" / f"{method}_{env}_n50_seed42.json"
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _load_mujoco(env: str) -> Dict[str, Any]:
    # Prefer timed all_methods mujoco_cem cell, then env/suite summaries, then GT caches.
    timed = R / "all_methods" / f"mujoco_cem_{env}_n50_seed42.json"
    if timed.is_file():
        d = json.loads(timed.read_text(encoding="utf-8"))
        if d.get("success_rate") is not None and d.get("error") is None:
            return d
    for cand in (R / env / "summary.json",):
        if cand.is_file():
            gt = json.loads(cand.read_text(encoding="utf-8")).get("mujoco_gt") or {}
            if gt:
                return gt
    suite = R / "summary_n50_seed42.json"
    if suite.is_file():
        d = json.loads(suite.read_text(encoding="utf-8"))
        gt = (d.get("per_env") or {}).get(env, {}).get("mujoco_gt") or {}
        if gt:
            return gt
    caches = {
        "pusht": ROOT
        / "artifacts/visualpatchworld/reports/gt90_sweep/pusht_cem_ab5_600x15_n50_proven.json",
        "cube": ROOT
        / "artifacts/visualpatchworld/reports/gt90_sweep/cube_library_shoot_budget100_n50.json",
        "tworoom": ROOT
        / "artifacts/visualpatchworld/reports/gt90_sweep/tworoom_cem_fair_fixed_n50.json",
        "reacher": ROOT
        / "artifacts/visualpatchworld/reports/reacher_improve_sim_cem_finger_frame_nowarm_n50.json",
    }
    c = caches.get(env)
    if c and c.is_file():
        return json.loads(c.read_text(encoding="utf-8"))
    return {}


def main() -> None:
    rows: List[Dict[str, Any]] = []
    for env, method, N, iters, p, planner in SPECS:
        if method.startswith("vpw"):
            d = _load_vpw(env, method)
        else:
            d = _load_mujoco(env)
        el = d.get("elapsed_s") or d.get("suite_elapsed_s")
        sr = d.get("success_rate")
        q = _est_queries(N, iters, p, planner)
        rows.append(
            {
                "env": env,
                "method": method,
                "success_rate": sr,
                "elapsed_s": el,
                "elapsed_min": (None if el is None else round(float(el) / 60.0, 2)),
                "est_sim_queries_per_plan_call": q,
                "query_frac_vs_full_mujoco": (
                    None
                    if p <= 0
                    else round(q / max(1, _est_queries(N, iters, 1.0, planner)), 3)
                ),
                "N": N,
                "iters": iters,
                "p": p,
                "planner": planner,
            }
        )

    # Relative wall-clock vs Induced within env (where both exist).
    by_env: Dict[str, Dict[str, Optional[float]]] = {}
    for r in rows:
        by_env.setdefault(r["env"], {})[r["method"]] = r["elapsed_s"]
    for r in rows:
        ind = by_env[r["env"]].get("vpw_oracle_induced")
        if ind and r["elapsed_s"]:
            r["wall_clock_vs_induced"] = round(float(r["elapsed_s"]) / float(ind), 2)
        else:
            r["wall_clock_vs_induced"] = None

    payload = {
        "seed": 42,
        "num_eval": 50,
        "note": (
            "Physics queries are analytical per planning call "
            "(Induced=0, Hybrid=ceil(0.3N)*iters, MuJoCo=N*iters). "
            "Wall-clock from fair seed-42 reports; PushT Induced uses parallel "
            "CEM while Hybrid uses serial evaluate_planning — interpret query "
            "column as the primary fair cost metric."
        ),
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    # Markdown/latex-friendly print
    print(f"{'env':8} {'method':22} {'sr':>6} {'min':>8} {'simQ/plan':>10} {'Qfrac':>6} {'vsInd':>6}")
    for r in rows:
        print(
            f"{r['env']:8} {r['method']:22} "
            f"{str(r['success_rate']):>6} "
            f"{str(r['elapsed_min']):>8} "
            f"{r['est_sim_queries_per_plan_call']:>10} "
            f"{str(r['query_frac_vs_full_mujoco']):>6} "
            f"{str(r['wall_clock_vs_induced']):>6}"
        )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
