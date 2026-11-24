#!/usr/bin/env python
"""Ceiling of a verified solver on PushT: PD pusher + true T geometry + the solver as the contact step."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/visualpatchworld"))

from visualpatchworld.llm_family.bank import load_bank  # noqa: E402
from visualpatchworld.llm_family.fit import fit_candidate, nrmse  # noqa: E402
from visualpatchworld.llm_family.traces import load_episodes, state_keys, state_scales  # noqa: E402
from pusht_contact_model import contact_step  # noqa: E402

TRUE = {"action_scale": 100.0, "kp": 100.0, "kv": 20.0, "radius": 15.0, "cog_y": 45.0,
        "mass": 1.0, "inertia": 3000.0, "substeps": 10}
SHAPE = [(-60.0, 60.0, 0.0, 30.0), (-15.0, 15.0, 30.0, 120.0)]


def make_step(solve):
    def step(state, action, p, meta):
        ax, ay = state["agent.x"], state["agent.y"]
        vx, vy = state.get("agent.vx", 0.0), state.get("agent.vy", 0.0)
        bx, by, th = state["block.x"], state["block.y"], state["block.angle"]
        a0 = max(-1.0, min(1.0, action[0]))
        a1 = max(-1.0, min(1.0, action[1]))
        tx, ty = ax + p["action_scale"] * a0, ay + p["action_scale"] * a1
        n = max(1, int(round(p["substeps"])))
        dt = 0.1 / n
        for _ in range(n):
            vx += (p["kp"] * (tx - ax) - p["kv"] * vx) * dt
            vy += (p["kp"] * (ty - ay) - p["kv"] * vy) * dt
            disc = {"x": ax, "y": ay, "vx": vx, "vy": vy, "r": p["radius"]}
            body = {"x": bx, "y": by, "ang": th, "vx": 0.0, "vy": 0.0, "w": 0.0, "shape": SHAPE,
                    "com": (0.0, p["cog_y"]), "mass": max(p["mass"], 1e-6), "inertia": max(p["inertia"], 1e-6)}
            out = solve(disc, body, dt, {"substeps": 1})
            bx, by, th = out["x"], out["y"], out["ang"]
            ax, ay = ax + vx * dt, ay + vy * dt
        s = dict(state)
        s.update({"agent.x": ax, "agent.y": ay, "agent.vx": vx, "agent.vy": vy,
                  "block.x": bx, "block.y": by, "block.angle": th % (2 * math.pi)})
        return s
    return step


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default=str(ROOT / "artifacts/visualpatchworld/llm_family/bank/qwen35_general_verified3.json"))
    ap.add_argument("--data", default=str(Path.home() / "le-wm-code/artifacts/visualpatchworld/data_stride1"))
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--identified_shape", action="store_true", help="identified grown boxes with radius 0")
    a = ap.parse_args()
    train = load_episodes(Path(a.data) / "pusht_oracle_traj_train.jsonl", max_episodes=40)
    val = load_episodes(Path(a.data) / "pusht_oracle_traj_val.jsonl", max_episodes=10)
    keys = state_keys(train)
    scales = state_scales(train, keys)
    if a.identified_shape:
        from visualpatchworld.llm_family.sysid import analyze_interactions
        SHAPE[:] = [tuple(b) for b in analyze_interactions(train, log=lambda m: None)["contact_shape"]["boxes_u_min_u_max_v_min_v_max"]]
        TRUE["radius"] = 0.0
        print("shape", SHAPE)
    tb_step, tb_params, _ = contact_step()
    c = SimpleNamespace(step=tb_step)
    print(f"textbook                     val1={nrmse(c, tb_params, val, keys, scales, k=1):.4f} "
          f"val5={nrmse(c, tb_params, val, keys, scales, k=5):.4f}")
    for e in load_bank(Path(a.bank)):
        if e.source != "general_solver":
            continue
        ns = {"math": math}
        exec(e.snippet, ns)
        step = make_step(ns["solve"])
        params = dict(TRUE)
        if a.fit:
            fixed = {"substeps": 10}
            cand = SimpleNamespace(step=lambda s, u, q, m, _st=step: _st(s, u, {**fixed, **q}, m),
                                   params={k: v for k, v in TRUE.items() if k != "substeps"})
            fr = fit_candidate(cand, train[:10], val, keys, scales, objective="multi_step",
                               rollout_k=5, restarts=1, max_nfev=60)
            params.update(fr.params)
        c = SimpleNamespace(step=step)
        print(f"{e.name:28s} val1={nrmse(c, params, val, keys, scales, k=1):.4f} "
              f"val5={nrmse(c, params, val, keys, scales, k=5):.4f}", flush=True)


if __name__ == "__main__":
    main()
