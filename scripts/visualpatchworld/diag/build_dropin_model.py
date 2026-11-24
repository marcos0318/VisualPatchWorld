#!/usr/bin/env python
"""Ceiling of the verified-solver route on PushT: identified pusher + identified shape (grown
boxes, r=0) + a verified LLM solver, with the body constants fitted; written as a planner model."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from visualpatchworld.llm_family.bank import load_bank  # noqa: E402
from visualpatchworld.llm_family.fit import fit_candidate, nrmse  # noqa: E402
from visualpatchworld.llm_family.traces import load_episodes, state_keys, state_scales  # noqa: E402

LLM = ROOT / "artifacts/visualpatchworld/llm_family"

STEP = '''# HYPOTHESIS: identified_pusher_verified_solver
# MECHANISM: identified PD pusher; contact with the identified grown box union solved by a verified quasi-static solver.
PARAMS = {"action_scale": 100.0, "kp": 100.0, "kv": 20.0, "substeps": 10, "dt": 0.01,
          "com_u": 0.0, "com_v": 40.0, "inertia": 2000.0}

def step(state, action, p, meta):
    import math
    ax, ay = state["agent.x"], state["agent.y"]
    vx, vy = state.get("agent.vx", 0.0), state.get("agent.vy", 0.0)
    bx, by, th = state["block.x"], state["block.y"], state["block.angle"]
    a0 = max(-1.0, min(1.0, action[0]))
    a1 = max(-1.0, min(1.0, action[1]))
    tx, ty = ax + p["action_scale"] * a0, ay + p["action_scale"] * a1
    n = max(1, int(round(p["substeps"])))
    dt = p["dt"]
    boxes = [tuple(b) for b in meta["contact_shape"]["boxes_u_min_u_max_v_min_v_max"]]
    for _ in range(n):
        vx += (p["kp"] * (tx - ax) - p["kv"] * vx) * dt
        vy += (p["kp"] * (ty - ay) - p["kv"] * vy) * dt
        disc = {"x": ax, "y": ay, "vx": vx, "vy": vy, "r": 0.0}
        body = {"x": bx, "y": by, "ang": th, "vx": 0.0, "vy": 0.0, "w": 0.0, "shape": boxes,
                "com": (p["com_u"], p["com_v"]), "mass": 1.0, "inertia": max(p["inertia"], 1.0)}
        out = SOLVER(disc, body, dt, {"substeps": 1})
        bx, by, th = out["x"], out["y"], out["ang"]
        ax, ay = ax + vx * dt, ay + vy * dt
    s = dict(state)
    s.update({"agent.x": ax, "agent.y": ay, "agent.vx": vx, "agent.vy": vy,
              "block.x": bx, "block.y": by, "block.angle": th % (2 * math.pi)})
    return s
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default=str(LLM / "bank/qwen35_general_verified4.json"))
    ap.add_argument("--solver", default="quasi_static_position_level")
    ap.add_argument("--template", default=str(LLM / "refine_vhelper3_pusht_qwen35_s0/models/offline_multi_step.py"))
    ap.add_argument("--out", default=str(LLM / "dropin_v4_pusht_qwen35/models/offline_multi_step.py"))
    ap.add_argument("--data", default=str(Path.home() / "le-wm-code/artifacts/visualpatchworld/data_stride1"))
    a = ap.parse_args()
    entry = next(e for e in load_bank(Path(a.bank)) if e.name == a.solver)
    src = STEP + "\n\n" + entry.snippet.replace("def solve(", "def SOLVER(", 1) + "\n"
    template = Path(a.template).read_text(encoding="utf-8")
    meta = json.loads(re.search(r"^STATIC_META = (.*)$", template, flags=re.M).group(1))
    ns: dict = {}
    exec(compile(src, "<dropin>", "exec"), ns)
    train = load_episodes(Path(a.data) / "pusht_oracle_traj_train.jsonl", max_episodes=40)
    val = load_episodes(Path(a.data) / "pusht_oracle_traj_val.jsonl", max_episodes=10)
    keys = state_keys(train)
    scales = state_scales(train, keys)
    for eps in (train, val):
        for ep in eps:
            for t in ep.transitions:
                t.meta.update(meta)
    params = dict(ns["PARAMS"])
    free = ["com_u", "com_v", "inertia"]
    fixed = {k: v for k, v in params.items() if k not in free}
    cand = SimpleNamespace(step=lambda s, u, q, m: ns["step"](s, u, {**fixed, **q}, m), params={k: params[k] for k in free})
    fr = fit_candidate(cand, train[:10], val, keys, scales, objective="multi_step", rollout_k=5, restarts=2, max_nfev=80)
    params.update(fr.params)
    c = SimpleNamespace(step=ns["step"])
    print(f"[dropin] {a.solver} params={ {k: round(v, 2) for k, v in params.items()} } "
          f"val1={nrmse(c, params, val, keys, scales, k=1):.4f} val5={nrmse(c, params, val, keys, scales, k=5):.4f}")
    out = re.sub(r"^CANDIDATE_SOURCE = .*$", lambda _: "CANDIDATE_SOURCE = " + repr(src), template, count=1, flags=re.M)
    out = re.sub(r"^FITTED_PARAMS = .*$", lambda _: "FITTED_PARAMS = " + json.dumps(params), out, count=1, flags=re.M)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(out, encoding="utf-8")
    print(f"[dropin] wrote {a.out}")


if __name__ == "__main__":
    main()
