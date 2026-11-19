#!/usr/bin/env python
"""A textbook contact model for PushT, written in the LLM-family program format.

Mechanism: the agent is a disc driven by a PD controller toward its position
plus a scaled action; the block is a planar rigid body whose shape is the
union of two rectangles. At each substep, disc-rectangle contacts receive
non-negative normal impulses (frictionless, inelastic, with Baumgarte-style
penetration bias) computed with the effective mass 1/m + (r x n)^2 / I, solved
by sequential impulses. Block velocity does not persist between substeps
(quasi-static pushing). The program uses only ``math``.

Ablations remove one ingredient at a time; ``--fit`` refits the named
constants with the pipeline's multi-step least squares.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/visualpatchworld"
sys.path.insert(0, str(ROOT / "src"))

CONTACT_SOURCE = r'''# HYPOTHESIS: pd_disc_pushes_rigid_t
# MECHANISM: PD-driven disc pusher; frictionless impulse contact against a rigid T (two rectangles); no block momentum between substeps.
PARAMS = {"action_scale": 100.0, "kp": 100.0, "kv": 20.0, "substeps": 10, "dt": 0.01,
          "radius": 15.0, "bar_len": 120.0, "bar_w": 30.0, "stem_len": 90.0, "stem_w": 30.0,
          "cog_y": 45.0, "mass": 1.0, "inertia": 3000.0, "bias": 0.0613, "slop": 0.1, "iters": 4}

def step(state, action, p, meta):
    import math
    ax, ay = state["agent.x"], state["agent.y"]
    vx, vy = state.get("agent.vx", 0.0), state.get("agent.vy", 0.0)
    bx, by, th = state["block.x"], state["block.y"], state["block.angle"]
    a0 = max(-1.0, min(1.0, action[0])) if len(action) > 0 else 0.0
    a1 = max(-1.0, min(1.0, action[1])) if len(action) > 1 else 0.0
    tx, ty = ax + p["action_scale"] * a0, ay + p["action_scale"] * a1
    n_sub = max(1, int(round(p["substeps"])))
    dt = p["dt"] * 10.0 / n_sub
    r, m, inertia = p["radius"], max(p["mass"], 1e-6), max(p["inertia"], 1e-6)
    hb, bw = 0.5 * p["bar_len"], p["bar_w"]
    hs, sl = 0.5 * p["stem_w"], p["stem_len"]
    rects = ((-hb, hb, 0.0, bw), (-hs, hs, bw, bw + sl))
    cy = p["cog_y"]
    bias_rate = p["bias"] / dt
    for _ in range(n_sub):
        vx += (p["kp"] * (tx - ax) - p["kv"] * vx) * dt
        vy += (p["kp"] * (ty - ay) - p["kv"] * vy) * dt
        c, s = math.cos(th), math.sin(th)
        dx, dy = ax - bx, ay - by
        lx, ly = c * dx + s * dy, -s * dx + c * dy
        contacts = []
        for x0, x1, y0, y1 in rects:
            qx, qy = min(max(lx, x0), x1), min(max(ly, y0), y1)
            ex, ey = lx - qx, ly - qy
            d = math.hypot(ex, ey)
            if d > 1e-9:
                if d >= r:
                    continue
                nx, ny, pen = ex / d, ey / d, r - d
            else:
                gaps = ((lx - x0, -1.0, 0.0), (x1 - lx, 1.0, 0.0), (ly - y0, 0.0, -1.0), (y1 - ly, 0.0, 1.0))
                g, nx, ny = min(gaps)
                qx, qy = lx + nx * g, ly + ny * g
                pen = r + g
            wnx, wny = c * nx - s * ny, s * nx + c * ny
            px, py = bx + c * qx - s * qy, by + s * qx + c * qy
            gx, gy = bx - s * cy, by + c * cy
            contacts.append([px - gx, py - gy, wnx, wny, pen, 0.0])
        vbx = vby = w = 0.0
        for _ in range(max(1, int(round(p["iters"])))):
            for ct in contacts:
                rx, ry, nx, ny, pen, acc = ct
                rn = rx * ny - ry * nx
                k = 1.0 / m + rn * rn / inertia
                vpx, vpy = vbx - w * ry, vby + w * rx
                vrel = (vx - vpx) * nx + (vy - vpy) * ny
                target = bias_rate * max(pen - p["slop"], 0.0)
                new = max(acc + (target - vrel) / k, 0.0)
                dj = new - acc
                ct[5] = new
                vbx -= dj * nx / m
                vby -= dj * ny / m
                w -= dj * rn / inertia
        gx, gy = bx - s * cy, by + c * cy
        gx, gy, th = gx + vbx * dt, gy + vby * dt, th + w * dt
        bx, by = gx + math.sin(th) * cy, gy - math.cos(th) * cy
        ax, ay = ax + vx * dt, ay + vy * dt
    new_state = dict(state)
    new_state.update({"agent.x": ax, "agent.y": ay, "agent.vx": vx, "agent.vy": vy,
                      "block.x": bx, "block.y": by, "block.angle": th % (2.0 * math.pi)})
    return new_state
'''

ABLATIONS = {
    "full": ({}, None),
    "one_substep": ({"substeps": 1}, None),
    "no_rotation": ({"inertia": 1e12}, None),
    "pivot_at_origin": ({"cog_y": 0.0}, None),
    "square_block": ({"square_half": 40.0},
                     ("rects = ((-hb, hb, 0.0, bw), (-hs, hs, bw, bw + sl))",
                      "rects = ((-p['square_half'], p['square_half'], p['cog_y'] - p['square_half'], p['cog_y'] + p['square_half']),)")),
}


def contact_step(ablation: str = "full"):
    overrides, patch = ABLATIONS[ablation]
    src = CONTACT_SOURCE if patch is None else CONTACT_SOURCE.replace(*patch)
    ns: dict = {}
    exec(compile(src, "<contact>", "exec"), ns)
    params = dict(ns["PARAMS"])
    params.update(overrides)
    return ns["step"], params, src


KEYS = ["agent.x", "agent.y", "block.x", "block.y", "block.angle", "agent.vx", "agent.vy"]


def _to_state(v):
    return dict(zip(KEYS, map(float, v)))


def _from_state(s):
    return np.array([s[k] for k in KEYS])


def write_model(source: str, params: dict, out: Path, template: Path, label: str) -> Path:
    src = template.read_text(encoding="utf-8")
    src = re.sub(r"^CANDIDATE_SOURCE = .*$", lambda _: "CANDIDATE_SOURCE = " + repr(source), src, count=1, flags=re.M)
    src = re.sub(r"^FITTED_PARAMS = .*$", lambda _: "FITTED_PARAMS = " + json.dumps(params), src, count=1, flags=re.M)
    src = re.sub(r'^"""LLM-proposed.*?"""', lambda _: f'"""Textbook PushT contact model ({label})."""', src, count=1, flags=re.S | re.M)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(src, encoding="utf-8")
    return out


def sim_check(n: int = 300, seed: int = 0) -> None:
    """One-env-step agreement with the simulator on random near-contact states."""
    import gymnasium as gym
    import stable_worldmodel.envs  # noqa: F401

    step, params, _ = contact_step()
    env = gym.make("swm/PushT-v1")
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    errs, moved = [], []
    t0 = time.time()
    for _ in range(n):
        bx, by, th = rng.uniform(150, 350), rng.uniform(150, 350), rng.uniform(0, 2 * np.pi)
        lx, ly = rng.uniform(-80, 80), rng.uniform(-30, 150)
        ax = bx + np.cos(th) * lx - np.sin(th) * ly
        ay = by + np.sin(th) * lx + np.cos(th) * ly
        st = np.array([ax, ay, bx, by, th, *rng.uniform(-150, 150, 2)])
        a = rng.uniform(-1, 1, 2)
        env.reset(options={"state": st})
        start = env.unwrapped._get_obs()
        env.step(a.astype(np.float32))
        true = env.unwrapped._get_obs()
        pred = _from_state(step(_to_state(start), list(a), params, {}))
        e = np.abs(pred - true)
        e[4] = min(e[4], 2 * np.pi - e[4])
        errs.append(e)
        moved.append(np.linalg.norm(true[2:4] - start[2:4]) > 0.5 or abs(true[4] - start[4]) > 0.01)
    errs, moved = np.array(errs), np.array(moved)
    print(f"[contact] {n} states, block moved in {moved.mean():.0%}; {1e3 * (time.time() - t0) / n:.1f} ms/state incl. sim")
    for name, mask in (("all", np.ones_like(moved)), ("block moved", moved)):
        med = np.median(errs[mask], axis=0)
        print(f"  median abs error ({name}): " + ", ".join(f"{k}={v:.3f}" for k, v in zip(KEYS, med)))


def main() -> None:
    from visualpatchworld.llm_family.fit import fit_candidate, nrmse
    from visualpatchworld.llm_family.traces import load_episodes, state_keys, state_scales

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sim_check", type=int, default=0)
    p.add_argument("--ablations", default=",".join(ABLATIONS))
    p.add_argument("--fit", default="", help="ablation:param+param,... constants to refit")
    p.add_argument("--data", default=str(Path.home() / "le-wm-code/artifacts/visualpatchworld/data_stride1"))
    p.add_argument("--fit_train_episodes", type=int, default=10)
    p.add_argument("--out_dir", default=str(ART / "contact_models"))
    args = p.parse_args()
    if args.sim_check:
        sim_check(args.sim_check)
    data = Path(args.data)
    train = load_episodes(data / "pusht_oracle_traj_train.jsonl", max_episodes=40)
    val = load_episodes(data / "pusht_oracle_traj_val.jsonl", max_episodes=10)
    keys = state_keys(train)
    scales = state_scales(train, keys)
    block_keys = [k for k in keys if k.startswith("block")]
    fit_spec = dict(item.split(":") for item in args.fit.split(",") if item)
    template = ART / "llm_family/v1_pusht_qwen35_s0/models/offline_one_step.py"
    report = {}
    for name in args.ablations.split(","):
        step, params, src = contact_step(name)
        if name in fit_spec:
            names = fit_spec[name].split("+")
            fixed = {k: v for k, v in params.items() if k not in names}

            def sub_step(state, action, q, meta, _fixed=fixed, _step=step):
                return _step(state, action, {**_fixed, **q}, meta)

            cand = SimpleNamespace(step=sub_step, params={k: params[k] for k in names})
            fr = fit_candidate(cand, train[: args.fit_train_episodes], val, keys, scales,
                               objective="multi_step", rollout_k=5, restarts=1, max_nfev=60)
            params.update(fr.params)
        c = SimpleNamespace(step=step)
        row = {"val1": nrmse(c, params, val, keys, scales, k=1), "val5": nrmse(c, params, val, keys, scales, k=5),
               "val5_block": nrmse(c, params, val, block_keys, scales, k=5),
               "fitted": fit_spec.get(name, ""), "params": params}
        out = Path(args.out_dir) / ("pusht_contact_textbook.py" if name == "full" else f"pusht_contact_{name}.py")
        write_model(src, params, out, template, name)
        row["model"] = str(out)
        report[name] = row
        print(f"[contact] {name:16s} val1={row['val1']:.4f} val5={row['val5']:.4f} block5={row['val5_block']:.4f} fitted={row['fitted']}", flush=True)
    (Path(args.out_dir) / "ablations.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
