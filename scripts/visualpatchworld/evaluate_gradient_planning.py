#!/usr/bin/env python
"""Gradient-based planning through the differentiable physics-sim world model.

The physics-sim is fully differentiable, so instead of CEM (which is sensitive to
the stiff penalty-contact response for off-distribution actions), we optimize the
action sequence by **gradient descent through the model rollout** (gradient-refined
shooting: a few restarts, each refined by Adam). This exploits the physics-sim's
accuracy with small local steps. Evaluated with receding-horizon execution in the
true PushT simulator + oracle re-observation (no simulator dynamics used for
planning -- only perception of the current state, which is allowed).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PATCHWORLD_ROOT = ROOT  # vendored package at repo root
for p in (str(SRC), str(PATCHWORLD_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from visualpatchworld.hdf5 import require_h5py, resolve_dataset_path  # noqa: E402
from visualpatchworld.physics_sim_dynamics import differentiable_rollout  # noqa: E402
from visualpatchworld.planning_eval import (  # noqa: E402
    _block_velocity_from_states,
    _global_index,
    pusht_goal_distance,
    pusht_success,
    rollout_sim_macro_values,
    select_lewm_eval_starts,
)
from visualpatchworld.sim_eval import _make_pusht_env, load_induced_model, resolve_generated_model_path  # noqa: E402


def _log(m):
    print(f"[visualpatchworld][{datetime.now().isoformat(timespec='seconds')}][grad-plan] {m}", flush=True)


def model_params(model):
    return {
        "agent_radius": model.AGENT_RADIUS,
        "tee_polygons": model.TEE_POLYGONS,
        "tee_cog": model.TEE_COG,
        "gain": model.GAIN,
        "substeps": model.SUBSTEPS,
        "k_n": model.K_N, "mu": model.MU, "k_t": model.K_T,
        "inv_i": model.INV_I, "c_lin": model.C_LIN, "c_ang": model.C_ANG,
    }


def plan_gradient(params, s0, vel0, goal, *, horizon, restarts, grad_steps, lr, rng, warm=None):
    goal_t = torch.tensor(np.asarray(goal, dtype=np.float64)[:5])
    s0 = np.asarray(s0, dtype=np.float64)
    init = rng.uniform(-0.6, 0.6, size=(restarts, horizon, 2))
    # Heuristic-init half the restarts to DRIVE THE AGENT TOWARD THE BLOCK so a
    # contact is established -- pure gradient planning has no signal to discover
    # contact (block cost is flat with no contact: the differentiable-sim
    # "contact gradient" problem). With contact seeded, gradients can refine the
    # push. We seed several push directions around the agent->block direction.
    rel = np.array([s0[2] - s0[0], s0[3] - s0[1]])
    dist = float(np.hypot(rel[0], rel[1]))
    if dist > 1e-6:
        base = rel / dist
        n_heur = restarts // 2
        for i in range(n_heur):
            ang = (i / max(1, n_heur - 1) - 0.5) * 1.6  # fan -0.8..0.8 rad around the direction
            c, s = math.cos(ang), math.sin(ang)
            d = np.array([c * base[0] - s * base[1], s * base[0] + c * base[1]])
            init[i] = np.clip(d[None, :] * 0.7 + rng.uniform(-0.15, 0.15, (horizon, 2)), -1.0, 1.0)
    if warm is not None:
        init[-1, : warm.shape[0]] = warm  # warm-start last restart with previous plan
    # raw -> tanh(raw) bounding; invert so tanh(raw) == init exactly
    raw = torch.tensor(np.arctanh(np.clip(init, -0.95, 0.95)), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([raw], lr=lr)
    last = max(1, horizon // 2)
    for _ in range(grad_steps):
        opt.zero_grad()
        acts = torch.tanh(raw)  # bound to (-1, 1)
        states = differentiable_rollout(params, s0, vel0, acts)  # (B,H,5)
        seg = states[:, -last:, :]  # encourage reaching/holding near the end
        pos = torch.sqrt(((seg[:, :, :4] - goal_t[:4]) ** 2).sum(2) + 1e-9)
        ang = torch.abs(torch.remainder(seg[:, :, 4] - goal_t[4] + math.pi, 2 * math.pi) - math.pi)
        cost = (pos + 10.0 * ang).min(dim=1).values  # best step in the window
        cost.sum().backward()
        opt.step()
    with torch.no_grad():
        acts = torch.tanh(raw)
        states = differentiable_rollout(params, s0, vel0, acts)
        seg = states[:, -last:, :]
        pos = torch.sqrt(((seg[:, :, :4] - goal_t[:4]) ** 2).sum(2) + 1e-9)
        ang = torch.abs(torch.remainder(seg[:, :, 4] - goal_t[4] + math.pi, 2 * math.pi) - math.pi)
        cost = (pos + 10.0 * ang).min(dim=1).values
        best = int(torch.argmin(cost))
        return acts[best].numpy()  # (H,2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="oracle")
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--dataset", default="pusht_expert_train.h5")
    ap.add_argument("--stablewm_home", default=os.environ.get("STABLEWM_HOME"))
    ap.add_argument("--num_eval", type=int, default=50)
    ap.add_argument("--goal_offset_steps", type=int, default=25)
    ap.add_argument("--eval_budget", type=int, default=50)
    ap.add_argument("--frameskip", type=int, default=1)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--receding_horizon", type=int, default=5)
    ap.add_argument("--restarts", type=int, default=24)
    ap.add_argument("--grad_steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    model_path = resolve_generated_model_path(Path(args.results_dir), args.source, None)
    _log(f"Loading physics-sim model: {model_path}")
    params = model_params(load_induced_model(model_path))

    dataset_path = resolve_dataset_path(args.dataset, args.stablewm_home)
    h5py = require_h5py()
    env = _make_pusht_env()
    successes, final_distances, starts = [], [], []
    with h5py.File(dataset_path, "r") as h5:
        states = h5["state"]
        eval_starts = select_lewm_eval_starts(h5, num_eval=args.num_eval, goal_offset_steps=args.goal_offset_steps, seed=args.seed)
        for trial_id, start in enumerate(eval_starts):
            g0 = _global_index(h5, start.episode_index, start.start_step)
            gg = _global_index(h5, start.episode_index, start.start_step + args.goal_offset_steps)
            current = np.asarray(states[g0], dtype=np.float64)
            goal = np.asarray(states[gg], dtype=np.float64)
            prev_block = np.asarray(states[g0 - 1], dtype=np.float64) if start.start_step > 0 else None
            rng = np.random.default_rng(args.seed + trial_id)
            steps_remaining = max(1, args.eval_budget // args.frameskip)
            warm = None
            while steps_remaining > 0:
                vel0 = _block_velocity_from_states(prev_block, current)
                plan = plan_gradient(
                    params, current[:5], vel0, goal,
                    horizon=args.horizon, restarts=args.restarts,
                    grad_steps=args.grad_steps, lr=args.lr, rng=rng, warm=warm,
                )
                execute = min(args.receding_horizon, steps_remaining, plan.shape[0])
                prev_block = current.copy()
                current = np.asarray(
                    rollout_sim_macro_values(env, current, plan[:execute].astype(np.float32), frameskip=args.frameskip),
                    dtype=np.float64,
                )
                steps_remaining -= execute
                warm = np.concatenate([plan[execute:], np.zeros((execute, 2))], axis=0)[: args.horizon]
            success = pusht_success(goal, current)
            successes.append(float(success))
            final_distances.append(pusht_goal_distance(goal, current))
            starts.append({"episode_index": int(start.episode_index), "start_step": int(start.start_step)})
            _log(f"trial {trial_id+1}/{len(eval_starts)} success={success} dist={final_distances[-1]:.1f}")

    summary = {
        "planner": "gradient_physics_sim",
        "num_eval": len(successes),
        "success_rate": 100.0 * float(np.mean(successes)),
        "mean_goal_distance": float(np.mean(final_distances)),
        "horizon": args.horizon, "receding_horizon": args.receding_horizon,
        "restarts": args.restarts, "grad_steps": args.grad_steps, "lr": args.lr,
        "frameskip": args.frameskip, "seed": args.seed, "model_path": str(model_path),
        "starts": starts,
    }
    out = Path(args.output) if args.output else ROOT / "artifacts" / "visualpatchworld" / "reports" / "pusht_gradient_planning.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _log(f"success_rate={summary['success_rate']:.1f} mean_goal_distance={summary['mean_goal_distance']:.2f}")
    print(json.dumps({k: v for k, v in summary.items() if k != "starts"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
