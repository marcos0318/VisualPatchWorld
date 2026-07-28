"""Multi-step rollout-loss fit for the collision contact law (+ soft gate).

The one-step least-squares fit (``pushing_mechanics_collision``) is accurate for
a single step (~2.4 px) but compounds to ~25 px over 5 steps, which is what caps
pure-model planning. This module refits the contact-response coefficients (and
the soft-gate width, angular decay, and contact-regime agent gain) by minimizing
a **K-step rollout loss** through a differentiable rollout of the model's own
predictions, using the penetration-graded soft gate (no gate-flip discontinuity).
Geometry (polygons, agent radius, CoG) is the perceived geometry; only the small
response law is optimized.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from . import collision
from .pusht_local_induction import (
    _angle_delta,
    _parse_action,
    _parse_observation_text,
    _state_from_graph,
)
from .pushing_mechanics_collision import (
    CONTACT_MARGIN,
    fit_collision_params,
    render_collision_model_code,
)

State = Tuple[float, float, float, float, float]


def _build_segments(trajectories, geo, k: int, margin: float):
    """Near-contact rollout segments: (start_state, vel0, actions[k], block_targets[k])."""
    polys, radius, cog = geo
    segs = []
    for traj in trajectories:
        ts = list(getattr(traj, "transitions", []))
        states = [_state_from_graph(_parse_observation_text(getattr(t, "observation", ""))) for t in ts]
        nexts = [_state_from_graph(_parse_observation_text(getattr(t, "next_observation", ""))) for t in ts]
        actions = [_parse_action(getattr(t, "action", "")) for t in ts]
        for i in range(len(ts) - k):
            s0 = states[i]
            c = collision.pusht_contact((s0[0], s0[1]), (s0[2], s0[3]), s0[4], agent_radius=radius, polygons=polys, cog=cog)
            if c["distance"] > radius + margin + 6.0:
                continue  # only start segments at/near contact
            if i > 0:
                ps = states[i - 1]
                vel0 = (s0[2] - ps[2], s0[3] - ps[3], _angle_delta(s0[4], ps[4]))
            else:
                vel0 = (0.0, 0.0, 0.0)
            acts = [actions[i + j] for j in range(k)]
            targets = [(nexts[i + j][2], nexts[i + j][3], nexts[i + j][4]) for j in range(k)]
            segs.append((s0, vel0, acts, targets))
    return segs


def fit_collision_rollout_params(
    trajectories: Sequence[Any],
    *,
    geometry: Optional[Mapping[str, Any]] = None,
    contact_margin: float = CONTACT_MARGIN,
    rollout_k: int = 6,
    steps: int = 150,
    lr: float = 0.004,
    angle_weight: float = 30.0,
    gate_width: float = 2.5,
    reg_lambda: float = 20.0,
    use_l34: bool = False,
    max_segments: int = 4000,
    seed: int = 0,
) -> Dict[str, Any]:
    import numpy as np
    import torch

    base = fit_collision_params(trajectories, contact_margin=contact_margin, geometry=geometry)
    polys_list = [[(float(x), float(y)) for x, y in poly] for poly in base["tee_polygons"]]
    radius = float(base["agent_radius"])
    cog = (float(base["tee_cog"][0]), float(base["tee_cog"][1]))
    geo = (polys_list, radius, cog)
    margin = float(contact_margin)

    segs = _build_segments(trajectories, geo, rollout_k, margin)
    if len(segs) > max_segments:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(segs), max_segments, replace=False)
        segs = [segs[i] for i in idx]
    if not segs:
        base["gate_width"] = gate_width
        return base

    torch.manual_seed(seed)
    # edges across all polygons: (E,2,2)
    edges = []
    for poly in polys_list:
        n = len(poly)
        for j in range(n):
            edges.append([poly[j], poly[(j + 1) % n]])
    edge_a = torch.tensor([[e[0][0], e[0][1]] for e in edges], dtype=torch.float64)  # (E,2)
    edge_b = torch.tensor([[e[1][0], e[1][1]] for e in edges], dtype=torch.float64)
    cog_t = torch.tensor(cog, dtype=torch.float64)

    B = len(segs)
    s0 = torch.tensor([s[0] for s in segs], dtype=torch.float64)            # (B,5)
    vel0 = torch.tensor([s[1] for s in segs], dtype=torch.float64)          # (B,3)
    acts = torch.tensor([s[2] for s in segs], dtype=torch.float64)          # (B,K,2)
    tgts = torch.tensor([s[3] for s in segs], dtype=torch.float64)          # (B,K,3)

    af = torch.tensor(base["agent_free"], dtype=torch.float64)
    beta = float(base["beta"])

    # learnable params (regularized toward the good one-step init for stability)
    W0 = torch.tensor(base["contact_coef"], dtype=torch.float64)
    ac0 = torch.tensor(base["agent_contact"], dtype=torch.float64)
    W = W0.clone().requires_grad_(True)   # (3,8)
    ac = ac0.clone().requires_grad_(True)  # (2,)
    beta_w = torch.tensor(float(base["beta_w"]), dtype=torch.float64, requires_grad=True)
    resist_raw = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)  # sigmoid -> [0,1]
    gw = torch.tensor(float(gate_width), dtype=torch.float64)  # fixed (small => smooth boundary only)
    # L3 (contact-resist) + L4 (non-penetration projection) are gated off by
    # default: empirically they rarely trigger on PushT (the response already
    # keeps the agent/block non-overlapping) and including them in the fit
    # perturbs W to a worse multi-step solution. Kept available for other envs.
    params_to_opt = [W, ac, beta_w] + ([resist_raw] if use_l34 else [])
    opt = torch.optim.Adam(params_to_opt, lr=lr)
    # per-feature scale to make the L2 penalty meaningful across very different
    # feature magnitudes (torque ~100s, pen ~10s, others ~ single digits)
    feat_scale = torch.tensor([1.0, 1.0, 1.0, 1.0, 100.0, 1.0, 10.0, 1.0], dtype=torch.float64)

    def contact(sx, sy, bx, by, ba):
        c, s = torch.cos(ba), torch.sin(ba)
        # rotate edges to world: (B,E,2)
        ex_a = bx[:, None] + c[:, None] * edge_a[:, 0] - s[:, None] * edge_a[:, 1]
        ey_a = by[:, None] + s[:, None] * edge_a[:, 0] + c[:, None] * edge_a[:, 1]
        ex_b = bx[:, None] + c[:, None] * edge_b[:, 0] - s[:, None] * edge_b[:, 1]
        ey_b = by[:, None] + s[:, None] * edge_b[:, 0] + c[:, None] * edge_b[:, 1]
        abx, aby = ex_b - ex_a, ey_b - ey_a
        apx, apy = sx[:, None] - ex_a, sy[:, None] - ey_a
        denom = abx * abx + aby * aby + 1e-9
        t = torch.clamp((apx * abx + apy * aby) / denom, 0.0, 1.0)
        qx, qy = ex_a + t * abx, ey_a + t * aby
        d = torch.sqrt((sx[:, None] - qx) ** 2 + (sy[:, None] - qy) ** 2 + 1e-9)
        dmin, ai = torch.min(d, dim=1)
        bi = torch.arange(d.shape[0])
        cpx, cpy = qx[bi, ai], qy[bi, ai]
        nx = (sx - cpx) / (dmin + 1e-9)
        ny = (sy - cpy) / (dmin + 1e-9)
        cogx = bx + c * cog_t[0] - s * cog_t[1]
        cogy = by + s * cog_t[0] + c * cog_t[1]
        return dmin, nx, ny, (cpx - cogx), (cpy - cogy)

    loss = torch.zeros((), dtype=torch.float64)
    for it in range(steps):
        opt.zero_grad()
        sx, sy = s0[:, 0].clone(), s0[:, 1].clone()
        bx, by, ba = s0[:, 2].clone(), s0[:, 3].clone(), s0[:, 4].clone()
        vbx, vby, vw = vel0[:, 0].clone(), vel0[:, 1].clone(), vel0[:, 2].clone()
        loss = torch.zeros((), dtype=torch.float64)
        resist = torch.sigmoid(resist_raw) if use_l34 else torch.zeros((), dtype=torch.float64)
        for kk in range(rollout_k):
            ax, ay = acts[:, kk, 0], acts[:, kk, 1]
            dmin, nx, ny, lx, ly = contact(sx, sy, bx, by, ba)
            pen = radius + margin - dmin
            gate = torch.sigmoid(pen / gw)
            gx = af[0] + gate * (ac[0] - af[0])
            gy = af[1] + gate * (ac[1] - af[1])
            dax, day = gx * ax, gy * ay  # commanded (drives the block response)
            push = torch.clamp(-(dax * nx + day * ny), min=0.0)
            torque = lx * day - ly * dax
            feats = torch.stack([dax, day, push * (-nx), push * (-ny), torque, push, pen, torch.ones_like(dax)], dim=1)  # (B,8)
            vbx = beta * vbx + gate * (feats @ W[0])
            vby = beta * vby + gate * (feats @ W[1])
            vw = beta_w * vw + gate * (feats @ W[2])
            # L3: contact-resisted agent (block resists motion into the surface)
            da_ax = dax + resist * gate * push * nx
            da_ay = day + resist * gate * push * ny
            nsx, nsy = sx + da_ax, sy + da_ay
            nbx, nby, nba = bx + vbx, by + vby, ba + vw
            if use_l34:
                # L4: non-penetration projection
                dmin2, nx2, ny2, _, _ = contact(nsx, nsy, nbx, nby, nba)
                overlap = torch.clamp(radius - dmin2, min=0.0)
                nbx = nbx - overlap * nx2
                nby = nby - overlap * ny2
            sx, sy = nsx, nsy
            bx, by, ba = nbx, nby, nba
            dx = bx - tgts[:, kk, 0]
            dy = by - tgts[:, kk, 1]
            da = torch.remainder(ba - tgts[:, kk, 2] + math.pi, 2 * math.pi) - math.pi
            loss = loss + (dx * dx + dy * dy).mean() + angle_weight * (da * da).mean()
        # L2 toward the one-step solution (scaled per feature) keeps the refit a
        # gentle correction instead of a divergent from-scratch fit
        reg = reg_lambda * ((((W - W0) * feat_scale) ** 2).sum() + ((ac - ac0) ** 2).sum())
        (loss + reg).backward()
        torch.nn.utils.clip_grad_norm_([W, ac, beta_w, resist_raw], 1.0)
        opt.step()

    params = dict(base)
    params["contact_coef"] = [[float(v) for v in row] for row in W.detach().tolist()]
    params["agent_contact"] = [float(v) for v in ac.detach().tolist()]
    params["beta_w"] = float(beta_w.detach())
    params["gate_width"] = float(gate_width)
    params["resist"] = float(torch.sigmoid(resist_raw).detach()) if use_l34 else 0.0
    params["project"] = 1.0 if use_l34 else 0.0
    params["rollout_k"] = rollout_k
    params["n_segments"] = len(segs)
    params["final_rollout_loss"] = float(loss.detach())
    return params


def generate_collision_rollout_model_code(
    trajectories: Sequence[Any],
    *,
    geometry_json: Optional[str] = None,
    **kwargs: Any,
) -> str:
    geometry = collision.load_perceived_geometry(geometry_json) if geometry_json else None
    if geometry is None:
        import sys

        print(
            "[pushing_mechanics_rollout] WARNING: no --geometry_json; using ORACLE geometry (validation only).",
            file=sys.stderr,
        )
    params = fit_collision_rollout_params(trajectories, geometry=geometry, **kwargs)
    return render_collision_model_code(params)
