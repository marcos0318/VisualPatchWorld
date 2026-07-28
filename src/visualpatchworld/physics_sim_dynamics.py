"""Write-the-simulation dynamics: a tiny rigid-body + penalty-contact integrator.

Instead of *regressing* the contact response, this *writes out a small physics
simulation* and identifies its few physical constants from perceived data
(system identification). The block is a rigid body (pos, vel, angle, angular vel)
with table damping; the pusher is a kinematic circle; contact is resolved by a
penalty (spring) normal force + Coulomb friction, using the **perceived collision
geometry**. This naturally captures inertia/momentum, contact make/break, and
stick/slip -- the things feature-regression struggled to reproduce -- because they
are *integrated*, not learned.

Parameters identified by a differentiable K-step rollout (torch): contact
stiffness k_n, friction mu, tangential stiffness k_t, inverse inertia inv_i, and
table linear/angular damping c_lin/c_ang. The agent gain (action -> pusher
displacement) is measured directly from data. Geometry comes from
``visual_geometry`` (perceived), not the simulator source.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

from . import collision
from .pusht_local_induction import _parse_action, _parse_observation_text, _state_from_graph
from .pushing_mechanics_rollout import _build_segments

SUBSTEPS = 4


def _measure_agent_gain(trajectories, geo, margin: float = 8.0):
    polys, radius, cog = geo
    num = [0.0, 0.0]; den = [0.0, 0.0]
    for traj in trajectories:
        for t in getattr(traj, "transitions", []):
            s = _state_from_graph(_parse_observation_text(getattr(t, "observation", "")))
            ns = _state_from_graph(_parse_observation_text(getattr(t, "next_observation", "")))
            ax, ay = _parse_action(getattr(t, "action", ""))
            num[0] += (ns[0] - s[0]) * ax; num[1] += (ns[1] - s[1]) * ay
            den[0] += ax * ax; den[1] += ay * ay
    gx = num[0] / den[0] if den[0] > 1e-9 else 40.0
    gy = num[1] / den[1] if den[1] > 1e-9 else 40.0
    return [float(gx), float(gy)]


def fit_physics_sim_params(
    trajectories: Sequence[Any],
    *,
    geometry: Optional[Mapping[str, Any]] = None,
    rollout_k: int = 6,
    steps: int = 250,
    lr: float = 0.03,
    angle_weight: float = 50.0,
    max_segments: int = 4000,
    seed: int = 0,
) -> Dict[str, Any]:
    import numpy as np
    import torch

    g = geometry if geometry is not None else collision.oracle_geometry()
    polys_list = [[(float(x), float(y)) for x, y in poly] for poly in g["polygons"]]
    radius = float(g["agent_radius"]); cog = (float(g["cog"][0]), float(g["cog"][1]))
    geo = (polys_list, radius, cog)
    gain = _measure_agent_gain(trajectories, geo)
    margin = 0.0  # penalty contact uses true penetration (radius - dist)

    segs = _build_segments(trajectories, (polys_list, radius, cog), rollout_k, 8.0)
    if len(segs) > max_segments:
        rng = np.random.default_rng(seed)
        segs = [segs[i] for i in rng.choice(len(segs), max_segments, replace=False)]
    if not segs:
        raise ValueError("no near-contact segments for physics-sim fit")

    torch.manual_seed(seed)
    edges = []
    for poly in polys_list:
        n = len(poly)
        for j in range(n):
            edges.append([poly[j], poly[(j + 1) % n]])
    ea = torch.tensor([e[0] for e in edges], dtype=torch.float64)
    eb = torch.tensor([e[1] for e in edges], dtype=torch.float64)
    cog_t = torch.tensor(cog, dtype=torch.float64)

    s0 = torch.tensor([s[0] for s in segs], dtype=torch.float64)
    vel0 = torch.tensor([s[1] for s in segs], dtype=torch.float64)
    acts = torch.tensor([s[2] for s in segs], dtype=torch.float64)
    tgts = torch.tensor([s[3] for s in segs], dtype=torch.float64)
    gxt, gyt = float(gain[0]), float(gain[1])

    # learnable (positive) params via exp/softplus
    log_kn = torch.tensor(math.log(1.0), dtype=torch.float64, requires_grad=True)
    log_mu = torch.tensor(math.log(0.5), dtype=torch.float64, requires_grad=True)
    log_kt = torch.tensor(math.log(0.5), dtype=torch.float64, requires_grad=True)
    log_invi = torch.tensor(math.log(1e-3), dtype=torch.float64, requires_grad=True)
    raw_clin = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    raw_cang = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([log_kn, log_mu, log_kt, log_invi, raw_clin, raw_cang], lr=lr)

    def contact(sx, sy, bx, by, ba):
        c, s = torch.cos(ba), torch.sin(ba)
        ax_w = bx[:, None] + c[:, None] * ea[:, 0] - s[:, None] * ea[:, 1]
        ay_w = by[:, None] + s[:, None] * ea[:, 0] + c[:, None] * ea[:, 1]
        bx_w = bx[:, None] + c[:, None] * eb[:, 0] - s[:, None] * eb[:, 1]
        by_w = by[:, None] + s[:, None] * eb[:, 0] + c[:, None] * eb[:, 1]
        abx, aby = bx_w - ax_w, by_w - ay_w
        apx, apy = sx[:, None] - ax_w, sy[:, None] - ay_w
        denom = abx * abx + aby * aby + 1e-9
        t = torch.clamp((apx * abx + apy * aby) / denom, 0.0, 1.0)
        qx, qy = ax_w + t * abx, ay_w + t * aby
        d = torch.sqrt((sx[:, None] - qx) ** 2 + (sy[:, None] - qy) ** 2 + 1e-9)
        dmin, ai = torch.min(d, dim=1)
        bi = torch.arange(d.shape[0])
        cpx, cpy = qx[bi, ai], qy[bi, ai]
        nx = (sx - cpx) / (dmin + 1e-9); ny = (sy - cpy) / (dmin + 1e-9)
        return dmin, nx, ny, cpx, cpy

    sp = torch.nn.functional.softplus
    loss = torch.zeros((), dtype=torch.float64)
    dt = 1.0 / SUBSTEPS
    for it in range(steps):
        opt.zero_grad()
        sx, sy = s0[:, 0].clone(), s0[:, 1].clone()
        bx, by, ba = s0[:, 2].clone(), s0[:, 3].clone(), s0[:, 4].clone()
        vbx, vby, vw = vel0[:, 0].clone(), vel0[:, 1].clone(), vel0[:, 2].clone()
        kn, mu, kt = torch.exp(log_kn), torch.exp(log_mu), torch.exp(log_kt)
        invi = torch.exp(log_invi); clin, cang = sp(raw_clin), sp(raw_cang)
        loss = torch.zeros((), dtype=torch.float64)
        for kk in range(rollout_k):
            ax, ay = acts[:, kk, 0], acts[:, kk, 1]
            pvx, pvy = gxt * ax, gyt * ay
            for _ in range(SUBSTEPS):
                sx = sx + pvx * dt; sy = sy + pvy * dt
                dmin, nx, ny, cpx, cpy = contact(sx, sy, bx, by, ba)
                pen = torch.clamp(radius - dmin, min=0.0)
                cogx = bx + torch.cos(ba) * cog_t[0] - torch.sin(ba) * cog_t[1]
                cogy = by + torch.sin(ba) * cog_t[0] + torch.cos(ba) * cog_t[1]
                rx, ry = cpx - cogx, cpy - cogy
                vcx = vbx - vw * ry; vcy = vby + vw * rx
                rvx, rvy = vcx - pvx, vcy - pvy
                Fn = kn * pen
                dotn = rvx * nx + rvy * ny
                rtx, rty = rvx - dotn * nx, rvy - dotn * ny
                Ftx, Fty = -kt * rtx, -kt * rty
                Ftmag = torch.sqrt(Ftx * Ftx + Fty * Fty + 1e-9)
                Ftmax = mu * Fn
                scale = torch.clamp(Ftmax / (Ftmag + 1e-9), max=1.0)
                Ftx, Fty = Ftx * scale, Fty * scale
                Fx = -Fn * nx + Ftx; Fy = -Fn * ny + Fty
                torque = rx * Fy - ry * Fx
                vbx = vbx + (Fx - clin * vbx) * dt
                vby = vby + (Fy - clin * vby) * dt
                vw = vw + (invi * torque - cang * vw) * dt
                bx = bx + vbx * dt; by = by + vby * dt; ba = ba + vw * dt
            dx = bx - tgts[:, kk, 0]; dy = by - tgts[:, kk, 1]
            da = torch.remainder(ba - tgts[:, kk, 2] + math.pi, 2 * math.pi) - math.pi
            loss = loss + (dx * dx + dy * dy).mean() + angle_weight * (da * da).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([log_kn, log_mu, log_kt, log_invi, raw_clin, raw_cang], 1.0)
        opt.step()

    return {
        "agent_radius": radius,
        "tee_polygons": [list(map(list, poly)) for poly in polys_list],
        "tee_cog": list(cog),
        "gain": gain,
        "substeps": SUBSTEPS,
        "k_n": float(torch.exp(log_kn).detach()),
        "mu": float(torch.exp(log_mu).detach()),
        "k_t": float(torch.exp(log_kt).detach()),
        "inv_i": float(torch.exp(log_invi).detach()),
        "c_lin": float(sp(raw_clin).detach()),
        "c_ang": float(sp(raw_cang).detach()),
        "rollout_k": rollout_k,
        "n_segments": len(segs),
        "final_rollout_loss": float(loss.detach()),
    }


_PHYSICS_SIM_TEMPLATE = r'''
from patchworld.worldmodel_base import BaseWorldModel
import copy
import json
import math
import re


class PushtPhysicsSimWorldModel(BaseWorldModel):
    """A tiny rigid-body + penalty-contact integrator (identified parameters)."""

    AGENT_RADIUS = __AGENT_RADIUS__
    TEE_POLYGONS = __TEE_POLYGONS_JSON__
    TEE_COG = __TEE_COG_JSON__
    GAIN = __GAIN_JSON__
    SUBSTEPS = __SUBSTEPS__
    K_N = __K_N__
    MU = __MU__
    K_T = __K_T__
    INV_I = __INV_I__
    C_LIN = __C_LIN__
    C_ANG = __C_ANG__

    def init_belief(self):
        return {"env": "pusht", "source": "physics_sim", "step": 0,
                "goal": "move block to the target T pose", "objects": {}, "relations": {}, "metadata": {}}

    def init_belief_from_observation(self, obs_text):
        return self.correct_belief(self.init_belief(), obs_text)

    def extract_valid_action_forms(self):
        return {k: [k + "[dx=<float>,dy=<float>,speed=<float>]"] for k in ("move_left","move_right","move_up","move_down","hold")}

    def _rotate(self, vx, vy, a):
        c, s = math.cos(a), math.sin(a)
        return c * vx - s * vy, s * vx + c * vy

    def _contact(self, ax, ay, bx, by, ba):
        best = None
        for poly in self.TEE_POLYGONS:
            n = len(poly)
            wp = []
            for v in poly:
                rx, ry = self._rotate(v[0], v[1], ba)
                wp.append((bx + rx, by + ry))
            for i in range(n):
                p1 = wp[i]; p2 = wp[(i + 1) % n]
                ex, ey = p2[0] - p1[0], p2[1] - p1[1]
                den = ex * ex + ey * ey
                t = 0.0 if den <= 1e-12 else ((ax - p1[0]) * ex + (ay - p1[1]) * ey) / den
                t = max(0.0, min(1.0, t))
                qx, qy = p1[0] + t * ex, p1[1] + t * ey
                d = math.hypot(ax - qx, ay - qy)
                if best is None or d < best[0]:
                    best = (d, qx, qy)
        d, qx, qy = best
        nx = (ax - qx) / (d + 1e-9); ny = (ay - qy) / (d + 1e-9)
        return d, nx, ny, qx, qy

    def parse_observation(self, obs_text):
        graph = self.init_belief(); section = ""
        for raw in str(obs_text or "").splitlines():
            line = raw.strip()
            if not line: continue
            if line.startswith("Environment:"): graph["env"] = line.split(":",1)[1].strip()
            elif line.startswith("Source:"): graph["source"] = line.split(":",1)[1].strip()
            elif line.startswith("Step:"):
                try: graph["step"] = int(float(line.split(":",1)[1].strip()))
                except Exception: graph["step"] = 0
            elif line.startswith("Goal:"): graph["goal"] = line.split(":",1)[1].strip()
            elif line.startswith("Objects:"): section = "objects"
            elif line.startswith("Relations:"): section = "relations"
            elif line.startswith("Metadata:"):
                try: graph["metadata"] = json.loads(line.split(":",1)[1].strip())
                except Exception: graph["metadata"] = {}
                section = ""
            elif section == "objects" and line.startswith("-"):
                o = self._parse_object(line)
                if o: graph["objects"][o["id"]] = o
            elif section == "relations" and line.startswith("-"):
                r = self._parse_relation(line)
                if r: graph["relations"][r["key"]] = r
        self._recompute_relations(graph)
        return graph

    def _parse_object(self, line):
        m = re.match(r"^-\s*(\w+):(\w+)\s*(.*)$", line)
        if not m: return None
        o = {"id": m.group(1), "category": m.group(2), "x":0.0,"y":0.0,"vx":0.0,"vy":0.0,"w":0.0,"conf":1.0,"angle":0.0}
        for k, v in re.findall(r"(\w+)=(-?\d+(?:\.\d+)?)", m.group(3)): o[k] = float(v)
        return o

    def _parse_relation(self, line):
        m = re.match(r"^-\s*(\w+)\s+(\w+(?:_\w+)*)\s+(\w+)\s*=\s*(.+)$", line)
        if not m: return None
        val = m.group(4).strip()
        try: val = float(val)
        except Exception: pass
        return {"subject":m.group(1),"predicate":m.group(2),"object":m.group(3),"value":val,"key":f"{m.group(1)}_{m.group(2)}_{m.group(3)}"}

    def correct_belief(self, prior, obs_text):
        observed = self.parse_observation(obs_text)
        merged = copy.deepcopy(prior) if isinstance(prior, dict) else self.init_belief()
        for k in ("env","source","step","goal","metadata"): merged[k] = observed.get(k, merged.get(k))
        prior_objs = merged.get("objects", {}) if isinstance(merged.get("objects"), dict) else {}
        new_objs = observed.get("objects", {})
        block = new_objs.get("block")
        if block is not None:
            pb = prior_objs.get("block")
            if isinstance(pb, dict) and float(observed.get("step", 0) or 0) > 0:
                if "vx" not in block or float(block.get("vx", 0.0)) == 0.0:
                    block["vx"] = float(block.get("x",0.0)) - float(pb.get("x",0.0))
                    block["vy"] = float(block.get("y",0.0)) - float(pb.get("y",0.0))
                    dth = float(block.get("angle",0.0)) - float(pb.get("angle",0.0))
                    block["w"] = (dth + math.pi) % (2.0*math.pi) - math.pi
        merged["objects"] = new_objs; merged["relations"] = observed.get("relations", {})
        return merged

    def predict_belief(self, belief, action):
        b = copy.deepcopy(belief) if isinstance(belief, dict) else self.init_belief()
        objects = b.setdefault("objects", {})
        agent = objects.setdefault("agent", {"id":"agent","category":"circular_pusher","x":0.0,"y":0.0,"vx":0.0,"vy":0.0,"conf":1.0})
        block = objects.setdefault("block", {"id":"block","category":"t_block","x":0.0,"y":0.0,"angle":0.0,"vx":0.0,"vy":0.0,"w":0.0,"conf":1.0})
        ax, ay = self._parse_action(action)
        sx, sy = float(agent.get("x",0.0)), float(agent.get("y",0.0))
        bx, by, ba = float(block.get("x",0.0)), float(block.get("y",0.0)), float(block.get("angle",0.0))
        vbx, vby, vw = float(block.get("vx",0.0)), float(block.get("vy",0.0)), float(block.get("w",0.0))
        pvx, pvy = self.GAIN[0]*ax, self.GAIN[1]*ay
        dt = 1.0 / self.SUBSTEPS
        for _ in range(self.SUBSTEPS):
            sx += pvx*dt; sy += pvy*dt
            d, nx, ny, qx, qy = self._contact(sx, sy, bx, by, ba)
            pen = self.AGENT_RADIUS - d
            if pen < 0.0: pen = 0.0
            cogx, cogy = self._rotate(self.TEE_COG[0], self.TEE_COG[1], ba)
            cogx += bx; cogy += by
            rx, ry = qx - cogx, qy - cogy
            vcx = vbx - vw*ry; vcy = vby + vw*rx
            rvx, rvy = vcx - pvx, vcy - pvy
            Fn = self.K_N * pen
            dotn = rvx*nx + rvy*ny
            rtx, rty = rvx - dotn*nx, rvy - dotn*ny
            Ftx, Fty = -self.K_T*rtx, -self.K_T*rty
            Ftmag = math.hypot(Ftx, Fty); Ftmax = self.MU*Fn
            if Ftmag > Ftmax and Ftmag > 1e-9:
                s = Ftmax/Ftmag; Ftx *= s; Fty *= s
            Fx = -Fn*nx + Ftx; Fy = -Fn*ny + Fty
            torque = rx*Fy - ry*Fx
            vbx += (Fx - self.C_LIN*vbx)*dt
            vby += (Fy - self.C_LIN*vby)*dt
            vw += (self.INV_I*torque - self.C_ANG*vw)*dt
            bx += vbx*dt; by += vby*dt; ba += vw*dt
        agent["x"], agent["y"] = self._clip(sx), self._clip(sy)
        agent["vx"], agent["vy"] = pvx, pvy
        block["x"], block["y"] = self._clip(bx), self._clip(by)
        block["angle"] = ba % (2.0*math.pi)
        block["vx"], block["vy"], block["w"] = vbx, vby, vw
        b["step"] = int(b.get("step",0) or 0) + 1
        self._recompute_relations(b)
        return b

    def _parse_action(self, action):
        m = re.match(r"^\w+\[dx=(-?\d+(?:\.\d+)?),dy=(-?\d+(?:\.\d+)?),speed=(-?\d+(?:\.\d+)?)\]$", str(action or "").strip())
        if not m: return 0.0, 0.0
        return float(m.group(1)), float(m.group(2))

    def _clip(self, v):
        return max(0.0, min(512.0, float(v)))

    def _recompute_relations(self, graph):
        objects = graph.get("objects", {}); agent = objects.get("agent"); block = objects.get("block"); rels = {}
        if agent and block:
            dx = float(block.get("x",0.0)) - float(agent.get("x",0.0)); dy = float(block.get("y",0.0)) - float(agent.get("y",0.0))
            dist = math.hypot(dx, dy)
            direction = ("right" if dx>=0 else "left") if abs(dx)>=abs(dy) else ("up" if dy>=0 else "down")
            rels["agent_distance_to_block"] = {"subject":"agent","predicate":"distance_to","object":"block","value":dist,"key":"agent_distance_to_block"}
            rels["agent_near_block"] = {"subject":"agent","predicate":"near","object":"block","value":float(dist<=35.0),"key":"agent_near_block"}
            rels["agent_relative_direction_to_block"] = {"subject":"agent","predicate":"relative_direction_to","object":"block","value":direction,"key":"agent_relative_direction_to_block"}
        graph["relations"] = rels

    def readout_observation(self, belief, action=""):
        b = belief if isinstance(belief, dict) else self.init_belief()
        objects = b.get("objects", {})
        agent = objects.get("agent", {"id":"agent","category":"circular_pusher","x":0.0,"y":0.0,"vx":0.0,"vy":0.0,"conf":1.0})
        block = objects.get("block", {"id":"block","category":"t_block","x":0.0,"y":0.0,"angle":0.0,"vx":0.0,"vy":0.0,"w":0.0,"conf":1.0})
        self._recompute_relations(b)
        lines = ["Environment: pusht", f"Source: {b.get('source','physics_sim')}", f"Step: {int(b.get('step',0) or 0)}",
                 f"Goal: {b.get('goal','move block to the target T pose')}", "Objects:",
                 f"- agent:circular_pusher x={float(agent.get('x',0.0)):.3f} y={float(agent.get('y',0.0)):.3f} vx={float(agent.get('vx',0.0)):.3f} vy={float(agent.get('vy',0.0)):.3f} conf={float(agent.get('conf',1.0)):.3f}",
                 f"- block:t_block x={float(block.get('x',0.0)):.3f} y={float(block.get('y',0.0)):.3f} angle={float(block.get('angle',0.0)):.3f} vx={float(block.get('vx',0.0)):.3f} vy={float(block.get('vy',0.0)):.3f} w={float(block.get('w',0.0)):.3f} conf={float(block.get('conf',1.0)):.3f}",
                 "Relations:"]
        for key in sorted(b.get("relations", {})):
            rel = b["relations"][key]; value = rel.get("value")
            vt = f"{float(value):.3f}" if isinstance(value, (int, float)) else str(value)
            lines.append(f"- {rel.get('subject')} {rel.get('predicate')} {rel.get('object')} = {vt}")
        lines.append("Metadata: {}")
        return "\n".join(lines)
'''.strip() + "\n"


def render_physics_sim_model_code(params: Mapping[str, Any]) -> str:
    return (
        _PHYSICS_SIM_TEMPLATE
        .replace("__AGENT_RADIUS__", repr(float(params["agent_radius"])))
        .replace("__TEE_POLYGONS_JSON__", json.dumps(params["tee_polygons"]))
        .replace("__TEE_COG_JSON__", json.dumps(params["tee_cog"]))
        .replace("__GAIN_JSON__", json.dumps(params["gain"]))
        .replace("__SUBSTEPS__", repr(int(params["substeps"])))
        .replace("__K_N__", repr(float(params["k_n"])))
        .replace("__MU__", repr(float(params["mu"])))
        .replace("__K_T__", repr(float(params["k_t"])))
        .replace("__INV_I__", repr(float(params["inv_i"])))
        .replace("__C_LIN__", repr(float(params["c_lin"])))
        .replace("__C_ANG__", repr(float(params["c_ang"])))
    )


def differentiable_rollout(params: Mapping[str, Any], s0, vel0, actions):
    """Batched, differentiable physics rollout for gradient-based planning.

    ``actions`` is a torch tensor (B, H, 2) (requires_grad); returns predicted
    full states (B, H, 5) = [agent_x, agent_y, block_x, block_y, block_angle],
    differentiable in ``actions``. Mirrors the deployed integrator exactly.
    """
    import torch

    polys = [[(float(x), float(y)) for x, y in poly] for poly in params["tee_polygons"]]
    radius = float(params["agent_radius"])
    cog = (float(params["tee_cog"][0]), float(params["tee_cog"][1]))
    gx, gy = float(params["gain"][0]), float(params["gain"][1])
    substeps = int(params["substeps"])
    kn, mu, kt = float(params["k_n"]), float(params["mu"]), float(params["k_t"])
    invi, clin, cang = float(params["inv_i"]), float(params["c_lin"]), float(params["c_ang"])

    edges = []
    for poly in polys:
        n = len(poly)
        for j in range(n):
            edges.append([poly[j], poly[(j + 1) % n]])
    ea = torch.tensor([e[0] for e in edges], dtype=torch.float64)
    eb = torch.tensor([e[1] for e in edges], dtype=torch.float64)
    cog_t = torch.tensor(cog, dtype=torch.float64)

    B, H = actions.shape[0], actions.shape[1]
    s0 = torch.as_tensor(s0, dtype=torch.float64)
    vel0 = torch.as_tensor(vel0, dtype=torch.float64)
    sx = s0[0].repeat(B); sy = s0[1].repeat(B)
    bx = s0[2].repeat(B); by = s0[3].repeat(B); ba = s0[4].repeat(B)
    vbx = vel0[0].repeat(B); vby = vel0[1].repeat(B); vw = vel0[2].repeat(B)
    dt = 1.0 / substeps

    def contact(sx, sy, bx, by, ba):
        c, s = torch.cos(ba), torch.sin(ba)
        ax_w = bx[:, None] + c[:, None] * ea[:, 0] - s[:, None] * ea[:, 1]
        ay_w = by[:, None] + s[:, None] * ea[:, 0] + c[:, None] * ea[:, 1]
        bx_w = bx[:, None] + c[:, None] * eb[:, 0] - s[:, None] * eb[:, 1]
        by_w = by[:, None] + s[:, None] * eb[:, 0] + c[:, None] * eb[:, 1]
        abx, aby = bx_w - ax_w, by_w - ay_w
        apx, apy = sx[:, None] - ax_w, sy[:, None] - ay_w
        denom = abx * abx + aby * aby + 1e-9
        t = torch.clamp((apx * abx + apy * aby) / denom, 0.0, 1.0)
        qx, qy = ax_w + t * abx, ay_w + t * aby
        d = torch.sqrt((sx[:, None] - qx) ** 2 + (sy[:, None] - qy) ** 2 + 1e-9)
        dmin, ai = torch.min(d, dim=1)
        bi = torch.arange(d.shape[0])
        cpx, cpy = qx[bi, ai], qy[bi, ai]
        nx = (sx - cpx) / (dmin + 1e-9); ny = (sy - cpy) / (dmin + 1e-9)
        return dmin, nx, ny, cpx, cpy

    out = []
    for kk in range(H):
        ax, ay = actions[:, kk, 0], actions[:, kk, 1]
        pvx, pvy = gx * ax, gy * ay
        for _ in range(substeps):
            sx = sx + pvx * dt; sy = sy + pvy * dt
            dmin, nx, ny, cpx, cpy = contact(sx, sy, bx, by, ba)
            pen = torch.clamp(radius - dmin, min=0.0)
            cogx = bx + torch.cos(ba) * cog_t[0] - torch.sin(ba) * cog_t[1]
            cogy = by + torch.sin(ba) * cog_t[0] + torch.cos(ba) * cog_t[1]
            rx, ry = cpx - cogx, cpy - cogy
            vcx = vbx - vw * ry; vcy = vby + vw * rx
            rvx, rvy = vcx - pvx, vcy - pvy
            Fn = kn * pen
            dotn = rvx * nx + rvy * ny
            rtx, rty = rvx - dotn * nx, rvy - dotn * ny
            Ftx, Fty = -kt * rtx, -kt * rty
            Ftmag = torch.sqrt(Ftx * Ftx + Fty * Fty + 1e-9)
            scale = torch.clamp(mu * Fn / (Ftmag + 1e-9), max=1.0)
            Ftx, Fty = Ftx * scale, Fty * scale
            Fx = -Fn * nx + Ftx; Fy = -Fn * ny + Fty
            torque = rx * Fy - ry * Fx
            vbx = vbx + (Fx - clin * vbx) * dt
            vby = vby + (Fy - clin * vby) * dt
            vw = vw + (invi * torque - cang * vw) * dt
            bx = bx + vbx * dt; by = by + vby * dt; ba = ba + vw * dt
        out.append(torch.stack([sx, sy, bx, by, ba], dim=1))
    return torch.stack(out, dim=1)  # (B,H,5)


def generate_physics_sim_model_code(trajectories: Sequence[Any], *, geometry_json: Optional[str] = None, **kwargs: Any) -> str:
    geometry = collision.load_perceived_geometry(geometry_json) if geometry_json else None
    if geometry is None:
        import sys
        print("[physics_sim] WARNING: no --geometry_json; using ORACLE geometry (validation only).", file=sys.stderr)
    return render_physics_sim_model_code(fit_physics_sim_params(trajectories, geometry=geometry, **kwargs))
