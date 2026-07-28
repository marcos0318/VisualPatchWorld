"""Faithful PushT model: PD-controlled agent + quasi-static block + swept contact.

Fixes the three structural gaps found in the ground-truth code
(`docs/pusht_ground_truth_gap.md`):

1. **PD agent with velocity state** -- the agent is a kinematic body chasing
   ``target = pos + scale*action`` via PD (``vel += (kp*(target-pos) - kv*vel)*dt``)
   over N sub-steps, with velocity carried across control steps. (Our older models
   used a memoryless linear ``gain*action`` and ignored the observed agent
   velocity.)
2. **Quasi-static block** -- ``space.damping=0`` means the block keeps no momentum
   (empirically it moves 0.000 px when not in contact). The block velocity does
   not carry; it moves only while pushed. (We drop the phantom block momentum.)
3. **Swept sub-step contact** -- contact is resolved at each of N sub-steps as the
   agent sweeps, not once per control step.

Structure (PD agent, quasi-static block, swept contact) is general physics; the
constants (scale, kp, kv, contact gains, friction) are identified from data by a
differentiable rollout. dt and sub-step count are discretization choices.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

from . import collision
from .pusht_local_induction import _angle_delta, _parse_action, _parse_observation_text, _state_from_graph

SUBSTEPS = 10
DT = 0.01


def _agent_vel_from_obs(text: str):
    g = _parse_observation_text(text)
    a = g.get("objects", {}).get("agent", {})
    return float(a.get("vx", 0.0)), float(a.get("vy", 0.0))


def _build_segments(trajectories, geo, k: int):
    """Rollout segments: (start_state5, agent_vel0, actions[k], agent_targets[k], block_targets[k])."""
    polys, radius, cog = geo
    segs = []
    for traj in trajectories:
        ts = list(getattr(traj, "transitions", []))
        if len(ts) <= k:
            continue
        states = [_state_from_graph(_parse_observation_text(getattr(t, "observation", ""))) for t in ts]
        nexts = [_state_from_graph(_parse_observation_text(getattr(t, "next_observation", ""))) for t in ts]
        avels = [_agent_vel_from_obs(getattr(t, "observation", "")) for t in ts]
        actions = [_parse_action(getattr(t, "action", "")) for t in ts]
        for i in range(len(ts) - k):
            s0 = states[i]
            c = collision.pusht_contact((s0[0], s0[1]), (s0[2], s0[3]), s0[4], agent_radius=radius, polygons=polys, cog=cog)
            if c["distance"] > radius + 30.0:
                continue  # focus on near/contact segments (free segments are trivial)
            atg = [(nexts[i + j][0], nexts[i + j][1]) for j in range(k)]
            btg = [(nexts[i + j][2], nexts[i + j][3], nexts[i + j][4]) for j in range(k)]
            acts = [actions[i + j] for j in range(k)]
            segs.append((s0, avels[i], acts, atg, btg))
    return segs


def fit_pd_physics_params(
    trajectories: Sequence[Any],
    *,
    geometry: Optional[Mapping[str, Any]] = None,
    val_trajectories: Optional[Sequence[Any]] = None,
    rollout_k: int = 6,
    val_rollout_k: int = 8,
    steps: int = 300,
    lr: float = 0.02,
    angle_weight: float = 50.0,
    agent_weight: float = 1.0,
    max_segments: int = 4000,
    restarts: int = 3,
    seed: int = 0,
) -> Dict[str, Any]:
    import numpy as np
    import torch

    g = geometry if geometry is not None else collision.oracle_geometry()
    polys = [[(float(x), float(y)) for x, y in poly] for poly in g["polygons"]]
    radius = float(g["agent_radius"]); cog = (float(g["cog"][0]), float(g["cog"][1]))
    geo = (polys, radius, cog)

    segs = _build_segments(trajectories, geo, rollout_k)
    if len(segs) > max_segments:
        rng = np.random.default_rng(seed)
        segs = [segs[i] for i in rng.choice(len(segs), max_segments, replace=False)]
    if not segs:
        raise ValueError("no segments for PD physics fit")

    # Held-out validation segments (longer horizon) for restart selection: the
    # rollout-loss optimum can drift away from the true parameters as the fit
    # overfits the contact approximation, so we select the restart by held-out
    # multi-step rollout error -- the metric that tracks planning quality.
    val_segs = None
    if val_trajectories:
        val_segs = _build_segments(val_trajectories, geo, val_rollout_k)
        if val_segs and len(val_segs) > max_segments:
            rng = np.random.default_rng(seed + 1)
            val_segs = [val_segs[i] for i in rng.choice(len(val_segs), max_segments, replace=False)]

    torch.manual_seed(seed)
    edges = []
    for poly in polys:
        n = len(poly)
        for j in range(n):
            edges.append([poly[j], poly[(j + 1) % n]])
    ea = torch.tensor([e[0] for e in edges], dtype=torch.float64)
    eb = torch.tensor([e[1] for e in edges], dtype=torch.float64)
    cog_t = torch.tensor(cog, dtype=torch.float64)

    s0 = torch.tensor([s[0] for s in segs], dtype=torch.float64)
    av0 = torch.tensor([s[1] for s in segs], dtype=torch.float64)
    acts = torch.tensor([s[2] for s in segs], dtype=torch.float64)
    atg = torch.tensor([s[3] for s in segs], dtype=torch.float64)
    btg = torch.tensor([s[4] for s in segs], dtype=torch.float64)

    if val_segs:
        vs0 = torch.tensor([s[0] for s in val_segs], dtype=torch.float64)
        vav0 = torch.tensor([s[1] for s in val_segs], dtype=torch.float64)
        vacts = torch.tensor([s[2] for s in val_segs], dtype=torch.float64)
        vbtg = torch.tensor([s[4] for s in val_segs], dtype=torch.float64)

    def contact(ax, ay, bx, by, ba):
        c, s = torch.cos(ba), torch.sin(ba)
        ax_w = bx[:, None] + c[:, None] * ea[:, 0] - s[:, None] * ea[:, 1]
        ay_w = by[:, None] + s[:, None] * ea[:, 0] + c[:, None] * ea[:, 1]
        bx_w = bx[:, None] + c[:, None] * eb[:, 0] - s[:, None] * eb[:, 1]
        by_w = by[:, None] + s[:, None] * eb[:, 0] + c[:, None] * eb[:, 1]
        abx, aby = bx_w - ax_w, by_w - ay_w
        apx, apy = ax[:, None] - ax_w, ay[:, None] - ay_w
        denom = abx * abx + aby * aby + 1e-9
        t = torch.clamp((apx * abx + apy * aby) / denom, 0.0, 1.0)
        qx, qy = ax_w + t * abx, ay_w + t * aby
        d = torch.sqrt((ax[:, None] - qx) ** 2 + (ay[:, None] - qy) ** 2 + 1e-9)
        dmin, ai = torch.min(d, dim=1)
        bi = torch.arange(d.shape[0])
        cpx, cpy = qx[bi, ai], qy[bi, ai]
        nx = (ax - cpx) / (dmin + 1e-9); ny = (ay - cpy) / (dmin + 1e-9)
        return dmin, nx, ny, cpx, cpy

    def _run_restart(rseed: int):
        """One full optimization from a (jittered) init; returns (params, final_loss)."""
        torch.manual_seed(rseed)
        gen = torch.Generator().manual_seed(rseed)
        jitter = 0.0 if rseed == seed else 0.2

        def _init(v: float):
            base = math.log(v)
            if jitter:
                base = base + float(torch.randn((), generator=gen)) * jitter
            return torch.tensor(base, dtype=torch.float64, requires_grad=True)

        # agent PD params (init near the diagnosed values; identified by fit)
        log_scale = _init(100.0); log_kp = _init(100.0); log_kv = _init(20.0)
        # quasi-static block contact params
        log_kn = _init(0.5); log_kt = _init(0.2); log_mu = _init(0.5); log_kr = _init(1e-3)
        params = [log_scale, log_kp, log_kv, log_kn, log_kt, log_mu, log_kr]
        opt = torch.optim.Adam(params, lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps), eta_min=lr * 0.05)

        loss = torch.zeros((), dtype=torch.float64)
        for _it in range(steps):
            opt.zero_grad()
            scale, kp, kv = torch.exp(log_scale), torch.exp(log_kp), torch.exp(log_kv)
            kn, kt, mu, kr = torch.exp(log_kn), torch.exp(log_kt), torch.exp(log_mu), torch.exp(log_kr)
            ax, ay = s0[:, 0].clone(), s0[:, 1].clone()
            bx, by, ba = s0[:, 2].clone(), s0[:, 3].clone(), s0[:, 4].clone()
            avx, avy = av0[:, 0].clone(), av0[:, 1].clone()
            loss = torch.zeros((), dtype=torch.float64)
            for kk in range(rollout_k):
                tx = ax + scale * acts[:, kk, 0]
                ty = ay + scale * acts[:, kk, 1]
                for _ in range(SUBSTEPS):
                    acc_x = kp * (tx - ax) - kv * avx
                    acc_y = kp * (ty - ay) - kv * avy
                    avx = avx + acc_x * DT; avy = avy + acc_y * DT
                    ax = ax + avx * DT; ay = ay + avy * DT
                    dmin, nx, ny, cpx, cpy = contact(ax, ay, bx, by, ba)
                    pen = torch.clamp(radius - dmin, min=0.0)
                    Fn = kn * pen
                    # tangential agent velocity at contact (friction drive)
                    dotv = avx * nx + avy * ny
                    tvx, tvy = avx - dotv * nx, avy - dotv * ny
                    Ftx, Fty = kt * tvx, kt * tvy
                    Ftmag = torch.sqrt(Ftx * Ftx + Fty * Fty + 1e-9)
                    fscale = torch.clamp(mu * Fn / (Ftmag + 1e-9), max=1.0)
                    Ftx, Fty = Ftx * fscale, Fty * fscale
                    Fx = -Fn * nx + Ftx; Fy = -Fn * ny + Fty   # force on block
                    cogx = bx + torch.cos(ba) * cog_t[0] - torch.sin(ba) * cog_t[1]
                    cogy = by + torch.sin(ba) * cog_t[0] + torch.cos(ba) * cog_t[1]
                    lx, ly = cpx - cogx, cpy - cogy
                    # quasi-static: block displaces proportional to force (no momentum carry)
                    bx = bx + Fx; by = by + Fy
                    ba = ba + kr * (lx * Fy - ly * Fx)
                adx = ax - atg[:, kk, 0]; ady = ay - atg[:, kk, 1]
                bdx = bx - btg[:, kk, 0]; bdy = by - btg[:, kk, 1]
                bda = torch.remainder(ba - btg[:, kk, 2] + math.pi, 2 * math.pi) - math.pi
                loss = loss + (bdx * bdx + bdy * bdy).mean() + angle_weight * (bda * bda).mean() \
                    + agent_weight * (adx * adx + ady * ady).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()

        result = {
            "agent_radius": radius,
            "tee_polygons": [list(map(list, poly)) for poly in polys],
            "tee_cog": list(cog),
            "substeps": SUBSTEPS, "dt": DT,
            "scale": float(torch.exp(log_scale).detach()),
            "kp": float(torch.exp(log_kp).detach()),
            "kv": float(torch.exp(log_kv).detach()),
            "k_n": float(torch.exp(log_kn).detach()),
            "k_t": float(torch.exp(log_kt).detach()),
            "mu": float(torch.exp(log_mu).detach()),
            "k_r": float(torch.exp(log_kr).detach()),
            "n_segments": len(segs),
            "final_loss": float(loss.detach()),
        }
        return result, float(loss.detach())

    @torch.no_grad()
    def _val_error(result: Dict[str, Any]) -> float:
        """Held-out multi-step block+angle rollout error (px-equivalent) for a fit."""
        if not val_segs:
            return float(result["final_loss"])
        scale = torch.tensor(result["scale"], dtype=torch.float64)
        kp = torch.tensor(result["kp"], dtype=torch.float64)
        kv = torch.tensor(result["kv"], dtype=torch.float64)
        kn = torch.tensor(result["k_n"], dtype=torch.float64)
        kt = torch.tensor(result["k_t"], dtype=torch.float64)
        mu = torch.tensor(result["mu"], dtype=torch.float64)
        kr = torch.tensor(result["k_r"], dtype=torch.float64)
        ax, ay = vs0[:, 0].clone(), vs0[:, 1].clone()
        bx, by, ba = vs0[:, 2].clone(), vs0[:, 3].clone(), vs0[:, 4].clone()
        avx, avy = vav0[:, 0].clone(), vav0[:, 1].clone()
        total = torch.zeros((), dtype=torch.float64)
        for kk in range(val_rollout_k):
            tx = ax + scale * vacts[:, kk, 0]
            ty = ay + scale * vacts[:, kk, 1]
            for _ in range(SUBSTEPS):
                acc_x = kp * (tx - ax) - kv * avx
                acc_y = kp * (ty - ay) - kv * avy
                avx = avx + acc_x * DT; avy = avy + acc_y * DT
                ax = ax + avx * DT; ay = ay + avy * DT
                dmin, nx, ny, cpx, cpy = contact(ax, ay, bx, by, ba)
                pen = torch.clamp(radius - dmin, min=0.0)
                Fn = kn * pen
                dotv = avx * nx + avy * ny
                tvx, tvy = avx - dotv * nx, avy - dotv * ny
                Ftx, Fty = kt * tvx, kt * tvy
                Ftmag = torch.sqrt(Ftx * Ftx + Fty * Fty + 1e-9)
                fscale = torch.clamp(mu * Fn / (Ftmag + 1e-9), max=1.0)
                Ftx, Fty = Ftx * fscale, Fty * fscale
                Fx = -Fn * nx + Ftx; Fy = -Fn * ny + Fty
                cogx = bx + torch.cos(ba) * cog_t[0] - torch.sin(ba) * cog_t[1]
                cogy = by + torch.sin(ba) * cog_t[0] + torch.cos(ba) * cog_t[1]
                lx, ly = cpx - cogx, cpy - cogy
                bx = bx + Fx; by = by + Fy
                ba = ba + kr * (lx * Fy - ly * Fx)
            bpos = torch.sqrt((bx - vbtg[:, kk, 0]) ** 2 + (by - vbtg[:, kk, 1]) ** 2 + 1e-9)
            bda = torch.remainder(ba - vbtg[:, kk, 2] + math.pi, 2 * math.pi) - math.pi
            total = total + bpos.mean() + 10.0 * bda.abs().mean()
        return float((total / val_rollout_k).detach())

    # Multi-restart parameter identification: the rollout loss is non-convex and,
    # because the model is an approximation of the true contact solver, its
    # training optimum can drift from the true parameters. Run several restarts
    # (first from the diagnosed init, the rest jittered) and select the one with
    # the lowest *held-out* multi-step rollout error -- the metric that tracks
    # planning quality (falls back to training loss when no val set is given).
    best_result: Optional[Dict[str, Any]] = None
    best_score = float("inf")
    train_losses: List[float] = []
    val_scores: List[float] = []
    for r in range(max(1, restarts)):
        result, floss = _run_restart(seed + r)
        score = _val_error(result)
        train_losses.append(floss)
        val_scores.append(score)
        if score < best_score:
            best_score, best_result = score, result
    assert best_result is not None
    best_result["restart_losses"] = train_losses
    best_result["restart_val_scores"] = val_scores
    best_result["selected_val_score"] = best_score
    best_result["n_restarts"] = max(1, restarts)
    return best_result


_PD_TEMPLATE = r'''
from patchworld.worldmodel_base import BaseWorldModel
import copy
import json
import math
import re


class PushtPDPhysicsWorldModel(BaseWorldModel):
    """Faithful PushT: PD agent (velocity state) + quasi-static block + swept contact."""

    AGENT_RADIUS = __AGENT_RADIUS__
    TEE_POLYGONS = __TEE_POLYGONS_JSON__
    TEE_COG = __TEE_COG_JSON__
    SUBSTEPS = __SUBSTEPS__
    DT = __DT__
    SCALE = __SCALE__
    KP = __KP__
    KV = __KV__
    K_N = __K_N__
    K_T = __K_T__
    MU = __MU__
    K_R = __K_R__

    def init_belief(self):
        return {"env": "pusht", "source": "pd_physics", "step": 0,
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
            n = len(poly); wp = []
            for v in poly:
                rx, ry = self._rotate(v[0], v[1], ba); wp.append((bx + rx, by + ry))
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
        merged["objects"] = observed.get("objects", {}); merged["relations"] = observed.get("relations", {})
        return merged

    def predict_belief(self, belief, action):
        b = copy.deepcopy(belief) if isinstance(belief, dict) else self.init_belief()
        objects = b.setdefault("objects", {})
        agent = objects.setdefault("agent", {"id":"agent","category":"circular_pusher","x":0.0,"y":0.0,"vx":0.0,"vy":0.0,"conf":1.0})
        block = objects.setdefault("block", {"id":"block","category":"t_block","x":0.0,"y":0.0,"angle":0.0,"vx":0.0,"vy":0.0,"w":0.0,"conf":1.0})
        adx, ady = self._parse_action(action)
        ax, ay = float(agent.get("x",0.0)), float(agent.get("y",0.0))
        avx, avy = float(agent.get("vx",0.0)), float(agent.get("vy",0.0))
        bx, by, ba = float(block.get("x",0.0)), float(block.get("y",0.0)), float(block.get("angle",0.0))
        tx, ty = ax + self.SCALE * adx, ay + self.SCALE * ady
        for _ in range(self.SUBSTEPS):
            acc_x = self.KP * (tx - ax) - self.KV * avx
            acc_y = self.KP * (ty - ay) - self.KV * avy
            avx += acc_x * self.DT; avy += acc_y * self.DT
            ax += avx * self.DT; ay += avy * self.DT
            d, nx, ny, qx, qy = self._contact(ax, ay, bx, by, ba)
            pen = self.AGENT_RADIUS - d
            if pen > 0.0:
                Fn = self.K_N * pen
                dotv = avx * nx + avy * ny
                tvx, tvy = avx - dotv * nx, avy - dotv * ny
                Ftx, Fty = self.K_T * tvx, self.K_T * tvy
                Ftmag = math.hypot(Ftx, Fty); Ftmax = self.MU * Fn
                if Ftmag > Ftmax and Ftmag > 1e-9:
                    sc = Ftmax / Ftmag; Ftx *= sc; Fty *= sc
                Fx = -Fn * nx + Ftx; Fy = -Fn * ny + Fty
                cogx, cogy = self._rotate(self.TEE_COG[0], self.TEE_COG[1], ba)
                cogx += bx; cogy += by
                lx, ly = qx - cogx, qy - cogy
                bx += Fx; by += Fy
                ba += self.K_R * (lx * Fy - ly * Fx)
        agent["x"], agent["y"] = self._clip(ax), self._clip(ay)
        agent["vx"], agent["vy"] = avx, avy
        block["x"], block["y"] = self._clip(bx), self._clip(by)
        block["angle"] = ba % (2.0 * math.pi)
        block["vx"], block["vy"], block["w"] = 0.0, 0.0, 0.0
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
        lines = ["Environment: pusht", f"Source: {b.get('source','pd_physics')}", f"Step: {int(b.get('step',0) or 0)}",
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


def render_pd_physics_model_code(params: Mapping[str, Any]) -> str:
    repl = {
        "__AGENT_RADIUS__": repr(float(params["agent_radius"])),
        "__TEE_POLYGONS_JSON__": json.dumps(params["tee_polygons"]),
        "__TEE_COG_JSON__": json.dumps(params["tee_cog"]),
        "__SUBSTEPS__": repr(int(params["substeps"])),
        "__DT__": repr(float(params["dt"])),
        "__SCALE__": repr(float(params["scale"])),
        "__KP__": repr(float(params["kp"])),
        "__KV__": repr(float(params["kv"])),
        "__K_N__": repr(float(params["k_n"])),
        "__K_T__": repr(float(params["k_t"])),
        "__MU__": repr(float(params["mu"])),
        "__K_R__": repr(float(params["k_r"])),
    }
    code = _PD_TEMPLATE
    for key, val in repl.items():
        code = code.replace(key, val)
    return code


def generate_pd_physics_model_code(trajectories: Sequence[Any], *, geometry_json: Optional[str] = None, **kwargs: Any) -> str:
    geometry = collision.load_perceived_geometry(geometry_json) if geometry_json else None
    if geometry is None:
        import sys
        print("[pd_physics] WARNING: no --geometry_json; using ORACLE geometry (validation only).", file=sys.stderr)
    return render_pd_physics_model_code(fit_pd_physics_params(trajectories, geometry=geometry, **kwargs))
