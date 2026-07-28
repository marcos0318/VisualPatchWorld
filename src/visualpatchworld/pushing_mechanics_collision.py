"""Collision-geometry contact dynamics for PushT (the §8 generalizable design).

This realizes the shared-substrate principle from
``docs/scientific_discovery_dynamics.md`` §8: the contact law is induced on a
*geometry-aware* basis derived from a general collision-detection primitive
(circle vs. polygons), using the **actual contact point** on the object boundary,
the local surface **normal**, and the **lever arm about the centre of gravity** --
the inputs the agent->centre proxy models lacked. Empirically this is the first
PushT contact law to beat the 18-feature ridge on one-step contact translation
*and* rotation.

The per-object geometry (T polygons, agent radius, CoG) is the only
environment-specific input and is perceivable; the collision detection and the
small response law are general.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

from . import collision
from .pusht_local_induction import (
    WeightedTransition,
    _angle_delta,
    _parse_action,
    _parse_observation_text,
    _state_from_graph,
    iter_weighted_transitions,
)

CONTACT_MARGIN = 8.0  # contact if agent centre within (radius + margin) of surface


def _geo_tuple(geometry: Optional[Mapping[str, Any]]):
    if geometry is None:
        g = collision.oracle_geometry()
    else:
        g = geometry
    polys = [[(float(x), float(y)) for x, y in poly] for poly in g["polygons"]]
    return polys, float(g["agent_radius"]), (float(g["cog"][0]), float(g["cog"][1]))


def _surface_distance(s, geo) -> float:
    polys, radius, cog = geo
    c = collision.pusht_contact((s[0], s[1]), (s[2], s[3]), s[4], agent_radius=radius, polygons=polys, cog=cog)
    return float(c["distance"])


def _in_contact(s, geo, margin: float = CONTACT_MARGIN) -> bool:
    return _surface_distance(s, geo) <= geo[1] + margin


def _fit_agent_gain(rows, *, contact: bool, margin: float, geo) -> List[float]:
    num = [0.0, 0.0]
    den = [0.0, 0.0]
    for row in rows:
        s = _state_from_graph(_parse_observation_text(row.observation))
        ns = _state_from_graph(_parse_observation_text(row.next_observation))
        ax, ay = _parse_action(row.action)
        if _in_contact(s, geo, margin) != contact:
            continue
        w = float(row.weight)
        num[0] += w * (ns[0] - s[0]) * ax
        num[1] += w * (ns[1] - s[1]) * ay
        den[0] += w * ax * ax
        den[1] += w * ay * ay
    gx = num[0] / den[0] if den[0] > 1e-9 else 200.0
    gy = num[1] / den[1] if den[1] > 1e-9 else 200.0
    return [float(gx), float(gy)]


def _fit_momentum_decay(trajectories, margin: float) -> List[float]:
    """Free-flight decay measured on clearly-coasting transitions.

    Identified on transitions where (a) the agent is clearly not contacting the
    block (large agent->block-origin distance) and (b) the block was actually
    moving last step. Using the contact gate alone fails at frame resolution
    because the pusher stays near the block whenever the block moves, so almost
    no "far + coasting" samples exist; a center-distance "clearly free" criterion
    recovers them.
    """
    num_l = den_l = num_a = den_a = 0.0
    for traj in trajectories:
        ts = list(getattr(traj, "transitions", []))
        for i in range(1, len(ts)):
            s = _state_from_graph(_parse_observation_text(getattr(ts[i], "observation", "")))
            ns = _state_from_graph(_parse_observation_text(getattr(ts[i], "next_observation", "")))
            ps = _state_from_graph(_parse_observation_text(getattr(ts[i - 1], "observation", "")))
            pns = _state_from_graph(_parse_observation_text(getattr(ts[i - 1], "next_observation", "")))
            # clearly free of the pusher (origin distance well beyond the T extent)
            if math.hypot(s[2] - s[0], s[3] - s[1]) < 130.0:
                continue
            cdx, cdy = ns[2] - s[2], ns[3] - s[3]
            pdx, pdy = pns[2] - ps[2], pns[3] - ps[3]
            if math.hypot(pdx, pdy) > 0.5:
                num_l += pdx * cdx + pdy * cdy
                den_l += pdx * pdx + pdy * pdy
            pw = _angle_delta(pns[4], ps[4])
            if abs(pw) > 0.003:
                cw = _angle_delta(ns[4], s[4])
                num_a += pw * cw
                den_a += pw * pw
    beta = max(0.0, min(0.98, num_l / den_l if den_l > 1e-6 else 0.85))
    beta_w = max(0.0, min(0.98, num_a / den_a if den_a > 1e-6 else 0.7))
    return [float(beta), float(beta_w)]


def _fit_contact_coef(rows, agent_gain_contact, *, margin: float, ridge: float, geo) -> List[List[float]]:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("collision dynamics fitting requires numpy") from exc
    polys, radius, cog = geo
    gx, gy = float(agent_gain_contact[0]), float(agent_gain_contact[1])
    feats: List[List[float]] = []
    targets: List[List[float]] = []
    weights: List[float] = []
    for row in rows:
        s = _state_from_graph(_parse_observation_text(row.observation))
        ns = _state_from_graph(_parse_observation_text(row.next_observation))
        ax, ay = _parse_action(row.action)
        if not _in_contact(s, geo, margin):
            continue
        da = (gx * ax, gy * ay)
        feats.append(collision.pusht_contact_features((s[0], s[1]), (s[2], s[3]), s[4], da, agent_radius=radius, polygons=polys, cog=cog))
        targets.append([ns[2] - s[2], ns[3] - s[3], _angle_delta(ns[4], s[4])])
        weights.append(float(row.weight))
    n = collision.N_PUSHT_CONTACT_FEATS
    if len(feats) < n:
        return [[0.0] * n for _ in range(3)]
    x = np.asarray(feats)
    y = np.asarray(targets)
    w = np.sqrt(np.clip(np.asarray(weights), 1e-6, None))[:, None]
    xw, yw = x * w, y * w
    coef = np.linalg.solve(xw.T @ xw + ridge * np.eye(x.shape[1]), xw.T @ yw).T
    return [[float(v) for v in r] for r in coef.tolist()]


def fit_collision_params(
    trajectories: Sequence[Any],
    *,
    ridge: float = 1e-2,
    contact_margin: float = CONTACT_MARGIN,
    weighted_transitions: Optional[Sequence[WeightedTransition]] = None,
    geometry: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Fit the geometry-aware contact law.

    ``geometry`` is the perceived geometry (from
    ``visual_geometry.detect_and_calibrate`` / the detect_geometry CLI). If None,
    falls back to the oracle (source) geometry -- validation only.
    """
    geo = _geo_tuple(geometry)
    polys, radius, cog = geo
    rows = (
        list(weighted_transitions)
        if weighted_transitions is not None
        else iter_weighted_transitions(trajectories, default_weight=1.0)
    )
    agent_free = _fit_agent_gain(rows, contact=False, margin=contact_margin, geo=geo)
    agent_contact = _fit_agent_gain(rows, contact=True, margin=contact_margin, geo=geo)
    decay = _fit_momentum_decay(trajectories, contact_margin)
    contact_coef = _fit_contact_coef(rows, agent_contact, margin=contact_margin, ridge=ridge, geo=geo)
    return {
        "agent_radius": radius,
        "tee_polygons": [list(map(list, poly)) for poly in polys],
        "tee_cog": list(cog),
        "geometry_source": "perceived" if geometry is not None else "oracle",
        "contact_margin": float(contact_margin),
        "gate_width": 0.0,  # 0 => hard gate (one-step fit); rollout fit sets >0
        "resist": 0.0,      # L3 contact resistance (rollout fit sets >0)
        "project": 0.0,     # L4 non-penetration projection (rollout fit sets 1)
        "agent_free": agent_free,
        "agent_contact": agent_contact,
        "beta": decay[0],
        "beta_w": decay[1],
        "contact_coef": contact_coef,
        "n_rows": len(rows),
    }


_COLLISION_TEMPLATE = r'''
from patchworld.worldmodel_base import BaseWorldModel
import copy
import json
import math
import re


class PushtCollisionWorldModel(BaseWorldModel):
    """Contact dynamics on a general circle-vs-polygon collision primitive."""

    AGENT_RADIUS = __AGENT_RADIUS__
    TEE_POLYGONS = __TEE_POLYGONS_JSON__
    TEE_COG = __TEE_COG_JSON__
    CONTACT_MARGIN = __CONTACT_MARGIN__
    GATE_WIDTH = __GATE_WIDTH__
    RESIST = __RESIST__
    PROJECT = __PROJECT__
    AGENT_FREE = __AGENT_FREE_JSON__
    AGENT_CONTACT = __AGENT_CONTACT_JSON__
    BETA = __BETA__
    BETA_W = __BETA_W__
    CONTACT_COEF = __CONTACT_COEF_JSON__

    def init_belief(self):
        return {"env": "pusht", "source": "collision", "step": 0,
                "goal": "move block to the target T pose", "objects": {}, "relations": {}, "metadata": {}}

    def init_belief_from_observation(self, obs_text: str):
        return self.correct_belief(self.init_belief(), obs_text)

    def extract_valid_action_forms(self):
        return {k: [k + "[dx=<float>,dy=<float>,speed=<float>]"] for k in ("move_left", "move_right", "move_up", "move_down", "hold")}

    # --- geometry primitives (general: circle vs convex polygons) ---
    def _rotate(self, vx, vy, angle):
        c, s = math.cos(angle), math.sin(angle)
        return (c * vx - s * vy, s * vx + c * vy)

    def _to_world(self, lx, ly, ox, oy, angle):
        rx, ry = self._rotate(lx, ly, angle)
        return (ox + rx, oy + ry)

    def _closest_on_seg(self, px, py, ax, ay, bx, by):
        abx, aby = bx - ax, by - ay
        denom = abx * abx + aby * aby
        t = 0.0 if denom <= 1e-12 else ((px - ax) * abx + (py - ay) * aby) / denom
        t = max(0.0, min(1.0, t))
        qx, qy = ax + t * abx, ay + t * aby
        return qx, qy, math.hypot(px - qx, py - qy)

    def _point_in_convex(self, px, py, poly):
        # ray casting: handles convex AND non-convex (perceived) polygons
        inside = False
        n = len(poly)
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi):
                inside = not inside
            j = i
        return inside

    def _contact(self, agent_x, agent_y, bx, by, angle):
        polys = []
        for poly in self.TEE_POLYGONS:
            polys.append([self._to_world(v[0], v[1], bx, by, angle) for v in poly])
        best = None
        for poly in polys:
            bq, bd = (poly[0][0], poly[0][1]), float("inf")
            n = len(poly)
            for i in range(n):
                qx, qy, d = self._closest_on_seg(agent_x, agent_y, poly[i][0], poly[i][1], poly[(i + 1) % n][0], poly[(i + 1) % n][1])
                if d < bd:
                    bd, bq = d, (qx, qy)
            inside = self._point_in_convex(agent_x, agent_y, poly)
            score = -1e9 if inside else bd
            if best is None or score < best[0]:
                best = (score, bq, bd, inside)
        _, q, d, inside = best
        if d > 1e-9:
            nx, ny = (agent_x - q[0]) / d, (agent_y - q[1]) / d
        else:
            nx, ny = 0.0, 0.0
        if inside:
            nx, ny = -nx, -ny
            pen = self.AGENT_RADIUS + d
        else:
            pen = self.AGENT_RADIUS - d
        cogx, cogy = self._to_world(self.TEE_COG[0], self.TEE_COG[1], bx, by, angle)
        return {"q": q, "n": (nx, ny), "d": d, "pen": pen, "lever": (q[0] - cogx, q[1] - cogy)}

    def _contact_features(self, agent_x, agent_y, bx, by, angle, dax, day):
        c = self._contact(agent_x, agent_y, bx, by, angle)
        nx, ny = c["n"]
        push = -(dax * nx + day * ny)
        if push < 0.0:
            push = 0.0
        lx, ly = c["lever"]
        torque = lx * day - ly * dax
        return [dax, day, push * (-nx), push * (-ny), torque, push, c["pen"], 1.0], c["d"]

    def parse_observation(self, obs_text: str):
        graph = self.init_belief()
        section = ""
        for raw in str(obs_text or "").splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith("Environment:"):
                graph["env"] = line.split(":", 1)[1].strip()
            elif line.startswith("Source:"):
                graph["source"] = line.split(":", 1)[1].strip()
            elif line.startswith("Step:"):
                try:
                    graph["step"] = int(float(line.split(":", 1)[1].strip()))
                except Exception:
                    graph["step"] = 0
            elif line.startswith("Goal:"):
                graph["goal"] = line.split(":", 1)[1].strip()
            elif line.startswith("Objects:"):
                section = "objects"
            elif line.startswith("Relations:"):
                section = "relations"
            elif line.startswith("Metadata:"):
                try:
                    graph["metadata"] = json.loads(line.split(":", 1)[1].strip())
                except Exception:
                    graph["metadata"] = {}
                section = ""
            elif section == "objects" and line.startswith("-"):
                obj = self._parse_object(line)
                if obj:
                    graph["objects"][obj["id"]] = obj
            elif section == "relations" and line.startswith("-"):
                rel = self._parse_relation(line)
                if rel:
                    graph["relations"][rel["key"]] = rel
        self._recompute_relations(graph)
        return graph

    def _parse_object(self, line):
        m = re.match(r"^-\s*(\w+):(\w+)\s*(.*)$", line)
        if not m:
            return None
        obj = {"id": m.group(1), "category": m.group(2), "x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "w": 0.0, "conf": 1.0, "angle": 0.0}
        for k, v in re.findall(r"(\w+)=(-?\d+(?:\.\d+)?)", m.group(3)):
            obj[k] = float(v)
        return obj

    def _parse_relation(self, line):
        m = re.match(r"^-\s*(\w+)\s+(\w+(?:_\w+)*)\s+(\w+)\s*=\s*(.+)$", line)
        if not m:
            return None
        val = m.group(4).strip()
        try:
            val = float(val)
        except Exception:
            pass
        return {"subject": m.group(1), "predicate": m.group(2), "object": m.group(3), "value": val, "key": f"{m.group(1)}_{m.group(2)}_{m.group(3)}"}

    def correct_belief(self, belief_prior, obs_text: str):
        observed = self.parse_observation(obs_text)
        merged = copy.deepcopy(belief_prior) if isinstance(belief_prior, dict) else self.init_belief()
        for key in ("env", "source", "step", "goal", "metadata"):
            merged[key] = observed.get(key, merged.get(key))
        prior_objs = merged.get("objects", {}) if isinstance(merged.get("objects"), dict) else {}
        new_objs = observed.get("objects", {})
        block = new_objs.get("block")
        if block is not None:
            prior_block = prior_objs.get("block")
            if isinstance(prior_block, dict) and float(observed.get("step", 0) or 0) > 0:
                if "vx" not in block or float(block.get("vx", 0.0)) == 0.0:
                    block["vx"] = float(block.get("x", 0.0)) - float(prior_block.get("x", 0.0))
                    block["vy"] = float(block.get("y", 0.0)) - float(prior_block.get("y", 0.0))
                    dth = float(block.get("angle", 0.0)) - float(prior_block.get("angle", 0.0))
                    block["w"] = (dth + math.pi) % (2.0 * math.pi) - math.pi
        merged["objects"] = new_objs
        merged["relations"] = observed.get("relations", {})
        return merged

    def predict_belief(self, belief, action: str):
        b = copy.deepcopy(belief) if isinstance(belief, dict) else self.init_belief()
        objects = b.setdefault("objects", {})
        agent = objects.setdefault("agent", {"id": "agent", "category": "circular_pusher", "x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "conf": 1.0})
        block = objects.setdefault("block", {"id": "block", "category": "t_block", "x": 0.0, "y": 0.0, "angle": 0.0, "vx": 0.0, "vy": 0.0, "w": 0.0, "conf": 1.0})
        ax, ay = self._parse_action(action)
        sx, sy = float(agent.get("x", 0.0)), float(agent.get("y", 0.0))
        bx, by, ba = float(block.get("x", 0.0)), float(block.get("y", 0.0)), float(block.get("angle", 0.0))
        c0 = self._contact(sx, sy, bx, by, ba)
        nx0, ny0 = c0["n"]
        pen = self.AGENT_RADIUS + self.CONTACT_MARGIN - c0["d"]
        if self.GATE_WIDTH > 0.0:
            # penetration-graded soft gate: contact turns on/off smoothly,
            # removing the gate-flip discontinuity that amplifies rollout drift
            gate = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, pen / self.GATE_WIDTH))))
        else:
            gate = 1.0 if c0["d"] <= self.AGENT_RADIUS + self.CONTACT_MARGIN else 0.0
        gx = self.AGENT_FREE[0] + gate * (self.AGENT_CONTACT[0] - self.AGENT_FREE[0])
        gy = self.AGENT_FREE[1] + gate * (self.AGENT_CONTACT[1] - self.AGENT_FREE[1])
        dax, day = gx * ax, gy * ay
        vbx = self.BETA * float(block.get("vx", 0.0))
        vby = self.BETA * float(block.get("vy", 0.0))
        vw = self.BETA_W * float(block.get("w", 0.0))
        if gate > 1e-4:
            feats, _ = self._contact_features(sx, sy, bx, by, ba, dax, day)
            d0 = sum(c * f for c, f in zip(self.CONTACT_COEF[0], feats))
            d1 = sum(c * f for c, f in zip(self.CONTACT_COEF[1], feats))
            d2 = sum(c * f for c, f in zip(self.CONTACT_COEF[2], feats))
            vbx += gate * d0
            vby += gate * d1
            vw += gate * d2
        # L3: contact-resisted agent -- the block resists the agent's commanded
        # motion *into* the surface (the pusher cannot drive through the block)
        into = -(dax * nx0 + day * ny0)
        if self.RESIST > 0.0 and into > 0.0:
            dax += self.RESIST * gate * into * nx0
            day += self.RESIST * gate * into * ny0
        nsx, nsy = self._clip(sx + dax), self._clip(sy + day)
        nbx = self._clip(bx + vbx)
        nby = self._clip(by + vby)
        nba = (ba + vw) % (2.0 * math.pi)
        # L4: non-penetration projection -- resolve any residual agent/block
        # overlap by pushing the block out along the contact normal; bounds drift
        if self.PROJECT > 0.0:
            c1 = self._contact(nsx, nsy, nbx, nby, nba)
            overlap = self.AGENT_RADIUS - c1["d"]
            if overlap > 0.0:
                nbx = self._clip(nbx - self.PROJECT * overlap * c1["n"][0])
                nby = self._clip(nby - self.PROJECT * overlap * c1["n"][1])
        agent["x"], agent["y"] = nsx, nsy
        agent["vx"], agent["vy"] = dax, day
        block["x"], block["y"], block["angle"] = nbx, nby, nba
        block["vx"], block["vy"], block["w"] = vbx, vby, vw
        b["step"] = int(b.get("step", 0) or 0) + 1
        self._recompute_relations(b)
        return b

    def _parse_action(self, action):
        m = re.match(r"^\w+\[dx=(-?\d+(?:\.\d+)?),dy=(-?\d+(?:\.\d+)?),speed=(-?\d+(?:\.\d+)?)\]$", str(action or "").strip())
        if not m:
            return 0.0, 0.0
        return float(m.group(1)), float(m.group(2))

    def _clip(self, v):
        return max(0.0, min(512.0, float(v)))

    def _recompute_relations(self, graph):
        objects = graph.get("objects", {})
        agent = objects.get("agent")
        block = objects.get("block")
        rels = {}
        if agent and block:
            dx = float(block.get("x", 0.0)) - float(agent.get("x", 0.0))
            dy = float(block.get("y", 0.0)) - float(agent.get("y", 0.0))
            dist = math.hypot(dx, dy)
            direction = ("right" if dx >= 0 else "left") if abs(dx) >= abs(dy) else ("up" if dy >= 0 else "down")
            rels["agent_distance_to_block"] = {"subject": "agent", "predicate": "distance_to", "object": "block", "value": dist, "key": "agent_distance_to_block"}
            rels["agent_near_block"] = {"subject": "agent", "predicate": "near", "object": "block", "value": float(dist <= 35.0), "key": "agent_near_block"}
            rels["agent_relative_direction_to_block"] = {"subject": "agent", "predicate": "relative_direction_to", "object": "block", "value": direction, "key": "agent_relative_direction_to_block"}
        graph["relations"] = rels

    def readout_observation(self, belief, action: str = ""):
        b = belief if isinstance(belief, dict) else self.init_belief()
        objects = b.get("objects", {})
        agent = objects.get("agent", {"id": "agent", "category": "circular_pusher", "x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "conf": 1.0})
        block = objects.get("block", {"id": "block", "category": "t_block", "x": 0.0, "y": 0.0, "angle": 0.0, "vx": 0.0, "vy": 0.0, "w": 0.0, "conf": 1.0})
        self._recompute_relations(b)
        lines = [
            "Environment: pusht",
            f"Source: {b.get('source', 'collision')}",
            f"Step: {int(b.get('step', 0) or 0)}",
            f"Goal: {b.get('goal', 'move block to the target T pose')}",
            "Objects:",
            f"- agent:circular_pusher x={float(agent.get('x', 0.0)):.3f} y={float(agent.get('y', 0.0)):.3f} vx={float(agent.get('vx', 0.0)):.3f} vy={float(agent.get('vy', 0.0)):.3f} conf={float(agent.get('conf', 1.0)):.3f}",
            f"- block:t_block x={float(block.get('x', 0.0)):.3f} y={float(block.get('y', 0.0)):.3f} angle={float(block.get('angle', 0.0)):.3f} vx={float(block.get('vx', 0.0)):.3f} vy={float(block.get('vy', 0.0)):.3f} w={float(block.get('w', 0.0)):.3f} conf={float(block.get('conf', 1.0)):.3f}",
            "Relations:",
        ]
        for key in sorted(b.get("relations", {})):
            rel = b["relations"][key]
            value = rel.get("value")
            value_text = f"{float(value):.3f}" if isinstance(value, (int, float)) else str(value)
            lines.append(f"- {rel.get('subject')} {rel.get('predicate')} {rel.get('object')} = {value_text}")
        lines.append("Metadata: {}")
        return "\n".join(lines)
'''.strip() + "\n"


def render_collision_model_code(params: Mapping[str, Any]) -> str:
    return (
        _COLLISION_TEMPLATE.replace("__AGENT_RADIUS__", repr(float(params["agent_radius"])))
        .replace("__TEE_POLYGONS_JSON__", json.dumps(params["tee_polygons"]))
        .replace("__TEE_COG_JSON__", json.dumps(params["tee_cog"]))
        .replace("__CONTACT_MARGIN__", repr(float(params["contact_margin"])))
        .replace("__GATE_WIDTH__", repr(float(params.get("gate_width", 0.0))))
        .replace("__RESIST__", repr(float(params.get("resist", 0.0))))
        .replace("__PROJECT__", repr(float(params.get("project", 0.0))))
        .replace("__AGENT_FREE_JSON__", json.dumps(params["agent_free"], sort_keys=True))
        .replace("__AGENT_CONTACT_JSON__", json.dumps(params["agent_contact"], sort_keys=True))
        .replace("__BETA__", repr(float(params["beta"])))
        .replace("__BETA_W__", repr(float(params["beta_w"])))
        .replace("__CONTACT_COEF_JSON__", json.dumps(params["contact_coef"], sort_keys=True))
    )


def generate_collision_model_code(
    trajectories: Sequence[Any],
    *,
    geometry_json: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """Generate the collision model. ``geometry_json`` points at perceived
    geometry from the detect_geometry CLI; if omitted, oracle geometry is used
    (validation only) and a warning is logged."""
    geometry = None
    if geometry_json:
        geometry = collision.load_perceived_geometry(geometry_json)
    else:
        import sys

        print(
            "[pushing_mechanics_collision] WARNING: no --geometry_json provided; "
            "using ORACLE (source-derived) geometry, which is for validation only. "
            "Run scripts/visualpatchworld/detect_geometry.py for the legitimate path.",
            file=sys.stderr,
        )
    return render_collision_model_code(fit_collision_params(trajectories, geometry=geometry, **kwargs))
