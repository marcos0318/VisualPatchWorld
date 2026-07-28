"""Physics-structured symbolic dynamics for PushT (quasi-static pushing + momentum).

Motivation (scientific-discovery framing, cf. NewtonBench arXiv:2510.07172):
the dominant induced-dynamics strategies in this repo (ridge / gated-MLP on 18
hand-crafted features) treat dynamics induction as *static function fitting* of
the next state from the currently observed state. Two structural facts about the
PushT ground-truth program make that hypothesis space misspecified:

1. **Block momentum is a hidden state variable.** Empirically, when the block is
   far from the agent (no contact) it still moves ~15 px if it was moving on the
   previous step versus ~1 px if it was stationary (corr(prev_move, cur_move) ~=
   0.62 on oracle train). No memoryless ``f(agent, block_pose, action)`` can be
   correct here. We make block velocity an explicit, round-tripped state field.

2. **The contact law is a quasi-static pushing law, not a generic linear map.**
   The block translates along the contact normal under the pushing penetration
   and rotates under the torque of the off-center contact force about the block
   centre (Mason's voting theorem / Lynch-Mason ellipsoidal limit surface). We
   regress the contact response on a small *physically-derived* basis instead of
   raw coordinates, which makes the "no motion when far" behaviour exact by
   construction (generic ridge hallucinates ~8 px of far-field block motion).

This module fits and renders an executable PatchWorld world model that:
  * gates contact by an estimated threshold,
  * predicts agent motion from the action (regime-specific linear gain),
  * carries a latent block velocity (vx, vy, w) with free-flight decay,
  * injects a contact impulse from a compact physics-feature regressor,
  * round-trips block velocity through the observation text so momentum persists
    through multi-step model rollouts.

The fit is fully deterministic and LLM-free, so it slots into the existing
unified-acquisition loop as a new hypothesis class (change the hypothesis class,
keep the data loop).
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .pusht_local_induction import (
    WeightedTransition,
    _angle_delta,
    _parse_action,
    _parse_observation_text,
    _state_from_graph,
    iter_weighted_transitions,
)

# Physics-feature basis used for the contact-regime block response.
#   f = [da_x, da_y, push*n_x, push*n_y, torque, push, 1]
# where da is the predicted agent displacement, n the agent->block unit normal,
# push = max(0, da . n) the penetrating component, and
# torque = (s - b) x da the moment of the contact force about the block centre.
N_PHYS_FEATS = 7


def _phys_feats(
    sx: float,
    sy: float,
    bx: float,
    by: float,
    dax: float,
    day: float,
) -> List[float]:
    rel_x = bx - sx
    rel_y = by - sy
    dist = math.hypot(rel_x, rel_y)
    if dist > 1e-6:
        nx, ny = rel_x / dist, rel_y / dist
    else:
        nx, ny = 0.0, 0.0
    push = dax * nx + day * ny
    push = push if push > 0.0 else 0.0
    torque = (sx - bx) * day - (sy - by) * dax
    return [dax, day, push * nx, push * ny, torque, push, 1.0]


def _estimate_contact_threshold(
    rows: Sequence[WeightedTransition], default: float = 75.0
) -> float:
    distances: List[float] = []
    for row in rows:
        sx, sy, bx, by, _ = _state_from_graph(_parse_observation_text(row.observation))
        nbx, nby = _state_from_graph(_parse_observation_text(row.next_observation))[2:4]
        dist = math.hypot(bx - sx, by - sy)
        if math.hypot(nbx - bx, nby - by) > 0.5:
            distances.append(dist)
    if len(distances) < 5:
        return default
    distances.sort()
    # Median contact distance + small margin, clamped to a physically plausible
    # band. The 90th percentile is too permissive: it pulls the gate into the
    # 80-95 px band where residual block motion is momentum, not contact, which
    # makes the model predict phantom pushes and misleads the planner.
    median_dist = distances[len(distances) // 2]
    return float(max(45.0, min(80.0, median_dist + 5.0)))


def _fit_agent_gain(
    rows: Sequence[WeightedTransition], threshold: float, *, contact: bool
) -> List[float]:
    """Diagonal least-squares gain mapping action -> agent displacement."""
    num = [0.0, 0.0]
    den = [0.0, 0.0]
    for row in rows:
        sx, sy, bx, by, _ = _state_from_graph(_parse_observation_text(row.observation))
        nsx, nsy = _state_from_graph(_parse_observation_text(row.next_observation))[:2]
        ax, ay = _parse_action(row.action)
        dist = math.hypot(bx - sx, by - sy)
        if (dist <= threshold) != contact:
            continue
        w = float(row.weight)
        num[0] += w * (nsx - sx) * ax
        num[1] += w * (nsy - sy) * ay
        den[0] += w * ax * ax
        den[1] += w * ay * ay
    gx = num[0] / den[0] if den[0] > 1e-9 else 200.0
    gy = num[1] / den[1] if den[1] > 1e-9 else 200.0
    return [float(gx), float(gy)]


def _fit_momentum_decay(trajectories: Sequence[Any], threshold: float) -> List[float]:
    """Free-flight decay: block displacement_t ~= beta * displacement_{t-1}."""
    num_lin = den_lin = 0.0
    num_ang = den_ang = 0.0
    for traj in trajectories:
        transitions = list(getattr(traj, "transitions", []))
        for i in range(1, len(transitions)):
            cur = transitions[i]
            prev = transitions[i - 1]
            sx, sy, bx, by, ba = _state_from_graph(
                _parse_observation_text(getattr(cur, "observation", ""))
            )
            ns = _state_from_graph(
                _parse_observation_text(getattr(cur, "next_observation", ""))
            )
            ps = _state_from_graph(
                _parse_observation_text(getattr(prev, "observation", ""))
            )
            pns = _state_from_graph(
                _parse_observation_text(getattr(prev, "next_observation", ""))
            )
            if math.hypot(bx - sx, by - sy) <= threshold:
                continue  # only free-flight steps identify pure decay
            cur_dx, cur_dy = ns[2] - bx, ns[3] - by
            prev_dx, prev_dy = pns[2] - ps[2], pns[3] - ps[3]
            num_lin += prev_dx * cur_dx + prev_dy * cur_dy
            den_lin += prev_dx * prev_dx + prev_dy * prev_dy
            cur_w = _angle_delta(ns[4], ba)
            prev_w = _angle_delta(pns[4], ps[4])
            num_ang += prev_w * cur_w
            den_ang += prev_w * prev_w
    beta = num_lin / den_lin if den_lin > 1e-6 else 0.0
    beta_w = num_ang / den_ang if den_ang > 1e-6 else 0.0
    beta = float(max(0.0, min(0.98, beta)))
    beta_w = float(max(0.0, min(0.98, beta_w)))
    return [beta, beta_w]


def _fit_contact_regressor(
    rows: Sequence[WeightedTransition],
    threshold: float,
    agent_gain_contact: Sequence[float],
    *,
    ridge: float = 1e-2,
) -> List[List[float]]:
    """Least-squares 3x7 map from physics features to (dbx, dby, dtheta)."""
    try:
        import numpy as np
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("pushing-mechanics fitting requires numpy") from exc

    feats: List[List[float]] = []
    targets: List[List[float]] = []
    weights: List[float] = []
    gx, gy = float(agent_gain_contact[0]), float(agent_gain_contact[1])
    for row in rows:
        sx, sy, bx, by, ba = _state_from_graph(_parse_observation_text(row.observation))
        ns = _state_from_graph(_parse_observation_text(row.next_observation))
        ax, ay = _parse_action(row.action)
        if math.hypot(bx - sx, by - sy) > threshold:
            continue
        # deploy-consistent agent displacement (predicted, not observed)
        dax, day = gx * ax, gy * ay
        feats.append(_phys_feats(sx, sy, bx, by, dax, day))
        targets.append([ns[2] - bx, ns[3] - by, _angle_delta(ns[4], ba)])
        weights.append(float(row.weight))
    if len(feats) < N_PHYS_FEATS:
        return [[0.0] * N_PHYS_FEATS for _ in range(3)]
    x = np.asarray(feats, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    w = np.sqrt(np.clip(np.asarray(weights, dtype=np.float64), 1e-6, None))[:, None]
    xw, yw = x * w, y * w
    reg = ridge * np.eye(x.shape[1], dtype=np.float64)
    coef = np.linalg.solve(xw.T @ xw + reg, xw.T @ yw).T
    return [[float(v) for v in row] for row in coef.tolist()]


def fit_pushing_mechanics_params(
    trajectories: Sequence[Any],
    *,
    contact_threshold: float = 75.0,
    ridge: float = 1e-2,
    weighted_transitions: Optional[Sequence[WeightedTransition]] = None,
) -> Dict[str, Any]:
    rows = (
        list(weighted_transitions)
        if weighted_transitions is not None
        else iter_weighted_transitions(trajectories, default_weight=1.0)
    )
    threshold = _estimate_contact_threshold(rows, contact_threshold)
    agent_free = _fit_agent_gain(rows, threshold, contact=False)
    agent_contact = _fit_agent_gain(rows, threshold, contact=True)
    decay = _fit_momentum_decay(trajectories, threshold)
    contact_coef = _fit_contact_regressor(rows, threshold, agent_contact, ridge=ridge)
    return {
        "contact_threshold": threshold,
        "agent_free": agent_free,
        "agent_contact": agent_contact,
        "beta": decay[0],
        "beta_w": decay[1],
        "contact_coef": contact_coef,
        "n_rows": len(rows),
    }


_PUSHING_MECHANICS_TEMPLATE = r'''
from patchworld.worldmodel_base import BaseWorldModel
import copy
import json
import math
import re


class PushtPushingMechanicsWorldModel(BaseWorldModel):
    """Quasi-static pushing law + latent block momentum (second-order)."""

    CONTACT_THRESHOLD = __CONTACT_THRESHOLD__
    AGENT_FREE = __AGENT_FREE_JSON__
    AGENT_CONTACT = __AGENT_CONTACT_JSON__
    BETA = __BETA__
    BETA_W = __BETA_W__
    CONTACT_COEF = __CONTACT_COEF_JSON__

    def init_belief(self):
        return {
            "env": "pusht",
            "source": "pushing_mechanics",
            "step": 0,
            "goal": "move block to the target T pose",
            "objects": {},
            "relations": {},
            "metadata": {},
        }

    def init_belief_from_observation(self, obs_text: str):
        return self.correct_belief(self.init_belief(), obs_text)

    def extract_valid_action_forms(self):
        return {
            "move_left": ["move_left[dx=<float>,dy=<float>,speed=<float>]"],
            "move_right": ["move_right[dx=<float>,dy=<float>,speed=<float>]"],
            "move_up": ["move_up[dx=<float>,dy=<float>,speed=<float>]"],
            "move_down": ["move_down[dx=<float>,dy=<float>,speed=<float>]"],
            "hold": ["hold[dx=<float>,dy=<float>,speed=<float>]"],
        }

    def parse_observation(self, obs_text: str):
        graph = self.init_belief()
        section = ""
        for raw_line in str(obs_text or "").splitlines():
            line = raw_line.strip()
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
        match = re.match(r"^-\s*(\w+):(\w+)\s*(.*)$", line)
        if not match:
            return None
        obj = {"id": match.group(1), "category": match.group(2), "x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "w": 0.0, "conf": 1.0, "angle": 0.0}
        for key, value in re.findall(r"(\w+)=(-?\d+(?:\.\d+)?)", match.group(3)):
            obj[key] = float(value)
        return obj

    def _parse_relation(self, line):
        match = re.match(r"^-\s*(\w+)\s+(\w+(?:_\w+)*)\s+(\w+)\s*=\s*(.+)$", line)
        if not match:
            return None
        value = match.group(4).strip()
        try:
            value = float(value)
        except Exception:
            pass
        return {"subject": match.group(1), "predicate": match.group(2), "object": match.group(3), "value": value, "key": f"{match.group(1)}_{match.group(2)}_{match.group(3)}"}

    def correct_belief(self, belief_prior, obs_text: str):
        observed = self.parse_observation(obs_text)
        merged = copy.deepcopy(belief_prior) if isinstance(belief_prior, dict) else self.init_belief()
        for key in ("env", "source", "step", "goal", "metadata"):
            merged[key] = observed.get(key, merged.get(key))
        prior_objs = merged.get("objects", {}) if isinstance(merged.get("objects"), dict) else {}
        new_objs = observed.get("objects", {})
        block = new_objs.get("block")
        if block is not None:
            # Estimate incoming block velocity from the change since the last
            # observed/believed block pose (finite difference). The observation
            # itself carries no block velocity, so the prior belief supplies it.
            prior_block = prior_objs.get("block")
            if isinstance(prior_block, dict) and float(observed.get("step", 0) or 0) > 0:
                if "vx" not in block or float(block.get("vx", 0.0)) == 0.0:
                    block["vx"] = float(block.get("x", 0.0)) - float(prior_block.get("x", 0.0))
                    block["vy"] = float(block.get("y", 0.0)) - float(prior_block.get("y", 0.0))
                    dtheta = float(block.get("angle", 0.0)) - float(prior_block.get("angle", 0.0))
                    dtheta = (dtheta + math.pi) % (2.0 * math.pi) - math.pi
                    block["w"] = dtheta
        merged["objects"] = new_objs
        merged["relations"] = observed.get("relations", {})
        return merged

    def predict_belief(self, belief, action: str):
        b = copy.deepcopy(belief) if isinstance(belief, dict) else self.init_belief()
        objects = b.setdefault("objects", {})
        agent = objects.setdefault("agent", {"id": "agent", "category": "circular_pusher", "x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "conf": 1.0})
        block = objects.setdefault("block", {"id": "block", "category": "t_block", "x": 0.0, "y": 0.0, "angle": 0.0, "vx": 0.0, "vy": 0.0, "w": 0.0, "conf": 1.0})
        ax, ay = self._parse_action(action)
        sx = float(agent.get("x", 0.0))
        sy = float(agent.get("y", 0.0))
        bx = float(block.get("x", 0.0))
        by = float(block.get("y", 0.0))
        ba = float(block.get("angle", 0.0))
        dist = math.hypot(bx - sx, by - sy)
        in_contact = dist <= self.CONTACT_THRESHOLD

        gain = self.AGENT_CONTACT if in_contact else self.AGENT_FREE
        dax = gain[0] * ax
        day = gain[1] * ay
        agent["x"] = self._clip(sx + dax)
        agent["y"] = self._clip(sy + day)
        agent["vx"] = dax
        agent["vy"] = day

        # free-flight momentum decay of the latent block velocity
        vbx = self.BETA * float(block.get("vx", 0.0))
        vby = self.BETA * float(block.get("vy", 0.0))
        vw = self.BETA_W * float(block.get("w", 0.0))

        if in_contact:
            feats = self._phys_feats(sx, sy, bx, by, dax, day)
            delta = [sum(c * f for c, f in zip(row, feats)) for row in self.CONTACT_COEF]
            vbx += delta[0]
            vby += delta[1]
            vw += delta[2]

        block["x"] = self._clip(bx + vbx)
        block["y"] = self._clip(by + vby)
        block["angle"] = (ba + vw) % (2.0 * math.pi)
        block["vx"] = vbx
        block["vy"] = vby
        block["w"] = vw

        b["step"] = int(b.get("step", 0) or 0) + 1
        self._recompute_relations(b)
        return b

    def _phys_feats(self, sx, sy, bx, by, dax, day):
        rel_x = bx - sx
        rel_y = by - sy
        dist = math.hypot(rel_x, rel_y)
        if dist > 1e-6:
            nx, ny = rel_x / dist, rel_y / dist
        else:
            nx, ny = 0.0, 0.0
        push = dax * nx + day * ny
        if push < 0.0:
            push = 0.0
        torque = (sx - bx) * day - (sy - by) * dax
        return [dax, day, push * nx, push * ny, torque, push, 1.0]

    def _parse_action(self, action):
        match = re.match(r"^\w+\[dx=(-?\d+(?:\.\d+)?),dy=(-?\d+(?:\.\d+)?),speed=(-?\d+(?:\.\d+)?)\]$", str(action or "").strip())
        if not match:
            return 0.0, 0.0
        return float(match.group(1)), float(match.group(2))

    def _clip(self, value):
        return max(0.0, min(512.0, float(value)))

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
            f"Source: {b.get('source', 'pushing_mechanics')}",
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


def render_pushing_mechanics_model_code(params: Mapping[str, Any]) -> str:
    return (
        _PUSHING_MECHANICS_TEMPLATE.replace(
            "__CONTACT_THRESHOLD__", repr(float(params["contact_threshold"]))
        )
        .replace("__AGENT_FREE_JSON__", json.dumps(params["agent_free"], sort_keys=True))
        .replace("__AGENT_CONTACT_JSON__", json.dumps(params["agent_contact"], sort_keys=True))
        .replace("__BETA__", repr(float(params["beta"])))
        .replace("__BETA_W__", repr(float(params["beta_w"])))
        .replace("__CONTACT_COEF_JSON__", json.dumps(params["contact_coef"], sort_keys=True))
    )


def generate_pushing_mechanics_model_code(
    trajectories: Sequence[Any], **kwargs: Any
) -> str:
    params = fit_pushing_mechanics_params(trajectories, **kwargs)
    return render_pushing_mechanics_model_code(params)
