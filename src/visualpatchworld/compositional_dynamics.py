"""Compositional object-centric dynamics for PushT (agent + contact modules + gate)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from statistics import median
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .pusht_local_induction import (
    WeightedTransition,
    _angle_delta,
    _compact_feature_vector,
    _feature_vector,
    _fit_weighted_ridge_matrix,
    _parse_action,
    _parse_observation_text,
    _state_from_graph,
    fit_contact_mlp_residual,
    iter_contact_multistep_pairs,
    iter_weighted_transitions,
)


@dataclass(frozen=True)
class AgentState:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0


@dataclass(frozen=True)
class BlockState:
    x: float
    y: float
    angle: float


@dataclass(frozen=True)
class ObjectStates:
    agent: AgentState
    block: BlockState

    @classmethod
    def from_graph(cls, graph: Mapping[str, Any]) -> ObjectStates:
        state = _state_from_graph(graph if "objects" in graph else _parse_observation_text(str(graph)))
        return cls(
            agent=AgentState(state[0], state[1]),
            block=BlockState(state[2], state[3], state[4]),
        )


class ContactGate:
    def __init__(self, threshold: float) -> None:
        self.threshold = float(threshold)

    def in_contact(self, agent: AgentState, block: BlockState) -> bool:
        return math.hypot(block.x - agent.x, block.y - agent.y) <= self.threshold

    def distance(self, agent: AgentState, block: BlockState) -> float:
        return float(math.hypot(block.x - agent.x, block.y - agent.y))


def _rows_to_buckets(
    rows: Sequence[WeightedTransition],
    contact_threshold: float,
) -> Tuple[List[WeightedTransition], List[WeightedTransition]]:
    near: List[WeightedTransition] = []
    far: List[WeightedTransition] = []
    for row in rows:
        state = _state_from_graph(_parse_observation_text(row.observation))
        dist = math.hypot(state[2] - state[0], state[3] - state[1])
        if dist <= contact_threshold:
            near.append(row)
        else:
            far.append(row)
    return near, far


def _fit_agent_rows(
    rows: Sequence[WeightedTransition],
    *,
    ridge: float,
) -> List[List[float]]:
    features: List[List[float]] = []
    targets: List[List[float]] = []
    weights: List[float] = []
    for row in rows:
        state = _state_from_graph(_parse_observation_text(row.observation))
        next_state = _state_from_graph(_parse_observation_text(row.next_observation))
        action = _parse_action(row.action)
        sx, sy, _, _, _ = state
        nsx, nsy, _, _, _ = next_state
        features.append(_feature_vector(state, action))
        targets.append([nsx - sx, nsy - sy])
        weights.append(float(row.weight))
    if not features:
        return [[0.0] * 18, [0.0] * 18]
    return _fit_weighted_ridge_matrix(features, targets, weights, ridge=ridge)


def _fit_block_rows(
    rows: Sequence[WeightedTransition],
    *,
    ridge: float,
) -> List[List[float]]:
    features: List[List[float]] = []
    targets: List[List[float]] = []
    weights: List[float] = []
    for row in rows:
        state = _state_from_graph(_parse_observation_text(row.observation))
        next_state = _state_from_graph(_parse_observation_text(row.next_observation))
        action = _parse_action(row.action)
        _, _, bx, by, ba = state
        _, _, nbx, nby, nba = next_state
        features.append(_feature_vector(state, action))
        targets.append([nbx - bx, nby - by, _angle_delta(nba, ba)])
        weights.append(float(row.weight))
    if not features:
        return [[0.0] * 18, [0.0] * 18, [0.0] * 18]
    return _fit_weighted_ridge_matrix(features, targets, weights, ridge=ridge)


def _estimate_contact_threshold(rows: Sequence[WeightedTransition], default: float = 70.0) -> float:
    contact_distances: List[float] = []
    for row in rows:
        state = _state_from_graph(_parse_observation_text(row.observation))
        next_state = _state_from_graph(_parse_observation_text(row.next_observation))
        sx, sy, bx, by, _ = state
        nbx, nby = next_state[2], next_state[3]
        dist = math.hypot(bx - sx, by - sy)
        block_move = math.hypot(nbx - bx, nby - by)
        if dist <= default and block_move > 0.25:
            contact_distances.append(dist)
    if len(contact_distances) >= 3:
        return float(max(35.0, min(90.0, median(contact_distances) + 5.0)))
    return default


def fit_compositional_gated_mlp_params(
    trajectories: Sequence[Any],
    *,
    ridge: float = 1e-3,
    contact_threshold: float = 70.0,
    weighted_transitions: Optional[Sequence[WeightedTransition]] = None,
    mlp_hidden_dim: int = 64,
    mlp_steps: int = 3000,
    mlp_lr: float = 0.01,
    mlp_ridge: float = 1e-4,
    mlp_rollout_steps: int = 2,
    multistep_weight: float = 0.5,
    seed: int = 42,
) -> Dict[str, Any]:
    """Fit separate agent-free, agent-contact, and block-contact modules."""

    rows = (
        list(weighted_transitions)
        if weighted_transitions is not None
        else iter_weighted_transitions(trajectories, default_weight=1.0)
    )
    contact_threshold = _estimate_contact_threshold(rows, contact_threshold)
    near, far = _rows_to_buckets(rows, contact_threshold)
    if not far:
        far = list(rows)
    if not near:
        near = list(rows)

    agent_free = _fit_agent_rows(far, ridge=ridge)
    agent_contact = _fit_agent_rows(near, ridge=ridge)
    block_contact = _fit_block_rows(near, ridge=ridge)

    contact_coef_for_mlp = [
        [0.0] * 18,
        [0.0] * 18,
        block_contact[0],
        block_contact[1],
        block_contact[2],
    ]
    multistep_pairs = (
        iter_contact_multistep_pairs(trajectories, contact_threshold=contact_threshold)
        if mlp_rollout_steps >= 2
        else None
    )
    mlp = fit_contact_mlp_residual(
        near,
        contact_coef_for_mlp,
        contact_threshold=contact_threshold,
        hidden_dim=mlp_hidden_dim,
        ridge=mlp_ridge,
        steps=mlp_steps,
        lr=mlp_lr,
        seed=seed,
        multistep_pairs=multistep_pairs,
        multistep_weight=multistep_weight,
    )

    return {
        "contact_threshold": contact_threshold,
        "agent_free": agent_free,
        "agent_contact": agent_contact,
        "block_contact": block_contact,
        "mlp": mlp,
        "n_far_rows": len(far),
        "n_near_rows": len(near),
        "mlp_rollout_steps": mlp_rollout_steps,
    }


def refit_compositional_contact_params(
    base_params: Mapping[str, Any],
    trajectories: Sequence[Any],
    extra_transitions: Sequence[WeightedTransition],
    *,
    ridge: float = 1e-3,
    mlp_hidden_dim: int = 64,
    mlp_steps: int = 3000,
    mlp_lr: float = 0.01,
    mlp_ridge: float = 1e-4,
    mlp_rollout_steps: int = 2,
    multistep_weight: float = 0.5,
    seed: int = 42,
) -> Dict[str, Any]:
    """Refit block-contact + MLP and contact-regime agent rows; freeze agent-free."""

    contact_threshold = float(base_params["contact_threshold"])
    agent_free = [list(row) for row in base_params["agent_free"]]
    agent_contact = [list(row) for row in base_params["agent_contact"]]

    base_rows = iter_weighted_transitions(trajectories, default_weight=1.0)
    merged = list(base_rows) + list(extra_transitions)
    near, _ = _rows_to_buckets(merged, contact_threshold)
    if not near:
        near = list(merged)

    agent_contact = _fit_agent_rows(near, ridge=ridge)
    block_contact = _fit_block_rows(near, ridge=ridge)
    contact_coef_for_mlp = [
        [0.0] * 18,
        [0.0] * 18,
        block_contact[0],
        block_contact[1],
        block_contact[2],
    ]
    multistep_pairs = (
        iter_contact_multistep_pairs(trajectories, contact_threshold=contact_threshold)
        if mlp_rollout_steps >= 2
        else None
    )
    mlp = fit_contact_mlp_residual(
        near,
        contact_coef_for_mlp,
        contact_threshold=contact_threshold,
        hidden_dim=mlp_hidden_dim,
        ridge=mlp_ridge,
        steps=mlp_steps,
        lr=mlp_lr,
        seed=seed,
        multistep_pairs=multistep_pairs,
        multistep_weight=multistep_weight,
    )
    return {
        "contact_threshold": contact_threshold,
        "agent_free": agent_free,
        "agent_contact": agent_contact,
        "block_contact": block_contact,
        "mlp": mlp,
        "n_near_rows": len(near),
        "n_extra_rows": len(extra_transitions),
        "mlp_rollout_steps": mlp_rollout_steps,
    }


_COMPOSITIONAL_GATED_MLP_TEMPLATE = r'''
from patchworld.worldmodel_base import BaseWorldModel
import copy
import json
import math
import re


class PushtCompositionalGatedMlpWorldModel(BaseWorldModel):
    """Object-centric dynamics: AgentModule + ContactModule composed by ContactGate."""

    CONTACT_THRESHOLD = __CONTACT_THRESHOLD__
    AGENT_FREE = __AGENT_FREE_JSON__
    AGENT_CONTACT = __AGENT_CONTACT_JSON__
    BLOCK_CONTACT = __BLOCK_CONTACT_JSON__
    MLP = __MLP_JSON__

    def init_belief(self):
        return {
            "env": "pusht",
            "source": "compositional_gated_mlp",
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
        obj = {"id": match.group(1), "category": match.group(2), "x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "conf": 1.0, "angle": 0.0}
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
        merged["objects"] = observed.get("objects", {})
        merged["relations"] = observed.get("relations", {})
        return merged

    def predict_belief(self, belief, action: str):
        b = copy.deepcopy(belief) if isinstance(belief, dict) else self.init_belief()
        objects = b.setdefault("objects", {})
        agent = objects.setdefault("agent", {"id": "agent", "category": "circular_pusher", "x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "conf": 1.0})
        block = objects.setdefault("block", {"id": "block", "category": "t_block", "x": 0.0, "y": 0.0, "angle": 0.0, "conf": 1.0})
        dx, dy = self._parse_action(action)
        feats = self._features(agent, block, dx, dy)
        dist = math.hypot(float(block.get("x", 0.0)) - float(agent.get("x", 0.0)), float(block.get("y", 0.0)) - float(agent.get("y", 0.0)))
        in_contact = dist <= self.CONTACT_THRESHOLD

        agent_coef = self.AGENT_CONTACT if in_contact else self.AGENT_FREE
        agent_delta = [sum(c * f for c, f in zip(row, feats)) for row in agent_coef]
        agent["x"] = self._clip(float(agent.get("x", 0.0)) + agent_delta[0])
        agent["y"] = self._clip(float(agent.get("y", 0.0)) + agent_delta[1])
        agent["vx"] = agent_delta[0]
        agent["vy"] = agent_delta[1]

        if in_contact:
            block_delta = [sum(c * f for c, f in zip(row, feats)) for row in self.BLOCK_CONTACT]
            residual = self._mlp_block_residual(agent, block, dx, dy)
            block_delta[0] += residual[0]
            block_delta[1] += residual[1]
            block_delta[2] += residual[2]
            block["x"] = self._clip(float(block.get("x", 0.0)) + block_delta[0])
            block["y"] = self._clip(float(block.get("y", 0.0)) + block_delta[1])
            block["angle"] = (float(block.get("angle", 0.0)) + block_delta[2]) % (2.0 * math.pi)

        b["step"] = int(b.get("step", 0) or 0) + 1
        self._recompute_relations(b)
        return b

    def _mlp_block_residual(self, agent, block, dx, dy):
        compact = self._compact_features(agent, block, dx, dy)
        hidden = []
        for row, bias in zip(self.MLP["w1"], self.MLP["b1"]):
            hidden.append(max(0.0, sum(w * x for w, x in zip(row, compact)) + bias))
        outputs = []
        for row, bias in zip(self.MLP["w2"], self.MLP["b2"]):
            outputs.append(sum(w * h for w, h in zip(row, hidden)) + bias)
        return outputs

    def _compact_features(self, agent, block, dx, dy):
        sx = float(agent.get("x", 0.0))
        sy = float(agent.get("y", 0.0))
        bx = float(block.get("x", 0.0))
        by = float(block.get("y", 0.0))
        ba = float(block.get("angle", 0.0))
        speed = math.hypot(dx, dy)
        dist = math.hypot(bx - sx, by - sy)
        return [sx / 512.0, sy / 512.0, bx / 512.0, by / 512.0, math.cos(ba), math.sin(ba), dist / 512.0, dx, dy, speed]

    def _parse_action(self, action):
        match = re.match(r"^\w+\[dx=(-?\d+(?:\.\d+)?),dy=(-?\d+(?:\.\d+)?),speed=(-?\d+(?:\.\d+)?)\]$", str(action or "").strip())
        if not match:
            return 0.0, 0.0
        return float(match.group(1)), float(match.group(2))

    def _features(self, agent, block, dx, dy):
        sx = float(agent.get("x", 0.0))
        sy = float(agent.get("y", 0.0))
        bx = float(block.get("x", 0.0))
        by = float(block.get("y", 0.0))
        ba = float(block.get("angle", 0.0))
        speed = math.hypot(dx, dy)
        rel_x = bx - sx
        rel_y = by - sy
        dist = math.hypot(rel_x, rel_y)
        near = 1.0 if dist <= 70.0 else 0.0
        inv_dist = 1.0 / max(dist, 1.0)
        return [
            1.0, dx, dy, speed,
            sx / 512.0, sy / 512.0, bx / 512.0, by / 512.0,
            math.cos(ba), math.sin(ba),
            rel_x / 512.0, rel_y / 512.0, dist / 512.0,
            near, dx * near, dy * near,
            dx * rel_x * inv_dist, dy * rel_y * inv_dist,
        ]

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
        block = objects.get("block", {"id": "block", "category": "t_block", "x": 0.0, "y": 0.0, "angle": 0.0, "conf": 1.0})
        self._recompute_relations(b)
        lines = [
            "Environment: pusht",
            f"Source: {b.get('source', 'compositional_gated_mlp')}",
            f"Step: {int(b.get('step', 0) or 0)}",
            f"Goal: {b.get('goal', 'move block to the target T pose')}",
            "Objects:",
            f"- agent:circular_pusher x={float(agent.get('x', 0.0)):.3f} y={float(agent.get('y', 0.0)):.3f} vx={float(agent.get('vx', 0.0)):.3f} vy={float(agent.get('vy', 0.0)):.3f} conf={float(agent.get('conf', 1.0)):.3f}",
            f"- block:t_block x={float(block.get('x', 0.0)):.3f} y={float(block.get('y', 0.0)):.3f} angle={float(block.get('angle', 0.0)):.3f} conf={float(block.get('conf', 1.0)):.3f}",
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


def render_compositional_gated_mlp_model_code(params: Mapping[str, Any]) -> str:
    return (
        _COMPOSITIONAL_GATED_MLP_TEMPLATE.replace(
            "__CONTACT_THRESHOLD__",
            repr(float(params["contact_threshold"])),
        )
        .replace("__AGENT_FREE_JSON__", json.dumps(params["agent_free"], sort_keys=True))
        .replace("__AGENT_CONTACT_JSON__", json.dumps(params["agent_contact"], sort_keys=True))
        .replace("__BLOCK_CONTACT_JSON__", json.dumps(params["block_contact"], sort_keys=True))
        .replace("__MLP_JSON__", json.dumps(params["mlp"], sort_keys=True))
    )


def generate_compositional_offline_model_code(trajectories: Sequence[Any], **kwargs: Any) -> str:
    params = fit_compositional_gated_mlp_params(trajectories, **kwargs)
    return render_compositional_gated_mlp_model_code(params)


def generate_compositional_contact_refit_model_code(
    trajectories: Sequence[Any],
    *,
    base_params: Mapping[str, Any],
    extra_transitions: Sequence[WeightedTransition],
    **kwargs: Any,
) -> str:
    params = refit_compositional_contact_params(
        base_params,
        trajectories,
        extra_transitions,
        **kwargs,
    )
    return render_compositional_gated_mlp_model_code(params)


def load_compositional_params_from_results_dir(results_dir: Any) -> Dict[str, Any]:
    from .modular_dynamics import load_gated_mlp_params_from_results_dir

    root_path = results_dir if hasattr(results_dir, "is_dir") else __import__("pathlib").Path(results_dir)
    candidates: List[Any] = []
    from pathlib import Path

    root = Path(root_path)
    for subdir in ("", "tool", "oracle"):
        model_dir = root / subdir / "generated_models" if subdir else root / "generated_models"
        if model_dir.is_dir():
            candidates.extend(model_dir.glob("*.py"))
    if not candidates:
        raise FileNotFoundError(f"No model under {root}")
    code = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0].read_text(encoding="utf-8")
    if "AGENT_FREE" in code:
        from visualpatchworld.patchworld_adapter import apply_pusht_patchworld_adapter, sanitize_pusht_model_code

        apply_pusht_patchworld_adapter()
        from patchworld.worldmodel_validator import load_model_from_code

        model, error = load_model_from_code(sanitize_pusht_model_code(code))
        if error or model is None:
            raise ValueError(error)
        return {
            "contact_threshold": float(model.CONTACT_THRESHOLD),
            "agent_free": model.AGENT_FREE,
            "agent_contact": model.AGENT_CONTACT,
            "block_contact": model.BLOCK_CONTACT,
            "mlp": model.MLP,
        }
    legacy = load_gated_mlp_params_from_results_dir(root)
    return {
        "contact_threshold": legacy["contact_threshold"],
        "agent_free": [legacy["coef_free"][0], legacy["coef_free"][1]],
        "agent_contact": [legacy["coef_contact"][0], legacy["coef_contact"][1]],
        "block_contact": [
            legacy["coef_contact"][2],
            legacy["coef_contact"][3],
            legacy["coef_contact"][4],
        ],
        "mlp": legacy["mlp"],
    }
