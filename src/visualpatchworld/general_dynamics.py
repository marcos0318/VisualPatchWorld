"""Environment-agnostic linear scaffold-fit dynamics for VisualPatchWorld."""

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .lewm_env_specs import get_env_spec
from .pusht_local_induction import _fit_weighted_ridge_matrix, _parse_observation_text


def _object_numeric_fields(obj: Mapping[str, Any]) -> List[str]:
    keys = ["x", "y"]
    if "z" in obj:
        keys.append("z")
    if "angle" in obj:
        keys.append("angle")
    return keys


def _state_vector(graph: Mapping[str, Any]) -> Tuple[List[str], List[float]]:
    objects = graph.get("objects", {})
    if not isinstance(objects, Mapping):
        return [], []
    keys: List[str] = []
    values: List[float] = []
    for object_id in sorted(objects):
        obj = objects[object_id]
        if not isinstance(obj, Mapping):
            continue
        for field in _object_numeric_fields(obj):
            keys.append(f"{object_id}.{field}")
            values.append(float(obj.get(field, 0.0)))
    return keys, values


def _parse_action_values(action: str, action_dim: int) -> List[float]:
    if action_dim == 2:
        match = re.match(
            r"^(?P<label>\w+)\[dx=(?P<dx>-?\d+(?:\.\d+)?),dy=(?P<dy>-?\d+(?:\.\d+)?),speed=(?P<speed>-?\d+(?:\.\d+)?)\]$",
            str(action).strip(),
        )
        if match:
            return [float(match.group("dx")), float(match.group("dy"))]
        return [0.0, 0.0]
    out = [0.0] * action_dim
    for match in re.finditer(r"a(\d+)=(-?\d+(?:\.\d+)?)", str(action)):
        idx = int(match.group(1))
        if 0 <= idx < action_dim:
            out[idx] = float(match.group(2))
    return out


def _feature_vector(state_keys: Sequence[str], state_values: Sequence[float], action_values: Sequence[float]) -> List[float]:
    by_key = {k: v for k, v in zip(state_keys, state_values)}
    feats = [1.0, *action_values]
    for key in state_keys:
        feats.append(by_key.get(key, 0.0))
    object_ids = sorted({k.split(".")[0] for k in state_keys})
    for i, left in enumerate(object_ids):
        lx = by_key.get(f"{left}.x", 0.0)
        ly = by_key.get(f"{left}.y", 0.0)
        lz = by_key.get(f"{left}.z")
        for right in object_ids[i + 1 :]:
            rx = by_key.get(f"{right}.x", 0.0)
            ry = by_key.get(f"{right}.y", 0.0)
            rz = by_key.get(f"{right}.z")
            if lz is None or rz is None:
                feats.append(math.hypot(lx - rx, ly - ry))
            else:
                feats.append(math.sqrt((lx - rx) ** 2 + (ly - ry) ** 2 + (lz - rz) ** 2))
    return feats


def fit_general_scaffold_params(trajectories: Sequence[Any], env_name: str) -> Dict[str, Any]:
    spec = get_env_spec(env_name)
    features: List[List[float]] = []
    targets: List[List[float]] = []
    state_keys: List[str] = []

    for traj in trajectories:
        for transition in getattr(traj, "transitions", []):
            obs = _parse_observation_text(getattr(transition, "observation", ""))
            nxt = _parse_observation_text(getattr(transition, "next_observation", ""))
            keys, cur = _state_vector(obs)
            _, nxt_vals = _state_vector(nxt)
            if not keys:
                continue
            if not state_keys:
                state_keys = keys
            action = _parse_action_values(getattr(transition, "action", ""), spec.action_dim)
            features.append(_feature_vector(state_keys, cur, action))
            targets.append([n - c for n, c in zip(nxt_vals, cur)])

    if not features:
        raise RuntimeError(f"No usable transitions to fit general scaffold for {env_name}")

    coef = _fit_weighted_ridge_matrix(features, targets, [1.0] * len(features), ridge=1e-3, default_dim=len(features[0]))
    return {
        "env": spec.name,
        "action_dim": spec.action_dim,
        "state_keys": state_keys,
        "coef": coef,
        "feature_dim": len(features[0]),
        "target_dim": len(state_keys),
    }


def _cube_xyz(graph: Mapping[str, Any], object_id: str) -> Tuple[float, float, float]:
    objects = graph.get("objects", {})
    obj = objects.get(object_id, {}) if isinstance(objects, Mapping) else {}
    if not isinstance(obj, Mapping):
        return 0.0, 0.0, 0.0
    return float(obj.get("x", 0.0)), float(obj.get("y", 0.0)), float(obj.get("z", 0.0))


def fit_cube_contact_scaffold_params(trajectories: Sequence[Any]) -> Dict[str, Any]:
    """Fit Cube effector kinematics + grasp/contact latch for the block.

    Empirically on LeWM Cube expert macros, block tracking is almost perfectly
    predicted by ``dist_3d ≤ τ`` AND ``a4 > 0`` (τ≈0.02, ~96% agreement). The
    previous scaffold searched the wrong grip polarity (``a4 ≤ 0``), which is
    why it underperformed PoE-World's grip-gated expert. Model:
    - effector Δxyz ~ linear in [1, a0..a4, effector xyz]
    - attach when near + grip; stay latched while grip held; block = effector
    """

    spec = get_env_spec("cube")
    eff_feats: List[List[float]] = []
    eff_targets: List[List[float]] = []
    samples: List[Dict[str, float]] = []

    for traj in trajectories:
        for transition in getattr(traj, "transitions", []):
            obs = _parse_observation_text(getattr(transition, "observation", ""))
            nxt = _parse_observation_text(getattr(transition, "next_observation", ""))
            ex, ey, ez = _cube_xyz(obs, "effector")
            nx, ny, nz = _cube_xyz(nxt, "effector")
            bx, by, bz = _cube_xyz(obs, "block")
            nbx, nby, nbz = _cube_xyz(nxt, "block")
            action = _parse_action_values(getattr(transition, "action", ""), spec.action_dim)
            while len(action) < 5:
                action.append(0.0)
            feats = [1.0, *action[:5], ex, ey, ez]
            eff_feats.append(feats)
            eff_targets.append([nx - ex, ny - ey, nz - ez])
            dist_3d = math.sqrt((ex - bx) ** 2 + (ey - by) ** 2 + (ez - bz) ** 2)
            samples.append(
                {
                    "a4": float(action[4]),
                    "dist_3d": dist_3d,
                    "block_moved": math.sqrt((nbx - bx) ** 2 + (nby - by) ** 2 + (nbz - bz) ** 2),
                    "track_err": math.sqrt((nbx - nx) ** 2 + (nby - ny) ** 2 + (nbz - nz) ** 2),
                }
            )

    if not eff_feats:
        raise RuntimeError("No usable transitions to fit Cube contact scaffold")

    eff_coef = _fit_weighted_ridge_matrix(
        eff_feats, eff_targets, [1.0] * len(eff_feats), ridge=1e-3, default_dim=len(eff_feats[0])
    )

    # Grip polarity + 3D proximity gate (PoE-World uses a4>0; validate on data).
    candidate_tau = [0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12]
    grip_modes = ("none", "a4_pos", "a4_neg")
    best_tau = 0.02
    best_score = -1.0
    best_grip = "a4_pos"
    for grip_mode in grip_modes:
        for tau in candidate_tau:
            correct = 0
            total = 0
            for sample in samples:
                near = sample["dist_3d"] <= tau
                if grip_mode == "a4_pos":
                    near = near and sample["a4"] > 0.0
                elif grip_mode == "a4_neg":
                    near = near and sample["a4"] < 0.0
                tracked = sample["track_err"] <= 0.03 and sample["block_moved"] > 1e-4
                if near == tracked:
                    correct += 1
                total += 1
            score = correct / max(total, 1)
            if score > best_score:
                best_score = score
                best_tau = tau
                best_grip = grip_mode

    return {
        "env": "cube",
        "action_dim": spec.action_dim,
        "effector_coef": eff_coef,
        "contact_threshold": float(best_tau),
        "grip_mode": best_grip,
        "require_grip": best_grip != "none",
        "use_latch": True,
        "contact_fit_score": float(best_score),
        "state_keys": [
            "block.x",
            "block.y",
            "block.z",
            "effector.x",
            "effector.y",
            "effector.z",
            "goal.x",
            "goal.y",
            "goal.z",
        ],
        "n_transitions": len(samples),
    }


def _params_literal(params: Mapping[str, Any]) -> str:
    """Serialize params as a Python literal embedded in generated model code."""
    return repr(params)


def generate_cube_contact_scaffold_model_code(trajectories: Sequence[Any]) -> str:
    params = fit_cube_contact_scaffold_params(trajectories)
    return _CUBE_CONTACT_TEMPLATE.replace("__PARAMS_JSON__", _params_literal(params))


def generate_general_scaffold_model_code(
    trajectories: Sequence[Any],
    env_name: str,
    *,
    force_linear: bool = False,
) -> str:
    # Cube needs an explicit contact/grasp gate; linear ridge alone underperforms
    # WorldCoder's hard snap prior under library shooting. Level-1 may still select
    # the linear template explicitly via force_linear=True for ablation / wrong-class.
    if get_env_spec(env_name).name == "cube" and not force_linear:
        return generate_cube_contact_scaffold_model_code(trajectories)
    params = fit_general_scaffold_params(trajectories, env_name)
    return _GENERAL_TEMPLATE.replace("__PARAMS_JSON__", _params_literal(params))


_GENERAL_TEMPLATE = r'''
from patchworld.worldmodel_base import BaseWorldModel
import copy
import json
import math
import re


class GeneralScaffoldFitWorldModel(BaseWorldModel):
    PARAMS = __PARAMS_JSON__
    STATE_KEYS = PARAMS["state_keys"]
    COEF = PARAMS["coef"]
    ACTION_DIM = int(PARAMS["action_dim"])
    ENV = PARAMS["env"]

    def init_belief(self):
        return {"env": self.ENV, "source": "general_scaffold_fit", "step": 0, "goal": "", "objects": {}, "relations": {}, "metadata": {}}

    def init_belief_from_observation(self, obs_text: str):
        return self.correct_belief(self.init_belief(), obs_text)

    def extract_valid_action_forms(self):
        if self.ACTION_DIM == 2:
            return {
                "move_left": ["move_left[dx=<float>,dy=<float>,speed=<float>]"],
                "move_right": ["move_right[dx=<float>,dy=<float>,speed=<float>]"],
                "move_up": ["move_up[dx=<float>,dy=<float>,speed=<float>]"],
                "move_down": ["move_down[dx=<float>,dy=<float>,speed=<float>]"],
                "hold": ["hold[dx=<float>,dy=<float>,speed=<float>]"],
            }
        forms = ["act[" + ",".join(f"a{i}=<float>" for i in range(self.ACTION_DIM)) + ",speed=<float>]"]
        return {"act": forms}

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
                except ValueError:
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
        return graph

    def _parse_object(self, line):
        match = re.match(r"^-\s*(\w+):(\w+)\s*(.*)$", line)
        if not match:
            return None
        obj = {"id": match.group(1), "category": match.group(2), "x": 0.0, "y": 0.0, "conf": 1.0}
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
        except ValueError:
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

    def _state_vector(self, graph):
        keys = []
        values = []
        objects = graph.get("objects", {})
        for object_id in sorted(objects):
            obj = objects[object_id]
            for field in ("x", "y", "z", "angle"):
                if field in obj:
                    keys.append(f"{object_id}.{field}")
                    values.append(float(obj.get(field, 0.0)))
        return keys, values

    def _apply_state_vector(self, graph, keys, values):
        objects = graph.setdefault("objects", {})
        for key, value in zip(keys, values):
            object_id, field = key.split(".", 1)
            obj = objects.setdefault(object_id, {"id": object_id, "category": object_id, "x": 0.0, "y": 0.0, "conf": 1.0})
            obj[field] = float(value)
        return graph

    def _parse_action(self, action: str):
        if self.ACTION_DIM == 2:
            match = re.match(r"^(?P<label>\w+)\[dx=(?P<dx>-?\d+(?:\.\d+)?),dy=(?P<dy>-?\d+(?:\.\d+)?),speed=(?P<speed>-?\d+(?:\.\d+)?)\]$", str(action).strip())
            if match:
                return float(match.group("dx")), float(match.group("dy"))
            return 0.0, 0.0
        out = [0.0] * self.ACTION_DIM
        for match in re.finditer(r"a(\d+)=(-?\d+(?:\.\d+)?)", str(action)):
            idx = int(match.group(1))
            if 0 <= idx < self.ACTION_DIM:
                out[idx] = float(match.group(2))
        return out

    def _features(self, keys, values, action_values):
        by_key = {k: v for k, v in zip(keys, values)}
        feats = [1.0, *action_values]
        for key in self.STATE_KEYS:
            feats.append(by_key.get(key, 0.0))
        object_ids = sorted({k.split(".")[0] for k in self.STATE_KEYS})
        for i, left in enumerate(object_ids):
            lx = by_key.get(f"{left}.x", 0.0)
            ly = by_key.get(f"{left}.y", 0.0)
            for right in object_ids[i + 1 :]:
                rx = by_key.get(f"{right}.x", 0.0)
                ry = by_key.get(f"{right}.y", 0.0)
                feats.append(math.hypot(lx - rx, ly - ry))
        return feats

    def predict_belief(self, belief, action: str):
        b = copy.deepcopy(belief) if isinstance(belief, dict) else self.init_belief()
        keys, values = self._state_vector(b)
        if not keys:
            keys = list(self.STATE_KEYS)
            values = [0.0] * len(keys)
        if self.ACTION_DIM == 2:
            dx, dy = self._parse_action(action)
            action_values = [dx, dy]
        else:
            parsed = self._parse_action(action)
            action_values = list(parsed) if isinstance(parsed, (list, tuple)) else [float(v) for v in parsed]
        feats = self._features(keys, values, action_values)
        delta = [sum(c * f for c, f in zip(row, feats)) for row in self.COEF]
        next_values = [v + d for v, d in zip(values, delta)]
        b = self._apply_state_vector(b, keys, next_values)
        b["step"] = int(b.get("step", 0)) + 1
        return b

    def readout_observation(self, belief, action: str):
        b = belief if isinstance(belief, dict) else self.init_belief()
        lines = [f"Environment: {b.get('env', self.ENV)}", f"Source: {b.get('source', 'general_scaffold_fit')}", f"Step: {int(b.get('step', 0))}", f"Goal: {b.get('goal', '')}", "Objects:"]
        for object_id in sorted(b.get("objects", {})):
            obj = b["objects"][object_id]
            parts = [f"- {object_id}:{obj.get('category', object_id)}", f"x={float(obj.get('x', 0.0)):.3f}", f"y={float(obj.get('y', 0.0)):.3f}"]
            if "z" in obj:
                parts.append(f"z={float(obj.get('z', 0.0)):.3f}")
            if "angle" in obj:
                parts.append(f"angle={float(obj.get('angle', 0.0)):.3f}")
            parts.append(f"conf={float(obj.get('conf', 1.0)):.3f}")
            lines.append(" ".join(parts))
        lines.append("Relations:")
        for rel in sorted(b.get("relations", {}).values(), key=lambda item: item.get("key", "")):
            lines.append(f"- {rel['subject']} {rel['predicate']} {rel['object']} = {rel.get('value')}")
        lines.append(f"Metadata: {json.dumps(b.get('metadata', {}), sort_keys=True, separators=(',', ':'))}")
        return "\n".join(lines)
'''


_CUBE_CONTACT_TEMPLATE = r'''
from patchworld.worldmodel_base import BaseWorldModel
import copy
import json
import math
import re


class CubeContactScaffoldWorldModel(BaseWorldModel):
    PARAMS = __PARAMS_JSON__
    EFF_COEF = PARAMS["effector_coef"]
    CONTACT_THRESHOLD = float(PARAMS["contact_threshold"])
    GRIP_MODE = str(PARAMS.get("grip_mode", "a4_pos"))
    USE_LATCH = bool(PARAMS.get("use_latch", True))
    ACTION_DIM = int(PARAMS["action_dim"])
    ENV = "cube"

    def init_belief(self):
        return {
            "env": self.ENV,
            "source": "cube_contact_scaffold_fit",
            "step": 0,
            "goal": "move block to goal pose",
            "objects": {},
            "relations": {},
            "metadata": {},
            "grasping": False,
        }

    def init_belief_from_observation(self, obs_text: str):
        return self.correct_belief(self.init_belief(), obs_text)

    def extract_valid_action_forms(self):
        forms = ["act[" + ",".join(f"a{i}=<float>" for i in range(self.ACTION_DIM)) + ",speed=<float>]"]
        return {"act": forms}

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
                except ValueError:
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
        return graph

    def _parse_object(self, line):
        match = re.match(r"^-\s*(\w+):(\w+)\s*(.*)$", line)
        if not match:
            return None
        obj = {"id": match.group(1), "category": match.group(2), "x": 0.0, "y": 0.0, "z": 0.0, "conf": 1.0}
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
        except ValueError:
            pass
        return {
            "subject": match.group(1),
            "predicate": match.group(2),
            "object": match.group(3),
            "value": value,
            "key": f"{match.group(1)}_{match.group(2)}_{match.group(3)}",
        }

    def correct_belief(self, belief_prior, obs_text: str):
        observed = self.parse_observation(obs_text)
        merged = copy.deepcopy(belief_prior) if isinstance(belief_prior, dict) else self.init_belief()
        for key in ("env", "source", "step", "goal", "metadata"):
            merged[key] = observed.get(key, merged.get(key))
        merged["objects"] = observed.get("objects", {})
        merged["relations"] = observed.get("relations", {})
        # Re-sync latch from observed proximity (oracle/tool replan resets grasp).
        objects = merged.get("objects", {})
        ex, ey, ez = self._xyz(objects, "effector")
        bx, by, bz = self._xyz(objects, "block")
        dist = math.sqrt((ex - bx) ** 2 + (ey - by) ** 2 + (ez - bz) ** 2)
        merged["grasping"] = bool(dist <= self.CONTACT_THRESHOLD)
        return merged

    def _parse_action(self, action: str):
        out = [0.0] * self.ACTION_DIM
        for match in re.finditer(r"a(\d+)=(-?\d+(?:\.\d+)?)", str(action)):
            idx = int(match.group(1))
            if 0 <= idx < self.ACTION_DIM:
                out[idx] = float(match.group(2))
        return out

    def _xyz(self, objects, object_id):
        obj = objects.get(object_id, {})
        return float(obj.get("x", 0.0)), float(obj.get("y", 0.0)), float(obj.get("z", 0.0))

    def _grip_active(self, a4: float) -> bool:
        if self.GRIP_MODE == "a4_neg":
            return a4 < 0.0
        if self.GRIP_MODE == "none":
            return True
        return a4 > 0.0  # a4_pos (LeWM Cube expert convention)

    def predict_belief(self, belief, action: str):
        b = copy.deepcopy(belief) if isinstance(belief, dict) else self.init_belief()
        objects = b.setdefault("objects", {})
        for object_id, category in (("effector", "gripper"), ("block", "cube"), ("goal", "goal_pose")):
            objects.setdefault(object_id, {"id": object_id, "category": category, "x": 0.0, "y": 0.0, "z": 0.0, "conf": 1.0})

        ex, ey, ez = self._xyz(objects, "effector")
        bx, by, bz = self._xyz(objects, "block")
        gx, gy, gz = self._xyz(objects, "goal")
        action_values = self._parse_action(action)
        while len(action_values) < 5:
            action_values.append(0.0)

        feats = [1.0, *action_values[:5], ex, ey, ez]
        delta = [sum(c * f for c, f in zip(row, feats)) for row in self.EFF_COEF]
        nex = ex + float(delta[0])
        ney = ey + float(delta[1])
        nez = ez + float(delta[2]) if len(delta) > 2 else ez

        dist_3d = math.sqrt((ex - bx) ** 2 + (ey - by) ** 2 + (ez - bz) ** 2)
        grip = self._grip_active(action_values[4])
        near = dist_3d <= self.CONTACT_THRESHOLD
        grasping = bool(b.get("grasping", False))
        if self.USE_LATCH:
            if near and grip:
                grasping = True
            elif not grip:
                grasping = False
            attached = grasping and grip
        else:
            attached = near and grip

        objects["effector"]["x"] = nex
        objects["effector"]["y"] = ney
        objects["effector"]["z"] = nez
        if attached:
            objects["block"]["x"] = nex
            objects["block"]["y"] = ney
            objects["block"]["z"] = nez
        else:
            objects["block"]["x"] = bx
            objects["block"]["y"] = by
            objects["block"]["z"] = bz
        objects["goal"]["x"] = gx
        objects["goal"]["y"] = gy
        objects["goal"]["z"] = gz
        b["grasping"] = attached

        block_goal = math.sqrt((objects["block"]["x"] - gx) ** 2 + (objects["block"]["y"] - gy) ** 2 + (objects["block"]["z"] - gz) ** 2)
        eff_block = math.sqrt((nex - objects["block"]["x"]) ** 2 + (ney - objects["block"]["y"]) ** 2 + (nez - objects["block"]["z"]) ** 2)
        b["relations"] = {
            "effector_distance_to_block": {
                "subject": "effector",
                "predicate": "distance_to",
                "object": "block",
                "value": eff_block,
                "key": "effector_distance_to_block",
            },
            "block_distance_to_goal": {
                "subject": "block",
                "predicate": "distance_to",
                "object": "goal",
                "value": block_goal,
                "key": "block_distance_to_goal",
            },
            "block_near_goal": {
                "subject": "block",
                "predicate": "near",
                "object": "goal",
                "value": 1.0 if block_goal <= 0.05 else 0.0,
                "key": "block_near_goal",
            },
        }
        b["step"] = int(b.get("step", 0)) + 1
        return b

    def readout_observation(self, belief, action: str):
        b = belief if isinstance(belief, dict) else self.init_belief()
        lines = [
            f"Environment: {b.get('env', self.ENV)}",
            f"Source: {b.get('source', 'cube_contact_scaffold_fit')}",
            f"Step: {int(b.get('step', 0))}",
            f"Goal: {b.get('goal', 'move block to goal pose')}",
            "Objects:",
        ]
        for object_id in sorted(b.get("objects", {})):
            obj = b["objects"][object_id]
            parts = [
                f"- {object_id}:{obj.get('category', object_id)}",
                f"x={float(obj.get('x', 0.0)):.3f}",
                f"y={float(obj.get('y', 0.0)):.3f}",
                f"z={float(obj.get('z', 0.0)):.3f}",
                f"conf={float(obj.get('conf', 1.0)):.3f}",
            ]
            lines.append(" ".join(parts))
        lines.append("Relations:")
        for rel in sorted(b.get("relations", {}).values(), key=lambda item: item.get("key", "")):
            lines.append(f"- {rel['subject']} {rel['predicate']} {rel['object']} = {rel.get('value')}")
        lines.append(f"Metadata: {json.dumps(b.get('metadata', {}), sort_keys=True, separators=(',', ':'))}")
        return "\n".join(lines)
'''
