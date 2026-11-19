
"""LLM-proposed, data-fitted world model: gripper_with_grasp_and_gravity

Mechanism (as stated by the proposing LLM): Gripper can grasp block when close; both subject to gravity and ground contact.
Fit objective: multi_step; fitted constants are in FITTED_PARAMS.
"""
from patchworld.worldmodel_base import BaseWorldModel
import copy
import json
import math
import re

CANDIDATE_SOURCE = '# HYPOTHESIS: gripper_with_grasp_and_gravity\n# MECHANISM: Gripper can grasp block when close; both subject to gravity and ground contact.\nPARAMS = {\n    "grip_threshold": 0.05,\n    "gravity": -9.8,\n    "ground_level": 0.02,\n    "dt": 0.1,\n    "gripper_speed": 1.0,\n    "release_distance": 0.1\n}\n\ndef step(state, action, p, meta):\n    import math\n    \n    # Compute distance between effector and block\n    dx = state["effector.x"] - state["block.x"]\n    dy = state["effector.y"] - state["block.y"]\n    dz = state["effector.z"] - state["block.z"]\n    distance = math.sqrt(dx*dx + dy*dy + dz*dz)\n    \n    # Determine if gripping\n    is_gripped = getattr(step, \'is_gripped\', False)\n    \n    # Update gripper position based on action\n    grip_action = action[4]\n    if grip_action > 0.5:\n        if distance < p["grip_threshold"]:\n            step.is_gripped = True\n    elif grip_action < -0.5:\n        step.is_gripped = False\n        \n    # If already gripped, check release condition\n    if getattr(step, \'is_gipped\', False) and distance > p["release_distance"]:\n        step.is_gripped = False\n            \n    # Move gripper\n    new_ex = state["effector.x"] + action[0] * p["gripper_speed"] * p["dt"]\n    new_ey = state["effector.y"] + action[1] * p["gripper_speed"] * p["dt"]\n    new_ez = state["effector.z"] + action[2] * p["gripper_speed"] * p["dt"]\n    \n    # Apply gravity to block if not gripped\n    new_bx, new_by, new_bz = state["block.x"], state["block.y"], state["block.z"]\n    if not getattr(step, \'is_gripped\', False):\n        new_bz += p["gravity"] * p["dt"] * p["dt"] / 2\n        if new_bz < p["ground_level"]:\n            new_bz = p["ground_level"]\n    else:\n        # Move block with gripper\n        new_bx = new_ex\n        new_by = new_ey\n        new_bz = new_ez\n\n    # Ensure block doesn\'t fall through floor even when not gripped\n    if new_bz < p["ground_level"]:\n        new_bz = p["ground_level"]\n\n    new_state = state.copy()\n    new_state["effector.x"] = new_ex\n    new_state["effector.y"] = new_ey\n    new_state["effector.z"] = new_ez\n    new_state["block.x"] = new_bx\n    new_state["block.y"] = new_by\n    new_state["block.z"] = new_bz\n    return new_state'
FITTED_PARAMS = {"dt": 0.2865443193628798, "gravity": -8.680228887236884, "grip_threshold": 0.05679798526881795, "gripper_speed": 0.28648115529697543, "ground_level": 0.019947368421052603, "release_distance": 0.12052317174516673}
ENV = 'cube'
ACTION_DIM = 5
STATE_KEYS = ["block.x", "block.y", "block.z", "effector.x", "effector.y", "effector.z", "goal.x", "goal.y", "goal.z"]

_ns = {"math": math}
exec(compile(CANDIDATE_SOURCE, "<candidate>", "exec"), _ns)
_step = _ns["step"]


class LLMFamilyWorldModel(BaseWorldModel):
    def init_belief(self):
        return {"env": ENV, "source": "llm_family", "step": 0, "goal": "", "objects": {}, "relations": {}, "metadata": {}}

    def init_belief_from_observation(self, obs_text: str):
        return self.correct_belief(self.init_belief(), obs_text)

    def extract_valid_action_forms(self):
        if ACTION_DIM == 2:
            form = "[dx=<float>,dy=<float>,speed=<float>]"
            return {k: [k + form] for k in ("move_left", "move_right", "move_up", "move_down", "hold")}
        return {"act": ["act[" + ",".join(f"a{i}=<float>" for i in range(ACTION_DIM)) + ",speed=<float>]"]}

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
                m = re.match(r"^-\s*(\w+):(\w+)\s*(.*)$", line)
                if m:
                    obj = {"id": m.group(1), "category": m.group(2), "x": 0.0, "y": 0.0, "conf": 1.0}
                    for key, value in re.findall(r"(\w+)=(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", m.group(3)):
                        obj[key] = float(value)
                    graph["objects"][obj["id"]] = obj
            elif section == "relations" and line.startswith("-"):
                m = re.match(r"^-\s*(\w+)\s+(\w+(?:_\w+)*)\s+(\w+)\s*=\s*(.+)$", line)
                if m:
                    value = m.group(4).strip()
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                    key = f"{m.group(1)}_{m.group(2)}_{m.group(3)}"
                    graph["relations"][key] = {"subject": m.group(1), "predicate": m.group(2), "object": m.group(3), "value": value, "key": key}
        return graph

    def correct_belief(self, belief_prior, obs_text: str):
        observed = self.parse_observation(obs_text)
        merged = copy.deepcopy(belief_prior) if isinstance(belief_prior, dict) else self.init_belief()
        for key in ("env", "source", "step", "goal", "metadata"):
            merged[key] = observed.get(key, merged.get(key))
        merged["objects"] = observed.get("objects", {})
        merged["relations"] = observed.get("relations", {})
        return merged

    def _parse_action(self, action):
        if ACTION_DIM == 2:
            m = re.match(r"^\w+\[dx=(-?\d+(?:\.\d+)?),dy=(-?\d+(?:\.\d+)?)", str(action).strip())
            return [float(m.group(1)), float(m.group(2))] if m else [0.0, 0.0]
        out = [0.0] * ACTION_DIM
        for m in re.finditer(r"a(\d+)=(-?\d+(?:\.\d+)?)", str(action)):
            i = int(m.group(1))
            if 0 <= i < ACTION_DIM:
                out[i] = float(m.group(2))
        return out

    def predict_belief(self, belief, action: str):
        b = copy.deepcopy(belief) if isinstance(belief, dict) else self.init_belief()
        objects = b.setdefault("objects", {})
        state = {}
        for key in STATE_KEYS:
            oid, fld = key.split(".", 1)
            state[key] = float(objects.get(oid, {}).get(fld, 0.0))
        try:
            nxt = _step(dict(state), self._parse_action(action), dict(FITTED_PARAMS), dict(b.get("metadata") or {}))
        except Exception:
            nxt = state
        for key in STATE_KEYS:
            oid, fld = key.split(".", 1)
            obj = objects.setdefault(oid, {"id": oid, "category": oid, "x": 0.0, "y": 0.0, "conf": 1.0})
            try:
                v = float(nxt.get(key, state[key]))
            except (TypeError, ValueError):
                v = state[key]
            obj[fld] = v if math.isfinite(v) else state[key]
        b["step"] = int(b.get("step", 0)) + 1
        return b

    def readout_observation(self, belief, action: str):
        b = belief if isinstance(belief, dict) else self.init_belief()
        lines = [f"Environment: {b.get('env', ENV)}", f"Source: {b.get('source', 'llm_family')}", f"Step: {int(b.get('step', 0))}", f"Goal: {b.get('goal', '')}", "Objects:"]
        for oid in sorted(b.get("objects", {})):
            obj = b["objects"][oid]
            parts = [f"- {oid}:{obj.get('category', oid)}", f"x={float(obj.get('x', 0.0)):.3f}", f"y={float(obj.get('y', 0.0)):.3f}"]
            if "z" in obj:
                parts.append(f"z={float(obj.get('z', 0.0)):.3f}")
            if "angle" in obj:
                parts.append(f"angle={float(obj.get('angle', 0.0)):.3f}")
            for fld in ("vx", "vy"):
                if fld in obj:
                    parts.append(f"{fld}={float(obj.get(fld, 0.0)):.3f}")
            parts.append(f"conf={float(obj.get('conf', 1.0)):.3f}")
            lines.append(" ".join(parts))
        lines.append("Relations:")
        for rel in sorted(b.get("relations", {}).values(), key=lambda item: item.get("key", "")):
            lines.append(f"- {rel['subject']} {rel['predicate']} {rel['object']} = {rel.get('value')}")
        lines.append(f"Metadata: {json.dumps(b.get('metadata', {}), sort_keys=True, separators=(',', ':'))}")
        return "\n".join(lines)
