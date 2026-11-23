
"""LLM-proposed, data-fitted world model: kinematic_coupling_rigid_grasp

Mechanism (as stated by the proposing LLM): Effector X gain depends on Z height (simulating arm Jacobian kinematics); Block is rigidly attached when grasped.
Fit objective: multi_step; fitted constants are in FITTED_PARAMS.
"""
from patchworld.worldmodel_base import BaseWorldModel
import copy
import json
import math
import re

CANDIDATE_SOURCE = '# HYPOTHESIS: kinematic_coupling_rigid_grasp\n# MECHANISM: Effector X gain depends on Z height (simulating arm Jacobian kinematics); Block is rigidly attached when grasped.\nPARAMS = {"eff_gain_x_base": 0.15, "eff_gain_x_z_dep": 0.1, "eff_gain_y": 0.11, "eff_gain_z": 0.10, "grasp_thresh": 0.3, "grasp_x_off": 0.01, "grasp_y_off": 0.0, "grasp_z_off": -0.02}\n\ndef step(state, action, p, meta):\n    import math\n    s = state\n    a = action\n    ns = {}\n    ns["goal.x"] = s["goal.x"]\n    ns["goal.y"] = s["goal.y"]\n    ns["goal.z"] = s["goal.z"]\n    \n    # Z-dependent gain for X movement\n    current_gain_x = p["eff_gain_x_base"] + p["eff_gain_x_z_dep"] * s["effector.z"]\n    dx = a[0] * current_gain_x\n    dy = a[1] * p["eff_gain_y"]\n    dz = a[2] * p["eff_gain_z"]\n    \n    ns["effector.x"] = s["effector.x"] + dx\n    ns["effector.y"] = s["effector.y"] + dy\n    ns["effector.z"] = s["effector.z"] + dz\n    \n    if a[4] > p["grasp_thresh"]:\n        ns["block.x"] = ns["effector.x"] + p["grasp_x_off"]\n        ns["block.y"] = ns["effector.y"] + p["grasp_y_off"]\n        ns["block.z"] = ns["effector.z"] + p["grasp_z_off"]\n    else:\n        ns["block.x"] = s["block.x"]\n        ns["block.y"] = s["block.y"]\n        ns["block.z"] = s["block.z"]\n    return ns'
FITTED_PARAMS = {"eff_gain_x_base": -0.013033902384729207, "eff_gain_x_z_dep": 0.4822801301877684, "eff_gain_y": 0.11058198126808745, "eff_gain_z": 0.10013328583660885, "grasp_thresh": 0.24676077964152468, "grasp_x_off": 0.002092129803638586, "grasp_y_off": 0.0013890598315941943, "grasp_z_off": -0.020505950851871353}
ENV = 'cube'
ACTION_DIM = 5
STATE_KEYS = ["block.x", "block.y", "block.z", "effector.x", "effector.y", "effector.z", "goal.x", "goal.y", "goal.z"]
STATIC_META = {}

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
            nxt = _step(dict(state), self._parse_action(action), dict(FITTED_PARAMS), {**STATIC_META, **(b.get("metadata") or {})})
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
