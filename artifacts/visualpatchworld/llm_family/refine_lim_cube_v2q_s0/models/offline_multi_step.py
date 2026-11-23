
"""LLM-proposed, data-fitted world model: config_dependent_velocity

Mechanism (as stated by the proposing LLM): Effector velocity gain varies linearly with arm height (effector.z) due to kinematic Jacobian changes, while block follows effector with a fixed push gain during contact.
Fit objective: multi_step; fitted constants are in FITTED_PARAMS.
"""
from patchworld.worldmodel_base import BaseWorldModel
import copy
import json
import math
import re

CANDIDATE_SOURCE = '# HYPOTHESIS: config_dependent_velocity\n# MECHANISM: Effector velocity gain varies linearly with arm height (effector.z) due to kinematic Jacobian changes, while block follows effector with a fixed push gain during contact.\nPARAMS = {"vel_scale_base": 0.1, "vel_scale_z": 0.5, "contact_thresh": 0.05, "push_gain": 0.9, "ground_level": 0.019, "gripper_close_thresh": 0.5}\n\ndef step(state, action, p, meta):\n    import math\n    s = dict(state)\n    ex, ey, ez = s["effector.x"], s["effector.y"], s["effector.z"]\n    bx, by, bz = s["block.x"], s["block.y"], s["block.z"]\n    gx, gy, gz = s["goal.x"], s["goal.y"], s["goal.z"]\n    \n    # Configuration-dependent velocity scale (captures kinematic singularities/gains)\n    vx_scale = p["vel_scale_base"] + p["vel_scale_z"] * ez\n    vy_scale = p["vel_scale_base"] + p["vel_scale_z"] * ez\n    vz_scale = p["vel_scale_base"] # Z scale might be independent\n    \n    vx_cmd = action[0] * vx_scale\n    vy_cmd = action[1] * vy_scale\n    vz_cmd = action[2] * vz_scale\n    gripper_closed = action[4] > p["gripper_close_thresh"]\n    \n    # Move effector\n    ex_new = ex + vx_cmd\n    ey_new = ey + vy_cmd\n    ez_new = ez + vz_cmd\n    \n    # Check contact between effector and block\n    dist = math.sqrt((ex-bx)**2 + (ey-by)**2 + (ez-bz)**2)\n    in_contact = dist < p["contact_thresh"]\n    \n    # Block moves only if gripper closed AND in contact\n    if gripper_closed and in_contact:\n        # Block follows effector with some lag/gain\n        bx_new = bx + p["push_gain"] * (ex_new - ex)\n        by_new = by + p["push_gain"] * (ey_new - ey)\n        bz_new = bz + p["push_gain"] * (ez_new - ez)\n    else:\n        bx_new, by_new, bz_new = bx, by, bz\n    \n    # Ground constraint for block (using meta limits)\n    min_bz = meta.get("state_limits", {}).get("block.z", {}).get("min", {}).get("value", p["ground_level"])\n    max_bz = meta.get("state_limits", {}).get("block.z", {}).get("observed_range", [0, 0.335])[1]\n    bz_new = max(min_bz, min(max_bz, bz_new))\n    \n    # Clamp to bounds\n    ex_new = max(0.248, min(0.567, ex_new))\n    ey_new = max(-0.339, min(0.294, ey_new))\n    ez_new = max(0.015, min(0.335, ez_new))\n    bx_new = max(0.251, min(0.557, bx_new))\n    by_new = max(-0.339, min(0.294, by_new))\n    \n    return {\n        "block.x": bx_new, "block.y": by_new, "block.z": bz_new,\n        "effector.x": ex_new, "effector.y": ey_new, "effector.z": ez_new,\n        "goal.x": gx, "goal.y": gy, "goal.z": gz\n    }'
FITTED_PARAMS = {"contact_thresh": 0.05, "gripper_close_thresh": 0.0, "ground_level": 0.019, "push_gain": 0.9867442279363716, "vel_scale_base": 0.08820911952632488, "vel_scale_z": 0.06820619076501859}
ENV = 'cube'
ACTION_DIM = 5
STATE_KEYS = ["block.x", "block.y", "block.z", "effector.x", "effector.y", "effector.z", "goal.x", "goal.y", "goal.z"]
STATIC_META = {"state_limits": {"block.z": {"action_model_r2": 0.999, "min": {"blocked_pushes": 733, "median_progress_of_predicted": 0.0, "value": 0.019}, "observed_range": [0.019, 0.335]}}}

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
