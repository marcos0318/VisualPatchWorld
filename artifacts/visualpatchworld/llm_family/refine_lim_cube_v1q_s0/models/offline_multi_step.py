
"""LLM-proposed, data-fitted world model: velocity_saturation

Mechanism (as stated by the proposing LLM): Effector integrates action as velocity command, clamped by axis-specific maximum speeds.
Fit objective: multi_step; fitted constants are in FITTED_PARAMS.
"""
from patchworld.worldmodel_base import BaseWorldModel
import copy
import json
import math
import re

CANDIDATE_SOURCE = '# HYPOTHESIS: velocity_saturation\n# MECHANISM: Effector integrates action as velocity command, clamped by axis-specific maximum speeds.\nPARAMS = {"scale_x": 0.2, "scale_y": 0.2, "scale_z": 0.2, "max_vel_x": 0.15, "max_vel_y": 0.15, "max_vel_z": 0.15, "grip_threshold": 0.2}\n\ndef step(state, action, p, meta):\n    import math\n    s = dict(state)\n    ex, ey, ez = s["effector.x"], s["effector.y"], s["effector.z"]\n    bx, by, bz = s["block.x"], s["block.y"], s["block.z"]\n    gx, gy, gz = s["goal.x"], s["goal.y"], s["goal.z"]\n    \n    # Compute commanded velocity per axis\n    cmd_x = action[0] * p["scale_x"]\n    cmd_y = action[1] * p["scale_y"]\n    cmd_z = action[2] * p["scale_z"]\n    \n    # Apply velocity saturation (clamping)\n    dx = max(-p["max_vel_x"], min(p["max_vel_x"], cmd_x))\n    dy = max(-p["max_vel_y"], min(p["max_vel_y"], cmd_y))\n    dz = max(-p["max_vel_z"], min(p["max_vel_z"], cmd_z))\n    \n    # Update effector position\n    ex_new = ex + dx\n    ey_new = ey + dy\n    ez_new = ez + dz\n    \n    # Enforce effector Z limits (hardcoded from current model)\n    ez_new = max(0.015, min(0.335, ez_new))\n    \n    # Grip logic\n    is_gripped = action[4] > p["grip_threshold"]\n    \n    if is_gripped:\n        dx_b = ex_new - ex\n        dy_b = ey_new - ey\n        dz_b = ez_new - ez\n        bx_new = bx + dx_b\n        by_new = by + dy_b\n        bz_new = bz + dz_b\n    else:\n        bx_new = bx\n        by_new = by\n        bz_new = bz\n    \n    # Enforce block.z limits from meta\n    bz_min = meta.get("state_limits", {}).get("block.z", {}).get("min", {}).get("value", 0.019)\n    bz_max = meta.get("state_limits", {}).get("block.z", {}).get("observed_range", [0.019, 0.335])[1]\n    bz_new = max(bz_min, min(bz_max, bz_new))\n    \n    return {\n        "block.x": bx_new, "block.y": by_new, "block.z": bz_new,\n        "effector.x": ex_new, "effector.y": ey_new, "effector.z": ez_new,\n        "goal.x": gx, "goal.y": gy, "goal.z": gz\n    }'
FITTED_PARAMS = {"grip_threshold": 0.2, "max_vel_x": 0.13060304433545833, "max_vel_y": 0.11721116755503648, "max_vel_z": 0.08269736958289768, "scale_x": 0.07840229846631816, "scale_y": 0.1099000726094989, "scale_z": 0.1115609534004839}
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
