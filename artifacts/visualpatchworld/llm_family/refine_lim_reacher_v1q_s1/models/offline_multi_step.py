
"""LLM-proposed, data-fitted world model: arm_spring_rest

Mechanism (as stated by the proposing LLM): Arm joints are velocity-controlled but subject to elastic restoring forces pulling them towards a rest configuration.
Fit objective: multi_step; fitted constants are in FITTED_PARAMS.
"""
from patchworld.worldmodel_base import BaseWorldModel
import copy
import json
import math
import re

CANDIDATE_SOURCE = '# HYPOTHESIS: arm_spring_rest\n# MECHANISM: Arm joints are velocity-controlled but subject to elastic restoring forces pulling them towards a rest configuration.\nPARAMS = {"gain_ax": 0.8, "gain_ay": 0.8, "rest_x": -3.0, "rest_y": 0.0, "stiff_x": 0.5, "stiff_y": 0.5, "link1": 0.12, "link2": 0.12, "off_x": 0.0, "off_y": 0.0}\n\ndef step(state, action, p, meta):\n    import math\n    s = state.copy()\n    \n    # Extract limits from meta\n    limits = meta.get("state_limits", {})\n    \n    # Update arm angles with action + spring force\n    # delta = gain * action + stiffness * (rest - current)\n    ax = state["arm.x"]\n    ay = state["arm.y"]\n    \n    delta_x = p["gain_ax"] * action[0] + p["stiff_x"] * (p["rest_x"] - ax)\n    delta_y = p["gain_ay"] * action[1] + p["stiff_y"] * (p["rest_y"] - ay)\n    \n    new_ax = ax + delta_x\n    new_ay = ay + delta_y\n    \n    # Enforce state limits if present in meta\n    if "arm.x" in limits:\n        lim = limits["arm.x"]\n        new_ax = max(lim["min"]["value"], min(lim["max"]["value"], new_ax))\n    if "arm.y" in limits:\n        lim = limits["arm.y"]\n        new_ay = max(lim["min"]["value"], min(lim["max"]["value"], new_ay))\n    \n    s["arm.x"] = new_ax\n    s["arm.y"] = new_ay\n    \n    # Forward Kinematics for finger\n    c1 = math.cos(new_ax)\n    s1 = math.sin(new_ax)\n    c2 = math.cos(new_ax + new_ay)\n    s2 = math.sin(new_ax + new_ay)\n    \n    s["finger.x"] = p["link1"] * c1 + p["link2"] * c2 + p["off_x"]\n    s["finger.y"] = p["link1"] * s1 + p["link2"] * s2 + p["off_y"]\n    \n    s["target.x"] = state["target.x"]\n    s["target.y"] = state["target.y"]\n    return s'
FITTED_PARAMS = {"gain_ax": 0.779425669310585, "gain_ay": 0.9361646569183975, "link1": 0.11856581496885038, "link2": 0.11730418405948283, "off_x": 0.00046063545203074274, "off_y": 0.0005480048803928939, "rest_x": 0.5703550805716896, "rest_y": -0.12537099889412692, "stiff_x": -0.0012318917204425304, "stiff_y": 0.00440246940572251}
ENV = 'reacher'
ACTION_DIM = 2
STATE_KEYS = ["arm.x", "arm.y", "finger.x", "finger.y", "target.x", "target.y"]
STATIC_META = {"state_limits": {"arm.y": {"action_model_r2": 0.958, "max": {"blocked_pushes": 18, "median_progress_of_predicted": 0.068, "value": 2.869}, "min": {"blocked_pushes": 7, "median_progress_of_predicted": -0.057, "value": -2.899}, "observed_range": [-2.899, 2.869]}}}

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
