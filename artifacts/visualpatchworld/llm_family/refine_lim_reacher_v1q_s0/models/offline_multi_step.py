
"""LLM-proposed, data-fitted world model: config_dependent_gain

Mechanism (as stated by the proposing LLM): Action-to-state gain flips sign based on configuration (arm.x value); this explains why same-sign actions produce opposite state changes in different regions.
Fit objective: multi_step; fitted constants are in FITTED_PARAMS.
"""
from patchworld.worldmodel_base import BaseWorldModel
import copy
import json
import math
import re

CANDIDATE_SOURCE = '# HYPOTHESIS: config_dependent_gain\n# MECHANISM: Action-to-state gain flips sign based on configuration (arm.x value); this explains why same-sign actions produce opposite state changes in different regions.\nPARAMS = {"link1_len": 0.12, "link2_len": 0.12, "joint0_gain_pos": 0.5, "joint0_gain_neg": -0.5, "joint1_gain": 0.5, "decay_x": 0.95, "decay_y": 0.95, "base_x": 0.0, "base_y": 0.0, "switch_threshold": -1.5, "fric_x": 0.01, "fric_y": 0.01}\n\ndef step(state, action, p, meta):\n    import math\n    new_state = {}\n    new_state["target.x"] = state["target.x"]\n    new_state["target.y"] = state["target.y"]\n    \n    def sgn(x):\n        return 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)\n    \n    # Configuration-dependent gain for joint0 (arm.x)\n    # When arm.x is below threshold, positive action decreases arm.x (gain is negative)\n    if state["arm.x"] < p["switch_threshold"]:\n        j0_gain = p["joint0_gain_neg"]\n    else:\n        j0_gain = p["joint0_gain_pos"]\n    \n    j1_gain = p["joint1_gain"]\n    \n    # Friction opposing action direction\n    fx = p["fric_x"] * sgn(action[0])\n    fy = p["fric_y"] * sgn(action[1])\n    \n    # Leaky integration with config-dependent gain\n    new_state["arm.x"] = state["arm.x"] * p["decay_x"] + j0_gain * action[0] - fx\n    new_state["arm.y"] = state["arm.y"] * p["decay_y"] + j1_gain * action[1] - fy\n    \n    # Enforce limits from meta\n    limits = meta.get("state_limits", {})\n    if "arm.y" in limits:\n        min_y = limits["arm.y"].get("min", {}).get("value", -float(\'inf\'))\n        max_y = limits["arm.y"].get("max", {}).get("value", float(\'inf\'))\n        new_state["arm.y"] = max(min_y, min(max_y, new_state["arm.y"]))\n    if "arm.x" in limits:\n        min_x = limits["arm.x"].get("min", {}).get("value", -float(\'inf\'))\n        max_x = limits["arm.x"].get("max", {}).get("value", float(\'inf\'))\n        new_state["arm.x"] = max(min_x, min(max_x, new_state["arm.x"]))\n    \n    j0 = new_state["arm.x"]\n    j1 = new_state["arm.y"]\n    \n    new_state["finger.x"] = p["base_x"] + p["link1_len"] * math.cos(j0) + p["link2_len"] * math.cos(j0 + j1)\n    new_state["finger.y"] = p["base_y"] + p["link1_len"] * math.sin(j0) + p["link2_len"] * math.sin(j0 + j1)\n    return new_state'
FITTED_PARAMS = {"base_x": 0.00015526870111884247, "base_y": 0.0008013532577210542, "decay_x": 1.0022127382404187, "decay_y": 0.9952872757933925, "fric_x": 0.006591122357639044, "fric_y": 0.0023118420508459574, "joint0_gain_neg": 0.793976517059716, "joint0_gain_pos": 1638.983633859189, "joint1_gain": 0.9440459900266761, "link1_len": 0.11786101678746422, "link2_len": 0.11799184058035503, "switch_threshold": 46.95412110368977}
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
