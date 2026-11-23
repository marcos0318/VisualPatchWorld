
"""LLM-proposed, data-fitted world model: substep_integration

Mechanism (as stated by the proposing LLM): The recorded transition spans multiple control cycles; substepping reduces integration error for stiff finger tracking.
Fit objective: multi_step; fitted constants are in FITTED_PARAMS.
"""
from patchworld.worldmodel_base import BaseWorldModel
import copy
import json
import math
import re

CANDIDATE_SOURCE = '# HYPOTHESIS: substep_integration\n# MECHANISM: The recorded transition spans multiple control cycles; substepping reduces integration error for stiff finger tracking.\nPARAMS = {"arm_gain": 0.5, "bias_x": 0.0, "bias_y": 0.0, "fric_x": 0.05, "fric_y": 0.05, "link1": 0.12, "link2": 0.12, "base_x": 0.0, "base_y": 0.0, "finger_k": 0.5}\n\ndef step(state, action, p, meta):\n    import math\n    ns = dict(state)\n    ax, ay = state["arm.x"], state["arm.y"]\n    fx, fy = state["finger.x"], state["finger.y"]\n    \n    # Substep parameters\n    N = 5\n    dt = 1.0 / N\n    \n    # Scale gains for substeps to approximate the full-step behavior\n    # If x_new = x + k(target-x), then k_sub = 1 - (1-k)^(1/N)\n    # However, we treat params as per-substep gains for optimization flexibility.\n    arm_gain_sub = p["arm_gain"] * dt\n    finger_k_sub = p["finger_k"] * dt\n    \n    for _ in range(N):\n        # Arm update\n        sign_x = 1.0 if action[0] > 0 else (-1.0 if action[0] < 0 else 0.0)\n        sign_y = 1.0 if action[1] > 0 else (-1.0 if action[1] < 0 else 0.0)\n        \n        eff_ax = action[0] + p["bias_x"] - p["fric_x"] * sign_x\n        eff_ay = action[1] + p["bias_y"] - p["fric_y"] * sign_y\n        \n        ax = ax + arm_gain_sub * eff_ax\n        ay = ay + arm_gain_sub * eff_ay\n        \n        # Limits (applied at each substep to prevent drift)\n        limits = meta.get("state_limits", {}).get("arm.y", {})\n        min_y = limits.get("min", {}).get("value", -3.0)\n        max_y = limits.get("max", {}).get("value", 3.0)\n        if ay < min_y: ay = min_y\n        if ay > max_y: ay = max_y\n        \n        # FK for finger target\n        c1, s1 = math.cos(ax), math.sin(ax)\n        c2, s2 = math.cos(ax + ay), math.sin(ax + ay)\n        dfx = p["base_x"] + p["link1"] * c1 + p["link2"] * c2\n        dfy = p["base_y"] + p["link1"] * s1 + p["link2"] * s2\n        \n        # Finger update\n        fx = fx + finger_k_sub * (dfx - fx)\n        fy = fy + finger_k_sub * (dfy - fy)\n    \n    ns["arm.x"], ns["arm.y"] = ax, ay\n    ns["finger.x"], ns["finger.y"] = fx, fy\n    ns["target.x"], ns["target.y"] = state["target.x"], state["target.y"]\n    return ns'
FITTED_PARAMS = {"arm_gain": 0.9025489163363148, "base_x": 0.00037720028970264654, "base_y": 0.0008573851313240111, "bias_x": -0.0013214030947131968, "bias_y": -4.9624483199125485e-05, "finger_k": 1.6551988401081905, "fric_x": 0.011944648890353609, "fric_y": -0.022173132733084824, "link1": 0.12089966307769054, "link2": 0.11832629771097009}
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
