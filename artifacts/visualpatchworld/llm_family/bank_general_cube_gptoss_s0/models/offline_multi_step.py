
"""LLM-proposed, data-fitted world model: grasp_attach_release_mechanism

Mechanism (as stated by the proposing LLM): The fourth action component is a binary grasp command (positive = close, negative = open). When the hand is within a distance threshold and the command is “close”, the block becomes rigidly attached to the effector (its pose follows the hand). The fifth action component is ignored. The effector moves by simple saturated translation of the first three actions.
Fit objective: multi_step; fitted constants are in FITTED_PARAMS.
"""
from patchworld.worldmodel_base import BaseWorldModel
import copy
import json
import math
import re

CANDIDATE_SOURCE = '# HYPOTHESIS: grasp_attach_release_mechanism\n# MECHANISM: The fourth action component is a binary grasp command (positive = close, negative = open). When the hand is within a distance threshold and the command is “close”, the block becomes rigidly attached to the effector (its pose follows the hand). The fifth action component is ignored. The effector moves by simple saturated translation of the first three actions.\nPARAMS = {\n    "dt": 0.05,\n    "v_max": 0.11,\n    "attach_thresh": 0.035,\n    "deadzone": 0.06\n}\n\ndef step(state, action, p, meta):\n    def clamp(v, lo, hi):\n        return max(lo, min(hi, v))\n    # ---- effector motion -------------------------------------------------\n    dx = clamp(action[0] * p["dt"], -p["v_max"]*p["dt"], p["v_max"]*p["dt"])\n    dy = clamp(action[1] * p["dt"], -p["v_max"]*p["dt"], p["v_max"]*p["dt"])\n    dz = clamp(action[2] * p["dt"], -p["v_max"]*p["dt"], p["v_max"]*p["dt"])\n    eff_x = state["effector.x"] + dx\n    eff_y = state["effector.y"] + dy\n    eff_z = state["effector.z"] + dz\n    # ---- grasp command ---------------------------------------------------\n    grasp_cmd = action[3]\n    close_cmd = grasp_cmd > p["deadzone"]\n    open_cmd = grasp_cmd < -p["deadzone"]\n    # ---- attachment logic -----------------------------------------------\n    # keep attachment state inside p (persistent across steps)\n    attached = p.get("attached", False)\n    # distance check\n    dist = ((eff_x - state["block.x"])**2 + (eff_y - state["block.y"])**2 + (eff_z - state["block.z"])**2)**0.5\n    if close_cmd and not attached and dist < p["attach_thresh"]:\n        attached = True\n    if open_cmd:\n        attached = False\n    # update block position if attached\n    if attached:\n        blk_x, blk_y, blk_z = eff_x, eff_y, eff_z\n    else:\n        blk_x, blk_y, blk_z = state["block.x"], state["block.y"], state["block.z"]\n    # store attachment flag\n    p["attached"] = attached\n    # ---- assemble next state ---------------------------------------------\n    return {\n        "block.x": blk_x, "block.y": blk_y, "block.z": blk_z,\n        "effector.x": eff_x, "effector.y": eff_y, "effector.z": eff_z,\n        "goal.x": state["goal.x"], "goal.y": state["goal.y"], "goal.z": state["goal.z"]\n    }'
FITTED_PARAMS = {"attach_thresh": 0.035, "deadzone": 0.06, "dt": 0.09094100999271329, "v_max": 1.0127308049540633}
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
