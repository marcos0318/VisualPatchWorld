
"""LLM-proposed, data-fitted world model: pid_position_control_effector

Mechanism (as stated by the proposing LLM): The first three action components are desired position set‑points for the effector; a PID controller (with parameters) generates a saturated velocity command that moves the effector toward the target. The fourth action is ignored; the block moves only when a grasp is formed via proximity (no explicit grasp command).
Fit objective: multi_step; fitted constants are in FITTED_PARAMS.
"""
from patchworld.worldmodel_base import BaseWorldModel
import copy
import json
import math
import re

CANDIDATE_SOURCE = '# HYPOTHESIS: pid_position_control_effector\n# MECHANISM: The first three action components are desired position set‑points for the effector; a PID controller (with parameters) generates a saturated velocity command that moves the effector toward the target. The fourth action is ignored; the block moves only when a grasp is formed via proximity (no explicit grasp command).\nPARAMS = {\n    "dt": 0.05,\n    "kp": 2.0,\n    "kd": 0.4,\n    "ki": 0.0,\n    "v_max": 0.12,\n    "integral_x": 0.0,\n    "integral_y": 0.0,\n    "integral_z": 0.0,\n    "prev_err_x": 0.0,\n    "prev_err_y": 0.0,\n    "prev_err_z": 0.0,\n    "grasp_thresh": 0.03\n}\n\ndef step(state, action, p, meta):\n    # helper\n    def clamp(v, lo, hi):\n        return max(lo, min(hi, v))\n    # PID for each axis (stateful integrals/prev_err are stored in p)\n    target_x, target_y, target_z = action[0], action[1], action[2]\n    err_x = target_x - state["effector.x"]\n    err_y = target_y - state["effector.y"]\n    err_z = target_z - state["effector.z"]\n    # update integrals\n    int_x = p["integral_x"] + err_x * p["dt"]\n    int_y = p["integral_y"] + err_y * p["dt"]\n    int_z = p["integral_z"] + err_z * p["dt"]\n    # derivative\n    der_x = (err_x - p["prev_err_x"]) / p["dt"]\n    der_y = (err_y - p["prev_err_y"]) / p["dt"]\n    der_z = (err_z - p["prev_err_z"]) / p["dt"]\n    # PID output (velocity command)\n    vel_x = p["kp"]*err_x + p["kd"]*der_x + p["ki"]*int_x\n    vel_y = p["kp"]*err_y + p["kd"]*der_y + p["ki"]*int_y\n    vel_z = p["kp"]*err_z + p["kd"]*der_z + p["ki"]*int_z\n    # saturate speed\n    vel_x = clamp(vel_x, -p["v_max"], p["v_max"])\n    vel_y = clamp(vel_y, -p["v_max"], p["v_max"])\n    vel_z = clamp(vel_z, -p["v_max"], p["v_max"])\n    # integrate\n    eff_x = state["effector.x"] + vel_x * p["dt"]\n    eff_y = state["effector.y"] + vel_y * p["dt"]\n    eff_z = state["effector.z"] + vel_z * p["dt"]\n    # update block if within grasp threshold (passive attachment)\n    blk_x, blk_y, blk_z = state["block.x"], state["block.y"], state["block.z"]\n    dist = ((eff_x - blk_x)**2 + (eff_y - blk_y)**2 + (eff_z - blk_z)**2)**0.5\n    if dist < p["grasp_thresh"]:\n        blk_x, blk_y, blk_z = eff_x, eff_y, eff_z\n    # store updated integrals/prev_err back into p for the next call (mutating p is allowed)\n    p["integral_x"], p["integral_y"], p["integral_z"] = int_x, int_y, int_z\n    p["prev_err_x"], p["prev_err_y"], p["prev_err_z"] = err_x, err_y, err_z\n    return {\n        "block.x": blk_x, "block.y": blk_y, "block.z": blk_z,\n        "effector.x": eff_x, "effector.y": eff_y, "effector.z": eff_z,\n        "goal.x": state["goal.x"], "goal.y": state["goal.y"], "goal.z": state["goal.z"]\n    }'
FITTED_PARAMS = {"dt": 0.006692347464108358, "grasp_thresh": 0.08991797860162284, "integral_x": -11.613091766368115, "integral_y": 0.8678022126286349, "integral_z": 3.1405161873112686, "kd": 0.002361293903579439, "ki": -0.12622057176326795, "kp": 8.177611677840282, "prev_err_x": 0.06386987155893382, "prev_err_y": -0.2221608311221648, "prev_err_z": 0.0066361774314200915, "v_max": 9.309139647580189}
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
