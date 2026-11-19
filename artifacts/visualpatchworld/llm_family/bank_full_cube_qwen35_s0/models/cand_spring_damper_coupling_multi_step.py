
"""LLM-proposed, data-fitted world model: spring_damper_coupling

Mechanism (as stated by the proposing LLM): Block is connected to effector via virtual spring-damper when grasping; spring pulls block toward effector, damper resists relative velocity.
Fit objective: multi_step; fitted constants are in FITTED_PARAMS.
"""
from patchworld.worldmodel_base import BaseWorldModel
import copy
import json
import math
import re

CANDIDATE_SOURCE = '# HYPOTHESIS: spring_damper_coupling\n# MECHANISM: Block is connected to effector via virtual spring-damper when grasping; spring pulls block toward effector, damper resists relative velocity.\nPARAMS = {"effector_gain": 0.1, "spring_k": 5.0, "damper_c": 2.0, "grasp_thresh": 0.0, "block_drag": 0.1}\n\ndef step(state, action, p, meta):\n    import math\n    new_state = dict(state)\n    \n    eff_x = state["effector.x"]\n    eff_y = state["effector.y"]\n    eff_z = state["effector.z"]\n    block_x = state["block.x"]\n    block_y = state["block.y"]\n    block_z = state["block.z"]\n    goal_x = state["goal.x"]\n    goal_y = state["goal.y"]\n    goal_z = state["goal.z"]\n    \n    block_vx = state.get("block.vx", 0.0)\n    block_vy = state.get("block.vy", 0.0)\n    block_vz = state.get("block.vz", 0.0)\n    eff_vx = state.get("effector.vx", 0.0)\n    eff_vy = state.get("effector.vy", 0.0)\n    eff_vz = state.get("effector.vz", 0.0)\n    \n    ax = action[0] if len(action) > 0 else 0.0\n    ay = action[1] if len(action) > 1 else 0.0\n    az = action[2] if len(action) > 2 else 0.0\n    agrasp = action[4] if len(action) > 4 else 0.0\n    \n    new_eff_x = eff_x + p["effector_gain"] * ax\n    new_eff_y = eff_y + p["effector_gain"] * ay\n    new_eff_z = eff_z + p["effector_gain"] * az\n    \n    new_eff_vx = (new_eff_x - eff_x)\n    new_eff_vy = (new_eff_y - eff_y)\n    new_eff_vz = (new_eff_z - eff_z)\n    \n    is_grasping = agrasp > p["grasp_thresh"]\n    \n    if is_grasping:\n        disp_x = block_x - eff_x\n        disp_y = block_y - eff_y\n        disp_z = block_z - eff_z\n        \n        rel_vx = block_vx - eff_vx\n        rel_vy = block_vy - eff_vy\n        rel_vz = block_vz - eff_vz\n        \n        spring_fx = -p["spring_k"] * disp_x\n        spring_fy = -p["spring_k"] * disp_y\n        spring_fz = -p["spring_k"] * disp_z\n        \n        damp_fx = -p["damper_c"] * rel_vx\n        damp_fy = -p["damper_c"] * rel_vy\n        damp_fz = -p["damper_c"] * rel_vz\n        \n        total_fx = spring_fx + damp_fx\n        total_fy = spring_fy + damp_fy\n        total_fz = spring_fz + damp_fz\n        \n        block_vx = block_vx + total_fx - p["block_drag"] * block_vx\n        block_vy = block_vy + total_fy - p["block_drag"] * block_vy\n        block_vz = block_vz + total_fz - p["block_drag"] * block_vz\n    else:\n        block_vx = block_vx - p["block_drag"] * block_vx\n        block_vy = block_vy - p["block_drag"] * block_vy\n        block_vz = block_vz - p["block_drag"] * block_vz\n    \n    new_block_x = block_x + block_vx\n    new_block_y = block_y + block_vy\n    new_block_z = block_z + block_vz\n    \n    new_state["effector.x"] = new_eff_x\n    new_state["effector.y"] = new_eff_y\n    new_state["effector.z"] = new_eff_z\n    new_state["block.x"] = new_block_x\n    new_state["block.y"] = new_block_y\n    new_state["block.z"] = new_block_z\n    new_state["effector.vx"] = new_eff_vx\n    new_state["effector.vy"] = new_eff_vy\n    new_state["effector.vz"] = new_eff_vz\n    new_state["block.vx"] = block_vx\n    new_state["block.vy"] = block_vy\n    new_state["block.vz"] = block_vz\n    new_state["goal.x"] = goal_x\n    new_state["goal.y"] = goal_y\n    new_state["goal.z"] = goal_z\n    \n    return new_state'
FITTED_PARAMS = {"block_drag": 0.1578870273551744, "damper_c": 0.593767249619271, "effector_gain": 0.09841510899218003, "grasp_thresh": 0.04116305363741329, "spring_k": 1.2628309210253716}
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
