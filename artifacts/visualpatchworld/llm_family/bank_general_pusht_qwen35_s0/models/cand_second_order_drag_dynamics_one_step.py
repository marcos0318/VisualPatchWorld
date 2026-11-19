
"""LLM-proposed, data-fitted world model: second_order_drag_dynamics

Mechanism (as stated by the proposing LLM): Agent has second-order dynamics with action as acceleration and linear velocity drag; block responds to contact force
Fit objective: one_step; fitted constants are in FITTED_PARAMS.
"""
from patchworld.worldmodel_base import BaseWorldModel
import copy
import json
import math
import re

CANDIDATE_SOURCE = '# HYPOTHESIS: second_order_drag_dynamics\n# MECHANISM: Agent has second-order dynamics with action as acceleration and linear velocity drag; block responds to contact force\nPARAMS = {"acc_scale": 400.0, "drag_coeff": 0.8, "contact_force_thresh": 30.0, "block_mass": 5.0, "dt": 0.1}\n\ndef step(state, action, p, meta):\n    import math\n    new_state = {}\n    \n    # Agent: second-order dynamics - action is acceleration command\n    acc_x = action[0] * p["acc_scale"]\n    acc_y = action[1] * p["acc_scale"]\n    \n    # Apply linear drag to velocity\n    drag_vx = -p["drag_coeff"] * state["agent.vx"]\n    drag_vy = -p["drag_coeff"] * state["agent.vy"]\n    \n    # Update agent velocity\n    new_vx = state["agent.vx"] + (acc_x + drag_vx) * p["dt"]\n    new_vy = state["agent.vy"] + (acc_y + drag_vy) * p["dt"]\n    \n    # Update agent position\n    new_x = state["agent.x"] + new_vx * p["dt"]\n    new_y = state["agent.y"] + new_vy * p["dt"]\n    \n    # Contact force estimate based on relative velocity\n    dx = state["block.x"] - state["agent.x"]\n    dy = state["block.y"] - state["agent.y"]\n    dist = math.sqrt(dx*dx + dy*dy)\n    \n    if dist < 50.0:\n        contact_force = math.sqrt(new_vx*new_vx + new_vy*new_vy)\n        if contact_force > p["contact_force_thresh"]:\n            force_ratio = (contact_force - p["contact_force_thresh"]) / p["contact_force_thresh"]\n            new_block_x = state["block.x"] + (new_vx / p["block_mass"]) * force_ratio * p["dt"]\n            new_block_y = state["block.y"] + (new_vy / p["block_mass"]) * force_ratio * p["dt"]\n            new_angle = state["block.angle"] + (action[0] * 0.01) * force_ratio\n        else:\n            new_block_x = state["block.x"]\n            new_block_y = state["block.y"]\n            new_angle = state["block.angle"]\n    else:\n        new_block_x = state["block.x"]\n        new_block_y = state["block.y"]\n        new_angle = state["block.angle"]\n    \n    new_state["agent.vx"] = new_vx\n    new_state["agent.vy"] = new_vy\n    new_state["agent.x"] = new_x\n    new_state["agent.y"] = new_y\n    new_state["block.x"] = new_block_x\n    new_state["block.y"] = new_block_y\n    new_state["block.angle"] = new_angle\n    \n    return new_state'
FITTED_PARAMS = {"acc_scale": 3355.1365230238607, "block_mass": 5.0, "contact_force_thresh": 4051.242572834401, "drag_coeff": 8.998978056461004, "dt": 0.10719492631942495}
ENV = 'pusht'
ACTION_DIM = 2
STATE_KEYS = ["agent.vx", "agent.vy", "agent.x", "agent.y", "block.angle", "block.x", "block.y"]

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
