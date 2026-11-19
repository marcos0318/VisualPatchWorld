
"""LLM-proposed, data-fitted world model: velocity_momentum_friction

Mechanism (as stated by the proposing LLM): Objects have hidden velocity; action applies acceleration; friction dissipates momentum; block slides when pushed
Fit objective: multi_step; fitted constants are in FITTED_PARAMS.
"""
from patchworld.worldmodel_base import BaseWorldModel
import copy
import json
import math
import re

CANDIDATE_SOURCE = '# HYPOTHESIS: velocity_momentum_friction\n# MECHANISM: Objects have hidden velocity; action applies acceleration; friction dissipates momentum; block slides when pushed\nPARAMS = {"eff_mass": 1.0, "block_mass": 2.0, "friction": 0.5, "force_scale": 0.2, "dt": 0.1}\n\ndef step(state, action, p, meta):\n    import math\n    new_state = dict(state)\n    \n    # Initialize velocities from meta or assume zero (hidden state approximation)\n    evx = meta.get("eff_vx", 0.0)\n    evy = meta.get("eff_vy", 0.0)\n    evz = meta.get("eff_vz", 0.0)\n    bvx = meta.get("block_vx", 0.0)\n    bvy = meta.get("block_vy", 0.0)\n    bvz = meta.get("block_vz", 0.0)\n    \n    # Action applies force to effector\n    evx += action[0] * p["force_scale"] / p["eff_mass"] * p["dt"]\n    evy += action[1] * p["force_scale"] / p["eff_mass"] * p["dt"]\n    evz += action[2] * p["force_scale"] / p["eff_mass"] * p["dt"]\n    \n    # Friction\n    evx *= (1 - p["friction"] * p["dt"])\n    evy *= (1 - p["friction"] * p["dt"])\n    evz *= (1 - p["friction"] * p["dt"])\n    \n    # Update effector position\n    ex = state["effector.x"] + evx * p["dt"]\n    ey = state["effector.y"] + evy * p["dt"]\n    ez = max(0.02, state["effector.z"] + evz * p["dt"])\n    new_state["effector.x"] = ex\n    new_state["effector.y"] = ey\n    new_state["effector.z"] = ez\n    \n    # Block moves only if effector pushes it (contact)\n    dx = ex - state["block.x"]\n    dy = ey - state["block.y"]\n    dz = ez - state["block.z"]\n    dist = math.sqrt(dx*dx + dy*dy + dz*dz)\n    \n    if dist < 0.02 and action[4] > 0:\n        # Pushed by effector\n        bvx = evx * 0.5\n        bvy = evy * 0.5\n        bvz = evz * 0.5\n    else:\n        # Friction stops block\n        bvx *= (1 - p["friction"] * p["dt"] * 2)\n        bvy *= (1 - p["friction"] * p["dt"] * 2)\n        bvz *= (1 - p["friction"] * p["dt"] * 2)\n    \n    bx = state["block.x"] + bvx * p["dt"]\n    by = state["block.y"] + bvy * p["dt"]\n    bz = max(0.02, state["block.z"] + bvz * p["dt"])\n    new_state["block.x"] = bx\n    new_state["block.y"] = by\n    new_state["block.z"] = bz\n    \n    # Store velocities in meta for next step (simulated hidden state)\n    new_state["_eff_vx"] = evx\n    new_state["_eff_vy"] = evy\n    new_state["_eff_vz"] = evz\n    new_state["_block_vx"] = bvx\n    new_state["_block_vy"] = bvy\n    new_state["_block_vz"] = bvz\n    \n    new_state["goal.x"] = state["goal.x"]\n    new_state["goal.y"] = state["goal.y"]\n    new_state["goal.z"] = state["goal.z"]\n    \n    return new_state'
FITTED_PARAMS = {"block_mass": 1.9873299962087188, "dt": 0.45803567238005455, "eff_mass": 0.6231740404734614, "force_scale": 0.29811020084238204, "friction": 0.12486024726542513}
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
