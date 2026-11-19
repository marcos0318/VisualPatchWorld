
"""LLM-proposed, data-fitted world model: spring_attached_objects

Mechanism (as stated by the proposing LLM): Block attached to effector via spring; both move under forces and constraints.
Fit objective: multi_step; fitted constants are in FITTED_PARAMS.
"""
from patchworld.worldmodel_base import BaseWorldModel
import copy
import json
import math
import re

CANDIDATE_SOURCE = '# HYPOTHESIS: spring_attached_objects\n# MECHANISM: Block attached to effector via spring; both move under forces and constraints.\nPARAMS = {\n    "spring_k": 10.0,\n    "spring_rest_length": 0.0,\n    "effector_mass": 1.0,\n    "block_mass": 0.5,\n    "gravity": -9.8,\n    "ground_level": 0.02,\n    "dt": 0.1,\n    "damping": 0.2\n}\n\ndef step(state, action, p, meta):\n    import math\n    \n    # Estimate velocities from previous states\n    prev_ex = state.get("prev_ex", state["effector.x"])\n    prev_ey = state.get("prev_ey", state["effector.y"])\n    prev_ez = state.get("prev_ez", state["effector.z"])\n    \n    prev_bx = state.get("prev_bx", state["block.x"])\n    prev_by = state.get("prev_by", state["block.y"])\n    prev_bz = state.get("prev_bz", state["block.z"])\n    \n    evx = (state["effector.x"] - prev_ex) / p["dt"]\n    evy = (state["effector.y"] - prev_ey) / p["dt"]\n    evz = (state["effector.z"] - prev_ez) / p["dt"]\n    \n    bvx = (state["block.x"] - prev_bx) / p["dt"]\n    bvy = (state["block.y"] - prev_by) / p["dt"]\n    bvz = (state["block.z"] - prev_bz) / p["dt"]\n    \n    # Forces on effector: action + spring force from block\n    # Action force\n    fe_x = action[0] * 5.0\n    fe_y = action[1] * 5.0\n    fe_z = action[2] * 5.0\n    \n    # Spring force from block\n    dx = state["effector.x"] - state["block.x"]\n    dy = state["effector.y"] - state["block.y"]\n    dz = state["effector.z"] - state["block.z"]\n    dist = math.sqrt(dx*dx + dy*dy + dz*dz)\n    if dist > 1e-6:\n        unit_dx = dx / dist\n        unit_dy = dy / dist\n        unit_dz = dz / dist\n        fs_mag = -p["spring_k"] * (dist - p["spring_rest_length"])\n        fs_x = fs_mag * unit_dx\n        fs_y = fs_mag * unit_dy\n        fs_z = fs_mag * unit_dz\n    else:\n        fs_x, fs_y, fs_z = 0.0, 0.0, 0.0\n    \n    # Total force on effector\n    total_fe_x = fe_x + fs_x\n    total_fe_y = fe_y + fs_y\n    total_fe_z = fe_z + fs_z\n    \n    # Acceleration and update velocity\n    aex = total_fe_x / p["effector_mass"]\n    aey = total_fe_y / p["effector_mass"]\n    aez = total_fe_z / p["effector_mass"]\n    \n    evx = (evx + aex * p["dt"]) * math.exp(-p["damping"] * p["dt"] / p["effector_mass"])\n    evy = (evy + aey * p["dt"]) * math.exp(-p["damping"] * p["dt"] / p["effector_mass"])\n    evz = (evz + aez * p["dt"]) * math.exp(-p["damping"] * p["dt"] / p["effector_mass"])\n    \n    # Forces on block: spring force from effector + gravity\n    fb_x = -fs_x\n    fb_y = -fs_y\n    fb_z = -fs_z + p["block_mass"] * p["gravity"]\n    \n    # Acceleration and update velocity\n    abx = fb_x / p["block_mass"]\n    aby = fb_y / p["block_mass"]\n    abz = fb_z / p["block_mass"]\n    \n    bvx = (bvx + abx * p["dt"]) * math.exp(-p["damping"] * p["dt"] / p["block_mass"])\n    bvy = (bvy + aby * p["dt"]) * math.exp(-p["damping"] * p["dt"] / p["block_mass"])\n    bvz = (bvz + abz * p["dt"]) * math.exp(-p["damping"] * p["dt"] / p["block_mass"])\n    \n    # Integrate positions\n    new_ex = state["effector.x"] + evx * p["dt"]\n    new_ey = state["effector.y"] + evy * p["dt"]\n    new_ez = state["effector.z"] + evz * p["dt"]\n    \n    new_bx = state["block.x"] + bvx * p["dt"]\n    new_by = state["block.y"] + bvy * p["dt"]\n    new_bz = state["block.z"] + bvz * p["dt"]\n    \n    # Ground constraint for block\n    if new_bz < p["ground_level"]:\n        new_bz = p["ground_level"]\n        if bvz < 0:\n            bvz = -bvz * 0.3  # bounce\n    \n    new_state = state.copy()\n    new_state["effector.x"] = new_ex\n    new_state["effector.y"] = new_ey\n    new_state["effector.z"] = new_ez\n    new_state["block.x"] = new_bx\n    new_state["block.y"] = new_by\n    new_state["block.z"] = new_bz\n    new_state["prev_ex"] = state["effector.x"]\n    new_state["prev_ey"] = state["effector.y"]\n    new_state["prev_ez"] = state["effector.z"]\n    new_state["prev_bx"] = state["block.x"]\n    new_state["prev_by"] = state["block.y"]\n    new_state["prev_bz"] = state["block.z"]\n    return new_state'
FITTED_PARAMS = {"block_mass": 0.10306629054642079, "damping": 0.9174193259451565, "dt": 0.13436360155022548, "effector_mass": 0.6759738425092093, "gravity": -6.892633398226523, "ground_level": 0.04432659057541606, "spring_k": 7.836101084596771, "spring_rest_length": -0.023081164039283338}
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
