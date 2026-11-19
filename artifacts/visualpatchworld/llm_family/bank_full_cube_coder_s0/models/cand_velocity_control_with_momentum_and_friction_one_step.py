
"""LLM-proposed, data-fitted world model: velocity_control_with_momentum_and_friction

Mechanism (as stated by the proposing LLM): Both effector and block have momentum; they interact through contact with friction and damping.
Fit objective: one_step; fitted constants are in FITTED_PARAMS.
"""
from patchworld.worldmodel_base import BaseWorldModel
import copy
import json
import math
import re

CANDIDATE_SOURCE = '# HYPOTHESIS: velocity_control_with_momentum_and_friction\n# MECHANISM: Both effector and block have momentum; they interact through contact with friction and damping.\nPARAMS = {\n    "effector_accel": 10.0,\n    "block_mass": 0.5,\n    "friction_coeff": 0.1,\n    "air_resistance": 0.2,\n    "contact_spring": 50.0,\n    "dt": 0.1\n}\n\ndef step(state, action, p, meta):\n    import math\n    next_state = state.copy()\n    \n    # Effector acceleration from action\n    ax_e = action[0] * p["effector_accel"]\n    ay_e = action[1] * p["effector_accel"]\n    az_e = action[2] * p["effector_accel"]\n    \n    # Get current velocities\n    vx_e = state.get("effector.vx", 0.0)\n    vy_e = state.get("effector.vy", 0.0)\n    vz_e = state.get("effector.vz", 0.0)\n    \n    vx_b = state.get("block.vx", 0.0)\n    vy_b = state.get("block.vy", 0.0)\n    vz_b = state.get("block.vz", 0.0)\n    \n    # Update effector velocity\n    vx_e += ax_e * p["dt"]\n    vy_e += ay_e * p["dt"]\n    vz_e += az_e * p["dt"]\n    \n    # Apply air resistance\n    vx_e *= (1 - p["air_resistance"] * p["dt"])\n    vy_e *= (1 - p["air_resistance"] * p["dt"])\n    vz_e *= (1 - p["air_resistance"] * p["dt"])\n    \n    # Calculate relative positions\n    ex = state["effector.x"] + vx_e * p["dt"]\n    ey = state["effector.y"] + vy_e * p["dt"]\n    ez = state["effector.z"] + vz_e * p["dt"]\n    \n    bx = state["block.x"] + vx_b * p["dt"]\n    by = state["block.y"] + vy_b * p["dt"]\n    bz = state["block.z"] + vz_b * p["dt"]\n    \n    # Contact interaction\n    dx = ex - bx\n    dy = ey - by\n    dz = ez - bz\n    dist = math.sqrt(dx*dx + dy*dy + dz*dz)\n    \n    if dist < 0.1:  # Interaction range\n        # Direction vectors\n        nx = dx / (dist + 1e-8)\n        ny = dy / (dist + 1e-8)\n        nz = dz / (dist + 1e-8)\n        \n        # Spring force\n        spring_force = (0.1 - dist) * p["contact_spring"]\n        fx = nx * spring_force\n        fy = ny * spring_force\n        fz = nz * spring_force\n        \n        # Apply to block\n        ax_b = fx / p["block_mass"]\n        ay_b = fy / p["block_mass"]\n        az_b = fz / p["block_mass"]\n        \n        vx_b += ax_b * p["dt"]\n        vy_b += ay_b * p["dt"]\n        vz_b += az_b * p["dt"]\n        \n        # Friction opposing relative motion\n        rvx = vx_e - vx_b\n        rvy = vy_e - vy_b\n        rvz = vz_e - vz_b\n        \n        rv_norm = math.sqrt(rvx*rvx + rvy*rvy + rvz*rvz)\n        if rv_norm > 1e-8:\n            friction_mag = p["friction_coeff"] * abs(spring_force)\n            fv_x = -rvx/rv_norm * friction_mag\n            fv_y = -rvy/rv_norm * friction_mag\n            fv_z = -rvz/rv_norm * friction_mag\n            \n            # Apply friction to both objects\n            vx_e += fv_x / p["block_mass"] * p["dt"]\n            vy_e += fv_y / p["block_mass"] * p["dt"]\n            vz_e += fv_z / p["block_mass"] * p["dt"]\n            \n            vx_b -= fv_x / p["block_mass"] * p["dt"]\n            vy_b -= fv_y / p["block_mass"] * p["dt"]\n            vz_b -= fv_z / p["block_mass"] * p["dt"]\n    \n    # Apply air resistance to block\n    vx_b *= (1 - p["air_resistance"] * p["dt"])\n    vy_b *= (1 - p["air_resistance"] * p["dt"])\n    vz_b *= (1 - p["air_resistance"] * p["dt"])\n    \n    # Update positions\n    next_state["effector.x"] = state["effector.x"] + vx_e * p["dt"]\n    next_state["effector.y"] = state["effector.y"] + vy_e * p["dt"]\n    next_state["effector.z"] = state["effector.z"] + vz_e * p["dt"]\n    next_state["effector.vx"] = vx_e\n    next_state["effector.vy"] = vy_e\n    next_state["effector.vz"] = vz_e\n    \n    next_state["block.x"] = state["block.x"] + vx_b * p["dt"]\n    next_state["block.y"] = state["block.y"] + vy_b * p["dt"]\n    next_state["block.z"] = state["block.z"] + vz_b * p["dt"]\n    next_state["block.vx"] = vx_b\n    next_state["block.vy"] = vy_b\n    next_state["block.vz"] = vz_b\n    \n    return next_state'
FITTED_PARAMS = {"air_resistance": -8.197866353728982, "block_mass": 2.654134598176079, "contact_spring": 54.59453432445644, "dt": 0.128995633930927, "effector_accel": 1.5890327072718384, "friction_coeff": 0.6893501355661079}
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
