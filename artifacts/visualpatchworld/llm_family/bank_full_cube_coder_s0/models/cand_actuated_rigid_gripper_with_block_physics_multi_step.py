
"""LLM-proposed, data-fitted world model: actuated_rigid_gripper_with_block_physics

Mechanism (as stated by the proposing LLM): Effector is a rigid body with velocity control; block behaves as independent object with gravity and contact.
Fit objective: multi_step; fitted constants are in FITTED_PARAMS.
"""
from patchworld.worldmodel_base import BaseWorldModel
import copy
import json
import math
import re

CANDIDATE_SOURCE = '# HYPOTHESIS: actuated_rigid_gripper_with_block_physics\n# MECHANISM: Effector is a rigid body with velocity control; block behaves as independent object with gravity and contact.\nPARAMS = {\n    "effector_speed": 0.5,\n    "block_mass": 0.2,\n    "gravity": -9.8,\n    "ground_level": 0.02,\n    "dt": 0.1,\n    "contact_stiffness": 100.0\n}\n\ndef step(state, action, p, meta):\n    import math\n    next_state = state.copy()\n    \n    # Effector velocity clamped to maximum speed\n    ax = action[0]\n    ay = action[1]\n    az = action[2]\n    norm = math.sqrt(ax*ax + ay*ay + az*az)\n    if norm > 1e-8:\n        scale = min(norm, 1.0) / norm\n        vx_e = ax * scale * p["effector_speed"]\n        vy_e = ay * scale * p["effector_speed"]\n        vz_e = az * scale * p["effector_speed"]\n    else:\n        vx_e, vy_e, vz_e = 0.0, 0.0, 0.0\n    \n    # Effector position update\n    new_ex = state["effector.x"] + vx_e * p["dt"]\n    new_ey = state["effector.y"] + vy_e * p["dt"]\n    new_ez = state["effector.z"] + vz_e * p["dt"]\n    \n    # Block physics simulation\n    bx = state["block.x"]\n    by = state["block.y"]\n    bz = state["block.z"]\n    vx_b = state.get("block.vx", 0.0)\n    vy_b = state.get("block.vy", 0.0)\n    vz_b = state.get("block.vz", 0.0)\n    \n    # Apply gravity\n    vz_b += p["gravity"] * p["dt"]\n    \n    # Contact force calculation (simplified)\n    dx = new_ex - bx\n    dy = new_ey - by\n    dz = new_ez - bz\n    dist = math.sqrt(dx*dx + dy*dy + dz*dz)\n    \n    if dist < 0.1:  # Interaction distance\n        # Normalized direction vector\n        nx = dx / (dist + 1e-8)\n        ny = dy / (dist + 1e-8)\n        nz = dz / (dist + 1e-8)\n        \n        # Spring-like repulsion/attraction\n        force_mag = (0.1 - dist) * p["contact_stiffness"]\n        fx = nx * force_mag\n        fy = ny * force_mag\n        fz = nz * force_mag\n        \n        # Apply force to block\n        ax_b = fx / p["block_mass"]\n        ay_b = fy / p["block_mass"]\n        az_b = fz / p["block_mass"]\n        \n        vx_b += ax_b * p["dt"]\n        vy_b += ay_b * p["dt"]\n        vz_b += az_b * p["dt"]\n    \n    # Ground collision for block\n    new_bz = bz + vz_b * p["dt"]\n    if new_bz < p["ground_level"]:\n        new_bz = p["ground_level"]\n        vz_b = 0.0  # Stop vertical motion\n    \n    new_bx = bx + vx_b * p["dt"]\n    new_by = by + vy_b * p["dt"]\n    \n    # Update states\n    next_state["effector.x"] = new_ex\n    next_state["effector.y"] = new_ey\n    next_state["effector.z"] = new_ez\n    \n    next_state["block.x"] = new_bx\n    next_state["block.y"] = new_by\n    next_state["block.z"] = new_bz\n    next_state["block.vx"] = vx_b\n    next_state["block.vy"] = vy_b\n    next_state["block.vz"] = vz_b\n    \n    return next_state'
FITTED_PARAMS = {"block_mass": 3.774292528465554, "contact_stiffness": 120.11431724222378, "dt": 0.15523478208160282, "effector_speed": 0.2951259677455646, "gravity": -0.447111723545255, "ground_level": 0.03874311787962311}
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
