
"""LLM-proposed, data-fitted world model: momentum_based_dynamics

Mechanism (as stated by the proposing LLM): Both effector and block have momentum; interactions via impulses or continuous forces.
Fit objective: one_step; fitted constants are in FITTED_PARAMS.
"""
from patchworld.worldmodel_base import BaseWorldModel
import copy
import json
import math
import re

CANDIDATE_SOURCE = '# HYPOTHESIS: momentum_based_dynamics\n# MECHANISM: Both effector and block have momentum; interactions via impulses or continuous forces.\nPARAMS = {\n    "effector_mass": 1.0,\n    "block_mass": 0.5,\n    "dt": 0.1,\n    "force_scale": 5.0,\n    "restitution": 0.3,\n    "damping": 0.1\n}\n\ndef step(state, action, p, meta):\n    import math\n    \n    # Estimate previous velocities from positions (assuming constant velocity)\n    prev_ex = state.get("prev_ex", state["effector.x"])\n    prev_ey = state.get("prev_ey", state["effector.y"])\n    prev_ez = state.get("prev_ez", state["effector.z"])\n    \n    prev_bx = state.get("prev_bx", state["block.x"])\n    prev_by = state.get("prev_by", state["block.y"])\n    prev_bz = state.get("prev_bz", state["block.z"])\n    \n    # Current velocities\n    evx = (state["effector.x"] - prev_ex) / p["dt"]\n    evy = (state["effector.y"] - prev_ey) / p["dt"]\n    evz = (state["effector.z"] - prev_ez) / p["dt"]\n    \n    bvx = (state["block.x"] - prev_bx) / p["dt"]\n    bvy = (state["block.y"] - prev_by) / p["dt"]\n    bvz = (state["block.z"] - prev_bz) / p["dt"]\n    \n    # Action forces on effector\n    fx = action[0] * p["force_scale"]\n    fy = action[1] * p["force_scale"]\n    fz = action[2] * p["force_scale"]\n    \n    # Update effector velocity with force and damping\n    evx = (evx + fx * p["dt"] / p["effector_mass"]) * math.exp(-p["damping"] * p["dt"] / p["effector_mass"])\n    evy = (evy + fy * p["dt"] / p["effector_mass"]) * math.exp(-p["damping"] * p["dt"] / p["effector_mass"])\n    evz = (evz + fz * p["dt"] / p["effector_mass"]) * math.exp(-p["damping"] * p["dt"] / p["effector_mass"])\n    \n    # Simple collision detection (overlap in x,y,z within sum of radii)\n    dx = state["effector.x"] - state["block.x"]\n    dy = state["effector.y"] - state["block.y"]\n    dz = state["effector.z"] - state["block.z"]\n    dist_sq = dx*dx + dy*dy + dz*dz\n    \n    # Assume radius of 0.02 for both objects\n    if dist_sq < (0.04)**2:\n        # Inelastic collision impulse exchange\n        total_mass = p["effector_mass"] + p["block_mass"]\n        # Relative velocity\n        rvx = evx - bvx\n        rvy = evy - bvy\n        rvz = evz - bvz\n        \n        # Impulse factor\n        impulse_factor = (1 + p["restitution"]) * p["block_mass"] / total_mass\n        \n        # Update velocities after collision\n        evx -= impulse_factor * rvx\n        evy -= impulse_factor * rvy\n        evz -= impulse_factor * rvz\n        \n        bvx += impulse_factor * (p["effector_mass"]/p["block_mass"]) * rvx\n        bvy += impulse_factor * (p["effector_mass"]/p["block_mass"]) * rvy\n        bvz += impulse_factor * (p["effector_mass"]/p["block_mass"]) * rvz\n    \n    # Integrate positions\n    new_ex = state["effector.x"] + evx * p["dt"]\n    new_ey = state["effector.y"] + evy * p["dt"]\n    new_ez = state["effector.z"] + evz * p["dt"]\n    \n    new_bx = state["block.x"] + bvx * p["dt"]\n    new_by = state["block.y"] + bvy * p["dt"]\n    new_bz = state["block.z"] + bvz * p["dt"]\n    \n    # Floor constraint\n    if new_bz < 0.02:\n        new_bz = 0.02\n        if bvz < 0:\n            bvz = -bvz * p["restitution"]\n    \n    new_state = state.copy()\n    new_state["effector.x"] = new_ex\n    new_state["effector.y"] = new_ey\n    new_state["effector.z"] = new_ez\n    new_state["block.x"] = new_bx\n    new_state["block.y"] = new_by\n    new_state["block.z"] = new_bz\n    new_state["prev_ex"] = state["effector.x"]\n    new_state["prev_ey"] = state["effector.y"]\n    new_state["prev_ez"] = state["effector.z"]\n    new_state["prev_bx"] = state["block.x"]\n    new_state["prev_by"] = state["block.y"]\n    new_state["prev_bz"] = state["block.z"]\n    return new_state'
FITTED_PARAMS = {"block_mass": 0.21585181802931225, "damping": -0.14639754908606983, "dt": 0.22780162232403178, "effector_mass": 1.6358560705865282, "force_scale": 3.186954223059543, "restitution": -0.05243382494641353}
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
