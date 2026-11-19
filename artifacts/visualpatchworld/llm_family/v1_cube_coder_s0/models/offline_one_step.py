
"""LLM-proposed, data-fitted world model: momentum_coupled

Mechanism (as stated by the proposing LLM): Objects have momentum; actions apply forces that change velocities, with instantaneous position-based coupling when in contact.
Fit objective: one_step; fitted constants are in FITTED_PARAMS.
"""
from patchworld.worldmodel_base import BaseWorldModel
import copy
import json
import math
import re

CANDIDATE_SOURCE = '# HYPOTHESIS: momentum_coupled\n# MECHANISM: Objects have momentum; actions apply forces that change velocities, with instantaneous position-based coupling when in contact.\nPARAMS = {"mass_effector": 1.0, "mass_block": 0.5, "force_scale": 1.0, "friction": 0.8, "dt": 0.05}\n\ndef step(state, action, p, meta):\n    next_state = {}\n    \n    # Effector force integration (F = ma -> dv = F/m * dt)\n    dvx_e = action[0] * p["force_scale"] / p["mass_effector"] * p["dt"]\n    dvy_e = action[1] * p["force_scale"] / p["mass_effector"] * p["dt"]\n    dvz_e = action[2] * p["force_scale"] / p["mass_effector"] * p["dt"]\n    \n    vx_e = state.get("effector.vx", 0) + dvx_e\n    vy_e = state.get("effector.vy", 0) + dvy_e\n    vz_e = state.get("effector.vz", 0) + dvz_e\n\n    # Apply friction to effector\n    vx_e *= p["friction"]\n    vy_e *= p["friction"]\n    vz_e *= p["friction"]\n\n    # Update effector position\n    next_state["effector.x"] = state["effector.x"] + vx_e * p["dt"]\n    next_state["effector.y"] = state["effector.y"] + vy_e * p["dt"]\n    next_state["effector.z"] = state["effector.z"] + vz_e * p["dt"]\n    next_state["effector.vx"] = vx_e\n    next_state["effector.vy"] = vy_e\n    next_state["effector.vz"] = vz_e\n    \n    # Block dynamics\n    dx = state["effector.x"] - state["block.x"]\n    dy = state["effector.y"] - state["block.y"]\n    dz = state["effector.z"] - state["block.z"]\n    dist = (dx**2 + dy**2 + dz**2)**0.5\n\n    # If in contact, transfer some velocity\n    if dist < 0.05:\n        # Assume perfect transfer for simplicity\n        block_vx = vx_e\n        block_vy = vy_e\n        block_vz = vz_e\n    else:\n        block_vx = state.get("block.vx", 0)\n        block_vy = state.get("block.vy", 0)\n        block_vz = state.get("block.vz", 0)\n\n    # Apply friction to block\n    block_vx *= p["friction"]\n    block_vy *= p["friction"]\n    block_vz *= p["friction"]\n\n    next_state["block.x"] = state["block.x"] + block_vx * p["dt"]\n    next_state["block.y"] = state["block.y"] + block_vy * p["dt"]\n    next_state["block.z"] = state["block.z"] + block_vz * p["dt"]\n    next_state["block.vx"] = block_vx\n    next_state["block.vy"] = block_vy\n    next_state["block.vz"] = block_vz\n\n    # Goal remains fixed\n    next_state["goal.x"] = state["goal.x"]\n    next_state["goal.y"] = state["goal.y"]\n    next_state["goal.z"] = state["goal.z"]\n\n    return next_state'
FITTED_PARAMS = {"dt": 0.5839385214147831, "force_scale": 0.2676277938957702, "friction": 0.8636312662101743, "mass_block": 0.9157836257341446, "mass_effector": 0.8044153551217631}
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
