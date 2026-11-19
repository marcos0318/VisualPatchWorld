
"""LLM-proposed, data-fitted world model: mass_damper_with_spring_coupling

Mechanism (as stated by the proposing LLM): The effector receives a force proportional to the first three action components; its motion follows a first‑order mass‑damper ODE. When the hand is within a coupling radius, a linear spring pulls the block toward the effector.
Fit objective: multi_step; fitted constants are in FITTED_PARAMS.
"""
from patchworld.worldmodel_base import BaseWorldModel
import copy
import json
import math
import re

CANDIDATE_SOURCE = '# HYPOTHESIS: mass_damper_with_spring_coupling\n# MECHANISM: The effector receives a force proportional to the first three action components; its motion follows a first‑order mass‑damper ODE. When the hand is within a coupling radius, a linear spring pulls the block toward the effector.\nPARAMS = {\n    "dt": 0.05,\n    "m_eff": 0.8,\n    "b_eff": 0.3,\n    "k_spring": 2.0,\n    "couple_thresh": 0.04,\n    "force_scale": 0.1\n}\n\ndef step(state, action, p, meta):\n    def clamp(v, lo, hi):\n        return max(lo, min(hi, v))\n    # ---- effector dynamics (mass‑damper) -------------------------------\n    fx = action[0] * p["force_scale"]\n    fy = action[1] * p["force_scale"]\n    fz = action[2] * p["force_scale"]\n    # approximate velocity as previous displacement (zero‑order)\n    # v_new = (force - b*v)/m * dt  (treated as incremental position)\n    # we neglect stored velocity and use a simple Euler step:\n    dx = clamp((fx - p["b_eff"]*0.0) / p["m_eff"] * p["dt"], -0.2, 0.2)\n    dy = clamp((fy - p["b_eff"]*0.0) / p["m_eff"] * p["dt"], -0.2, 0.2)\n    dz = clamp((fz - p["b_eff"]*0.0) / p["m_eff"] * p["dt"], -0.2, 0.2)\n    eff_x = state["effector.x"] + dx\n    eff_y = state["effector.y"] + dy\n    eff_z = state["effector.z"] + dz\n    # ---- spring coupling to block --------------------------------------\n    blk_x, blk_y, blk_z = state["block.x"], state["block.y"], state["block.z"]\n    dist = ((eff_x - blk_x)**2 + (eff_y - blk_y)**2 + (eff_z - blk_z)**2)**0.5\n    if dist < p["couple_thresh"]:\n        # linear spring force pulling block toward hand\n        fx_s = -p["k_spring"] * (blk_x - eff_x)\n        fy_s = -p["k_spring"] * (blk_y - eff_y)\n        fz_s = -p["k_spring"] * (blk_z - eff_z)\n        # simple Euler update (mass of block assumed 1)\n        blk_x += fx_s * p["dt"]\n        blk_y += fy_s * p["dt"]\n        blk_z += fz_s * p["dt"]\n    # ---- assemble next state -------------------------------------------\n    return {\n        "block.x": blk_x, "block.y": blk_y, "block.z": blk_z,\n        "effector.x": eff_x, "effector.y": eff_y, "effector.z": eff_z,\n        "goal.x": state["goal.x"], "goal.y": state["goal.y"], "goal.z": state["goal.z"]\n    }'
FITTED_PARAMS = {"b_eff": 0.1909355574358352, "couple_thresh": 0.053283415063180364, "dt": 0.592331565209904, "force_scale": 0.06519032062566736, "k_spring": 1.4101206825517407, "m_eff": 0.8512740433666506}
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
