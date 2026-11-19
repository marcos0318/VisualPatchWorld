
"""LLM-proposed, data-fitted world model: ik_task_pd

Mechanism (as stated by the proposing LLM): Task‑space error (finger vs target) is turned into joint velocities via Jacobian pseudo‑inverse and a proportional gain; joints are integrated.
Fit objective: one_step; fitted constants are in FITTED_PARAMS.
"""
from patchworld.worldmodel_base import BaseWorldModel
import copy
import json
import math
import re

CANDIDATE_SOURCE = '# HYPOTHESIS: ik_task_pd\n# MECHANISM: Task‑space error (finger vs target) is turned into joint velocities via Jacobian pseudo‑inverse and a proportional gain; joints are integrated.\nPARAMS = {"k_p": 1.2, "dt": 0.05, "L1": 0.15, "L2": 0.15}\n\ndef step(state, action, p, meta):\n    # ignore raw action, use PD on task error\n    # current fingertip from current joints\n    th0 = state["arm.x"]\n    th1 = state["arm.y"]\n    fx = p["L1"] * math.cos(th0) + p["L2"] * math.cos(th0 + th1)\n    fy = p["L1"] * math.sin(th0) + p["L2"] * math.sin(th0 + th1)\n    # error to target\n    ex = state["target.x"] - fx\n    ey = state["target.y"] - fy\n    # Jacobian\n    j11 = -p["L1"] * math.sin(th0) - p["L2"] * math.sin(th0 + th1)\n    j12 = -p["L2"] * math.sin(th0 + th1)\n    j21 =  p["L1"] * math.cos(th0) + p["L2"] * math.cos(th0 + th1)\n    j22 =  p["L2"] * math.cos(th0 + th1)\n    # determinant\n    det = j11 * j22 - j12 * j21\n    # pseudo‑inverse (2x2)\n    if abs(det) < 1e-8:\n        inv11 = inv12 = inv21 = inv22 = 0.0\n    else:\n        inv11 =  j22 / det\n        inv12 = -j12 / det\n        inv21 = -j21 / det\n        inv22 =  j11 / det\n    # joint velocity = J⁺ * (k_p * error)\n    vx = p["k_p"] * (inv11 * ex + inv12 * ey)\n    vy = p["k_p"] * (inv21 * ex + inv22 * ey)\n    # integrate joints\n    new_th0 = th0 + vx * p["dt"]\n    new_th1 = th1 + vy * p["dt"]\n    # recompute fingertip\n    new_fx = p["L1"] * math.cos(new_th0) + p["L2"] * math.cos(new_th0 + new_th1)\n    new_fy = p["L1"] * math.sin(new_th0) + p["L2"] * math.sin(new_th0 + new_th1)\n    new_state = {\n        "arm.x": new_th0,\n        "arm.y": new_th1,\n        "finger.x": new_fx,\n        "finger.y": new_fy,\n        "target.x": state["target.x"],\n        "target.y": state["target.y"],\n    }\n    return new_state'
FITTED_PARAMS = {"L1": 0.1190277749052117, "L2": 0.11532553209847834, "dt": -0.024283901606779144, "k_p": -9.0269891417834e-05}
ENV = 'reacher'
ACTION_DIM = 2
STATE_KEYS = ["arm.x", "arm.y", "finger.x", "finger.y", "target.x", "target.y"]

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
