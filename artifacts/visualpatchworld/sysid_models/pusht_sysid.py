
"""LLM-proposed, data-fitted world model: pd_pusher_impulse_body_identified_shape

Mechanism (as stated by the proposing LLM): PD-tracking point pusher; frictionless impulse contact with a quasi-static rigid body whose grown shape was identified from data; rotation about a fitted centre of mass; several substeps per control step.
Fit objective: multi_step; fitted constants are in FITTED_PARAMS.
"""
from patchworld.worldmodel_base import BaseWorldModel
import copy
import json
import math
import re

CANDIDATE_SOURCE = '# HYPOTHESIS: pd_pusher_impulse_body_identified_shape\n# MECHANISM: PD-tracking point pusher; frictionless impulse contact with a quasi-static rigid body whose grown shape was identified from data; rotation about a fitted centre of mass; several substeps per control step.\nPARAMS = {"action_scale": 10.0, "kp": 10.0, "kv": 1.0, "dt": 0.01, "substeps": 10,\n          "cog_u": 0.0, "cog_v": 0.0, "inertia": 1000.0, "bias": 0.05, "radius": 5.0, "slop": 0.0, "iters": 4}\n\ndef step(state, action, p, meta):\n    import math\n    shape = meta.get("contact_shape", {})\n    P, B = shape.get("pusher", "agent"), shape.get("body", "block")\n    boxes = shape.get("boxes_u_min_u_max_v_min_v_max", [])\n    ax, ay = state[P + ".x"], state[P + ".y"]\n    vx, vy = state.get(P + ".vx", 0.0), state.get(P + ".vy", 0.0)\n    bx, by, th = state[B + ".x"], state[B + ".y"], state[B + ".angle"]\n    a0 = max(-1.0, min(1.0, action[0])) if len(action) > 0 else 0.0\n    a1 = max(-1.0, min(1.0, action[1])) if len(action) > 1 else 0.0\n    tx, ty = ax + p["action_scale"] * a0, ay + p["action_scale"] * a1\n    n_sub = max(1, int(round(p["substeps"])))\n    dt = p["dt"]\n    inertia = max(abs(p["inertia"]), 1e-6)\n    cu, cv = p["cog_u"], p["cog_v"]\n    r = max(p["radius"], 0.0)\n    for _ in range(n_sub):\n        vx += (p["kp"] * (tx - ax) - p["kv"] * vx) * dt\n        vy += (p["kp"] * (ty - ay) - p["kv"] * vy) * dt\n        c, s = math.cos(th), math.sin(th)\n        dx, dy = ax - bx, ay - by\n        lu, lv = c * dx + s * dy, -s * dx + c * dy\n        contacts = []\n        gx, gy = bx + c * cu - s * cv, by + s * cu + c * cv\n        for g0, g1, h0, h1 in boxes:\n            u0, u1 = min(g0 + r, 0.5 * (g0 + g1)), max(g1 - r, 0.5 * (g0 + g1))\n            v0, v1 = min(h0 + r, 0.5 * (h0 + h1)), max(h1 - r, 0.5 * (h0 + h1))\n            qu, qv = min(max(lu, u0), u1), min(max(lv, v0), v1)\n            eu, ev = lu - qu, lv - qv\n            d = math.hypot(eu, ev)\n            if d > 1e-9:\n                if d >= r:\n                    continue\n                nu, nv, pen = eu / d, ev / d, r - d\n            else:\n                g, nu, nv = min((lu - u0, -1.0, 0.0), (u1 - lu, 1.0, 0.0), (lv - v0, 0.0, -1.0), (v1 - lv, 0.0, 1.0))\n                qu, qv, pen = lu + nu * g, lv + nv * g, r + g\n            wnx, wny = c * nu - s * nv, s * nu + c * nv\n            px, py = bx + c * qu - s * qv, by + s * qu + c * qv\n            contacts.append([px - gx, py - gy, wnx, wny, max(pen - p["slop"], 0.0), 0.0])\n        vbx = vby = w = 0.0\n        for _ in range(max(1, int(round(p["iters"])))):\n            for ct in contacts:\n                rx, ry, nx, ny, pen, acc = ct\n                rn = rx * ny - ry * nx\n                k = 1.0 + rn * rn / inertia\n                vrel = (vx - (vbx - w * ry)) * nx + (vy - (vby + w * rx)) * ny\n                new = max(acc + (p["bias"] * pen / dt - vrel) / k, 0.0)\n                dj = new - acc\n                ct[5] = new\n                vbx -= dj * nx\n                vby -= dj * ny\n                w -= dj * rn / inertia\n        gx, gy = bx + c * cu - s * cv, by + s * cu + c * cv\n        gx, gy, th = gx + vbx * dt, gy + vby * dt, th + w * dt\n        c2, s2 = math.cos(th), math.sin(th)\n        bx, by = gx - (c2 * cu - s2 * cv), gy - (s2 * cu + c2 * cv)\n        ax, ay = ax + vx * dt, ay + vy * dt\n    out = dict(state)\n    out.update({P + ".x": ax, P + ".y": ay, B + ".x": bx, B + ".y": by, B + ".angle": th % (2.0 * math.pi)})\n    if P + ".vx" in out:\n        out[P + ".vx"], out[P + ".vy"] = vx, vy\n    return out\n'
FITTED_PARAMS = {"action_scale": 99.9998235392995, "bias": 0.17706227154970317, "cog_u": 0.003588798928766726, "cog_v": 45.02617324868934, "dt": 0.010016266179841115, "inertia": 3000.844905901005, "iters": 4.0, "kp": 100.00016533918853, "kv": 19.999861784442064, "radius": 17.5, "slop": 0.0, "substeps": 10.0}
ENV = 'pusht'
ACTION_DIM = 2
STATE_KEYS = ["agent.vx", "agent.vy", "agent.x", "agent.y", "block.angle", "block.x", "block.y"]
STATIC_META = {"contact_shape": {"body": "block", "boxes_u_min_u_max_v_min_v_max": [[-75.8, 75.2, -14.9, 44.7], [-29.7, 30.0, 14.9, 134.9]], "identified_from": "state traces (contact analysis)", "note": "boxes are the body shape grown by the pusher radius: the pusher CENTRE touches the body when it reaches a box boundary", "pose_convention": "a body-frame point (u, v) is at world (block.x + u*cos(block.angle) - v*sin(block.angle), block.y + u*sin(block.angle) + v*cos(block.angle))", "pusher": "agent", "type": "union_of_boxes_in_body_frame"}}

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
            nxt = _step(dict(state), self._parse_action(action), dict(FITTED_PARAMS), {**STATIC_META, **(b.get("metadata") or {})})
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
