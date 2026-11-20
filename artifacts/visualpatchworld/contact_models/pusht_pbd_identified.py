
"""LLM-proposed, data-fitted world model: quasi_static_position_projection

Mechanism (as stated by the proposing LLM): identified PD pusher; each substep, if the pusher centre is inside the grown shape, translate and rotate the body (about its CoM, split by effective mass) so the pusher lies on the boundary.
Fit objective: multi_step; fitted constants are in FITTED_PARAMS.
"""
from patchworld.worldmodel_base import BaseWorldModel
import copy
import json
import math
import re

CANDIDATE_SOURCE = '# HYPOTHESIS: quasi_static_position_projection\n# MECHANISM: identified PD pusher; each substep, if the pusher centre is inside the grown shape, translate and rotate the body (about its CoM, split by effective mass) so the pusher lies on the boundary.\nPARAMS = {"com_u": 0.0, "com_v": 0.0, "inertia_over_mass": 1000.0, "iters": 2}\n\ndef step(state, action, p, meta):\n    import math\n    pd = meta["pusher_free_dynamics"]["constants"]; n_sub = meta["pusher_free_dynamics"]["substeps"]\n    boxes = meta["contact_shape"]["boxes_u_min_u_max_v_min_v_max"]\n    ax, ay, vx, vy = state["agent.x"], state["agent.y"], state["agent.vx"], state["agent.vy"]\n    bx, by, th = state["block.x"], state["block.y"], state["block.angle"]\n    a0, a1 = max(-1.0, min(1.0, action[0])), max(-1.0, min(1.0, action[1]))\n    tx, ty = ax + pd["action_scale"] * a0, ay + pd["action_scale"] * a1\n    dt = pd["dt"]; J = max(abs(p["inertia_over_mass"]), 1e-6); cu, cv = p["com_u"], p["com_v"]\n    for _ in range(n_sub):\n        vx += (pd["kp"] * (tx - ax) - pd["kv"] * vx) * dt\n        vy += (pd["kp"] * (ty - ay) - pd["kv"] * vy) * dt\n        ax += vx * dt; ay += vy * dt\n        for _ in range(max(1, int(round(p["iters"])))):\n            c, s = math.cos(th), math.sin(th)\n            dx, dy = ax - bx, ay - by\n            u, v = c * dx + s * dy, -s * dx + c * dy\n            best = None\n            for u0, u1, v0, v1 in boxes:\n                if u0 < u < u1 and v0 < v < v1:\n                    cand = min((u - u0, -1.0, 0.0), (u1 - u, 1.0, 0.0), (v - v0, 0.0, -1.0), (v1 - v, 0.0, 1.0))\n                    if best is None or cand[0] < best[0]:\n                        best = cand\n            if best is None:\n                break\n            g, nu, nv = best\n            nx, ny = c * nu - s * nv, s * nu + c * nv\n            gx, gy = bx + c * cu - s * cv, by + s * cu + c * cv\n            rx, ry = ax - gx, ay - gy\n            rn = rx * ny - ry * nx\n            lam = g / (1.0 + rn * rn / J)\n            gx, gy, th = gx - lam * nx, gy - lam * ny, th - lam * rn / J\n            c, s = math.cos(th), math.sin(th)\n            bx, by = gx - (c * cu - s * cv), gy - (s * cu + c * cv)\n    return {"agent.x": ax, "agent.y": ay, "agent.vx": vx, "agent.vy": vy, "block.x": bx, "block.y": by, "block.angle": th}\n'
FITTED_PARAMS = {"com_u": -0.24810093989467935, "com_v": 43.14992047262977, "inertia_over_mass": 3123.088312698482, "iters": 2.2024079563102226}
ENV = 'pusht'
ACTION_DIM = 2
STATE_KEYS = ["agent.vx", "agent.vy", "agent.x", "agent.y", "block.angle", "block.x", "block.y"]
STATIC_META = {"contact_shape": {"body": "block", "boxes_u_min_u_max_v_min_v_max": [[-75.8, 75.2, -14.9, 44.7], [-29.7, 30.0, 14.9, 134.9]], "identified_from": "state traces (contact analysis)", "note": "boxes are the body shape grown by the pusher radius: the pusher CENTRE touches the body when it reaches a box boundary", "pose_convention": "a body-frame point (u, v) is at world (block.x + u*cos(block.angle) - v*sin(block.angle), block.y + u*sin(block.angle) + v*cos(block.angle))", "pusher": "agent", "type": "union_of_boxes_in_body_frame"}, "estimated_control_period": {"identified_from": "x' - x = T * mean velocity, least squares", "seconds_per_transition": 0.1095}, "pusher_free_dynamics": {"alternatives": {"direct_velocity_command": 0.073, "first_order_velocity_lag": 0.016, "pd_position_servo": 0.0}, "constants": {"action_scale": 99.99943, "dt": 0.01, "kp": 100.00062, "kv": 19.99998}, "contact_transition_nrmse": 0.0, "heldout_free_nrmse": 0.0, "identified_from": "4000 transitions where block did not move", "model": "pd_position_servo", "note": "nRMSE is per field over pusher position and velocity, normalized by their spread. If the error on contact transitions is close to the free-space error, contact does not slow the pusher (it is effectively kinematic / much heavier than the body).", "pusher": "agent", "substeps": 10, "update": "target = pos + action_scale*action (held for the transition); each substep: vel += (kp*(target - pos) - kv*vel)*dt; pos += vel*dt"}}

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
