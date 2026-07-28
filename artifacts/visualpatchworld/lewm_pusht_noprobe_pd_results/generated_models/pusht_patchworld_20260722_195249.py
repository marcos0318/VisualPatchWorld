from patchworld.worldmodel_base import BaseWorldModel
import copy
import json
import math
import re


def _vpw_as_float(value, default=0.0):
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    return default


def _vpw_coerce_belief_numerics(belief):
    if not isinstance(belief, dict):
        return belief
    import copy

    b = copy.deepcopy(belief)
    numeric_obj_keys = ("x", "y", "vx", "vy", "angle", "conf")
    for obj in b.get("objects", {}).values():
        if isinstance(obj, dict):
            for key in numeric_obj_keys:
                if key in obj:
                    obj[key] = _vpw_as_float(obj[key])
    for rel in b.get("relations", {}).values():
        if isinstance(rel, dict) and "value" in rel:
            value = rel["value"]
            if not isinstance(value, str):
                rel["value"] = _vpw_as_float(value)
    step = b.get("step")
    if isinstance(step, dict) and "value" in step and not isinstance(step["value"], str):
        step["value"] = int(_vpw_as_float(step["value"]))
    return b

class PushtPDPhysicsWorldModel(BaseWorldModel):
    """Faithful PushT: PD agent (velocity state) + quasi-static block + swept contact."""

    AGENT_RADIUS = 14.517714364188057
    TEE_POLYGONS = [[[-62.21304595943937, -1.094368758497538], [-62.21304595943937, 28.905631241502462], [-16.213045959439373, 32.90563124150246], [-18.213045959439373, 116.90563124150246], [13.786954040560627, 118.90563124150246], [15.786954040560627, 32.90563124150246], [59.78695404056063, 28.905631241502462], [59.78695404056063, -1.094368758497538]]]
    TEE_COG = [-0.21304595943937343, 40.90563124150246]
    SUBSTEPS = 10
    DT = 0.01
    SCALE = 202.0833563481335
    KP = 175.80876279037972
    KV = 11.29132142738647
    K_N = 0.5559332024420538
    K_T = 0.17987548197620928
    MU = 0.42826359873565556
    K_R = 0.0007157666507291768

    def init_belief(self):
        return {"env": "pusht", "source": "pd_physics", "step": 0,
                "goal": "move block to the target T pose", "objects": {}, "relations": {}, "metadata": {}}

    def init_belief_from_observation(self, obs_text):
        return self.correct_belief(self.init_belief(), obs_text)

    def extract_valid_action_forms(self):
        return {k: [k + "[dx=<float>,dy=<float>,speed=<float>]"] for k in ("move_left","move_right","move_up","move_down","hold")}

    def _rotate(self, vx, vy, a):
        c, s = math.cos(a), math.sin(a)
        return c * vx - s * vy, s * vx + c * vy

    def _contact(self, ax, ay, bx, by, ba):
        best = None
        for poly in self.TEE_POLYGONS:
            n = len(poly); wp = []
            for v in poly:
                rx, ry = self._rotate(v[0], v[1], ba); wp.append((bx + rx, by + ry))
            for i in range(n):
                p1 = wp[i]; p2 = wp[(i + 1) % n]
                ex, ey = p2[0] - p1[0], p2[1] - p1[1]
                den = ex * ex + ey * ey
                t = 0.0 if den <= 1e-12 else ((ax - p1[0]) * ex + (ay - p1[1]) * ey) / den
                t = max(0.0, min(1.0, t))
                qx, qy = p1[0] + t * ex, p1[1] + t * ey
                d = math.hypot(ax - qx, ay - qy)
                if best is None or d < best[0]:
                    best = (d, qx, qy)
        d, qx, qy = best
        nx = (ax - qx) / (d + 1e-9); ny = (ay - qy) / (d + 1e-9)
        return d, nx, ny, qx, qy

    def parse_observation(self, obs_text):
        graph = self.init_belief(); section = ""
        for raw in str(obs_text or "").splitlines():
            line = raw.strip()
            if not line: continue
            if line.startswith("Environment:"): graph["env"] = line.split(":",1)[1].strip()
            elif line.startswith("Source:"): graph["source"] = line.split(":",1)[1].strip()
            elif line.startswith("Step:"):
                try: graph["step"] = int(float(line.split(":",1)[1].strip()))
                except Exception: graph["step"] = 0
            elif line.startswith("Goal:"): graph["goal"] = line.split(":",1)[1].strip()
            elif line.startswith("Objects:"): section = "objects"
            elif line.startswith("Relations:"): section = "relations"
            elif line.startswith("Metadata:"):
                try: graph["metadata"] = json.loads(line.split(":",1)[1].strip())
                except Exception: graph["metadata"] = {}
                section = ""
            elif section == "objects" and line.startswith("-"):
                o = self._parse_object(line)
                if o: graph["objects"][o["id"]] = o
            elif section == "relations" and line.startswith("-"):
                r = self._parse_relation(line)
                if r: graph["relations"][r["key"]] = r
        self._recompute_relations(graph)
        return graph

    def _parse_object(self, line):
        m = re.match(r"^-\s*(\w+):(\w+)\s*(.*)$", line)
        if not m: return None
        o = {"id": m.group(1), "category": m.group(2), "x":0.0,"y":0.0,"vx":0.0,"vy":0.0,"w":0.0,"conf":1.0,"angle":0.0}
        for k, v in re.findall(r"(\w+)=(-?\d+(?:\.\d+)?)", m.group(3)): o[k] = float(v)
        return o

    def _parse_relation(self, line):
        m = re.match(r"^-\s*(\w+)\s+(\w+(?:_\w+)*)\s+(\w+)\s*=\s*(.+)$", line)
        if not m: return None
        val = m.group(4).strip()
        try: val = float(val)
        except Exception: pass
        return {"subject":m.group(1),"predicate":m.group(2),"object":m.group(3),"value":val,"key":f"{m.group(1)}_{m.group(2)}_{m.group(3)}"}

    def correct_belief(self, prior, obs_text):
        observed = self.parse_observation(obs_text)
        merged = copy.deepcopy(prior) if isinstance(prior, dict) else self.init_belief()
        for k in ("env","source","step","goal","metadata"): merged[k] = observed.get(k, merged.get(k))
        merged["objects"] = observed.get("objects", {}); merged["relations"] = observed.get("relations", {})
        return merged

    def predict_belief(self, belief, action):
        b = copy.deepcopy(belief) if isinstance(belief, dict) else self.init_belief()
        objects = b.setdefault("objects", {})
        agent = objects.setdefault("agent", {"id":"agent","category":"circular_pusher","x":0.0,"y":0.0,"vx":0.0,"vy":0.0,"conf":1.0})
        block = objects.setdefault("block", {"id":"block","category":"t_block","x":0.0,"y":0.0,"angle":0.0,"vx":0.0,"vy":0.0,"w":0.0,"conf":1.0})
        adx, ady = self._parse_action(action)
        ax, ay = float(agent.get("x",0.0)), float(agent.get("y",0.0))
        avx, avy = float(agent.get("vx",0.0)), float(agent.get("vy",0.0))
        bx, by, ba = float(block.get("x",0.0)), float(block.get("y",0.0)), float(block.get("angle",0.0))
        tx, ty = ax + self.SCALE * adx, ay + self.SCALE * ady
        for _ in range(self.SUBSTEPS):
            acc_x = self.KP * (tx - ax) - self.KV * avx
            acc_y = self.KP * (ty - ay) - self.KV * avy
            avx += acc_x * self.DT; avy += acc_y * self.DT
            ax += avx * self.DT; ay += avy * self.DT
            d, nx, ny, qx, qy = self._contact(ax, ay, bx, by, ba)
            pen = self.AGENT_RADIUS - d
            if pen > 0.0:
                Fn = self.K_N * pen
                dotv = avx * nx + avy * ny
                tvx, tvy = avx - dotv * nx, avy - dotv * ny
                Ftx, Fty = self.K_T * tvx, self.K_T * tvy
                Ftmag = math.hypot(Ftx, Fty); Ftmax = self.MU * Fn
                if Ftmag > Ftmax and Ftmag > 1e-9:
                    sc = Ftmax / Ftmag; Ftx *= sc; Fty *= sc
                Fx = -Fn * nx + Ftx; Fy = -Fn * ny + Fty
                cogx, cogy = self._rotate(self.TEE_COG[0], self.TEE_COG[1], ba)
                cogx += bx; cogy += by
                lx, ly = qx - cogx, qy - cogy
                bx += Fx; by += Fy
                ba += self.K_R * (lx * Fy - ly * Fx)
        agent["x"], agent["y"] = self._clip(ax), self._clip(ay)
        agent["vx"], agent["vy"] = avx, avy
        block["x"], block["y"] = self._clip(bx), self._clip(by)
        block["angle"] = ba % (2.0 * math.pi)
        block["vx"], block["vy"], block["w"] = 0.0, 0.0, 0.0
        b["step"] = int(b.get("step",0) or 0) + 1
        self._recompute_relations(b)
        return b

    def _parse_action(self, action):
        m = re.match(r"^\w+\[dx=(-?\d+(?:\.\d+)?),dy=(-?\d+(?:\.\d+)?),speed=(-?\d+(?:\.\d+)?)\]$", str(action or "").strip())
        if not m: return 0.0, 0.0
        return float(m.group(1)), float(m.group(2))

    def _clip(self, v):
        return max(0.0, min(512.0, float(v)))

    def _recompute_relations(self, graph):
        objects = graph.get("objects", {}); agent = objects.get("agent"); block = objects.get("block"); rels = {}
        if agent and block:
            dx = float(block.get("x",0.0)) - float(agent.get("x",0.0)); dy = float(block.get("y",0.0)) - float(agent.get("y",0.0))
            dist = math.hypot(dx, dy)
            direction = ("right" if dx>=0 else "left") if abs(dx)>=abs(dy) else ("up" if dy>=0 else "down")
            rels["agent_distance_to_block"] = {"subject":"agent","predicate":"distance_to","object":"block","value":dist,"key":"agent_distance_to_block"}
            rels["agent_near_block"] = {"subject":"agent","predicate":"near","object":"block","value":float(dist<=35.0),"key":"agent_near_block"}
            rels["agent_relative_direction_to_block"] = {"subject":"agent","predicate":"relative_direction_to","object":"block","value":direction,"key":"agent_relative_direction_to_block"}
        graph["relations"] = rels

    def readout_observation(self, belief, action=""):
        b = belief if isinstance(belief, dict) else self.init_belief()
        objects = b.get("objects", {})
        agent = objects.get("agent", {"id":"agent","category":"circular_pusher","x":0.0,"y":0.0,"vx":0.0,"vy":0.0,"conf":1.0})
        block = objects.get("block", {"id":"block","category":"t_block","x":0.0,"y":0.0,"angle":0.0,"vx":0.0,"vy":0.0,"w":0.0,"conf":1.0})
        self._recompute_relations(b)
        lines = ["Environment: pusht", f"Source: {b.get('source','pd_physics')}", f"Step: {int(b.get('step',0) or 0)}",
                 f"Goal: {b.get('goal','move block to the target T pose')}", "Objects:",
                 f"- agent:circular_pusher x={float(agent.get('x',0.0)):.3f} y={float(agent.get('y',0.0)):.3f} vx={float(agent.get('vx',0.0)):.3f} vy={float(agent.get('vy',0.0)):.3f} conf={float(agent.get('conf',1.0)):.3f}",
                 f"- block:t_block x={float(block.get('x',0.0)):.3f} y={float(block.get('y',0.0)):.3f} angle={float(block.get('angle',0.0)):.3f} vx={float(block.get('vx',0.0)):.3f} vy={float(block.get('vy',0.0)):.3f} w={float(block.get('w',0.0)):.3f} conf={float(block.get('conf',1.0)):.3f}",
                 "Relations:"]
        for key in sorted(b.get("relations", {})):
            rel = b["relations"][key]; value = rel.get("value")
            vt = f"{float(value):.3f}" if isinstance(value, (int, float)) else str(value)
            lines.append(f"- {rel.get('subject')} {rel.get('predicate')} {rel.get('object')} = {vt}")
        lines.append("Metadata: {}")
        return "\n".join(lines)
