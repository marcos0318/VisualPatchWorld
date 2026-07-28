from patchworld.worldmodel_base import BaseWorldModel
import json
import math
import re
import copy


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

class PushtScaffoldFitWorldModel(BaseWorldModel):
    COEF = [[-0.08277034726100539, 208.13381153784576, -2.194780662661986, 3.7761337880433055, 1.6468026453718319, 0.4793221532897739, -3.7110586130889183, 0.9532064924188733, -0.15542349184472953, 0.4105649557718768, -5.357861093219174, 0.47388445960996417, -0.7678697653337004, -0.14423929063298777, -9.229128342244481, 0.6483597000195871, -3.4726439754081238, -1.5477629754550106], [-1.7787935194405498, 0.6113067534042114, 204.26186723262092, 1.404300873900446, 0.5288871690881043, 2.3033459710707787, 0.702570387308958, -1.0242058029189083, -0.6468460601420234, 0.23043099518311222, 0.17368335656262046, -3.327551844942981, 0.8824950352591253, 0.035210290483752876, 1.6200910458220252, -7.185607001475864, -0.1386584495647436, -0.7839849320147165], [-8.504452806144789, 9.325586916470817, 6.789162482068118, 4.749510857136454, -3.3466572156513563, 5.06260382470349, -2.958582114018257, 1.858022324727074, -0.7397155751453206, 1.035658560583972, 0.3880750765918226, -3.2045815241632662, 25.38896909866783, 1.5143523945174202, 52.245424510569684, -11.718626330755473, 4.216409666806014, -16.20552720405897], [17.425836330103774, -0.227933739593295, 14.598527762143949, 4.444233906066893, -6.257254179659409, -12.61653894603473, -9.906477515188556, -6.8483917346390895, 1.9598809039215317, -4.084101978685479, -3.64922322543169, 5.768147297174349, 1.505078586471211, 0.9115672924316707, -6.336471439431224, 25.319748041628138, -2.7158501826667445, -6.820458154545584], [-0.214662701489625, 0.045159577470175696, 0.0119548241470011, 0.0018774551138864158, 0.043008856130434356, 0.06423254872264997, 0.10990626961611293, 0.05910715787601589, 0.08282250369148797, -0.12003597805497314, 0.06689741200712956, -0.005125393366416253, 0.3500776740398154, 0.02241358732159508, 0.05594569794698398, 0.24587468331254558, 0.1351141131715723, -0.08625100606559452]]

    def init_belief(self):
        return {
            "env": "pusht",
            "source": "scaffold_fit",
            "step": 0,
            "goal": "move block to the target T pose",
            "objects": {},
            "relations": {},
            "metadata": {},
        }

    def init_belief_from_observation(self, obs_text: str):
        return self.correct_belief(self.init_belief(), obs_text)

    def extract_valid_action_forms(self):
        return {
            "move_left": ["move_left[dx=<float>,dy=<float>,speed=<float>]"],
            "move_right": ["move_right[dx=<float>,dy=<float>,speed=<float>]"],
            "move_up": ["move_up[dx=<float>,dy=<float>,speed=<float>]"],
            "move_down": ["move_down[dx=<float>,dy=<float>,speed=<float>]"],
            "hold": ["hold[dx=<float>,dy=<float>,speed=<float>]"],
        }

    def parse_observation(self, obs_text: str):
        graph = self.init_belief()
        section = ""
        for raw_line in str(obs_text or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("Environment:"):
                graph["env"] = line.split(":", 1)[1].strip()
                section = ""
            elif line.startswith("Source:"):
                graph["source"] = line.split(":", 1)[1].strip()
                section = ""
            elif line.startswith("Step:"):
                try:
                    graph["step"] = int(float(line.split(":", 1)[1].strip()))
                except Exception:
                    graph["step"] = 0
                section = ""
            elif line.startswith("Goal:"):
                graph["goal"] = line.split(":", 1)[1].strip()
                section = ""
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
                obj = self._parse_object(line)
                if obj:
                    graph["objects"][obj["id"]] = obj
            elif section == "relations" and line.startswith("-"):
                rel = self._parse_relation(line)
                if rel:
                    graph["relations"][rel["key"]] = rel
        self._recompute_relations(graph)
        return graph

    def _parse_object(self, line):
        match = re.match(r"^-\s*(\w+):(\w+)\s*(.*)$", line)
        if not match:
            return None
        obj = {"id": match.group(1), "category": match.group(2), "x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "conf": 1.0}
        for key, value in re.findall(r"(\w+)=(-?\d+(?:\.\d+)?)", match.group(3)):
            obj[key] = float(value)
        return obj

    def _parse_relation(self, line):
        match = re.match(r"^-\s*(\w+)\s+(\w+(?:_\w+)*)\s+(\w+)\s*=\s*(.+)$", line)
        if not match:
            return None
        value = match.group(4).strip()
        try:
            value = float(value)
        except Exception:
            pass
        return {"subject": match.group(1), "predicate": match.group(2), "object": match.group(3), "value": value, "key": f"{match.group(1)}_{match.group(2)}_{match.group(3)}"}

    def correct_belief(self, belief_prior, obs_text: str):
        observed = self.parse_observation(obs_text)
        merged = copy.deepcopy(belief_prior) if isinstance(belief_prior, dict) else self.init_belief()
        for key in ("env", "source", "step", "goal", "metadata"):
            merged[key] = observed.get(key, merged.get(key))
        merged["objects"] = observed.get("objects", {})
        merged["relations"] = observed.get("relations", {})
        return merged

    def predict_belief(self, belief, action: str):
        b = copy.deepcopy(belief) if isinstance(belief, dict) else self.init_belief()
        objects = b.setdefault("objects", {})
        agent = objects.setdefault("agent", {"id": "agent", "category": "circular_pusher", "x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "conf": 1.0})
        block = objects.setdefault("block", {"id": "block", "category": "t_block", "x": 0.0, "y": 0.0, "angle": 0.0, "conf": 1.0})
        dx, dy = self._parse_action(action)
        feats = self._features(agent, block, dx, dy)
        delta = [sum(c * f for c, f in zip(row, feats)) for row in self.COEF]
        agent["x"] = self._clip(float(agent.get("x", 0.0)) + delta[0])
        agent["y"] = self._clip(float(agent.get("y", 0.0)) + delta[1])
        agent["vx"] = delta[0]
        agent["vy"] = delta[1]
        block["x"] = self._clip(float(block.get("x", 0.0)) + delta[2])
        block["y"] = self._clip(float(block.get("y", 0.0)) + delta[3])
        block["angle"] = (float(block.get("angle", 0.0)) + delta[4]) % (2.0 * math.pi)
        b["step"] = int(b.get("step", 0) or 0) + 1
        self._recompute_relations(b)
        return b

    def _parse_action(self, action):
        match = re.match(r"^\w+\[dx=(-?\d+(?:\.\d+)?),dy=(-?\d+(?:\.\d+)?),speed=(-?\d+(?:\.\d+)?)\]$", str(action or "").strip())
        if not match:
            return 0.0, 0.0
        return float(match.group(1)), float(match.group(2))

    def _features(self, agent, block, dx, dy):
        sx = float(agent.get("x", 0.0))
        sy = float(agent.get("y", 0.0))
        bx = float(block.get("x", 0.0))
        by = float(block.get("y", 0.0))
        ba = float(block.get("angle", 0.0))
        speed = math.hypot(dx, dy)
        rel_x = bx - sx
        rel_y = by - sy
        dist = math.hypot(rel_x, rel_y)
        near = 1.0 if dist <= 70.0 else 0.0
        inv_dist = 1.0 / max(dist, 1.0)
        return [
            1.0,
            dx,
            dy,
            speed,
            sx / 512.0,
            sy / 512.0,
            bx / 512.0,
            by / 512.0,
            math.cos(ba),
            math.sin(ba),
            rel_x / 512.0,
            rel_y / 512.0,
            dist / 512.0,
            near,
            dx * near,
            dy * near,
            dx * rel_x * inv_dist,
            dy * rel_y * inv_dist,
        ]

    def _clip(self, value):
        return max(0.0, min(512.0, float(value)))

    def _recompute_relations(self, graph):
        objects = graph.get("objects", {})
        agent = objects.get("agent")
        block = objects.get("block")
        rels = {}
        if agent and block:
            dx = float(block.get("x", 0.0)) - float(agent.get("x", 0.0))
            dy = float(block.get("y", 0.0)) - float(agent.get("y", 0.0))
            dist = math.hypot(dx, dy)
            direction = ("right" if dx >= 0 else "left") if abs(dx) >= abs(dy) else ("up" if dy >= 0 else "down")
            rels["agent_distance_to_block"] = {"subject": "agent", "predicate": "distance_to", "object": "block", "value": dist, "key": "agent_distance_to_block"}
            rels["agent_near_block"] = {"subject": "agent", "predicate": "near", "object": "block", "value": float(dist <= 35.0), "key": "agent_near_block"}
            rels["agent_relative_direction_to_block"] = {"subject": "agent", "predicate": "relative_direction_to", "object": "block", "value": direction, "key": "agent_relative_direction_to_block"}
        graph["relations"] = rels

    def readout_observation(self, belief, action: str = ""):
        b = belief if isinstance(belief, dict) else self.init_belief()
        objects = b.get("objects", {})
        agent = objects.get("agent", {"id": "agent", "category": "circular_pusher", "x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "conf": 1.0})
        block = objects.get("block", {"id": "block", "category": "t_block", "x": 0.0, "y": 0.0, "angle": 0.0, "conf": 1.0})
        self._recompute_relations(b)
        lines = [
            "Environment: pusht",
            f"Source: {b.get('source', 'scaffold_fit')}",
            f"Step: {int(b.get('step', 0) or 0)}",
            f"Goal: {b.get('goal', 'move block to the target T pose')}",
            "Objects:",
            f"- agent:circular_pusher x={float(agent.get('x', 0.0)):.3f} y={float(agent.get('y', 0.0)):.3f} vx={float(agent.get('vx', 0.0)):.3f} vy={float(agent.get('vy', 0.0)):.3f} conf={float(agent.get('conf', 1.0)):.3f}",
            f"- block:t_block x={float(block.get('x', 0.0)):.3f} y={float(block.get('y', 0.0)):.3f} angle={float(block.get('angle', 0.0)):.3f} conf={float(block.get('conf', 1.0)):.3f}",
            "Relations:",
        ]
        for key in sorted(b.get("relations", {})):
            rel = b["relations"][key]
            value = rel.get("value")
            value_text = f"{float(value):.3f}" if isinstance(value, (int, float)) else str(value)
            lines.append(f"- {rel.get('subject')} {rel.get('predicate')} {rel.get('object')} = {value_text}")
        lines.append("Metadata: {}")
        return "\n".join(lines)
