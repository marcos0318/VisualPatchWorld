
from patchworld.worldmodel_base import BaseWorldModel
import copy
import json
import math
import re


class GeneralScaffoldFitWorldModel(BaseWorldModel):
    PARAMS = {'env': 'reacher', 'action_dim': 2, 'state_keys': ['arm.x', 'arm.y', 'finger.x', 'finger.y', 'target.x', 'target.y'], 'coef': [[0.004034944730425469, 0.5806237491814549, -0.08656237841290719, 0.0025683924714832084, -0.0002952798975789962, -0.006422917214313639, -0.04000846257435176, 0.052050268509116135, 0.007924030827619472, 0.004240731048110726, -0.005314679697138589, -0.014561110349824559], [0.009112185518648463, -0.07680298323451454, 0.8781418035256444, 0.0006000870268902745, -0.009993579301490526, -0.014241996883593122, -0.008802890112741242, 0.011790882739428058, 0.0024293500373803543, 0.01189181278798432, -0.014866539008185995, -0.014801016615162576], [0.005556631267945559, 0.0034010204781070503, 0.005302515354816787, 0.00010151523342155579, -0.00041183333127946335, -0.03184871574206997, 0.007015477751587474, -0.00772702972032891, 0.009433221595539812, -0.005993913956649489, 0.004167461424881826, -0.005164352526071639], [-0.000593127120171598, -0.01616242032092393, -0.004986623321015645, 0.00010072207525143039, -0.00011902947568953538, -0.008895186791258801, -0.02821847458254464, 0.010338968306105163, 0.001732995060722577, 0.0005512361376916643, -6.853516132677085e-05, -0.002229585175151644], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0]], 'feature_dim': 12, 'target_dim': 6}
    STATE_KEYS = PARAMS["state_keys"]
    COEF = PARAMS["coef"]
    ACTION_DIM = int(PARAMS["action_dim"])
    ENV = PARAMS["env"]

    def init_belief(self):
        return {"env": self.ENV, "source": "general_scaffold_fit", "step": 0, "goal": "", "objects": {}, "relations": {}, "metadata": {}}

    def init_belief_from_observation(self, obs_text: str):
        return self.correct_belief(self.init_belief(), obs_text)

    def extract_valid_action_forms(self):
        if self.ACTION_DIM == 2:
            return {
                "move_left": ["move_left[dx=<float>,dy=<float>,speed=<float>]"],
                "move_right": ["move_right[dx=<float>,dy=<float>,speed=<float>]"],
                "move_up": ["move_up[dx=<float>,dy=<float>,speed=<float>]"],
                "move_down": ["move_down[dx=<float>,dy=<float>,speed=<float>]"],
                "hold": ["hold[dx=<float>,dy=<float>,speed=<float>]"],
            }
        forms = ["act[" + ",".join(f"a{i}=<float>" for i in range(self.ACTION_DIM)) + ",speed=<float>]"]
        return {"act": forms}

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
                obj = self._parse_object(line)
                if obj:
                    graph["objects"][obj["id"]] = obj
            elif section == "relations" and line.startswith("-"):
                rel = self._parse_relation(line)
                if rel:
                    graph["relations"][rel["key"]] = rel
        return graph

    def _parse_object(self, line):
        match = re.match(r"^-\s*(\w+):(\w+)\s*(.*)$", line)
        if not match:
            return None
        obj = {"id": match.group(1), "category": match.group(2), "x": 0.0, "y": 0.0, "conf": 1.0}
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
        except ValueError:
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

    def _state_vector(self, graph):
        keys = []
        values = []
        objects = graph.get("objects", {})
        for object_id in sorted(objects):
            obj = objects[object_id]
            for field in ("x", "y", "z", "angle"):
                if field in obj:
                    keys.append(f"{object_id}.{field}")
                    values.append(float(obj.get(field, 0.0)))
        return keys, values

    def _apply_state_vector(self, graph, keys, values):
        objects = graph.setdefault("objects", {})
        for key, value in zip(keys, values):
            object_id, field = key.split(".", 1)
            obj = objects.setdefault(object_id, {"id": object_id, "category": object_id, "x": 0.0, "y": 0.0, "conf": 1.0})
            obj[field] = float(value)
        return graph

    def _parse_action(self, action: str):
        if self.ACTION_DIM == 2:
            match = re.match(r"^(?P<label>\w+)\[dx=(?P<dx>-?\d+(?:\.\d+)?),dy=(?P<dy>-?\d+(?:\.\d+)?),speed=(?P<speed>-?\d+(?:\.\d+)?)\]$", str(action).strip())
            if match:
                return float(match.group("dx")), float(match.group("dy"))
            return 0.0, 0.0
        out = [0.0] * self.ACTION_DIM
        for match in re.finditer(r"a(\d+)=(-?\d+(?:\.\d+)?)", str(action)):
            idx = int(match.group(1))
            if 0 <= idx < self.ACTION_DIM:
                out[idx] = float(match.group(2))
        return out

    def _features(self, keys, values, action_values):
        by_key = {k: v for k, v in zip(keys, values)}
        feats = [1.0, *action_values]
        for key in self.STATE_KEYS:
            feats.append(by_key.get(key, 0.0))
        object_ids = sorted({k.split(".")[0] for k in self.STATE_KEYS})
        for i, left in enumerate(object_ids):
            lx = by_key.get(f"{left}.x", 0.0)
            ly = by_key.get(f"{left}.y", 0.0)
            for right in object_ids[i + 1 :]:
                rx = by_key.get(f"{right}.x", 0.0)
                ry = by_key.get(f"{right}.y", 0.0)
                feats.append(math.hypot(lx - rx, ly - ry))
        return feats

    def predict_belief(self, belief, action: str):
        b = copy.deepcopy(belief) if isinstance(belief, dict) else self.init_belief()
        keys, values = self._state_vector(b)
        if not keys:
            keys = list(self.STATE_KEYS)
            values = [0.0] * len(keys)
        if self.ACTION_DIM == 2:
            dx, dy = self._parse_action(action)
            action_values = [dx, dy]
        else:
            parsed = self._parse_action(action)
            action_values = list(parsed) if isinstance(parsed, (list, tuple)) else [float(v) for v in parsed]
        feats = self._features(keys, values, action_values)
        delta = [sum(c * f for c, f in zip(row, feats)) for row in self.COEF]
        next_values = [v + d for v, d in zip(values, delta)]
        b = self._apply_state_vector(b, keys, next_values)
        b["step"] = int(b.get("step", 0)) + 1
        return b

    def readout_observation(self, belief, action: str):
        b = belief if isinstance(belief, dict) else self.init_belief()
        lines = [f"Environment: {b.get('env', self.ENV)}", f"Source: {b.get('source', 'general_scaffold_fit')}", f"Step: {int(b.get('step', 0))}", f"Goal: {b.get('goal', '')}", "Objects:"]
        for object_id in sorted(b.get("objects", {})):
            obj = b["objects"][object_id]
            parts = [f"- {object_id}:{obj.get('category', object_id)}", f"x={float(obj.get('x', 0.0)):.3f}", f"y={float(obj.get('y', 0.0)):.3f}"]
            if "z" in obj:
                parts.append(f"z={float(obj.get('z', 0.0)):.3f}")
            if "angle" in obj:
                parts.append(f"angle={float(obj.get('angle', 0.0)):.3f}")
            parts.append(f"conf={float(obj.get('conf', 1.0)):.3f}")
            lines.append(" ".join(parts))
        lines.append("Relations:")
        for rel in sorted(b.get("relations", {}).values(), key=lambda item: item.get("key", "")):
            lines.append(f"- {rel['subject']} {rel['predicate']} {rel['object']} = {rel.get('value')}")
        lines.append(f"Metadata: {json.dumps(b.get('metadata', {}), sort_keys=True, separators=(',', ':'))}")
        return "\n".join(lines)
