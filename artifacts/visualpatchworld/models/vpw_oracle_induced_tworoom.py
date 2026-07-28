
from patchworld.worldmodel_base import BaseWorldModel
import copy
import json
import math
import re


class GeneralScaffoldFitWorldModel(BaseWorldModel):
    PARAMS = {'env': 'tworoom', 'action_dim': 2, 'state_keys': ['agent.x', 'agent.y', 'target.x', 'target.y'], 'coef': [[-2.631334724960411, 22.25766534497274, -0.2254026075713889, 0.004898266106959955, 0.0021998759295700355, 0.0058367775053525615, 0.0050514420411166315, 0.004904192161567755], [0.2551327284323348, -0.04534246281795004, 24.93193375687106, 0.0002765144661269834, -0.0032260855381161736, 1.0221119991311269e-05, 8.379547843985846e-06, 7.967384331217101e-05], [0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0], [0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0]], 'feature_dim': 8, 'target_dim': 4}
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
