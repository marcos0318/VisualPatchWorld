
from patchworld.worldmodel_base import BaseWorldModel
import copy
import json
import math
import re


class ReacherJointFitWorldModel(BaseWorldModel):
    PARAMS = {"action_dim": 2, "env": "reacher", "fk_coef": [[1.589898791400459e-06, 0.1200022889401073, -7.953540712577214e-06, 0.11999141689036036, -5.1758088133757465e-06], [-1.955746372421621e-06, 1.0653427093811712e-06, 0.1199933381950109, 1.920984928251689e-06, 0.1200121296296775]], "joint_coef": [[-0.0026323028384033333, 0.5809998937468643, -0.08693271216980795], [-0.0017652098704016642, -0.07575697530779532, 0.8785295442991662]]}
    JOINT_COEF = PARAMS["joint_coef"]
    FK_COEF = PARAMS["fk_coef"]
    ACTION_DIM = 2
    ENV = "reacher"

    def init_belief(self):
        return {"env": self.ENV, "source": "reacher_joint_fit", "step": 0, "goal": "", "objects": {}, "relations": {}, "metadata": {}}

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

    def _parse_action(self, action: str):
        match = re.match(r"^(?P<label>\w+)\[dx=(?P<dx>-?\d+(?:\.\d+)?),dy=(?P<dy>-?\d+(?:\.\d+)?),speed=(?P<speed>-?\d+(?:\.\d+)?)\]$", str(action).strip())
        if match:
            return float(match.group("dx")), float(match.group("dy"))
        return 0.0, 0.0

    def _fk(self, q1: float, q2: float):
        feats = [1.0, math.cos(q1), math.sin(q1), math.cos(q1 + q2), math.sin(q1 + q2)]
        fx = sum(f * c for f, c in zip(feats, self.FK_COEF[0]))
        fy = sum(f * c for f, c in zip(feats, self.FK_COEF[1]))
        return fx, fy

    def _predict_qpos(self, q1: float, q2: float, action: str):
        dx, dy = self._parse_action(action)
        feats = [1.0, dx, dy]
        dq1 = sum(f * c for f, c in zip(feats, self.JOINT_COEF[0]))
        dq2 = sum(f * c for f, c in zip(feats, self.JOINT_COEF[1]))
        return q1 + dq1, q2 + dq2

    def predict_belief(self, belief, action: str):
        merged = copy.deepcopy(belief)
        arm = merged.setdefault("objects", {}).setdefault("arm", {"id": "arm", "category": "joint_state", "x": 0.0, "y": 0.0, "conf": 1.0})
        q1 = float(arm.get("x", 0.0))
        q2 = float(arm.get("y", 0.0))
        nq1, nq2 = self._predict_qpos(q1, q2, action)
        arm["x"] = nq1
        arm["y"] = nq2
        fx, fy = self._fk(nq1, nq2)
        finger = merged.setdefault("objects", {}).setdefault("finger", {"id": "finger", "category": "end_effector", "x": 0.0, "y": 0.0, "conf": 1.0})
        finger["x"] = fx
        finger["y"] = fy
        target = merged.get("objects", {}).get("target")
        if isinstance(target, dict):
            tx = float(target.get("x", 0.0))
            ty = float(target.get("y", 0.0))
            dist = math.hypot(fx - tx, fy - ty)
            merged["relations"] = {
                "finger_distance_to_target": {"subject": "finger", "predicate": "distance_to", "object": "target", "value": dist, "key": "finger_distance_to_target"},
                "finger_near_target": {"subject": "finger", "predicate": "near", "object": "target", "value": float(dist <= 0.05), "key": "finger_near_target"},
            }
        merged["step"] = int(merged.get("step", 0)) + 1
        return merged

    def predict_next(self, belief, action: str):
        return self.predict_belief(belief, action)

    def readout_observation(self, belief, action: str):
        return self._render_graph(belief)

    def _render_graph(self, belief):
        objects = belief.get("objects", {})
        lines = [
            f"Environment: {belief.get('env', self.ENV)}",
            f"Source: {belief.get('source', 'reacher_joint_fit')}",
            f"Step: {int(belief.get('step', 0))}",
            f"Goal: {belief.get('goal', 'match finger to target')}",
            "Objects:",
        ]
        for obj_id in ("arm", "finger", "target"):
            obj = objects.get(obj_id)
            if not isinstance(obj, dict):
                continue
            parts = [f"- {obj_id}:{obj.get('category', obj_id)}", f"x={float(obj.get('x', 0.0)):.3f}", f"y={float(obj.get('y', 0.0)):.3f}", f"conf={float(obj.get('conf', 1.0)):.3f}"]
            lines.append(" ".join(parts))
        lines.append("Relations:")
        for rel in sorted(belief.get("relations", {}).values(), key=lambda item: item.get("key", "")):
            val = rel.get("value")
            if isinstance(val, float):
                val_text = f"{val:.3f}"
            elif isinstance(val, bool):
                val_text = "true" if val else "false"
            else:
                val_text = str(val)
            lines.append(f"- {rel['subject']} {rel['predicate']} {rel['object']} = {val_text}")
        meta = belief.get("metadata", {})
        lines.append(f"Metadata: {json.dumps(meta, sort_keys=True, separators=(',', ':'))}")
        return "\n".join(lines)
