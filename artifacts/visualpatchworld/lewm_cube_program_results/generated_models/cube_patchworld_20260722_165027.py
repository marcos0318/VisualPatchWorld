
from patchworld.worldmodel_base import BaseWorldModel
import copy
import json
import math
import re


class CubeContactScaffoldWorldModel(BaseWorldModel):
    PARAMS = {'env': 'cube', 'action_dim': 5, 'effector_coef': [[-0.0049144246145864, 0.10898138255198796, -4.6324535364799895e-05, -0.018458808365415053, 0.0003006922373134427, 0.00027266289061108874, -0.004113088578278332, -0.0012244784977579165, 0.041955956965937825], [0.0022777638554294837, -0.0009033266194689076, 0.1083469569508638, 0.006744826521370297, -0.0011972814277649215, -0.0010045378076596124, -0.00021327503171501573, -0.005784061461370377, -0.013694946535120791], [0.005100202674763761, -0.0013081688436805525, -0.00017737362841454803, 0.0985451212958564, 0.00013300810391261506, -0.001221900277657648, -0.027932438023502017, 0.0005149265899775533, 0.0039023452707887727]], 'contact_threshold': 0.015, 'grip_mode': 'a4_pos', 'require_grip': True, 'use_latch': True, 'contact_fit_score': 0.9588235294117647, 'state_keys': ['block.x', 'block.y', 'block.z', 'effector.x', 'effector.y', 'effector.z', 'goal.x', 'goal.y', 'goal.z'], 'n_transitions': 2040}
    EFF_COEF = PARAMS["effector_coef"]
    CONTACT_THRESHOLD = float(PARAMS["contact_threshold"])
    GRIP_MODE = str(PARAMS.get("grip_mode", "a4_pos"))
    USE_LATCH = bool(PARAMS.get("use_latch", True))
    ACTION_DIM = int(PARAMS["action_dim"])
    ENV = "cube"

    def init_belief(self):
        return {
            "env": self.ENV,
            "source": "cube_contact_scaffold_fit",
            "step": 0,
            "goal": "move block to goal pose",
            "objects": {},
            "relations": {},
            "metadata": {},
            "grasping": False,
        }

    def init_belief_from_observation(self, obs_text: str):
        return self.correct_belief(self.init_belief(), obs_text)

    def extract_valid_action_forms(self):
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
        obj = {"id": match.group(1), "category": match.group(2), "x": 0.0, "y": 0.0, "z": 0.0, "conf": 1.0}
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
        return {
            "subject": match.group(1),
            "predicate": match.group(2),
            "object": match.group(3),
            "value": value,
            "key": f"{match.group(1)}_{match.group(2)}_{match.group(3)}",
        }

    def correct_belief(self, belief_prior, obs_text: str):
        observed = self.parse_observation(obs_text)
        merged = copy.deepcopy(belief_prior) if isinstance(belief_prior, dict) else self.init_belief()
        for key in ("env", "source", "step", "goal", "metadata"):
            merged[key] = observed.get(key, merged.get(key))
        merged["objects"] = observed.get("objects", {})
        merged["relations"] = observed.get("relations", {})
        # Re-sync latch from observed proximity (oracle/tool replan resets grasp).
        objects = merged.get("objects", {})
        ex, ey, ez = self._xyz(objects, "effector")
        bx, by, bz = self._xyz(objects, "block")
        dist = math.sqrt((ex - bx) ** 2 + (ey - by) ** 2 + (ez - bz) ** 2)
        merged["grasping"] = bool(dist <= self.CONTACT_THRESHOLD)
        return merged

    def _parse_action(self, action: str):
        out = [0.0] * self.ACTION_DIM
        for match in re.finditer(r"a(\d+)=(-?\d+(?:\.\d+)?)", str(action)):
            idx = int(match.group(1))
            if 0 <= idx < self.ACTION_DIM:
                out[idx] = float(match.group(2))
        return out

    def _xyz(self, objects, object_id):
        obj = objects.get(object_id, {})
        return float(obj.get("x", 0.0)), float(obj.get("y", 0.0)), float(obj.get("z", 0.0))

    def _grip_active(self, a4: float) -> bool:
        if self.GRIP_MODE == "a4_neg":
            return a4 < 0.0
        if self.GRIP_MODE == "none":
            return True
        return a4 > 0.0  # a4_pos (LeWM Cube expert convention)

    def predict_belief(self, belief, action: str):
        b = copy.deepcopy(belief) if isinstance(belief, dict) else self.init_belief()
        objects = b.setdefault("objects", {})
        for object_id, category in (("effector", "gripper"), ("block", "cube"), ("goal", "goal_pose")):
            objects.setdefault(object_id, {"id": object_id, "category": category, "x": 0.0, "y": 0.0, "z": 0.0, "conf": 1.0})

        ex, ey, ez = self._xyz(objects, "effector")
        bx, by, bz = self._xyz(objects, "block")
        gx, gy, gz = self._xyz(objects, "goal")
        action_values = self._parse_action(action)
        while len(action_values) < 5:
            action_values.append(0.0)

        feats = [1.0, *action_values[:5], ex, ey, ez]
        delta = [sum(c * f for c, f in zip(row, feats)) for row in self.EFF_COEF]
        nex = ex + float(delta[0])
        ney = ey + float(delta[1])
        nez = ez + float(delta[2]) if len(delta) > 2 else ez

        dist_3d = math.sqrt((ex - bx) ** 2 + (ey - by) ** 2 + (ez - bz) ** 2)
        grip = self._grip_active(action_values[4])
        near = dist_3d <= self.CONTACT_THRESHOLD
        grasping = bool(b.get("grasping", False))
        if self.USE_LATCH:
            if near and grip:
                grasping = True
            elif not grip:
                grasping = False
            attached = grasping and grip
        else:
            attached = near and grip

        objects["effector"]["x"] = nex
        objects["effector"]["y"] = ney
        objects["effector"]["z"] = nez
        if attached:
            objects["block"]["x"] = nex
            objects["block"]["y"] = ney
            objects["block"]["z"] = nez
        else:
            objects["block"]["x"] = bx
            objects["block"]["y"] = by
            objects["block"]["z"] = bz
        objects["goal"]["x"] = gx
        objects["goal"]["y"] = gy
        objects["goal"]["z"] = gz
        b["grasping"] = attached

        block_goal = math.sqrt((objects["block"]["x"] - gx) ** 2 + (objects["block"]["y"] - gy) ** 2 + (objects["block"]["z"] - gz) ** 2)
        eff_block = math.sqrt((nex - objects["block"]["x"]) ** 2 + (ney - objects["block"]["y"]) ** 2 + (nez - objects["block"]["z"]) ** 2)
        b["relations"] = {
            "effector_distance_to_block": {
                "subject": "effector",
                "predicate": "distance_to",
                "object": "block",
                "value": eff_block,
                "key": "effector_distance_to_block",
            },
            "block_distance_to_goal": {
                "subject": "block",
                "predicate": "distance_to",
                "object": "goal",
                "value": block_goal,
                "key": "block_distance_to_goal",
            },
            "block_near_goal": {
                "subject": "block",
                "predicate": "near",
                "object": "goal",
                "value": 1.0 if block_goal <= 0.05 else 0.0,
                "key": "block_near_goal",
            },
        }
        b["step"] = int(b.get("step", 0)) + 1
        return b

    def readout_observation(self, belief, action: str):
        b = belief if isinstance(belief, dict) else self.init_belief()
        lines = [
            f"Environment: {b.get('env', self.ENV)}",
            f"Source: {b.get('source', 'cube_contact_scaffold_fit')}",
            f"Step: {int(b.get('step', 0))}",
            f"Goal: {b.get('goal', 'move block to goal pose')}",
            "Objects:",
        ]
        for object_id in sorted(b.get("objects", {})):
            obj = b["objects"][object_id]
            parts = [
                f"- {object_id}:{obj.get('category', object_id)}",
                f"x={float(obj.get('x', 0.0)):.3f}",
                f"y={float(obj.get('y', 0.0)):.3f}",
                f"z={float(obj.get('z', 0.0)):.3f}",
                f"conf={float(obj.get('conf', 1.0)):.3f}",
            ]
            lines.append(" ".join(parts))
        lines.append("Relations:")
        for rel in sorted(b.get("relations", {}).values(), key=lambda item: item.get("key", "")):
            lines.append(f"- {rel['subject']} {rel['predicate']} {rel['object']} = {rel.get('value')}")
        lines.append(f"Metadata: {json.dumps(b.get('metadata', {}), sort_keys=True, separators=(',', ':'))}")
        return "\n".join(lines)
