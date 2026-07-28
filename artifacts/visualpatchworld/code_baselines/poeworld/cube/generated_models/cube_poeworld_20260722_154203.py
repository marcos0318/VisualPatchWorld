from patchworld.worldmodel_base import BaseWorldModel
import copy
import json
import math
import re

from patchworld.worldmodel_base import BaseWorldModel
import copy
import json
import re

def _poe_parse_action_fields(action):
    action = str(action or "")
    fields = {}
    for key, value in re.findall(r"(\w+)=([-?\d.]+)", action):
        try:
            fields[key] = float(value)
        except (TypeError, ValueError):
            continue
    return fields

def poe_expert_3(belief, action):
    """Block moves with effector when grasped (a4 positive), based on effector motion."""
    fields = _poe_parse_action_fields(action)
    block = belief["objects"]["block"]
    effector = belief["objects"]["effector"]
    
    # Only move block if a4 is positive (grasping)
    if fields["a4"] > 0:
        # Calculate effector's movement
        dx = fields["a0"] * fields["speed"] * 0.1
        dy = fields["a1"] * fields["speed"] * 0.1
        dz = fields["a2"] * fields["speed"] * 0.1
        
        return {"block": {"x": block["x"] + dx, "y": block["y"] + dy, "z": block["z"] + dz}}
    return {}

class PoeExpert3WorldModel(BaseWorldModel):
    ENV = 'cube'
    ACTION_FORMS = {'act': ['act[a0=<float>,a1=<float>,a2=<float>,a3=<float>,a4=<float>,speed=<float>]']}

    def init_belief(self):
        return {"env": self.ENV, "source": "poeworld_expert", "step": 0, "goal": "", "objects": {}, "relations": {}, "metadata": {}}

    def init_belief_from_observation(self, obs_text: str):
        return self.correct_belief(self.init_belief(), obs_text)

    def parse_observation(self, obs_text: str):
        belief = self.init_belief()
        section = ""
        for raw_line in str(obs_text or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("Environment:"):
                belief["env"] = line.split(":", 1)[1].strip()
            elif line.startswith("Source:"):
                belief["source"] = line.split(":", 1)[1].strip()
            elif line.startswith("Step:"):
                try:
                    belief["step"] = int(float(line.split(":", 1)[1].strip()))
                except ValueError:
                    belief["step"] = 0
            elif line.startswith("Goal:"):
                belief["goal"] = line.split(":", 1)[1].strip()
            elif line.startswith("Objects:"):
                section = "objects"
            elif line.startswith("Relations:"):
                section = "relations"
            elif line.startswith("Metadata:"):
                try:
                    belief["metadata"] = json.loads(line.split(":", 1)[1].strip())
                except Exception:
                    belief["metadata"] = {}
                section = ""
            elif section == "objects" and line.startswith("-"):
                match = re.match(r"^-\s*(\w+):(\w+)\s*(.*)$", line)
                if match:
                    obj = {"id": match.group(1), "category": match.group(2), "x": 0.0, "y": 0.0, "conf": 1.0}
                    for key, value in re.findall(r"(\w+)=(-?\d+(?:\.\d+)?)", match.group(3)):
                        obj[key] = float(value)
                    belief["objects"][obj["id"]] = obj
            elif section == "relations" and line.startswith("-"):
                match = re.match(r"^-\s*(.+?)\s*=\s*(.+)$", line)
                if match:
                    belief["relations"][match.group(1).strip()] = {"key": match.group(1).strip(), "value": match.group(2).strip()}
        return belief

    def correct_belief(self, belief_prior, obs_text: str):
        parsed = self.parse_observation(obs_text)
        merged = copy.deepcopy(belief_prior if isinstance(belief_prior, dict) else self.init_belief())
        for key, value in parsed.items():
            if key in ("objects", "relations", "metadata") and isinstance(value, dict):
                merged.setdefault(key, {}).update(value)
            else:
                merged[key] = value
        return merged

    def extract_valid_action_forms(self):
        return self.ACTION_FORMS

    def predict_belief(self, belief, action: str):
        updated = copy.deepcopy(belief if isinstance(belief, dict) else self.init_belief())
        fn = globals().get("poe_expert_3")
        if callable(fn):
            patch = fn(updated, action)
            if isinstance(patch, dict):
                objs = patch.get("objects", patch)
                if isinstance(objs, dict):
                    for obj_id, fields in objs.items():
                        if obj_id == "objects" or not isinstance(fields, dict):
                            continue
                        updated.setdefault("objects", {}).setdefault(obj_id, {}).update(fields)
        updated["step"] = int(updated.get("step", 0)) + 1
        return updated

    def readout_observation(self, belief, action: str = "") -> str:
        return str(belief)

def _weighted_blend_beliefs(beliefs, weights):
    if not beliefs:
        return {}
    primary = copy.deepcopy(beliefs[0])
    if len(beliefs) == 1:
        return primary
    w = [max(0.0, float(x)) for x in weights]
    total = sum(w) or 1.0
    w = [x / total for x in w]
    objects = primary.setdefault("objects", {})
    numeric_keys = ("x", "y", "z", "angle", "vx", "vy", "conf")
    for obj_id, obj in list(objects.items()):
        if not isinstance(obj, dict):
            continue
        for key in numeric_keys:
            if key not in obj:
                continue
            acc = 0.0
            used = 0.0
            for belief, weight in zip(beliefs, w):
                other = belief.get("objects", {}).get(obj_id, {})
                if isinstance(other, dict) and key in other:
                    try:
                        acc += weight * float(other[key])
                        used += weight
                    except (TypeError, ValueError):
                        pass
            if used > 0:
                obj[key] = acc / used
    return primary


class PoeworldCombinedWorldModel(BaseWorldModel):
    WEIGHTS = [1.0]

    def __init__(self):
        self._experts = [PoeExpert3WorldModel()]
        self._primary = self._experts[0]

    def init_belief(self):
        return self._primary.init_belief()

    def init_belief_from_observation(self, obs_text: str):
        return self._primary.init_belief_from_observation(obs_text)

    def parse_observation(self, obs_text: str):
        return self._primary.parse_observation(obs_text)

    def correct_belief(self, belief_prior, obs_text: str):
        return self._primary.correct_belief(belief_prior, obs_text)

    def extract_valid_action_forms(self):
        return self._primary.extract_valid_action_forms()

    def predict_belief(self, belief, action: str):
        preds = [expert.predict_belief(copy.deepcopy(belief), action) for expert in self._experts]
        return _weighted_blend_beliefs(preds, self.WEIGHTS)

    def readout_observation(self, belief, action: str = "") -> str:
        return self._primary.readout_observation(belief, action)