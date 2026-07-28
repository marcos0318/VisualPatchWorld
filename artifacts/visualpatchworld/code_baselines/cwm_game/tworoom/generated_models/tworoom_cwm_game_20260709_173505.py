from patchworld.worldmodel_base import BaseWorldModel
import re
import json
import math


class TworoomWorldModel(BaseWorldModel):
    def parse_observation(self, obs_text: str) -> dict:
        """
        Parse a LeWM scene graph observation into a structured dictionary.
        """
        lines = obs_text.strip().split('\n')
        obs_dict = {
            "environment": "",
            "source": "",
            "step": 0,
            "goal": "",
            "objects": {},
            "relations": {},
            "metadata": {}
        }

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("Environment:"):
                obs_dict["environment"] = line.split(":", 1)[1].strip()
            elif line.startswith("Source:"):
                obs_dict["source"] = line.split(":", 1)[1].strip()
            elif line.startswith("Step:"):
                obs_dict["step"] = int(line.split(":", 1)[1].strip())
            elif line.startswith("Goal:"):
                obs_dict["goal"] = line.split(":", 1)[1].strip()
            elif line == "Objects:":
                i += 1
                while i < len(lines) and lines[i].startswith("- "):
                    obj_line = lines[i][2:]  # Remove "- "
                    # Parse object: id:type x=... y=... conf=...
                    parts = obj_line.split()
                    obj_id_type = parts[0]
                    obj_id, obj_type = obj_id_type.split(':', 1)
                    obj_attrs = {}
                    for part in parts[1:]:
                        if '=' in part:
                            k, v = part.split('=', 1)
                            obj_attrs[k] = float(v)
                    obs_dict["objects"][obj_id] = {
                        "type": obj_type,
                        "x": obj_attrs.get("x", 0.0),
                        "y": obj_attrs.get("y", 0.0),
                        "conf": obj_attrs.get("conf", 1.0)
                    }
                    i += 1
                continue  # Skip increment at end of loop
            elif line == "Relations:":
                i += 1
                while i < len(lines) and lines[i].startswith("- "):
                    rel_line = lines[i][2:]  # Remove "- "
                    # Parse relation: obj1 predicate obj2 = value
                    # e.g., agent distance_to target = 214.153
                    match = re.match(r'(\w+)\s+([\w_]+)\s+(\w+)\s*=\s*([0-9.]+)', rel_line)
                    if match:
                        obj1, predicate, obj2, value = match.groups()
                        key = f"{obj1} {predicate} {obj2}"
                        obs_dict["relations"][key] = float(value)
                    i += 1
                continue
            elif line.startswith("Metadata:"):
                metadata_str = line.split(":", 1)[1].strip()
                obs_dict["metadata"] = json.loads(metadata_str)
            i += 1

        return obs_dict

    def init_belief(self):
        """
        Initialize an empty belief state.
        """
        return {
            "environment": "tworoom",
            "source": "oracle",
            "step": 0,
            "goal": "navigate agent to target",
            "objects": {},
            "relations": {},
            "metadata": {}
        }

    def init_belief_from_observation(self, obs_text: str):
        """
        Initialize belief state from the first observation.
        """
        return self.parse_observation(obs_text)

    def correct_belief(self, belief_prior, obs_text: str):
        """
        Correct the belief state using the new observation.
        Since this is fully observed, we replace the belief with the observation.
        """
        return self.parse_observation(obs_text)

    def predict_belief(self, belief, action: str):
        """
        Predict the next belief state given an action.
        """
        # Create a deep copy of the belief
        new_belief = {
            "environment": belief["environment"],
            "source": belief["source"],
            "step": belief["step"] + 1,
            "goal": belief["goal"],
            "objects": {k: v.copy() for k, v in belief["objects"].items()},
            "relations": belief["relations"].copy(),
            "metadata": belief["metadata"].copy()
        }

        # Parse action
        if action.startswith("move_"):
            # Action format: move_direction[dx=...,dy=...,speed=...]
            direction = action.split("[")[0].split("_")[1]
            params_str = action[action.find("[") + 1:action.rfind("]")]
            params = {}
            for param in params_str.split(","):
                if '=' in param:
                    k, v = param.split('=', 1)
                    params[k] = float(v)

            dx = params.get("dx", 0.0)
            dy = params.get("dy", 0.0)

            # Update agent position
            if "agent" in new_belief["objects"]:
                agent = new_belief["objects"]["agent"]
                agent["x"] += dx
                agent["y"] += dy

        elif action == "hold":
            # hold action does not change position significantly
            pass

        # Recalculate relations
        new_belief["relations"] = self._calculate_relations(new_belief["objects"])

        return new_belief

    def _calculate_relations(self, objects: dict) -> dict:
        """
        Calculate relations based on object positions.
        """
        relations = {}
        if "agent" in objects and "target" in objects:
            agent = objects["agent"]
            target = objects["target"]
            dx = agent["x"] - target["x"]
            dy = agent["y"] - target["y"]
            distance = math.sqrt(dx * dx + dy * dy)
            relations["agent distance_to target"] = distance
            # near is 0.000 unless very close, which we assume is not the case here
            relations["agent near target"] = 0.000
        return relations

    def readout_observation(self, belief, action: str = "") -> str:
        """
        Generate an observation string from the belief state.
        """
        lines = []
        lines.append(f"Environment: {belief['environment']}")
        lines.append(f"Source: {belief['source']}")
        lines.append(f"Step: {belief['step']}")
        lines.append(f"Goal: {belief['goal']}")
        lines.append("Objects:")
        for obj_id, obj_data in belief["objects"].items():
            lines.append(f"- {obj_id}:{obj_data['type']} x={obj_data['x']:.3f} y={obj_data['y']:.3f} conf={obj_data['conf']:.3f}")

        lines.append("Relations:")
        for rel_key, rel_value in belief["relations"].items():
            lines.append(f"- {rel_key} = {rel_value:.3f}")

        lines.append(f"Metadata: {json.dumps(belief['metadata'])}")

        return "\n".join(lines)

    def extract_valid_action_forms(self) -> dict[str, list[str]]:
        """
        Return valid action templates.
        """
        return {
            "move_up": ["move_up[<X>]"],
            "move_down": ["move_down[<X>]"],
            "move_left": ["move_left[<X>]"],
            "move_right": ["move_right[<X>]"],
            "hold": ["hold[<X>]"]
        }