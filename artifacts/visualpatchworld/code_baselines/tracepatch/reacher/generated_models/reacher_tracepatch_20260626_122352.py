from patchworld.worldmodel_base import BaseWorldModel
import re
import math

class ReacherWorldModel(BaseWorldModel):
    def init_belief(self) -> dict:
        return {
            "environment": "reacher",
            "source": "oracle",
            "step": 0,
            "goal": "match finger to target",
            "objects": {
                "arm": {"x": 0.0, "y": 0.0, "conf": 1.0},
                "finger": {"x": 0.0, "y": 0.0, "conf": 1.0},
                "target": {"x": 0.0, "y": 0.0, "conf": 1.0}
            },
            "relations": {
                "finger distance_to target": 0.0,
                "finger near target": 0.0
            },
            "metadata": {}
        }

    def init_belief_from_observation(self, obs_text: str) -> dict:
        belief = self.init_belief()
        return self.correct_belief(belief, obs_text)

    def parse_observation(self, obs_text: str) -> dict:
        lines = obs_text.strip().split('\n')
        obs_dict = {}
        current_section = None
        objects = {}
        relations = {}
        metadata = {}

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("Environment:"):
                obs_dict["environment"] = line.split(":", 1)[1].strip()
            elif line.startswith("Source:"):
                obs_dict["source"] = line.split(":", 1)[1].strip()
            elif line.startswith("Step:"):
                obs_dict["step"] = int(line.split(":", 1)[1].strip())
            elif line.startswith("Goal:"):
                obs_dict["goal"] = line.split(":", 1)[1].strip()
            elif line == "Objects:":
                current_section = "objects"
            elif line == "Relations:":
                current_section = "relations"
            elif line.startswith("Metadata:"):
                current_section = "metadata"
                metadata_str = line.split(":", 1)[1].strip()
                # Parse JSON-like metadata
                metadata_match = re.search(r'\{.*\}', metadata_str)
                if metadata_match:
                    metadata_content = metadata_match.group(0)
                    # Simple parsing for key-value pairs
                    for pair in metadata_content.strip('{}').split(','):
                        if ':' in pair:
                            key, value = pair.split(':', 1)
                            key = key.strip().strip('"')
                            value = value.strip().strip('"')
                            if value.isdigit():
                                metadata[key] = int(value)
                            else:
                                try:
                                    metadata[key] = float(value)
                                except ValueError:
                                    metadata[key] = value
            elif current_section == "objects":
                # Parse object line: "- arm:joint_state x=-1.657 y=-1.858 conf=1.000"
                obj_match = re.match(r'- (\w+):(\w+) (x=)(-?\d+\.\d+) (y=)(-?\d+\.\d+) (conf=)(\d+\.\d+)', line)
                if obj_match:
                    obj_id = obj_match.group(1)
                    obj_type = obj_match.group(2)
                    x = float(obj_match.group(4))
                    y = float(obj_match.group(6))
                    conf = float(obj_match.group(8))
                    objects[obj_id] = {"x": x, "y": y, "conf": conf}
            elif current_section == "relations":
                # Parse relation line: "- finger distance_to target = 0.310"
                rel_match = re.match(r'- (\w+) ([\w_]+) (\w+) = (-?\d+\.\d+)', line)
                if rel_match:
                    obj1 = rel_match.group(1)
                    relation = rel_match.group(2) + " " + rel_match.group(3)
                    obj2 = rel_match.group(3)
                    value = float(rel_match.group(4))
                    relations[f"{obj1} {relation}"] = value

        obs_dict["objects"] = objects
        obs_dict["relations"] = relations
        obs_dict["metadata"] = metadata
        return obs_dict

    def correct_belief(self, belief_prior: dict, obs_text: str) -> dict:
        obs_dict = self.parse_observation(obs_text)
        belief = belief_prior.copy()
        
        # Update belief with observed values
        for key in ["environment", "source", "step", "goal"]:
            if key in obs_dict:
                belief[key] = obs_dict[key]
        
        # Update objects
        for obj_id, obj_data in obs_dict["objects"].items():
            if obj_id not in belief["objects"]:
                belief["objects"][obj_id] = {}
            belief["objects"][obj_id].update(obj_data)
        
        # Update relations
        belief["relations"].update(obs_dict["relations"])
        
        # Update metadata
        belief["metadata"].update(obs_dict["metadata"])
        
        return belief

    def predict_belief(self, belief: dict, action: str) -> dict:
        # Create a copy of the belief to avoid modifying the original
        belief_next = {key: value.copy() if isinstance(value, dict) else value 
                       for key, value in belief.items()}
        
        # Parse action
        action_match = re.match(r'(\w+)\[dx=([+-]?\d+\.\d+),dy=([+-]?\d+\.\d+),speed=([+-]?\d+\.\d+)\]', action)
        if not action_match:
            # If action doesn't match expected format, return unchanged belief
            return belief_next
            
        action_type = action_match.group(1)
        dx = float(action_match.group(2))
        dy = float(action_match.group(3))
        speed = float(action_match.group(4))
        
        # Update step
        belief_next["step"] += 1
        
        # Apply movement to finger (end effector)
        if "finger" in belief_next["objects"]:
            belief_next["objects"]["finger"]["x"] += dx
            belief_next["objects"]["finger"]["y"] += dy
            
        # Apply movement to arm (joint_state) - assuming it moves in the same direction but potentially with different magnitude
        if "arm" in belief_next["objects"]:
            # Simple model: arm moves proportionally to finger movement
            # This is a simplification based on the examples
            arm_dx = dx * 0.5  # Heuristic scaling factor
            arm_dy = dy * 0.5  # Heuristic scaling factor
            belief_next["objects"]["arm"]["x"] += arm_dx
            belief_next["objects"]["arm"]["y"] += arm_dy
            
        # Recalculate relations
        if "finger" in belief_next["objects"] and "target" in belief_next["objects"]:
            finger_x = belief_next["objects"]["finger"]["x"]
            finger_y = belief_next["objects"]["finger"]["y"]
            target_x = belief_next["objects"]["target"]["x"]
            target_y = belief_next["objects"]["target"]["y"]
            
            # Calculate distance
            distance = math.sqrt((finger_x - target_x)**2 + (finger_y - target_y)**2)
            belief_next["relations"]["finger distance_to target"] = distance
            
            # Near relation (using a threshold of 0.1 as in examples)
            belief_next["relations"]["finger near target"] = 1.0 if distance < 0.1 else 0.0
            
        return belief_next

    def readout_observation(self, belief: dict, action: str = "") -> str:
        # Build observation string following the template
        obs_lines = []
        obs_lines.append(f"Environment: {belief['environment']}")
        obs_lines.append(f"Source: {belief['source']}")
        obs_lines.append(f"Step: {belief['step']}")
        obs_lines.append(f"Goal: {belief['goal']}")
        obs_lines.append("Objects:")
        
        # Sort objects for consistent ordering
        for obj_id in sorted(belief["objects"].keys()):
            obj = belief["objects"][obj_id]
            obs_lines.append(f"- {obj_id}:end_effector x={obj['x']:.3f} y={obj['y']:.3f} conf={obj['conf']:.3f}")
            
        obs_lines.append("Relations:")
        # Sort relations for consistent ordering
        for rel_key in sorted(belief["relations"].keys()):
            value = belief["relations"][rel_key]
            obs_lines.append(f"- {rel_key} = {value:.3f}")
            
        obs_lines.append(f"Metadata: {str(belief['metadata'])}")
        
        return "\n".join(obs_lines)

    def extract_valid_action_forms(self) -> dict[str, list[str]]:
        return {
            "move_up": ["move_up[dx=<X>,dy=<Y>,speed=<SPEED>]"],
            "move_down": ["move_down[dx=<X>,dy=<Y>,speed=<SPEED>]"],
            "move_left": ["move_left[dx=<X>,dy=<Y>,speed=<SPEED>]"],
            "move_right": ["move_right[dx=<X>,dy=<Y>,speed=<SPEED>]"],
            "hold": ["hold[dx=<X>,dy=<Y>,speed=<SPEED>]"]
        }

    # Backward-compatible wrappers
    def transition(self, belief: dict, action: str) -> dict:
        return self.predict_belief(belief, action)

    def predict_next(self, state: dict, action: str) -> dict:
        return self.predict_belief(state, action)

    def predict_observation(self, belief: dict, action: str) -> str:
        belief_next_prior = self.predict_belief(belief, action)
        return self.readout_observation(belief_next_prior, action)