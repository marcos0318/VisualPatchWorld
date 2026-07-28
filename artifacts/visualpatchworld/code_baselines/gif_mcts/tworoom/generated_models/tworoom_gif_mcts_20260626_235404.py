from patchworld.worldmodel_base import BaseWorldModel
import re
import math


class TworoomWorldModel(BaseWorldModel):
    def __init__(self):
        super().__init__()
        self.door_coords = [112.0, 49.0, 0.0, 0.0, 0.0, 0.0]  # Default door coordinates
        self.room_width = 224.0
        self.room_height = 224.0

    def parse_observation(self, obs_text: str) -> dict:
        """Parse LeWM text scene graph into structured belief."""
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
            elif line.startswith("Objects:"):
                i += 1
                while i < len(lines) and lines[i].startswith("- "):
                    obj_line = lines[i]
                    # Parse object line: "- agent:navigator x=22.395 y=204.454 conf=1.000"
                    match = re.match(r'- ([^:]+):([^ ]+) x=([0-9.-]+) y=([0-9.-]+) conf=([0-9.]+)', obj_line)
                    if match:
                        obj_id, obj_type, x, y, conf = match.groups()
                        obs_dict["objects"][obj_id] = {
                            "type": obj_type,
                            "x": float(x),
                            "y": float(y),
                            "conf": float(conf)
                        }
                    i += 1
                continue  # Skip increment at end of loop
            elif line.startswith("Relations:"):
                i += 1
                while i < len(lines) and lines[i].startswith("- "):
                    rel_line = lines[i]
                    # Parse relation line: "- agent distance_to target = 214.153"
                    match = re.match(r'- ([^ ]+) ([^ ]+) ([^ ]+) = ([0-9.]+)', rel_line)
                    if match:
                        subj, rel_type, obj, value = match.groups()
                        obs_dict["relations"][f"{subj}_{rel_type}_{obj}"] = float(value)
                    i += 1
                continue
            elif line.startswith("Metadata:"):
                metadata_str = line.split(":", 1)[1].strip()
                try:
                    import json
                    obs_dict["metadata"] = json.loads(metadata_str)
                except:
                    obs_dict["metadata"] = {}
            i += 1

        return obs_dict

    def init_belief(self):
        """Initialize belief state with default values."""
        return {
            "environment": "tworoom",
            "source": "unknown",
            "step": 0,
            "goal": "navigate agent to target",
            "objects": {},
            "relations": {},
            "metadata": {
                "door_coords": self.door_coords.copy(),
                "episode_index": 0,
                "next_raw_index": 0,
                "raw_index": 0
            }
        }

    def init_belief_from_observation(self, obs_text: str):
        """Initialize belief state from an observation."""
        belief = self.parse_observation(obs_text)
        # Ensure objects is a dict keyed by object id
        if "objects" in belief and isinstance(belief["objects"], list):
            # Convert list to dict if needed (should not be needed per spec)
            obj_dict = {}
            for obj in belief["objects"]:
                obj_id = obj.get("id", "unknown")
                obj_dict[obj_id] = obj
            belief["objects"] = obj_dict
        return belief

    def correct_belief(self, belief_prior, obs_text: str):
        """Correct belief state using new observation."""
        obs = self.parse_observation(obs_text)
        
        # Update belief with observed values
        belief = belief_prior.copy()
        belief.update({k: v for k, v in obs.items() if k not in ["objects"]})
        
        # Update objects
        for obj_id, obj_data in obs["objects"].items():
            if obj_id in belief["objects"]:
                belief["objects"][obj_id].update(obj_data)
            else:
                belief["objects"][obj_id] = obj_data.copy()
                
        # Update relations
        belief["relations"] = obs["relations"]
        
        # Update metadata
        belief["metadata"].update(obs["metadata"])
        
        return belief

    def predict_belief(self, belief, action: str):
        """Predict next belief state given current belief and action."""
        # Parse action
        action_match = re.match(r'([a-z_]+)$$[^$$]*$$', action)
        if not action_match:
            return belief
            
        action_type = action_match.group(1)
        
        # Extract dx, dy from action string
        dx_match = re.search(r'dx=([0-9.-]+)', action)
        dy_match = re.search(r'dy=([0-9.-]+)', action)
        
        dx = float(dx_match.group(1)) if dx_match else 0.0
        dy = float(dy_match.group(1)) if dy_match else 0.0
        
        # Create new belief
        new_belief = {
            "environment": belief["environment"],
            "source": belief["source"],
            "step": belief["step"] + 1,
            "goal": belief["goal"],
            "objects": {},
            "relations": {},
            "metadata": belief["metadata"].copy()
        }
        
        # Copy and update objects
        for obj_id, obj_data in belief["objects"].items():
            new_obj = obj_data.copy()
            if obj_id == "agent" and action_type in ["move_up", "move_down", "move_left", "move_right", "hold"]:
                # Update agent position
                new_obj["x"] = max(0, min(self.room_width, new_obj["x"] + dx))
                new_obj["y"] = max(0, min(self.room_height, new_obj["y"] + dy))
            new_belief["objects"][obj_id] = new_obj
            
        # Update relations
        if "agent" in new_belief["objects"] and "target" in new_belief["objects"]:
            agent = new_belief["objects"]["agent"]
            target = new_belief["objects"]["target"]
            distance = math.sqrt((agent["x"] - target["x"])**2 + (agent["y"] - target["y"])**2)
            new_belief["relations"]["agent_distance_to_target"] = distance
            new_belief["relations"]["agent_near_target"] = 1.0 if distance < 10.0 else 0.0
            
        return new_belief

    def readout_observation(self, belief, action: str = "") -> str:
        """Generate observation text from belief state."""
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
            parts = rel_key.split("_")
            if len(parts) == 3:
                subj, rel_type, obj = parts
                lines.append(f"- {subj} {rel_type} {obj} = {rel_value:.3f}")
                
        metadata_str = str(belief["metadata"]).replace("'", '"')
        lines.append(f"Metadata: {metadata_str}")
        
        return "\n".join(lines)

    def extract_valid_action_forms(self) -> dict[str, list[str]]:
        """Return valid action templates for this environment."""
        return {
            "move_up": ["move_up[<X>]"],
            "move_down": ["move_down[<X>]"],
            "move_left": ["move_left[<X>]"],
            "move_right": ["move_right[<X>]"],
            "hold": ["hold[<X>]"]
        }