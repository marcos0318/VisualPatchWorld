from patchworld.worldmodel_base import BaseWorldModel
import re
from typing import Dict, Any, List, Tuple

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

class PushtWorldModel(BaseWorldModel):
    def __init__(self):
        super().__init__()
        self.valid_actions = {
            "move_up": ["move_up[<X>]"],
            "move_down": ["move_down[<X>]"],
            "move_left": ["move_left[<X>]"],
            "move_right": ["move_right[<X>]"],
            "hold": ["hold[<X>]"]
        }
        
    def parse_observation(self, obs_text: str) -> dict:
        """Parse the observation text into structured data."""
        lines = obs_text.strip().split('\n')
        parsed = {
            "environment": "",
            "source": "",
            "step": 0,
            "goal": "",
            "objects": {},
            "relations": [],
            "metadata": {}
        }
        
        current_section = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("Environment:"):
                parsed["environment"] = line.split(":", 1)[1].strip()
            elif line.startswith("Source:"):
                parsed["source"] = line.split(":", 1)[1].strip()
            elif line.startswith("Step:"):
                parsed["step"] = int(line.split(":", 1)[1].strip())
            elif line.startswith("Goal:"):
                parsed["goal"] = line.split(":", 1)[1].strip()
            elif line.startswith("Objects:"):
                current_section = "objects"
            elif line.startswith("Relations:"):
                current_section = "relations"
            elif line.startswith("Metadata:"):
                metadata_str = line.split(":", 1)[1].strip()
                # Parse metadata JSON-like string
                metadata_str = metadata_str.strip('{}')
                metadata_parts = metadata_str.split(',')
                metadata = {}
                for part in metadata_parts:
                    if ':' in part:
                        key, value = part.split(':', 1)
                        key = key.strip().strip('"')
                        value = value.strip().strip('"')
                        if key in ["episode_index", "raw_index"]:
                            metadata[key] = int(value)
                        else:
                            metadata[key] = value
                parsed["metadata"] = metadata
                current_section = None
            elif line.startswith("- ") and current_section == "objects":
                # Parse object line
                obj_desc = line[2:]  # Remove "- "
                # Extract object type and id
                obj_type_id_part, rest = obj_desc.split(' ', 1)
                obj_type, obj_id = obj_type_id_part.split(':')
                obj_data = {"type": obj_type, "id": obj_id}
                
                # Parse properties
                props = rest.split()
                for prop in props:
                    if '=' in prop:
                        key, value = prop.split('=', 1)
                        # Convert numeric values
                        try:
                            if '.' in value:
                                obj_data[key] = float(value)
                            else:
                                obj_data[key] = int(value)
                        except ValueError:
                            obj_data[key] = value
                
                # Use object id as key
                parsed["objects"][obj_id] = obj_data
            elif line.startswith("- ") and current_section == "relations":
                # Parse relation line
                relation_desc = line[2:]  # Remove "- "
                parsed["relations"].append(relation_desc)
                
        return parsed

    def init_belief(self) -> dict:
        """Initialize an empty belief state."""
        return {
            "objects": {},
            "step": 0,
            "goal": ""
        }

    def init_belief_from_observation(self, obs_text: str) -> dict:
        """Initialize belief state from an observation."""
        parsed = self.parse_observation(obs_text)
        belief = self.init_belief()
        belief["objects"] = {obj_id: obj.copy() for obj_id, obj in parsed["objects"].items()}
        belief["step"] = parsed["step"]
        belief["goal"] = parsed["goal"]
        return belief

    def correct_belief(self, belief_prior: dict, obs_text: str) -> dict:
        """Correct the belief state using the observation."""
        # For this simple model, we just replace the belief with the observation
        # In a more sophisticated model, we might do sensor fusion or other corrections
        return self.init_belief_from_observation(obs_text)

    def _apply_action_to_object(self, obj: dict, action: str, params: dict) -> dict:
        """Apply an action to update object state."""
        # Create a copy of the object to avoid modifying the original
        updated_obj = obj.copy()
        
        # Extract velocity multipliers based on action type
        dx, dy = 0.0, 0.0
        if "move_up" in action:
            dy = 1.0
        elif "move_down" in action:
            dy = -1.0
        elif "move_left" in action:
            dx = -1.0
        elif "move_right" in action:
            dx = 1.0
        elif "hold" in action:
            # Holding action has minimal movement
            dx = params.get("dx", 0.0)
            dy = params.get("dy", 0.0)
        
        # Apply speed factor if available
        speed = params.get("speed", 1.0)
        dx *= speed
        dy *= speed
        
        # Update position based on velocity
        if "x" in updated_obj and "y" in updated_obj:
            updated_obj["x"] = updated_obj["x"] + dx * 100  # Scale for visible movement
            updated_obj["y"] = updated_obj["y"] + dy * 100
            
        # Update velocity
        if "vx" in updated_obj and "vy" in updated_obj:
            updated_obj["vx"] = dx * 50  # Scale appropriately
            updated_obj["vy"] = dy * 50
            
        return updated_obj

    def _parse_action(self, action: str) -> Tuple[str, Dict[str, float]]:
        """Parse action string into action type and parameters."""
        action_type = action.split('[')[0]
        params = {}
        if '[' in action and ']' in action:
            param_str = action[action.find('[')+1:action.rfind(']')]
            if param_str and '=' in param_str:
                param_parts = param_str.split(',')
                for part in param_parts:
                    if '=' in part:
                        key, value = part.split('=', 1)
                        try:
                            params[key.strip()] = float(value.strip())
                        except ValueError:
                            params[key.strip()] = value.strip()
        return action_type, params

    def predict_belief(self, belief: dict, action: str) -> dict:
        """Predict the next belief state given an action."""
        # Create a copy of the belief to avoid modifying the original
        next_belief = {key: value.copy() if isinstance(value, dict) else value 
                      for key, value in belief.items()}
        
        # Parse the action
        action_type, params = self._parse_action(action)
        
        # Apply action to all objects, but primarily affect the agent
        for obj_id, obj in belief["objects"].items():
            if obj["type"] == "circular_pusher":  # This is the agent
                next_belief["objects"][obj_id] = self._apply_action_to_object(obj, action_type, params)
            else:  # For other objects, assume minimal change unless pushed
                next_belief["objects"][obj_id] = obj.copy()
                
        # Update step counter
        next_belief["step"] = belief["step"] + 1
        
        return next_belief

    def readout_observation(self, belief: dict, action: str = "") -> str:
        belief = _vpw_coerce_belief_numerics(belief)
        """Convert belief state back to observation text format."""
        obs_lines = []
        obs_lines.append(f"Environment: pusht")
        obs_lines.append(f"Source: predicted")
        obs_lines.append(f"Step: {belief['step']}")
        obs_lines.append(f"Goal: {belief['goal']}")
        obs_lines.append("Objects:")
        
        # Sort objects by id for consistent ordering
        for obj_id in sorted(belief["objects"].keys()):
            obj = belief["objects"][obj_id]
            obj_props = []
            for key in sorted(obj.keys()):
                if key not in ["type", "id"]:
                    value = obj[key]
                    if isinstance(value, float):
                        obj_props.append(f"{key}={value:.3f}")
                    else:
                        obj_props.append(f"{key}={value}")
            
            obj_line = f"- {obj['type']}:{obj_id} " + " ".join(obj_props)
            obs_lines.append(obj_line)
            
        # Add placeholder relations
        obs_lines.append("Relations:")
        obs_lines.append("- agent distance_to block = 100.000")
        obs_lines.append("- agent near block = 0.000")
        obs_lines.append("- agent relative_direction_to block = up")
        
        # Add placeholder metadata
        obs_lines.append('Metadata: {"episode_index":0,"raw_index":0}')
        
        return "\n".join(obs_lines)

    def extract_valid_action_forms(self) -> dict[str, list[str]]:
        """Return valid action forms."""
        return self.valid_actions