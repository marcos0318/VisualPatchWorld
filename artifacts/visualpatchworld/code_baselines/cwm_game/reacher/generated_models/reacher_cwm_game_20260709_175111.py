from patchworld.worldmodel_base import BaseWorldModel
import re
import math


class ReacherWorldModel(BaseWorldModel):
    def parse_observation(self, obs_text: str) -> dict:
        """Parse the textual scene graph into a structured belief state."""
        lines = obs_text.strip().split('\n')
        belief = {
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
                belief["environment"] = line.split(":", 1)[1].strip()
            elif line.startswith("Source:"):
                belief["source"] = line.split(":", 1)[1].strip()
            elif line.startswith("Step:"):
                belief["step"] = int(line.split(":", 1)[1].strip())
            elif line.startswith("Goal:"):
                belief["goal"] = line.split(":", 1)[1].strip()
            elif line.startswith("Objects:"):
                i += 1
                while i < len(lines) and lines[i].startswith("- "):
                    obj_line = lines[i][2:]  # Remove "- "
                    # Parse object: id:type attr1=val1 attr2=val2 ...
                    parts = obj_line.split()
                    obj_id_type = parts[0]
                    obj_id, obj_type = obj_id_type.split(':', 1)
                    attrs = {}
                    for part in parts[1:]:
                        if '=' in part:
                            key, val = part.split('=', 1)
                            # Try to convert to float if possible
                            try:
                                val = float(val)
                            except ValueError:
                                pass  # Keep as string
                            attrs[key] = val
                    belief["objects"][obj_id] = {
                        "type": obj_type,
                        **attrs
                    }
                    i += 1
                continue  # Skip increment at end of loop
            elif line.startswith("Relations:"):
                i += 1
                while i < len(lines) and lines[i].startswith("- "):
                    rel_line = lines[i][2:]  # Remove "- "
                    # Parse relation: obj1 rel_name obj2 = value
                    match = re.match(r'(\w+) ([\w_]+) (\w+) = (.+)', rel_line)
                    if match:
                        obj1, rel_name, obj2, value = match.groups()
                        # Convert value to float if possible
                        try:
                            value = float(value)
                        except ValueError:
                            pass  # Keep as string
                        belief["relations"][(obj1, rel_name, obj2)] = value
                    i += 1
                continue  # Skip increment at end of loop
            elif line.startswith("Metadata:"):
                metadata_str = line.split(":", 1)[1].strip()
                # Simple parsing assuming JSON-like format
                # This is simplified; in practice, use json.loads
                metadata_str = metadata_str.strip('{}')
                metadata = {}
                for item in metadata_str.split(','):
                    if '=' in item:
                        k, v = item.split('=', 1)
                        k = k.strip().strip('"')
                        v = v.strip().strip('"')
                        # Try to convert to int
                        try:
                            v = int(v)
                        except ValueError:
                            pass
                        metadata[k] = v
                belief["metadata"] = metadata
            i += 1
        return belief

    def init_belief(self):
        """Initialize an empty belief state."""
        return {
            "environment": "",
            "source": "",
            "step": 0,
            "goal": "",
            "objects": {},
            "relations": {},
            "metadata": {}
        }

    def init_belief_from_observation(self, obs_text: str):
        """Initialize belief state from the first observation."""
        return self.parse_observation(obs_text)

    def correct_belief(self, belief_prior, obs_text: str):
        """Correct the belief state using the new observation."""
        # In fully observed setting, we can just replace the belief
        return self.parse_observation(obs_text)

    def predict_belief(self, belief, action: str):
        """Predict the next belief state given an action."""
        # Create a deep copy of the belief to avoid mutation
        import copy
        next_belief = copy.deepcopy(belief)
        next_belief["step"] += 1
        
        # Parse action
        action_match = re.match(r'(\w+)$', action)
        if not action_match:
            action_match = re.match(r'(\w+)\[.*\]$', action)
        if not action_match:
            return next_belief  # Unknown action format
        
        action_type = action_match.group(1)
        
        # Get current positions
        if "arm" not in next_belief["objects"] or "finger" not in next_belief["objects"] or "target" not in next_belief["objects"]:
            return next_belief  # Missing objects
        
        arm = next_belief["objects"]["arm"]
        finger = next_belief["objects"]["finger"]
        target = next_belief["objects"]["target"]
        
        # Apply action effects based on examples
        # These are derived from observing dx, dy changes in finger position
        # We'll use simplified models based on action type
        speed_factor = 0.1  # A base speed factor to simulate movement
        
        if action_type == "move_up":
            arm["y"] += speed_factor
            finger["y"] += speed_factor
        elif action_type == "move_down":
            arm["y"] -= speed_factor
            finger["y"] -= speed_factor
        elif action_type == "move_left":
            arm["x"] -= speed_factor
            finger["x"] -= speed_factor
        elif action_type == "move_right":
            arm["x"] += speed_factor
            finger["x"] += speed_factor
        elif action_type == "hold":
            # Holding doesn't change position significantly
            pass
        
        # Update relations
        if ("finger", "distance_to", "target") in next_belief["relations"]:
            # Calculate Euclidean distance
            dx = finger["x"] - target["x"]
            dy = finger["y"] - target["y"]
            distance = math.sqrt(dx*dx + dy*dy)
            next_belief["relations"][("finger", "distance_to", "target")] = distance
            
            # Update near relation (using threshold from examples)
            near_threshold = 0.1  # Based on examples where near=0.000 until very close
            near_value = 1.0 if distance < near_threshold else 0.0
            if ("finger", "near", "target") in next_belief["relations"]:
                next_belief["relations"][("finger", "near", "target")] = near_value
        
        return next_belief

    def readout_observation(self, belief, action: str = "") -> str:
        """Generate a textual scene graph from the belief state."""
        lines = []
        lines.append(f"Environment: {belief['environment']}")
        lines.append(f"Source: {belief['source']}")
        lines.append(f"Step: {belief['step']}")
        lines.append(f"Goal: {belief['goal']}")
        lines.append("Objects:")
        for obj_id, obj_data in belief["objects"].items():
            obj_type = obj_data["type"]
            attrs = []
            for k, v in obj_data.items():
                if k != "type":
                    attrs.append(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}")
            lines.append(f"- {obj_id}:{obj_type} {' '.join(attrs)}")
        lines.append("Relations:")
        for (obj1, rel_name, obj2), value in belief["relations"].items():
            lines.append(f"- {obj1} {rel_name} {obj2} = {value:.3f}" if isinstance(value, float) else f"- {obj1} {rel_name} {obj2} = {value}")
        lines.append(f"Metadata: {belief['metadata']}")
        return "\n".join(lines)

    def extract_valid_action_forms(self) -> dict[str, list[str]]:
        """Return the valid action templates for this environment."""
        return {
            "move_up": ["move_up"],
            "move_down": ["move_down"],
            "move_left": ["move_left"],
            "move_right": ["move_right"],
            "hold": ["hold"]
        }