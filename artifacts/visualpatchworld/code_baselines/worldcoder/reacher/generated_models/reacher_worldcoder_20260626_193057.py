from patchworld.worldmodel_base import BaseWorldModel
import re
import math

class ReacherWorldModel(BaseWorldModel):
    def parse_observation(self, obs_text: str) -> dict:
        """Parse the textual observation into a structured dictionary."""
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
        
        current_section = None
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
            elif line.startswith("Objects:"):
                current_section = "objects"
            elif line.startswith("Relations:"):
                current_section = "relations"
            elif line.startswith("Metadata:"):
                metadata_str = line.split(":", 1)[1].strip()
                # Parse JSON-like metadata
                metadata_str = metadata_str.replace('true', 'True').replace('false', 'False')
                obs_dict["metadata"] = eval(metadata_str) if metadata_str else {}
            elif line.startswith("- ") and current_section == "objects":
                # Parse object line like "- finger:end_effector x=0.085 y=-0.199 conf=1.000"
                parts = line[2:].split()
                obj_id_type = parts[0]  # e.g., "finger:end_effector"
                obj_id = obj_id_type.split(":")[0]  # e.g., "finger"
                properties = {}
                for part in parts[1:]:
                    if '=' in part:
                        key, value = part.split('=', 1)
                        if key in ['x', 'y']:
                            properties[key] = float(value)
                        elif key == 'conf':
                            properties[key] = float(value)
                        else:
                            properties[key] = value
                obs_dict["objects"][obj_id] = properties
            elif line.startswith("- ") and current_section == "relations":
                # Parse relation line like "- finger distance_to target = 0.390"
                # We'll store as "finger-target-distance_to": 0.390
                parts = line[2:].split()
                if len(parts) >= 5 and parts[1] == "distance_to" and parts[2] == "target":
                    obj1 = parts[0]
                    relation_type = "distance_to"
                    obj2 = "target"
                    value = float(parts[4])
                    key = f"{obj1}-{obj2}-{relation_type}"
                    obs_dict["relations"][key] = value
                elif len(parts) >= 5 and parts[1] == "near" and parts[2] == "target":
                    obj1 = parts[0]
                    relation_type = "near"
                    obj2 = "target"
                    value = float(parts[4])
                    key = f"{obj1}-{obj2}-{relation_type}"
                    obs_dict["relations"][key] = value
                    
        return obs_dict

    def init_belief(self):
        """Initialize an empty belief state."""
        return {
            "objects": {},
            "step": 0
        }

    def init_belief_from_observation(self, obs_text: str):
        """Initialize belief state from an observation."""
        obs = self.parse_observation(obs_text)
        belief = {
            "objects": obs["objects"],
            "step": obs["step"]
        }
        return belief

    def correct_belief(self, belief_prior, obs_text: str):
        """Correct the belief state using the observation."""
        obs = self.parse_observation(obs_text)
        # For simplicity, we'll just replace the belief with the observation
        # In a more sophisticated model, we might do some kind of filtering
        belief_corrected = {
            "objects": obs["objects"],
            "step": obs["step"]
        }
        return belief_corrected

    def predict_belief(self, belief, action: str):
        """Predict the next belief state given an action."""
        # Parse action like "move_right[dx=0.3590,dy=-0.3079,speed=0.4729]"
        action_type = action.split('[')[0]
        params_str = action[action.find('[')+1:action.rfind(']')]
        params = {}
        if params_str:
            for param in params_str.split(','):
                if '=' in param:
                    key, value = param.split('=', 1)
                    params[key] = float(value)
        
        # Copy current belief
        new_belief = {
            "objects": {k: v.copy() for k, v in belief["objects"].items()},
            "step": belief["step"] + 1
        }
        
        # Apply action effects
        if "finger" in new_belief["objects"] and "arm" in new_belief["objects"]:
            finger = new_belief["objects"]["finger"]
            arm = new_belief["objects"]["arm"]
            
            # All actions seem to move the finger, so we'll apply dx, dy to finger
            if "dx" in params and "dy" in params:
                finger["x"] += params["dx"]
                finger["y"] += params["dy"]
                
                # Update arm position based on finger movement
                # This is a simplification; in reality, the arm joint state determines finger position
                arm["x"] += params["dx"] * 0.1  # Scale factor for joint state change
                arm["y"] += params["dy"] * 0.1
                
        return new_belief

    def readout_observation(self, belief, action: str = "") -> str:
        """Generate a textual observation from the belief state."""
        # Reconstruct the observation text from belief
        obs_lines = []
        obs_lines.append("Environment: reacher")
        obs_lines.append("Source: predicted")
        obs_lines.append(f"Step: {belief['step']}")
        obs_lines.append("Goal: match finger to target")
        
        obs_lines.append("Objects:")
        for obj_id, properties in belief["objects"].items():
            # Reconstruct object type from id
            obj_type = "end_effector" if obj_id == "finger" else "goal" if obj_id == "target" else "joint_state"
            if obj_id == "arm":
                obj_type = "joint_state"
            line = f"- {obj_id}:{obj_type}"
            for key in ["x", "y", "conf"]:
                if key in properties:
                    if key in ["x", "y"]:
                        line += f" {key}={properties[key]:.3f}"
                    else:
                        line += f" {key}={properties[key]:.3f}"
            obs_lines.append(line)
            
        # Calculate relations
        obs_lines.append("Relations:")
        if "finger" in belief["objects"] and "target" in belief["objects"]:
            finger = belief["objects"]["finger"]
            target = belief["objects"]["target"]
            dx = finger["x"] - target["x"]
            dy = finger["y"] - target["y"]
            distance = math.sqrt(dx*dx + dy*dy)
            obs_lines.append(f"- finger distance_to target = {distance:.3f}")
            # Simplification: near is always 0.000
            obs_lines.append("- finger near target = 0.000")
            
        # Add empty metadata line
        obs_lines.append("Metadata: {}")
        
        return "\n".join(obs_lines)

    def extract_valid_action_forms(self) -> dict[str, list[str]]:
        """Return the valid action forms for this environment."""
        return {
            "hold": ["hold[<X>]"],
            "move_down": ["move_down[<X>]"],
            "move_left": ["move_left[<X>]"],
            "move_right": ["move_right[<X>]"],
            "move_up": ["move_up[<X>]"]
        }