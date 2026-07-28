from patchworld.worldmodel_base import BaseWorldModel
import re
import math

class CubeWorldModel(BaseWorldModel):
    def parse_observation(self, obs_text: str) -> dict:
        """Parse LeWM text scene graph into structured observation."""
        obs = {
            "environment": "",
            "source": "",
            "step": 0,
            "goal": "",
            "objects": {},
            "relations": {},
            "metadata": {}
        }
        
        lines = obs_text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("Environment:"):
                obs["environment"] = line.split(":", 1)[1].strip()
            elif line.startswith("Source:"):
                obs["source"] = line.split(":", 1)[1].strip()
            elif line.startswith("Step:"):
                obs["step"] = int(line.split(":", 1)[1].strip())
            elif line.startswith("Goal:"):
                obs["goal"] = line.split(":", 1)[1].strip()
            elif line.startswith("- "):
                if "distance_to" in line or "near" in line:
                    # Relation line
                    parts = line[2:].split(" = ")
                    if len(parts) == 2:
                        relation_parts = parts[0].split()
                        if len(relation_parts) >= 3:
                            obj1 = relation_parts[0]
                            relation_type = relation_parts[1]
                            obj2 = relation_parts[2]
                            value = float(parts[1])
                            obs["relations"][f"{obj1}_{relation_type}_{obj2}"] = value
                else:
                    # Object line
                    parts = line[2:].split(" ", 1)
                    if len(parts) == 2:
                        obj_id_type = parts[0].rstrip(':')
                        if ':' in obj_id_type:
                            obj_id, obj_type = obj_id_type.split(':', 1)
                        else:
                            # Handle case where there's no type specified
                            obj_id = obj_id_type
                            obj_type = "unknown"
                        
                        attrs = parts[1]
                        obj_data = {"type": obj_type}
                        # Extract x, y, z, conf values
                        coord_matches = re.findall(r'([xyz])=([\-0-9.]+)', attrs)
                        for key, value in coord_matches:
                            obj_data[key] = float(value)
                        
                        conf_match = re.search(r'conf=([0-9.]+)', attrs)
                        if conf_match:
                            obj_data["conf"] = float(conf_match.group(1))
                        
                        obs["objects"][obj_id] = obj_data
            elif line.startswith("Metadata:"):
                try:
                    import json
                    metadata_str = line.split(":", 1)[1].strip()
                    obs["metadata"] = json.loads(metadata_str)
                except:
                    pass  # If JSON parsing fails, leave metadata empty
        
        return obs

    def init_belief(self):
        """Initialize an empty belief state."""
        return {
            "objects": {},
            "step": 0
        }

    def init_belief_from_observation(self, obs_text: str):
        """Initialize belief state from initial observation."""
        obs = self.parse_observation(obs_text)
        belief = {
            "objects": {},
            "step": obs["step"]
        }
        
        # Copy object states
        for obj_id, obj_data in obs["objects"].items():
            belief["objects"][obj_id] = obj_data.copy()
            
        return belief

    def correct_belief(self, belief_prior, obs_text: str):
        """Correct belief state using latest observation."""
        obs = self.parse_observation(obs_text)
        belief = belief_prior.copy()
        belief["step"] = obs["step"]
        
        # Update object states from observation
        for obj_id, obj_data in obs["objects"].items():
            if obj_id in belief["objects"]:
                belief["objects"][obj_id].update(obj_data)
            else:
                belief["objects"][obj_id] = obj_data.copy()
                
        return belief

    def predict_belief(self, belief, action: str):
        """Predict next belief state given action."""
        # Parse action parameters
        action_params = {}
        match = re.search(r"act\[([^\]]+)\]", action)
        if match:
            params_str = match.group(1)
            param_pairs = params_str.split(",")
            for pair in param_pairs:
                if "=" in pair:
                    key, value = pair.split("=")
                    action_params[key] = float(value)
        
        # Create new belief state
        new_belief = {
            "objects": {},
            "step": belief["step"] + 1
        }
        
        # Copy existing objects
        for obj_id, obj_data in belief["objects"].items():
            new_belief["objects"][obj_id] = obj_data.copy()
        
        # Apply action effect: move effector
        if "effector" in new_belief["objects"] and len(action_params) >= 3:
            effector = new_belief["objects"]["effector"]
            # Use a0, a1, a2 as x, y, z movement deltas
            if "a0" in action_params:
                effector["x"] = effector.get("x", 0) + action_params["a0"] * 0.01  # Scale down movement
            if "a1" in action_params:
                effector["y"] = effector.get("y", 0) + action_params["a1"] * 0.01
            if "a2" in action_params:
                effector["z"] = effector.get("z", 0) + action_params["a2"] * 0.01
                
            # If effector is close to block, move block with it
            if "block" in new_belief["objects"]:
                block = new_belief["objects"]["block"]
                effector = new_belief["objects"]["effector"]
                distance = math.sqrt(
                    (effector.get("x", 0) - block.get("x", 0))**2 +
                    (effector.get("y", 0) - block.get("y", 0))**2 +
                    (effector.get("z", 0) - block.get("z", 0))**2
                )
                # If close enough (less than threshold), move block with effector
                if distance < 0.05:  # Threshold for gripping
                    block["x"] = effector["x"]
                    block["y"] = effector["y"]
                    block["z"] = effector["z"]
        
        return new_belief

    def readout_observation(self, belief, action: str = "") -> str:
        """Generate text observation from belief state."""
        obs_lines = []
        obs_lines.append("Environment: cube")
        obs_lines.append("Source: predicted")
        obs_lines.append(f"Step: {belief['step']}")
        obs_lines.append("Goal: move block to goal pose")
        obs_lines.append("Objects:")
        
        # Sort objects for consistent output
        for obj_id in sorted(belief["objects"].keys()):
            obj = belief["objects"][obj_id]
            line = f"- {obj_id}:{obj.get('type', 'unknown')}"
            if "x" in obj:
                line += f" x={obj['x']:.3f}"
            if "y" in obj:
                line += f" y={obj['y']:.3f}"
            if "z" in obj:
                line += f" z={obj['z']:.3f}"
            if "conf" in obj:
                line += f" conf={obj['conf']:.3f}"
            else:
                line += " conf=1.000"
            obs_lines.append(line)
        
        # Compute and add relations
        obs_lines.append("Relations:")
        if "block" in belief["objects"] and "goal" in belief["objects"]:
            block = belief["objects"]["block"]
            goal = belief["objects"]["goal"]
            distance = math.sqrt(
                (block.get("x", 0) - goal.get("x", 0))**2 +
                (block.get("y", 0) - goal.get("y", 0))**2 +
                (block.get("z", 0) - goal.get("z", 0))**2
            )
            obs_lines.append(f"- block distance_to goal = {distance:.3f}")
            # Check if block is near goal (arbitrary threshold)
            near_value = 1.0 if distance < 0.1 else 0.0
            obs_lines.append(f"- block near goal = {near_value:.3f}")
            
        if "effector" in belief["objects"] and "block" in belief["objects"]:
            effector = belief["objects"]["effector"]
            block = belief["objects"]["block"]
            distance = math.sqrt(
                (effector.get("x", 0) - block.get("x", 0))**2 +
                (effector.get("y", 0) - block.get("y", 0))**2 +
                (effector.get("z", 0) - block.get("z", 0))**2
            )
            obs_lines.append(f"- effector distance_to block = {distance:.3f}")
        
        obs_lines.append("Metadata: {}")
        
        return "\n".join(obs_lines)

    def extract_valid_action_forms(self) -> dict[str, list[str]]:
        """Return valid action forms for this environment."""
        return {
            "act": ["act[a0=<float>,a1=<float>,a2=<float>,a3=<float>,a4=<float>,speed=<float>]"]
        }