from patchworld.worldmodel_base import BaseWorldModel
import re
from typing import Dict, Any, List, Tuple

class ReacherWorldModel(BaseWorldModel):
    def parse_observation(self, obs_text: str) -> dict:
        """Parse LeWM text scene graph into structured observation."""
        lines = obs_text.strip().split('\n')
        obs = {
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
                obs["environment"] = line.split(":", 1)[1].strip()
            elif line.startswith("Source:"):
                obs["source"] = line.split(":", 1)[1].strip()
            elif line.startswith("Step:"):
                obs["step"] = int(line.split(":", 1)[1].strip())
            elif line.startswith("Goal:"):
                obs["goal"] = line.split(":", 1)[1].strip()
            elif line.startswith("Objects:"):
                i += 1
                while i < len(lines) and lines[i].strip().startswith("-"):
                    obj_line = lines[i].strip()[2:]  # Remove "- "
                    parts = obj_line.split()
                    obj_id = parts[0].rstrip(':')
                    obj_attrs = {}
                    for part in parts[1:]:
                        if '=' in part:
                            key, value = part.split('=', 1)
                            # Handle conf values that might have multiple parts
                            if key == "conf":
                                obj_attrs[key] = float(value)
                            else:
                                try:
                                    obj_attrs[key] = float(value)
                                except ValueError:
                                    obj_attrs[key] = value
                    obs["objects"][obj_id] = obj_attrs
                    i += 1
                continue
            elif line.startswith("Relations:"):
                i += 1
                while i < len(lines) and lines[i].strip().startswith("-"):
                    rel_line = lines[i].strip()[2:]  # Remove "- "
                    # Parse relation like "finger distance_to target = 0.310"
                    if " = " in rel_line:
                        rel_part, value = rel_line.split(" = ", 1)
                        rel_parts = rel_part.split()
                        if len(rel_parts) >= 3:
                            obj1 = rel_parts[0]
                            relation = "_".join(rel_parts[1:-1])
                            obj2 = rel_parts[-1]
                            rel_key = f"{obj1}_{relation}_{obj2}"
                            try:
                                obs["relations"][rel_key] = float(value)
                            except ValueError:
                                obs["relations"][rel_key] = value
                    i += 1
                continue
            elif line.startswith("Metadata:"):
                metadata_str = line.split(":", 1)[1].strip()
                # Simple parsing for JSON-like string
                metadata_str = metadata_str.strip("{}")
                metadata_parts = metadata_str.split(",")
                for part in metadata_parts:
                    if ":" in part:
                        key, value = part.split(":", 1)
                        key = key.strip().strip('"')
                        value = value.strip().strip('"')
                        try:
                            obs["metadata"][key] = int(value)
                        except ValueError:
                            obs["metadata"][key] = value
            i += 1
            
        return obs

    def init_belief(self):
        """Initialize empty belief state."""
        return {
            "objects": {},
            "relations": {},
            "step": 0
        }

    def init_belief_from_observation(self, obs_text: str):
        """Initialize belief state from observation."""
        obs = self.parse_observation(obs_text)
        belief = {
            "objects": {},
            "relations": {},
            "step": obs["step"]
        }
        
        # Copy objects with their attributes
        for obj_id, attrs in obs["objects"].items():
            belief["objects"][obj_id] = attrs.copy()
            
        # Copy relations
        for rel_key, value in obs["relations"].items():
            belief["relations"][rel_key] = value
            
        return belief

    def correct_belief(self, belief_prior, obs_text: str):
        """Correct belief state with new observation."""
        obs = self.parse_observation(obs_text)
        belief = belief_prior.copy()
        
        # Update objects
        for obj_id, attrs in obs["objects"].items():
            if obj_id not in belief["objects"]:
                belief["objects"][obj_id] = {}
            belief["objects"][obj_id].update(attrs)
            
        # Update relations
        belief["relations"].update(obs["relations"])
        
        # Update step
        belief["step"] = obs["step"]
        
        return belief

    def predict_belief(self, belief, action: str):
        """Predict next belief state given action."""
        belief_next = {
            "objects": {},
            "relations": belief["relations"].copy(),
            "step": belief["step"] + 1
        }
        
        # Copy all objects first
        for obj_id, attrs in belief["objects"].items():
            belief_next["objects"][obj_id] = attrs.copy()
            
        # Parse action
        if "[" in action and "]" in action:
            action_type = action.split("[")[0]
            params_str = action[action.find("[")+1:action.find("]")]
            params = {}
            for param in params_str.split(","):
                if "=" in param:
                    key, value = param.split("=")
                    params[key] = float(value)
                    
            # Apply action effects based on type
            if action_type in ["move_up", "move_down", "move_left", "move_right"]:
                # These actions affect the arm joint state and finger position
                if "arm" in belief_next["objects"]:
                    arm = belief_next["objects"]["arm"]
                    if "x" not in arm:
                        arm["x"] = 0.0
                    if "y" not in arm:
                        arm["y"] = 0.0
                        
                    # Update arm joint state based on direction
                    if action_type == "move_up":
                        arm["y"] += params.get("dy", 0)
                    elif action_type == "move_down":
                        arm["y"] += params.get("dy", 0)
                    elif action_type == "move_left":
                        arm["x"] += params.get("dx", 0)
                    elif action_type == "move_right":
                        arm["x"] += params.get("dx", 0)
                        
                # Finger position might also change
                if "finger" in belief_next["objects"]:
                    finger = belief_next["objects"]["finger"]
                    if "x" not in finger:
                        finger["x"] = 0.0
                    if "y" not in finger:
                        finger["y"] = 0.0
                        
                    # Update finger position
                    if "dx" in params:
                        finger["x"] += params["dx"]
                    if "dy" in params:
                        finger["y"] += params["dy"]
                        
            elif action_type == "hold":
                # Hold action might slightly adjust positions
                if "arm" in belief_next["objects"]:
                    arm = belief_next["objects"]["arm"]
                    if "x" not in arm:
                        arm["x"] = 0.0
                    if "y" not in arm:
                        arm["y"] = 0.0
                    if "dx" in params:
                        arm["x"] += params["dx"] * 0.1  # Small adjustment
                    if "dy" in params:
                        arm["y"] += params["dy"] * 0.1
                        
                if "finger" in belief_next["objects"]:
                    finger = belief_next["objects"]["finger"]
                    if "x" not in finger:
                        finger["x"] = 0.0
                    if "y" not in finger:
                        finger["y"] = 0.0
                    if "dx" in params:
                        finger["x"] += params["dx"] * 0.1
                    if "dy" in params:
                        finger["y"] += params["dy"] * 0.1
                        
        # Update relations based on new positions
        if "finger" in belief_next["objects"] and "target" in belief_next["objects"]:
            finger = belief_next["objects"]["finger"]
            target = belief_next["objects"]["target"]
            if "x" in finger and "y" in finger and "x" in target and "y" in target:
                dx = finger["x"] - target["x"]
                dy = finger["y"] - target["y"]
                distance = (dx**2 + dy**2)**0.5
                
                # Update distance relation
                belief_next["relations"]["finger_distance_to_target"] = distance
                
                # Update near relation (arbitrary threshold)
                belief_next["relations"]["finger_near_target"] = 1.0 if distance < 0.1 else 0.0
                
        return belief_next

    def readout_observation(self, belief, action: str = "") -> str:
        """Generate observation text from belief state."""
        obs_lines = []
        obs_lines.append(f"Environment: reacher")
        obs_lines.append(f"Source: predicted")
        obs_lines.append(f"Step: {belief['step']}")
        obs_lines.append(f"Goal: match finger to target")
        obs_lines.append("Objects:")
        
        # Sort objects for consistent output
        for obj_id in sorted(belief["objects"].keys()):
            attrs = belief["objects"][obj_id]
            attr_strs = []
            for key in sorted(attrs.keys()):
                value = attrs[key]
                if isinstance(value, float):
                    attr_strs.append(f"{key}={value:.3f}")
                else:
                    attr_strs.append(f"{key}={value}")
            obs_lines.append(f"- {obj_id}: {' '.join(attr_strs)}")
            
        obs_lines.append("Relations:")
        for rel_key in sorted(belief["relations"].keys()):
            value = belief["relations"][rel_key]
            if isinstance(value, float):
                obs_lines.append(f"- {rel_key} = {value:.3f}")
            else:
                obs_lines.append(f"- {rel_key} = {value}")
                
        obs_lines.append('Metadata: {"episode_index":0,"next_raw_index":0,"raw_index":0}')
        
        return "\n".join(obs_lines)

    def extract_valid_action_forms(self) -> dict[str, list[str]]:
        """Return valid action forms for this environment."""
        return {
            "move_up": ["move_up[dx=<X>,dy=<Y>,speed=<SPEED>]"],
            "move_down": ["move_down[dx=<X>,dy=<Y>,speed=<SPEED>]"],
            "move_left": ["move_left[dx=<X>,dy=<Y>,speed=<SPEED>]"],
            "move_right": ["move_right[dx=<X>,dy=<Y>,speed=<SPEED>]"],
            "hold": ["hold[dx=<X>,dy=<Y>,speed=<SPEED>]"]
        }