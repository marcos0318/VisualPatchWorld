from patchworld.worldmodel_base import BaseWorldModel
import re
from typing import Dict, Any, List
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
        
        # Parse environment
        env_match = re.search(r'Environment:\s*(\w+)', obs_text)
        if env_match:
            obs["environment"] = env_match.group(1)
            
        # Parse source
        source_match = re.search(r'Source:\s*(\w+)', obs_text)
        if source_match:
            obs["source"] = source_match.group(1)
            
        # Parse step
        step_match = re.search(r'Step:\s*(\d+)', obs_text)
        if step_match:
            obs["step"] = int(step_match.group(1))
            
        # Parse goal
        goal_match = re.search(r'Goal:\s*(.+)', obs_text)
        if goal_match:
            obs["goal"] = goal_match.group(1).strip()
            
        # Parse objects
        objects_section = re.search(r'Objects:\n(.*?)(?=Relations:|Metadata:|$)', obs_text, re.DOTALL)
        if objects_section:
            object_lines = objects_section.group(1).strip().split('\n')
            for line in object_lines:
                if line.startswith('- '):
                    # Parse object line: - id:type x=... y=... z=... conf=...
                    obj_match = re.match(r'-\s+([^:]+):(\w+)\s+(.*)', line)
                    if obj_match:
                        obj_id, obj_type, attrs_text = obj_match.groups()
                        attrs = {}
                        for attr in attrs_text.split():
                            if '=' in attr:
                                key, val = attr.split('=', 1)
                                try:
                                    attrs[key] = float(val)
                                except ValueError:
                                    attrs[key] = val
                        obs["objects"][obj_id] = {
                            "type": obj_type,
                            **attrs
                        }
        
        # Parse relations
        relations_section = re.search(r'Relations:\n(.*?)(?=Metadata:|$)', obs_text, re.DOTALL)
        if relations_section:
            relation_lines = relations_section.group(1).strip().split('\n')
            for line in relation_lines:
                if line.startswith('- '):
                    # Parse relation line: - obj1 relation obj2 = value
                    rel_match = re.match(r'-\s+(\w+)\s+(.+?)\s+(\w+)\s*=\s*(.+)', line)
                    if rel_match:
                        obj1, relation, obj2, value = rel_match.groups()
                        key = f"{obj1} {relation} {obj2}"
                        try:
                            obs["relations"][key] = float(value)
                        except ValueError:
                            obs["relations"][key] = value
        
        # Parse metadata
        metadata_match = re.search(r'Metadata:\s*(\{.*\})', obs_text)
        if metadata_match:
            try:
                import json
                obs["metadata"] = json.loads(metadata_match.group(1))
            except json.JSONDecodeError:
                obs["metadata"] = {}
                
        return obs

    def init_belief(self) -> dict:
        """Initialize empty belief state."""
        return {
            "objects": {},
            "step": 0,
            "goal": ""
        }

    def init_belief_from_observation(self, obs_text: str) -> dict:
        """Initialize belief from first observation."""
        obs = self.parse_observation(obs_text)
        belief = self.init_belief()
        belief["objects"] = {k: v.copy() for k, v in obs["objects"].items()}
        belief["step"] = obs["step"]
        belief["goal"] = obs["goal"]
        return belief

    def correct_belief(self, belief_prior: dict, obs_text: str) -> dict:
        """Correct belief with new observation."""
        obs = self.parse_observation(obs_text)
        belief = belief_prior.copy()
        
        # Update objects
        for obj_id, obj_data in obs["objects"].items():
            if obj_id in belief["objects"]:
                # Update existing object
                belief["objects"][obj_id].update(obj_data)
            else:
                # Add new object
                belief["objects"][obj_id] = obj_data.copy()
                
        # Remove objects not in observation (if any)
        obs_obj_ids = set(obs["objects"].keys())
        belief_obj_ids = set(belief["objects"].keys())
        for to_remove in belief_obj_ids - obs_obj_ids:
            del belief["objects"][to_remove]
            
        belief["step"] = obs["step"]
        belief["goal"] = obs["goal"]
        return belief

    def predict_belief(self, belief: dict, action: str) -> dict:
        """Predict next belief state given action."""
        next_belief = {"objects": {}, "step": belief["step"] + 1, "goal": belief["goal"]}
        
        # Parse action parameters
        action_match = re.search(r'act\[a0=([^,]+),a1=([^,]+),a2=([^,]+),a3=([^,]+),a4=([^,]+),speed=([^\]]+)\]', action)
        if not action_match:
            # If action doesn't match expected format, just copy current state
            next_belief["objects"] = {k: v.copy() for k, v in belief["objects"].items()}
            return next_belief
            
        a0, a1, a2, a3, a4, speed = [float(x) for x in action_match.groups()]
        
        # Get current positions
        block = belief["objects"].get("block", {}).copy()
        effector = belief["objects"].get("effector", {}).copy()
        
        if not block or not effector:
            # If we don't have required objects, just copy current state
            next_belief["objects"] = {k: v.copy() for k, v in belief["objects"].items()}
            return next_belief
        
        # Calculate effector movement based on action
        # a0, a1, a2 control x, y, z movement of effector
        # a3, a4 are additional parameters (possibly for gripper control)
        effector_speed = min(speed, 1.0) * 0.1  # Scale speed appropriately
        
        new_effector_x = effector["x"] + a0 * effector_speed
        new_effector_y = effector["y"] + a1 * effector_speed
        new_effector_z = effector["z"] + a2 * effector_speed
        
        # Clamp effector to reasonable bounds (assuming workspace limits)
        new_effector_x = max(-1.0, min(1.0, new_effector_x))
        new_effector_y = max(-1.0, min(1.0, new_effector_y))
        new_effector_z = max(0.0, min(1.0, new_effector_z))
        
        # Check if effector is close enough to block to move it (grasping distance ~0.05)
        distance_to_block = math.sqrt(
            (new_effector_x - block["x"])**2 + 
            (new_effector_y - block["y"])**2 + 
            (new_effector_z - block["z"])**2
        )
        
        # If effector is close to block, move block with it
        if distance_to_block < 0.05:
            # Move block with effector
            block_dx = new_effector_x - effector["x"]
            block_dy = new_effector_y - effector["y"]
            block_dz = new_effector_z - effector["z"]
            
            new_block_x = block["x"] + block_dx
            new_block_y = block["y"] + block_dy
            new_block_z = block["z"] + block_dz
            
            # Clamp block to reasonable bounds
            new_block_x = max(-1.0, min(1.0, new_block_x))
            new_block_y = max(-1.0, min(1.0, new_block_y))
            new_block_z = max(0.0, min(1.0, new_block_z))
            
            block["x"] = new_block_x
            block["y"] = new_block_y
            block["z"] = new_block_z
        else:
            # Block stays in place
            pass
            
        # Update effector position
        effector["x"] = new_effector_x
        effector["y"] = new_effector_y
        effector["z"] = new_effector_z
        
        # Copy other object properties
        for obj_id, obj_data in belief["objects"].items():
            if obj_id == "block":
                next_belief["objects"][obj_id] = block.copy()
            elif obj_id == "effector":
                next_belief["objects"][obj_id] = effector.copy()
            else:
                next_belief["objects"][obj_id] = obj_data.copy()
                
        return next_belief

    def readout_observation(self, belief: dict, action: str = "") -> str:
        """Generate observation text from belief state."""
        # This is a simplified readout that doesn't reproduce the full text format
        # but captures the essential information
        obs_lines = []
        obs_lines.append(f"Environment: cube")
        obs_lines.append(f"Source: oracle")
        obs_lines.append(f"Step: {belief['step']}")
        obs_lines.append(f"Goal: {belief['goal']}")
        obs_lines.append("Objects:")
        
        for obj_id, obj_data in belief["objects"].items():
            attrs = []
            obj_type = obj_data["type"]
            for k, v in obj_data.items():
                if k != "type":
                    attrs.append(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}")
            obs_lines.append(f"- {obj_id}:{obj_type} {' '.join(attrs)}")
            
        # We don't reconstruct relations or metadata in this simple implementation
        obs_lines.append("Relations:")
        obs_lines.append("Metadata: {}")
        
        return "\n".join(obs_lines)

    def extract_valid_action_forms(self) -> dict[str, list[str]]:
        """Define valid action forms for this environment."""
        return {
            "act": [r"act\[a0=.*?,a1=.*?,a2=.*?,a3=.*?,a4=.*?,speed=.*?\]"]
        }