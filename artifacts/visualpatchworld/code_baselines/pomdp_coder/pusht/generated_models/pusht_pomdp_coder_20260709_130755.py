from patchworld.worldmodel_base import BaseWorldModel
import re
from typing import Dict, Any, Tuple

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
        # Action space constants
        self.ACTION_TYPES = {
            "move_up": (0, 1),
            "move_down": (0, -1),
            "move_left": (-1, 0),
            "move_right": (1, 0),
            "hold": (0, 0)
        }
        
        # Noise parameters (explicit uncertainty assumptions)
        self.POSITION_NOISE = 0.1  # Small noise for numerical stability
        self.AGENT_SPEED_FACTOR = 0.8  # How much of intended movement is executed
        
    def parse_observation(self, obs_text: str) -> dict:
        """Parse scene graph text into structured observation."""
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
        
        current_section = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("Environment:"):
                obs["environment"] = line.split(":", 1)[1].strip()
            elif line.startswith("Source:"):
                obs["source"] = line.split(":", 1)[1].strip()
            elif line.startswith("Step:"):
                obs["step"] = int(line.split(":", 1)[1].strip())
            elif line.startswith("Goal:"):
                obs["goal"] = line.split(":", 1)[1].strip()
            elif line.startswith("Objects:"):
                current_section = "objects"
            elif line.startswith("Relations:"):
                current_section = "relations"
            elif line.startswith("Metadata:"):
                import json
                obs["metadata"] = json.loads(line.split(":", 1)[1].strip())
            elif line.startswith("- ") and current_section == "objects":
                # Parse object line like: "- agent:circular_pusher x=372.936 y=252.025 vx=-21.177 vy=-21.407 conf=1.000"
                parts = line[2:].split()
                obj_id_type = parts[0]
                obj_id, obj_type = obj_id_type.split(':')
                
                attrs = {}
                for part in parts[1:]:
                    if '=' in part:
                        k, v = part.split('=', 1)
                        attrs[k] = float(v)
                
                obs["objects"][obj_id] = {
                    "type": obj_type,
                    "x": attrs.get("x", 0.0),
                    "y": attrs.get("y", 0.0),
                    "vx": attrs.get("vx", 0.0),
                    "vy": attrs.get("vy", 0.0),
                    "conf": attrs.get("conf", 1.0)
                }
            elif line.startswith("- ") and current_section == "relations":
                # Parse relation line like: "- agent distance_to block = 77.447"
                relation_text = line[2:]
                if " distance_to " in relation_text:
                    parts = relation_text.split(" distance_to ")
                    agent_id = parts[0]
                    rest = parts[1].split(" = ")
                    target_id = rest[0]
                    distance = float(rest[1])
                    obs["relations"][f"{agent_id}_distance_to_{target_id}"] = distance
                elif " near " in relation_text:
                    parts = relation_text.split(" near ")
                    agent_id = parts[0]
                    rest = parts[1].split(" = ")
                    target_id = rest[0]
                    nearness = float(rest[1])
                    obs["relations"][f"{agent_id}_near_{target_id}"] = nearness
                elif " relative_direction_to " in relation_text:
                    parts = relation_text.split(" relative_direction_to ")
                    agent_id = parts[0]
                    rest = parts[1].split(" = ")
                    target_id = rest[0]
                    direction = rest[1]
                    obs["relations"][f"{agent_id}_relative_direction_to_{target_id}"] = direction
        
        return obs

    def init_belief(self) -> dict:
        """Initialize empty belief state."""
        return {
            "step": 0,
            "objects": {}
        }

    def init_belief_from_observation(self, obs_text: str) -> dict:
        """Initialize belief state from first observation."""
        obs = self.parse_observation(obs_text)
        belief = {
            "step": obs["step"],
            "objects": {}
        }
        
        for obj_id, obj_data in obs["objects"].items():
            belief["objects"][obj_id] = {
                "type": obj_data["type"],
                "x": obj_data["x"],
                "y": obj_data["y"],
                "vx": obj_data["vx"],
                "vy": obj_data["vy"],
                "conf": obj_data["conf"]
            }
            
        return belief

    def correct_belief(self, belief_prior: dict, obs_text: str) -> dict:
        """Correct belief state with new observation (identity for deterministic)."""
        return self.init_belief_from_observation(obs_text)

    def predict_belief(self, belief: dict, action: str) -> dict:
        """Predict next belief state given action."""
        next_belief = {
            "step": belief["step"] + 1,
            "objects": {}
        }
        
        # Parse action
        action_type = None
        for act_type in self.ACTION_TYPES:
            if action.startswith(act_type):
                action_type = act_type
                break
        
        if action_type is None:
            # Unknown action, return unchanged belief
            next_belief["objects"] = belief["objects"].copy()
            return next_belief
            
        dx, dy = self.ACTION_TYPES[action_type]
        
        # Apply action to agent
        for obj_id, obj_data in belief["objects"].items():
            next_belief["objects"][obj_id] = obj_data.copy()
            
            if obj_data["type"] == "circular_pusher":
                # Update agent position based on action
                new_x = obj_data["x"] + dx * self.AGENT_SPEED_FACTOR
                new_y = obj_data["y"] + dy * self.AGENT_SPEED_FACTOR
                
                # Update velocity
                next_belief["objects"][obj_id]["vx"] = dx * self.AGENT_SPEED_FACTOR * 10  # Scale factor
                next_belief["objects"][obj_id]["vy"] = dy * self.AGENT_SPEED_FACTOR * 10
                
                # Apply boundary constraints (assuming world is 512x512)
                new_x = max(0, min(512, new_x))
                new_y = max(0, min(512, new_y))
                
                next_belief["objects"][obj_id]["x"] = new_x
                next_belief["objects"][obj_id]["y"] = new_y
                
            # For block:t_block, keep current state (passive object)
            # In a full POMDP, we would model how the block moves when pushed
        
        return next_belief

    def readout_observation(self, belief: dict, action: str = "") -> str:
        belief = _vpw_coerce_belief_numerics(belief)
        """Generate observation text from belief state."""
        obs_lines = [
            "Environment: pusht",
            "Source: oracle",
            f"Step: {belief['step']}",
            "Goal: move block to the target T pose",
            "Objects:"
        ]
        
        # Sort objects by ID for consistent ordering
        for obj_id in sorted(belief["objects"].keys()):
            obj = belief["objects"][obj_id]
            obs_lines.append(
                f"- {obj_id}:{obj['type']} x={obj['x']:.3f} y={obj['y']:.3f} "
                f"vx={obj['vx']:.3f} vy={obj['vy']:.3f} conf={obj['conf']:.3f}"
            )
        
        obs_lines.append("Relations:")
        
        # Add simple relations if both agent and block exist
        agent_obj = None
        block_obj = None
        agent_id = None
        block_id = None
        
        for obj_id, obj in belief["objects"].items():
            if obj["type"] == "circular_pusher":
                agent_obj = obj
                agent_id = obj_id
            elif obj["type"] == "t_block":
                block_obj = obj
                block_id = obj_id
        
        if agent_obj and block_obj and agent_id and block_id:
            # Calculate distance
            import math
            dx = agent_obj["x"] - block_obj["x"]
            dy = agent_obj["y"] - block_obj["y"]
            distance = math.sqrt(dx*dx + dy*dy)
            
            # Determine relative direction
            if abs(dx) > abs(dy):
                direction = "right" if dx > 0 else "left"
            else:
                direction = "down" if dy > 0 else "up"
                
            near = 1.0 if distance < 20 else 0.0  # Threshold for "near"
            
            obs_lines.append(f"- {agent_id} distance_to {block_id} = {distance:.3f}")
            obs_lines.append(f"- {agent_id} near {block_id} = {near:.3f}")
            obs_lines.append(f"- {agent_id} relative_direction_to {block_id} = {direction}")
        
        obs_lines.append('Metadata: {"episode_index":0,"raw_index":0}')
        
        return "\n".join(obs_lines)

    def extract_valid_action_forms(self) -> dict[str, list[str]]:
        """Return valid action templates."""
        return {
            "move_up": ["move_up[dx=0.0000,dy=1.0000,speed=1.0000]"],
            "move_down": ["move_down[dx=0.0000,dy=-1.0000,speed=1.0000]"],
            "move_left": ["move_left[dx=-1.0000,dy=0.0000,speed=1.0000]"],
            "move_right": ["move_right[dx=1.0000,dy=0.0000,speed=1.0000]"],
            "hold": ["hold[dx=0.0000,dy=0.0000,speed=0.0000]"]
        }