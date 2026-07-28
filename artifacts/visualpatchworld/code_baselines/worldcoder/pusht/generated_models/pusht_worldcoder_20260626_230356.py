from patchworld.worldmodel_base import BaseWorldModel
import re
import math

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
    def parse_observation(self, obs_text: str) -> dict:
        """Parse LeWM text scene graph into structured belief state."""
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
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("Environment:"):
                parsed["environment"] = line.split(":", 1)[1].strip()
            elif line.startswith("Source:"):
                parsed["source"] = line.split(":", 1)[1].strip()
            elif line.startswith("Step:"):
                parsed["step"] = int(line.split(":", 1)[1].strip())
            elif line.startswith("Goal:"):
                parsed["goal"] = line.split(":", 1)[1].strip()
            elif line.startswith("Objects:"):
                i += 1
                while i < len(lines) and lines[i].strip().startswith("-"):
                    obj_line = lines[i].strip()[2:]  # Remove "- "
                    # Parse object: type:id x=... y=... vx=... vy=... conf=...
                    parts = obj_line.split()
                    obj_type_id = parts[0]
                    obj_id = obj_type_id.split(":")[1] if ":" in obj_type_id else obj_type_id
                    attrs = {}
                    for part in parts[1:]:
                        if '=' in part:
                            k, v = part.split('=', 1)
                            attrs[k] = float(v)
                    parsed["objects"][obj_id] = {
                        "type": obj_type_id.split(":")[0],
                        "x": attrs.get("x", 0.0),
                        "y": attrs.get("y", 0.0),
                        "vx": attrs.get("vx", 0.0),
                        "vy": attrs.get("vy", 0.0),
                        "conf": attrs.get("conf", 1.0),
                        "angle": attrs.get("angle", 0.0)
                    }
                    i += 1
                continue
            elif line.startswith("Relations:"):
                i += 1
                while i < len(lines) and lines[i].strip().startswith("-"):
                    rel_line = lines[i].strip()[2:]  # Remove "- "
                    parsed["relations"].append(rel_line)
                    i += 1
                continue
            elif line.startswith("Metadata:"):
                meta_str = line.split(":", 1)[1].strip()
                try:
                    import json
                    parsed["metadata"] = json.loads(meta_str)
                except:
                    parsed["metadata"] = {}
                i += 1
                continue
            i += 1
        return parsed

    def init_belief(self):
        """Initialize empty belief state."""
        return {
            "objects": {},
            "step": 0,
            "relations": []
        }

    def init_belief_from_observation(self, obs_text: str):
        """Initialize belief state from initial observation."""
        parsed = self.parse_observation(obs_text)
        belief = {
            "objects": parsed["objects"],
            "step": parsed["step"],
            "relations": parsed["relations"]
        }
        return belief

    def correct_belief(self, belief_prior, obs_text: str):
        """Correct belief state with new observation."""
        parsed = self.parse_observation(obs_text)
        # Update object states
        for obj_id, obj_data in parsed["objects"].items():
            if obj_id in belief_prior["objects"]:
                # Direct update since we're correcting with ground truth
                belief_prior["objects"][obj_id] = obj_data.copy()
            else:
                belief_prior["objects"][obj_id] = obj_data.copy()
        
        # Update step and relations
        belief_prior["step"] = parsed["step"]
        belief_prior["relations"] = parsed["relations"]
        return belief_prior

    def predict_belief(self, belief, action: str):
        """Predict next belief state given action."""
        # Action format: move_direction[params] or hold[params]
        action_type = action.split("[")[0]
        next_belief = {
            "objects": {},
            "step": belief["step"] + 1,
            "relations": belief["relations"]  # Will be updated after object changes
        }
        
        # Copy all objects first
        for obj_id, obj_data in belief["objects"].items():
            next_belief["objects"][obj_id] = obj_data.copy()
        
        # Apply action to agent
        agent_id = None
        for obj_id, obj_data in belief["objects"].items():
            if obj_data["type"] == "agent":
                agent_id = obj_id
                break
        
        if not agent_id:
            return next_belief  # No agent to act on
        
        agent = next_belief["objects"][agent_id]
        
        # Extract parameters from action string
        params = {}
        param_match = re.search(r"\[([^\]]+)\]", action)
        if param_match:
            param_str = param_match.group(1)
            for pair in param_str.split(","):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    params[k] = float(v)
        
        # Store original position for block interaction calculation
        original_agent_x = agent["x"]
        original_agent_y = agent["y"]
        
        # Get block info for interaction
        block_id = None
        block = None
        for obj_id, obj_data in belief["objects"].items():
            if obj_data["type"] == "block":
                block_id = obj_id
                block = obj_data
                break
        
        # Apply movement based on action type and parameters
        if action_type in ["move_up", "move_down", "move_left", "move_right", "hold"]:
            # Use the dx, dy parameters if available, otherwise use speed-based movement
            if "dx" in params and "dy" in params:
                # Use the provided dx, dy for movement
                agent["x"] += params["dx"] * 200  # Scale factor to match observed behavior
                agent["y"] += params["dy"] * 200
                # Calculate velocity based on position change
                agent["vx"] = params["dx"] * 200 * 20  # Approximate velocity calculation
                agent["vy"] = params["dy"] * 200 * 20
            else:
                # Fallback to speed-based movement (simplified)
                speed = params.get("speed", 1.0) * 50  # Scale factor
                if action_type == "move_up":
                    agent["y"] -= speed * 0.5
                    agent["vy"] = -speed
                    agent["vx"] = 0
                elif action_type == "move_down":
                    agent["y"] += speed * 0.5
                    agent["vy"] = speed
                    agent["vx"] = 0
                elif action_type == "move_left":
                    agent["x"] -= speed * 0.5
                    agent["vx"] = -speed
                    agent["vy"] = 0
                elif action_type == "move_right":
                    agent["x"] += speed * 0.5
                    agent["vx"] = speed
                    agent["vy"] = 0
                elif action_type == "hold":
                    agent["vx"] *= 0.8  # Gradually reduce velocity
                    agent["vy"] *= 0.8
                    # Apply small position change based on current velocity
                    agent["x"] += agent["vx"] * 0.05
                    agent["y"] += agent["vy"] * 0.05
        
        # Handle block interaction - if agent moves close to block, block may move
        if block_id and block:
            # Calculate distance between agent and block
            dx = block["x"] - agent["x"]
            dy = block["y"] - agent["y"]
            distance = math.sqrt(dx*dx + dy*dy)
            
            # If agent was close to block and moved significantly, block might move
            orig_dx = block["x"] - original_agent_x
            orig_dy = block["y"] - original_agent_y
            orig_distance = math.sqrt(orig_dx*orig_dx + orig_dy*orig_dy)
            
            # Check if agent moved toward block and was already close
            if orig_distance < 100 and distance > orig_distance + 5:
                # Agent moved away from block - no interaction
                pass
            elif orig_distance < 100:  # Agent is close to block
                # Copy block to next belief if it exists
                if block_id in next_belief["objects"]:
                    next_belief["objects"][block_id] = block.copy()
        
        return next_belief

    def readout_observation(self, belief, action: str = "") -> str:
        """Generate text observation from belief state."""
        belief = _vpw_coerce_belief_numerics(belief)

        obs_lines = [
            "Environment: pusht",
            "Source: model",
            f"Step: {belief['step']}",
            "Goal: move block to the target T pose",
            "Objects:"
        ]
        
        # Sort objects by ID for consistent output
        for obj_id in sorted(belief["objects"].keys()):
            obj = belief["objects"][obj_id]
            if "angle" in obj:
                line = f"- {obj['type']}:{obj_id} x={obj['x']:.3f} y={obj['y']:.3f} vx={obj['vx']:.3f} vy={obj['vy']:.3f} conf={obj['conf']:.3f} angle={obj['angle']:.3f}"
            else:
                line = f"- {obj['type']}:{obj_id} x={obj['x']:.3f} y={obj['y']:.3f} vx={obj['vx']:.3f} vy={obj['vy']:.3f} conf={obj['conf']:.3f}"
            obs_lines.append(line)
        
        obs_lines.append("Relations:")
        # Recompute relations
        agent_obj = None
        block_obj = None
        agent_id = None
        block_id = None
        for obj_id, obj in belief["objects"].items():
            if obj["type"] == "agent":
                agent_obj = obj
                agent_id = obj_id
            elif obj["type"] == "block":
                block_obj = obj
                block_id = obj_id
        
        if agent_obj and block_obj:
            dx = block_obj["x"] - agent_obj["x"]
            dy = block_obj["y"] - agent_obj["y"]
            distance = math.sqrt(dx*dx + dy*dy)
            
            # Determine relative direction
            if abs(dx) > abs(dy):
                direction = "right" if dx > 0 else "left"
            else:
                direction = "down" if dy > 0 else "up"
            
            near = 1.0 if distance < 35.0 else 0.0  # Threshold based on observations
            
            obs_lines.append(f"- agent distance_to block = {distance:.3f}")
            obs_lines.append(f"- agent near block = {near:.3f}")
            obs_lines.append(f"- agent relative_direction_to block = {direction}")
        
        obs_lines.append('Metadata: {"episode_index":0,"raw_index":0}')
        return "\n".join(obs_lines)

    def extract_valid_action_forms(self) -> dict[str, list[str]]:
        """Return valid action templates."""
        return {
            "move_up": ["move_up[dx=<X>,dy=<Y>,speed=<Z>]"],
            "move_down": ["move_down[dx=<X>,dy=<Y>,speed=<Z>]"],
            "move_left": ["move_left[dx=<X>,dy=<Y>,speed=<Z>]"],
            "move_right": ["move_right[dx=<X>,dy=<Y>,speed=<Z>]"],
            "hold": ["hold[dx=<X>,dy=<Y>,speed=<Z>]"]
        }