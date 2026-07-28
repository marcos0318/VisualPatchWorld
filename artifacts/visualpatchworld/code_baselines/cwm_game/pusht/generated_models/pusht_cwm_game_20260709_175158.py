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
        """Parse a LeWM scene graph into a structured belief state."""
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
                    obj_line = lines[i]
                    # Parse object: - agent:circular_pusher x=372.936 y=252.025 vx=-21.177 vy=-21.407 conf=1.000
                    match = re.match(r'- ([^ ]+) (.+)', obj_line.strip())
                    if match:
                        obj_id_type = match.group(1)
                        props_str = match.group(2)
                        # Split properties
                        props = {}
                        for prop in props_str.split():
                            if '=' in prop:
                                k, v = prop.split('=', 1)
                                # Try to convert to number
                                try:
                                    v = float(v)
                                except ValueError:
                                    pass
                                props[k] = v
                        obj_id, obj_type = obj_id_type.split(':')
                        belief["objects"][obj_id] = {
                            "type": obj_type,
                            **props
                        }
                    i += 1
                continue  # Skip i increment at end of loop
            elif line.startswith("Relations:"):
                i += 1
                while i < len(lines) and lines[i].startswith("- "):
                    rel_line = lines[i]
                    # Parse relation: - agent distance_to block = 77.447
                    match = re.match(r'- ([^ ]+) ([^ ]+) ([^ ]+) = (.+)', rel_line.strip())
                    if match:
                        subj, rel, obj, val = match.groups()
                        try:
                            val = float(val)
                        except ValueError:
                            pass
                        belief["relations"][(subj, rel, obj)] = val
                    i += 1
                continue
            elif line.startswith("Metadata:"):
                meta_str = line.split(":", 1)[1].strip()
                # Simple parsing assuming JSON-like format
                # {"episode_index":327,"raw_index":36000}
                meta_str = meta_str.strip("{}")
                meta = {}
                for pair in meta_str.split(","):
                    if ':' in pair:
                        k, v = pair.split(":", 1)
                        k = k.strip().strip('"')
                        v = v.strip().strip('"')
                        try:
                            v = int(v)
                        except ValueError:
                            try:
                                v = float(v)
                            except ValueError:
                                pass
                        meta[k] = v
                belief["metadata"] = meta
            i += 1
        return belief

    def init_belief(self) -> dict:
        """Initialize an empty belief state."""
        return {
            "environment": "pusht",
            "source": "oracle",
            "step": 0,
            "goal": "move block to the target T pose",
            "objects": {},
            "relations": {},
            "metadata": {}
        }

    def init_belief_from_observation(self, obs_text: str) -> dict:
        """Initialize belief state from the first observation."""
        return self.parse_observation(obs_text)

    def correct_belief(self, belief_prior: dict, obs_text: str) -> dict:
        """Correct belief state using the new observation."""
        # In fully observed setting, we just replace the belief
        return self.parse_observation(obs_text)

    def predict_belief(self, belief: dict, action: str) -> dict:
        """Predict next belief state given an action."""
        # Deep copy belief
        import copy
        new_belief = copy.deepcopy(belief)
        new_belief["step"] += 1

        # Extract action parameters
        action_match = re.match(r'([a-z_]+)$$(?:dx=([+-]?[0-9]*\.?[0-9]+),dy=([+-]?[0-9]*\.?[0-9]+),speed=([+-]?[0-9]*\.?[0-9]+))?$$', action)
        if not action_match:
            # Fallback for simple actions without parameters
            action_match = re.match(r'([a-z_]+)(?:$$[^$$]*$$)?', action)
            if not action_match:
                return new_belief
            act_type = action_match.group(1)
            dx, dy, speed = 0.0, 0.0, 0.0
        else:
            act_type = action_match.group(1)
            dx = float(action_match.group(2)) if action_match.group(2) else 0.0
            dy = float(action_match.group(3)) if action_match.group(3) else 0.0
            speed = float(action_match.group(4)) if action_match.group(4) else 0.0

        # Get agent
        agent = new_belief["objects"].get("agent")
        if not agent:
            return new_belief

        # Time step
        dt = 1.0  # Unit time step

        # Update agent velocity based on action parameters
        if speed > 0 and (dx != 0 or dy != 0):
            # Use the provided dx, dy, speed for movement
            magnitude = math.sqrt(dx*dx + dy*dy)
            if magnitude > 0:
                # Normalize and scale by speed
                agent["vx"] = (dx / magnitude) * speed * 100.0  # Scale factor based on empirical data
                agent["vy"] = (dy / magnitude) * speed * 100.0
        else:
            # Direction-based movement for actions without parameters
            movement_deltas = {
                "move_up": {"dx": 0.0, "dy": 1.0},
                "move_down": {"dx": 0.0, "dy": -1.0},
                "move_left": {"dx": -1.0, "dy": 0.0},
                "move_right": {"dx": 1.0, "dy": 0.0},
                "hold": {"dx": 0.0, "dy": 0.0}
            }
            
            delta = movement_deltas.get(act_type, {"dx": 0.0, "dy": 0.0})
            # Apply movement with scaling factor based on empirical data
            force_scale = 50.0
            agent["vx"] = delta["dx"] * force_scale
            agent["vy"] = delta["dy"] * force_scale

        # Apply damping to velocity (friction)
        damping = 0.8
        agent["vx"] *= damping
        agent["vy"] *= damping

        # Update agent position
        agent["x"] = agent.get("x", 0.0) + agent["vx"] * dt
        agent["y"] = agent.get("y", 0.0) + agent["vy"] * dt

        # Update block positions if being pushed
        block = new_belief["objects"].get("block")
        if block and agent:
            # Compute distance between agent and block
            dx_agent_block = block["x"] - agent["x"]
            dy_agent_block = block["y"] - agent["y"]
            dist = math.sqrt(dx_agent_block*dx_agent_block + dy_agent_block*dy_agent_block)

            # If close enough, apply some movement to block
            push_threshold = 60.0
            if dist < push_threshold:
                # Apply a fraction of agent's movement to block based on proximity
                push_factor = max(0.0, (push_threshold - dist) / push_threshold) * 0.3
                block["x"] += agent["vx"] * dt * push_factor
                block["y"] += agent["vy"] * dt * push_factor
                # Rotate block slightly based on movement
                if "angle" in block:
                    block["angle"] += (agent["vx"] * 0.001 + agent["vy"] * 0.001)

        return new_belief

    def readout_observation(self, belief: dict, action: str = "") -> str:
        belief = _vpw_coerce_belief_numerics(belief)
        """Convert belief state back to LeWM scene graph text."""
        lines = []
        lines.append(f"Environment: {belief['environment']}")
        lines.append(f"Source: {belief['source']}")
        lines.append(f"Step: {belief['step']}")
        lines.append(f"Goal: {belief['goal']}")
        lines.append("Objects:")
        for obj_id, obj in belief["objects"].items():
            props = []
            for k in ["x", "y", "vx", "vy", "angle", "conf"]:
                if k in obj:
                    if isinstance(obj[k], float):
                        props.append(f"{k}={obj[k]:.3f}")
                    else:
                        props.append(f"{k}={obj[k]}")
            lines.append(f"- {obj_id}:{obj['type']} {' '.join(props)}")

        lines.append("Relations:")
        # Recompute relations
        agent = belief["objects"].get("agent")
        block = belief["objects"].get("block")
        if agent and block:
            dx = block["x"] - agent["x"]
            dy = block["y"] - agent["y"]
            dist = math.sqrt(dx*dx + dy*dy)
            lines.append(f"- agent distance_to block = {dist:.3f}")
            near = 1.0 if dist < 20 else 0.0
            lines.append(f"- agent near block = {near:.3f}")
            # Determine direction
            if abs(dx) > abs(dy):
                direction = "right" if dx > 0 else "left"
            else:
                direction = "down" if dy > 0 else "up"
            lines.append(f"- agent relative_direction_to block = {direction}")

        lines.append(f"Metadata: {str(belief['metadata'])}")
        return "\n".join(lines)

    def extract_valid_action_forms(self) -> dict[str, list[str]]:
        """Return valid action templates."""
        return {
            "move_up": ["move_up[dx=<X>,dy=<Y>,speed=<Z>]", "move_up[<X>]"],
            "move_down": ["move_down[dx=<X>,dy=<Y>,speed=<Z>]", "move_down[<X>]"],
            "move_left": ["move_left[dx=<X>,dy=<Y>,speed=<Z>]", "move_left[<X>]"],
            "move_right": ["move_right[dx=<X>,dy=<Y>,speed=<Z>]", "move_right[<X>]"],
            "hold": ["hold[dx=<X>,dy=<Y>,speed=<Z>]", "hold[<X>]"]
        }