from patchworld.worldmodel_base import BaseWorldModel
import re
import math


class ReacherWorldModel(BaseWorldModel):
    def __init__(self):
        super().__init__()
        # Action types and their expected parameter patterns
        self.action_forms = {
            "move_up": ["move_up[dx=<float>,dy=<float>,speed=<float>]"],
            "move_down": ["move_down[dx=<float>,dy=<float>,speed=<float>]"],
            "move_left": ["move_left[dx=<float>,dy=<float>,speed=<float>]"],
            "move_right": ["move_right[dx=<float>,dy=<float>,speed=<float>]"],
            "hold": ["hold[dx=<float>,dy=<float>,speed=<float>]"]
        }

    def parse_observation(self, obs_text: str) -> dict:
        """Parse the textual observation into structured belief state."""
        lines = obs_text.strip().split('\n')
        parsed = {
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
                parsed["environment"] = line.split(":", 1)[1].strip()
            elif line.startswith("Source:"):
                parsed["source"] = line.split(":", 1)[1].strip()
            elif line.startswith("Step:"):
                parsed["step"] = int(line.split(":", 1)[1].strip())
            elif line.startswith("Goal:"):
                parsed["goal"] = line.split(":", 1)[1].strip()
            elif line.startswith("Objects:"):
                i += 1
                while i < len(lines) and lines[i].startswith("- "):
                    obj_line = lines[i][2:]  # Remove "- "
                    # Parse object: type:id key=value key=value ...
                    parts = obj_line.split()
                    obj_type_id = parts[0]
                    obj_id = obj_type_id.split(":")[1] if ":" in obj_type_id else obj_type_id
                    attrs = {}
                    for part in parts[1:]:
                        if '=' in part:
                            k, v = part.split('=', 1)
                            attrs[k] = float(v)
                    parsed["objects"][obj_id] = attrs
                    i += 1
                continue  # Skip increment at end since we already incremented
            elif line.startswith("Relations:"):
                i += 1
                while i < len(lines) and lines[i].startswith("- "):
                    rel_line = lines[i][2:]  # Remove "- "
                    # Format: obj1 relation obj2 = value
                    # Example: finger distance_to target = 0.310
                    parts = rel_line.split()
                    if len(parts) >= 4 and parts[1] == "distance_to":
                        obj1 = parts[0]
                        obj2 = parts[2]
                        value = float(parts[4])
                        parsed["relations"][(obj1, obj2)] = value
                    i += 1
                continue
            elif line.startswith("Metadata:"):
                # Simple JSON-like parsing
                meta_str = line.split(":", 1)[1].strip()
                # Remove outer braces and split by comma
                meta_str = meta_str.strip("{}")
                meta_items = meta_str.split(",")
                for item in meta_items:
                    if ':' in item:
                        k, v = item.split(":", 1)
                        k = k.strip().strip('"')
                        v = v.strip().strip('"')
                        if v.isdigit():
                            v = int(v)
                        parsed["metadata"][k] = v
            i += 1
        return parsed

    def init_belief(self) -> dict:
        """Initialize an empty belief state."""
        return {
            "environment": "reacher",
            "source": "oracle",
            "step": 0,
            "goal": "match finger to target",
            "objects": {},
            "relations": {}
        }

    def init_belief_from_observation(self, obs_text: str) -> dict:
        """Initialize belief from the first observation."""
        return self.parse_observation(obs_text)

    def correct_belief(self, belief_prior: dict, obs_text: str) -> dict:
        """Update belief with new observation."""
        # In this deterministic model, we fully replace the belief with the new observation
        return self.parse_observation(obs_text)

    def predict_belief(self, belief: dict, action: str) -> dict:
        """Predict next belief given current belief and action."""
        # Copy current belief
        next_belief = {k: v for k, v in belief.items()}
        next_belief["objects"] = {k: {kk: vv for kk, vv in v.items()} for k, v in belief["objects"].items()}

        # Parse action
        match = re.match(r"(\w+)$|(\w+)\[dx=([+-]?\d+\.\d+),dy=([+-]?\d+\.\d+),speed=([+-]?\d+\.\d+)\]", action)
        if not match:
            return next_belief  # Unknown action format, no change

        action_type = match.group(1) or match.group(2)
        if action_type not in ["move_up", "move_down", "move_left", "move_right", "hold"]:
            return next_belief  # Unknown action type

        dx = float(match.group(3)) if match.group(3) else 0.0
        dy = float(match.group(4)) if match.group(4) else 0.0

        # Apply movement to finger
        if "finger" in next_belief["objects"]:
            finger = next_belief["objects"]["finger"]
            finger["x"] = finger.get("x", 0.0) + dx
            finger["y"] = finger.get("y", 0.0) + dy

        # Update arm joint state based on finger movement
        # Heuristic: arm joint state changes proportionally to finger movement
        # From examples, it seems like arm.x and arm.y change when finger moves
        if "arm" in next_belief["objects"]:
            arm = next_belief["objects"]["arm"]
            # These scaling factors are derived from inspecting the examples
            arm_dx = dx * 5.0  # Approximate scaling factor
            arm_dy = dy * 5.0  # Approximate scaling factor
            arm["x"] = arm.get("x", 0.0) + arm_dx
            arm["y"] = arm.get("y", 0.0) + arm_dy

        # Update step count
        next_belief["step"] = belief.get("step", 0) + 1

        # Recalculate relations
        if "finger" in next_belief["objects"] and "target" in next_belief["objects"]:
            finger = next_belief["objects"]["finger"]
            target = next_belief["objects"]["target"]
            fx, fy = finger.get("x", 0.0), finger.get("y", 0.0)
            tx, ty = target.get("x", 0.0), target.get("y", 0.0)
            distance = math.sqrt((fx - tx) ** 2 + (fy - ty) ** 2)
            next_belief["relations"] = {("finger", "target"): distance}

        return next_belief

    def readout_observation(self, belief: dict, action: str = "") -> str:
        """Convert belief state back to textual observation."""
        obs_lines = []
        obs_lines.append(f"Environment: {belief.get('environment', 'reacher')}")
        obs_lines.append(f"Source: {belief.get('source', 'oracle')}")
        obs_lines.append(f"Step: {belief.get('step', 0)}")
        obs_lines.append(f"Goal: {belief.get('goal', 'match finger to target')}")
        obs_lines.append("Objects:")
        for obj_id, attrs in belief["objects"].items():
            attr_strs = [f"{k}={v:.3f}" for k, v in attrs.items()]
            obs_lines.append(f"- {obj_id}:{obj_id} {' '.join(attr_strs)}")
        obs_lines.append("Relations:")
        for (obj1, obj2), value in belief["relations"].items():
            obs_lines.append(f"- {obj1} distance_to {obj2} = {value:.3f}")
        # Metadata placeholder
        obs_lines.append('Metadata: {"episode_index":0,"next_raw_index":0,"raw_index":0}')
        return "\n".join(obs_lines)

    def extract_valid_action_forms(self) -> dict[str, list[str]]:
        """Return valid action templates."""
        return self.action_forms