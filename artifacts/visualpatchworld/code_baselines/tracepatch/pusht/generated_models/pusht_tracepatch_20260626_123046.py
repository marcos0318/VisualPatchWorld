import math
import re
from typing import Dict, Any, Tuple

from patchworld.worldmodel_base import BaseWorldModel


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
    def init_belief(self) -> dict:
        return {
            "environment": "",
            "source": "",
            "step": 0,
            "goal": "",
            "objects": {},
            "relations": {},
            "metadata": {},
        }

    def init_belief_from_observation(self, obs_text: str) -> dict:
        belief = self.init_belief()
        return self.correct_belief(belief, obs_text)

    def parse_observation(self, obs_text: str) -> dict:
        lines = obs_text.strip().split('\n')
        belief = self.init_belief()
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("Environment:"):
                belief["environment"] = line.split(":", 1)[1].strip()
                current_section = "environment"
            elif line.startswith("Source:"):
                belief["source"] = line.split(":", 1)[1].strip()
                current_section = "source"
            elif line.startswith("Step:"):
                belief["step"] = int(line.split(":", 1)[1].strip())
                current_section = "step"
            elif line.startswith("Goal:"):
                belief["goal"] = line.split(":", 1)[1].strip()
                current_section = "goal"
            elif line == "Objects:":
                current_section = "objects"
                belief["objects"] = {}
            elif line.startswith("- ") and current_section == "objects":
                # Parse object line
                obj_line = line[2:]  # Remove "- "
                # Example: agent:circular_pusher x=372.936 y=252.025 vx=-21.177 vy=-21.407 conf=1.000
                obj_type_id, rest = obj_line.split(" ", 1)
                obj_type, obj_id = obj_type_id.split(":")
                obj_data = {"type": obj_type, "id": obj_id}
                # Parse key=value pairs
                parts = rest.split()
                for part in parts:
                    if '=' in part:
                        key, value = part.split('=', 1)
                        obj_data[key] = float(value)
                belief["objects"][obj_id] = obj_data
            elif line == "Relations:":
                current_section = "relations"
                belief["relations"] = {}
            elif line.startswith("- ") and current_section == "relations":
                # Parse relation line
                rel_line = line[2:]  # Remove "- "
                # Example: agent distance_to block = 77.447
                if " = " in rel_line:
                    rel_key, rel_value = rel_line.split(" = ", 1)
                    belief["relations"][rel_key] = float(rel_value)
            elif line.startswith("Metadata:"):
                current_section = "metadata"
                # Parse JSON-like metadata
                metadata_str = line.split(":", 1)[1].strip()
                # Simple parsing for known fields
                meta_dict = {}
                # Remove braces
                meta_content = metadata_str.strip("{}")
                for pair in meta_content.split(","):
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        meta_dict[k.strip()] = int(v.strip())
                    elif ":" in pair:
                        k, v = pair.split(":", 1)
                        meta_dict[k.strip().strip('"')] = int(v.strip())
                belief["metadata"] = meta_dict

        return belief

    def correct_belief(self, belief_prior: dict, obs_text: str) -> dict:
        # Assimilate actual observation evidence into prior belief.
        obs_dict = self.parse_observation(obs_text)
        # Update all top-level fields
        for key in ["environment", "source", "step", "goal", "metadata"]:
            if key in obs_dict:
                belief_prior[key] = obs_dict[key]
        # Update objects
        if "objects" in obs_dict:
            for obj_id, obj_data in obs_dict["objects"].items():
                if obj_id in belief_prior["objects"]:
                    # Merge, preferring observed data
                    belief_prior["objects"][obj_id].update(obj_data)
                else:
                    belief_prior["objects"][obj_id] = obj_data.copy()
        # Update relations
        if "relations" in obs_dict:
            belief_prior["relations"].update(obs_dict["relations"])
        return belief_prior

    def predict_belief(self, belief: dict, action: str) -> dict:
        # Predict step only (no observation correction): b' = f(b, a)
        # Apply action-conditioned dynamics and return prior belief.
        belief_next = self.deep_copy_belief(belief)
        belief_next["step"] += 1

        # Parse action
        action_match = re.match(r"(\w+)\[([^\]]+)\]", action)
        if not action_match:
            # Unknown action format, return unchanged
            return belief_next

        action_type = action_match.group(1)
        action_params_str = action_match.group(2)
        # Parse parameters
        params = {}
        for param in action_params_str.split(","):
            if "=" in param:
                k, v = param.split("=", 1)
                params[k.strip()] = float(v.strip())

        dx = params.get("dx", 0.0)
        dy = params.get("dy", 0.0)
        speed = params.get("speed", 0.0)

        # Update agent position based on action
        if "agent" in belief_next["objects"]:
            agent = belief_next["objects"]["agent"]
            # Update agent position
            new_x = agent.get("x", 0) + dx
            new_y = agent.get("y", 0) + dy
            # Update velocity (simplified model)
            # We assume velocity is instantaneous change, so we set it to dx/dt, dy/dt
            # But since we don't have dt, we'll just set it to dx, dy scaled by some factor or keep previous
            # For simplicity, we'll update vx, vy to reflect the movement
            agent["x"] = new_x
            agent["y"] = new_y
            # Update velocity based on movement
            agent["vx"] = dx * 10.0  # arbitrary scaling
            agent["vy"] = dy * 10.0

        # Update relations based on new positions
        self._update_relations(belief_next)

        return belief_next

    def _update_relations(self, belief: dict):
        # Update the relations based on current object positions
        if "agent" in belief["objects"] and "block" in belief["objects"]:
            agent = belief["objects"]["agent"]
            block = belief["objects"]["block"]
            ax = agent.get("x", 0)
            ay = agent.get("y", 0)
            bx = block.get("x", 0)
            by = block.get("y", 0)

            # Compute distance
            dx_rel = bx - ax
            dy_rel = by - ay
            distance = math.hypot(dx_rel, dy_rel)
            belief["relations"]["agent distance_to block"] = distance

            # Compute relative direction
            if abs(dx_rel) > abs(dy_rel):
                if dx_rel > 0:
                    direction = "right"
                else:
                    direction = "left"
            else:
                if dy_rel > 0:
                    direction = "down"
                else:
                    direction = "up"
            belief["relations"]["agent relative_direction_to block"] = direction

            # Compute nearness (thresholded)
            near_threshold = 20.0  # arbitrary threshold
            near = 1.0 if distance < near_threshold else 0.0
            belief["relations"]["agent near block"] = near

    def deep_copy_belief(self, belief: dict) -> dict:
        # Simple deep copy for belief state
        import copy
        return copy.deepcopy(belief)

    def readout_observation(self, belief: dict, action: str = "") -> str:
        
        belief = _vpw_coerce_belief_numerics(belief)
# Readout authority.
        # Reuse the same template/sections that parse_observation expects.
        lines = []
        lines.append(f"Environment: {belief.get('environment', '')}")
        lines.append(f"Source: {belief.get('source', '')}")
        lines.append(f"Step: {belief.get('step', 0)}")
        lines.append(f"Goal: {belief.get('goal', '')}")
        lines.append("Objects:")
        for obj_id in sorted(belief["objects"].keys()):
            obj = belief["objects"][obj_id]
            # Format with fixed precision
            obj_line = f"- {obj.get('type', '')}:{obj_id}"
            # Order fields consistently
            fields = []
            for key in ["x", "y", "vx", "vy", "angle", "conf"]:
                if key in obj:
                    fields.append(f"{key}={float(obj[key]):.3f}")
            obj_line += " " + " ".join(fields)
            lines.append(obj_line)
        lines.append("Relations:")
        for rel_key in sorted(belief["relations"].keys()):
            rel_value = belief["relations"][rel_key]
            lines.append(f"- {rel_key} = {float(rel_value):.3f}")
        lines.append(f"Metadata: {belief.get('metadata', {})}")
        return "\n".join(lines)

    # Backward-compatible wrappers expected by existing evaluators.
    def transition(self, belief: dict, action: str) -> dict:
        return self.predict_belief(belief, action)

    def predict_next(self, state: dict, action: str) -> dict:
        return self.predict_belief(state, action)

    def predict_observation(self, belief: dict, action: str) -> str:
        belief_next_prior = self.predict_belief(belief, action)
        return self.readout_observation(belief_next_prior, action)

    def extract_valid_action_forms(self) -> dict[str, list[str]]:
        # Return a mapping of action types to parameter templates
        return {
            "move_left": ["dx=<float>,dy=<float>,speed=<float>"],
            "move_right": ["dx=<float>,dy=<float>,speed=<float>"],
            "move_up": ["dx=<float>,dy=<float>,speed=<float>"],
            "move_down": ["dx=<float>,dy=<float>,speed=<float>"],
            "hold": ["dx=<float>,dy=<float>,speed=<float>"],
        }