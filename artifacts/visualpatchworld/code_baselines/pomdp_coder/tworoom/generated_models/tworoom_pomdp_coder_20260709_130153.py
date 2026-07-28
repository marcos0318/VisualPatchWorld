from patchworld.worldmodel_base import BaseWorldModel
import re
import json
from typing import Dict, Any, Tuple


class TworoomWorldModel(BaseWorldModel):
    """
    World model for the tworoom environment based on Curtis et al. (CoRL 2025).
    This model handles parsing scene graphs, tracking agent/target positions,
    and predicting state transitions in a deterministic manner for planning.
    """

    def __init__(self):
        super().__init__()
        # Action patterns for parsing
        self.action_patterns = {
            "move_up": re.compile(r"move_up\[dx=([+-]?\d+\.\d+),dy=([+-]?\d+\.\d+),speed=([+-]?\d+\.\d+)\]"),
            "move_down": re.compile(r"move_down\[dx=([+-]?\d+\.\d+),dy=([+-]?\d+\.\d+),speed=([+-]?\d+\.\d+)\]"),
            "move_left": re.compile(r"move_left\[dx=([+-]?\d+\.\d+),dy=([+-]?\d+\.\d+),speed=([+-]?\d+\.\d+)\]"),
            "move_right": re.compile(r"move_right\[dx=([+-]?\d+\.\d+),dy=([+-]?\d+\.\d+),speed=([+-]?\d+\.\d+)\]"),
            "hold": re.compile(r"hold\[dx=([+-]?\d+\.\d+),dy=([+-]?\d+\.\d+),speed=([+-]?\d+\.\d+)\]")
        }

    def parse_observation(self, obs_text: str) -> dict:
        """
        Parse the textual scene graph into structured belief state.
        """
        lines = obs_text.strip().split('\n')
        obs_dict = {
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
            elif line == "Objects:":
                current_section = "objects"
            elif line == "Relations:":
                current_section = "relations"
            elif line.startswith("- "):
                if current_section == "objects":
                    # Parse object line: agent:navigator x=22.395 y=204.454 conf=1.000
                    parts = line[2:].split()
                    obj_id_type = parts[0]  # e.g., agent:navigator
                    obj_id = obj_id_type.split(":")[0]  # e.g., agent
                    coords = {}
                    for part in parts[1:]:
                        if '=' in part:
                            k, v = part.split('=')
                            coords[k] = float(v)
                    obs_dict["objects"][obj_id] = coords
                elif current_section == "relations":
                    # Parse relation line: agent distance_to target = 214.153
                    # We ignore relations for now as they are derivable
                    pass
            elif line.startswith("Metadata:"):
                metadata_str = line.split(":", 1)[1].strip()
                obs_dict["metadata"] = json.loads(metadata_str)

        return obs_dict

    def init_belief(self) -> dict:
        """
        Initialize an empty belief state.
        """
        return {
            "objects": {},
            "step": 0
        }

    def init_belief_from_observation(self, obs_text: str) -> dict:
        """
        Initialize belief state from the first observation.
        """
        obs = self.parse_observation(obs_text)
        belief = self.init_belief()
        belief["objects"] = obs["objects"]
        belief["step"] = obs["step"]
        return belief

    def correct_belief(self, belief_prior: dict, obs_text: str) -> dict:
        """
        Correct belief state using the new observation.
        In this deterministic setting, we simply update to the observed state.
        """
        obs = self.parse_observation(obs_text)
        belief = belief_prior.copy()
        belief["objects"] = obs["objects"]
        belief["step"] = obs["step"]
        return belief

    def predict_belief(self, belief: dict, action: str) -> dict:
        """
        Predict the next belief state given an action.
        This is the core transition function.
        """
        next_belief = {
            "objects": {},
            "step": belief["step"] + 1
        }

        # Copy over objects that don't change
        for obj_id, obj_state in belief["objects"].items():
            next_belief["objects"][obj_id] = obj_state.copy()

        # Apply action to agent
        agent_state = next_belief["objects"].get("agent")
        if agent_state is None:
            # If no agent in belief, cannot apply action
            return next_belief

        # Parse action
        for action_type, pattern in self.action_patterns.items():
            match = pattern.match(action)
            if match:
                dx, dy, speed = map(float, match.groups())
                if action_type == "hold":
                    # hold action does not change position significantly
                    pass
                else:
                    # Use the actual dx, dy values from the action
                    agent_state["x"] += dx * 25.0  # Scale factor to match observed transitions
                    agent_state["y"] += dy * 25.0  # Scale factor to match observed transitions
                break

        return next_belief

    def readout_observation(self, belief: dict, action: str = "") -> str:
        """
        Generate a textual observation from the belief state.
        This is used for providing observations to the agent.
        """
        obs_lines = [
            "Environment: tworoom",
            "Source: oracle",
            f"Step: {belief['step']}",
            "Goal: navigate agent to target"
        ]

        obs_lines.append("Objects:")
        for obj_id, obj_state in belief["objects"].items():
            obj_type = "navigator" if obj_id == "agent" else "goal"
            line = f"- {obj_id}:{obj_type} x={obj_state['x']:.3f} y={obj_state['y']:.3f} conf=1.000"
            obs_lines.append(line)

        # Compute relations
        obs_lines.append("Relations:")
        agent_state = belief["objects"].get("agent")
        target_state = belief["objects"].get("target")
        if agent_state and target_state:
            dx = agent_state["x"] - target_state["x"]
            dy = agent_state["y"] - target_state["y"]
            distance = (dx**2 + dy**2)**0.5
            near = 1.0 if distance < 10.0 else 0.0  # Threshold for 'near'
            obs_lines.append(f"- agent distance_to target = {distance:.3f}")
            obs_lines.append(f"- agent near target = {near:.3f}")

        # Add dummy metadata
        obs_lines.append('Metadata: {"door_coords":[112.0,49.0,0.0,0.0,0.0,0.0],"episode_index":0,"next_raw_index":0,"raw_index":0}')

        return "\n".join(obs_lines)

    def extract_valid_action_forms(self) -> dict[str, list[str]]:
        """
        Return valid action templates for the environment.
        """
        return {
            "move_up": ["move_up[dx=<X>,dy=<Y>,speed=<S>]"],
            "move_down": ["move_down[dx=<X>,dy=<Y>,speed=<S>]"],
            "move_left": ["move_left[dx=<X>,dy=<Y>,speed=<S>]"],
            "move_right": ["move_right[dx=<X>,dy=<Y>,speed=<S>]"],
            "hold": ["hold[dx=<X>,dy=<Y>,speed=<S>]"]
        }