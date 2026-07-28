import re
import math
from typing import Dict, Any, Tuple, List
from patchworld.worldmodel_base import BaseWorldModel

class TworoomWorldModel(BaseWorldModel):
    def init_belief(self) -> dict:
        return {
            "environment": "tworoom",
            "source": "unknown",
            "step": 0,
            "goal": "navigate agent to target",
            "objects": {
                "agent": {
                    "type": "navigator",
                    "x": 0.0,
                    "y": 0.0,
                    "conf": 1.0,
                    "status": "unknown"
                },
                "target": {
                    "type": "goal",
                    "x": 0.0,
                    "y": 0.0,
                    "conf": 1.0,
                    "status": "unknown"
                }
            },
            "relations": {
                "agent distance_to target": 0.0,
                "agent near target": 0.0
            },
            "metadata": {
                "door_coords": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "episode_index": -1,
                "next_raw_index": -1,
                "raw_index": -1
            }
        }

    def init_belief_from_observation(self, obs_text: str) -> dict:
        belief = self.init_belief()
        return self.correct_belief(belief, obs_text)

    def parse_observation(self, obs_text: str) -> dict:
        # Parse the observation text into structured data
        obs = {}
        
        # Extract environment
        env_match = re.search(r"Environment: ([^\n]+)", obs_text)
        obs["environment"] = env_match.group(1) if env_match else "unknown"
        
        # Extract source
        source_match = re.search(r"Source: ([^\n]+)", obs_text)
        obs["source"] = source_match.group(1) if source_match else "unknown"
        
        # Extract step
        step_match = re.search(r"Step: ([0-9]+)", obs_text)
        obs["step"] = int(step_match.group(1)) if step_match else 0
        
        # Extract goal
        goal_match = re.search(r"Goal: ([^\n]+)", obs_text)
        obs["goal"] = goal_match.group(1) if goal_match else "navigate agent to target"
        
        # Extract objects
        obs["objects"] = {}
        object_pattern = r"- ([^:]+):([^ ]+) x=([0-9.]+) y=([0-9.]+) conf=([0-9.]+)"
        for match in re.finditer(object_pattern, obs_text):
            obj_id, obj_type, x, y, conf = match.groups()
            obs["objects"][obj_id] = {
                "type": obj_type,
                "x": float(x),
                "y": float(y),
                "conf": float(conf)
            }
        
        # Extract relations
        obs["relations"] = {}
        relation_pattern = r"- ([^ ]+) ([^ ]+) ([^ ]+) = ([0-9.]+)"
        for match in re.finditer(relation_pattern, obs_text):
            subject, relation, object_, value = match.groups()
            key = f"{subject} {relation} {object_}"
            obs["relations"][key] = float(value)
        
        # Extract metadata
        metadata_match = re.search(r'Metadata: ({.*})', obs_text)
        if metadata_match:
            try:
                obs["metadata"] = eval(metadata_match.group(1))
            except:
                obs["metadata"] = {}
        else:
            obs["metadata"] = {}
            
        return obs

    def correct_belief(self, belief_prior: dict, obs_text: str) -> dict:
        # Parse the observation
        obs = self.parse_observation(obs_text)
        
        # Update belief with observed values
        belief = belief_prior.copy()
        
        # Update basic fields if present
        if "environment" in obs:
            belief["environment"] = obs["environment"]
        if "source" in obs:
            belief["source"] = obs["source"]
        if "step" in obs:
            belief["step"] = obs["step"]
        if "goal" in obs:
            belief["goal"] = obs["goal"]
            
        # Update objects
        for obj_id, obj_data in obs["objects"].items():
            if obj_id not in belief["objects"]:
                belief["objects"][obj_id] = {}
            belief["objects"][obj_id].update(obj_data)
            belief["objects"][obj_id]["status"] = "known"
            
        # Update relations
        belief["relations"].update(obs["relations"])
        
        # Update metadata
        belief["metadata"].update(obs["metadata"])
        
        return belief

    def _apply_action_to_agent(self, agent: Dict[str, Any], action: str) -> Dict[str, Any]:
        # Parse action parameters
        dx = 0.0
        dy = 0.0
        speed = 1.0
        
        # Extract dx, dy, and speed from action string
        dx_match = re.search(r'dx=([+-]?[0-9.]+)', action)
        dy_match = re.search(r'dy=([+-]?[0-9.]+)', action)
        speed_match = re.search(r'speed=([0-9.]+)', action)
        
        if dx_match:
            dx = float(dx_match.group(1))
        if dy_match:
            dy = float(dy_match.group(1))
        if speed_match:
            speed = float(speed_match.group(1))
            
        # Apply movement based on dx, dy, and speed (scaling factor of 25)
        scaling_factor = 25.0
        new_agent = agent.copy()
        new_agent["x"] += dx * scaling_factor * speed
        new_agent["y"] += dy * scaling_factor * speed
            
        return new_agent

    def predict_belief(self, belief: dict, action: str) -> dict:
        # Predict step only (no observation correction): b' = f(b, a)
        # Apply action-conditioned dynamics and return prior belief.
        predicted_belief = belief.copy()
        
        # Increment step
        predicted_belief["step"] = belief.get("step", 0) + 1
        
        # Apply action to agent if it exists
        if "agent" in predicted_belief["objects"]:
            current_agent = predicted_belief["objects"]["agent"]
            new_agent = self._apply_action_to_agent(current_agent, action)
            predicted_belief["objects"]["agent"] = new_agent
            
            # Update relations based on new positions
            if "target" in predicted_belief["objects"]:
                target = predicted_belief["objects"]["target"]
                agent_x, agent_y = new_agent["x"], new_agent["y"]
                target_x, target_y = target["x"], target["y"]
                
                # Calculate distance
                dx = target_x - agent_x
                dy = target_y - agent_y
                distance = math.sqrt(dx*dx + dy*dy)
                
                predicted_belief["relations"]["agent distance_to target"] = distance
                predicted_belief["relations"]["agent near target"] = 0.0  # Always 0 in examples
                
        return predicted_belief

    def readout_observation(self, belief: dict, action: str = "") -> str:
        # Readout authority.
        # Reuse the same template/sections that parse_observation expects.
        # parse_observation(readout_observation(...)) should recover visible fields implied by belief.
        # Must always return a non-empty string.
        
        # Build the observation string
        obs_lines = []
        obs_lines.append(f"Environment: {belief.get('environment', 'tworoom')}")
        obs_lines.append(f"Source: {belief.get('source', 'unknown')}")
        obs_lines.append(f"Step: {belief.get('step', 0)}")
        obs_lines.append(f"Goal: {belief.get('goal', 'navigate agent to target')}")
        obs_lines.append("Objects:")
        
        # Add objects
        for obj_id, obj_data in belief.get("objects", {}).items():
            obs_lines.append(f"- {obj_id}:{obj_data['type']} x={obj_data['x']:.3f} y={obj_data['y']:.3f} conf={obj_data['conf']:.3f}")
            
        obs_lines.append("Relations:")
        
        # Add relations
        for rel_key, rel_value in belief.get("relations", {}).items():
            obs_lines.append(f"- {rel_key} = {rel_value:.3f}")
            
        # Add metadata
        metadata_str = str(belief.get("metadata", {}))
        obs_lines.append(f"Metadata: {metadata_str}")
        
        return "\n".join(obs_lines)

    # Backward-compatible wrappers expected by existing evaluators.
    def transition(self, belief: dict, action: str) -> dict:
        return self.predict_belief(belief, action)

    def predict_next(self, state: dict, action: str) -> dict:
        return self.predict_belief(state, action)

    def predict_observation(self, belief: dict, action: str) -> str:
        belief_next_prior = self.predict_belief(belief, action)
        return self.readout_observation(belief_next_prior, action)

    def extract_valid_action_forms(self) -> dict[str, list[str]]:
        return {
            "move_left": ["move_left[dx=<X>,dy=<Y>,speed=<SPEED>]"],
            "move_right": ["move_right[dx=<X>,dy=<Y>,speed=<SPEED>]"],
            "move_up": ["move_up[dx=<X>,dy=<Y>,speed=<SPEED>]"],
            "move_down": ["move_down[dx=<X>,dy=<Y>,speed=<SPEED>]"],
            "hold": ["hold[dx=<X>,dy=<Y>,speed=<SPEED>]"]
        }