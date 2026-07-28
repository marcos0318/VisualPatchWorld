from patchworld.worldmodel_base import BaseWorldModel
import re
import math

class CubeWorldModel(BaseWorldModel):
    def __init__(self):
        super().__init__()
        # Effector movement speed per action step (empirically estimated)
        self.EFFECTOR_SPEED = 0.08
        # Threshold for considering the block "near" the goal
        self.NEAR_THRESHOLD = 0.05
        # Noise level in position estimation (for probabilistic modeling)
        self.POSITION_NOISE = 0.01

    def parse_observation(self, obs_text: str) -> dict:
        """
        Parse the scene graph text into structured belief state.
        """
        lines = obs_text.strip().split('\n')
        obs_dict = {
            "environment": "",
            "source": "",
            "step": 0,
            "goal": "",
            "objects": {},
            "relations": {},
            "metadata": {}
        }

        obj_lines = []
        relation_lines = []
        metadata_line = ""

        for line in lines:
            if line.startswith("Environment:"):
                obs_dict["environment"] = line.split(":", 1)[1].strip()
            elif line.startswith("Source:"):
                obs_dict["source"] = line.split(":", 1)[1].strip()
            elif line.startswith("Step:"):
                obs_dict["step"] = int(line.split(":", 1)[1].strip())
            elif line.startswith("Goal:"):
                obs_dict["goal"] = line.split(":", 1)[1].strip()
            elif line.startswith("Objects:"):
                continue  # Skip this line
            elif line.startswith("- ") and (" x=" in line or " conf=" in line):
                obj_lines.append(line[2:])  # Remove "- "
            elif line.startswith("Relations:"):
                continue  # Skip this line
            elif line.startswith("- ") and (" distance_to " in line or " near " in line):
                relation_lines.append(line[2:])  # Remove "- "
            elif line.startswith("Metadata:"):
                metadata_line = line.split(":", 1)[1].strip()

        # Parse objects
        for obj_line in obj_lines:
            # Example: block:cube x=0.425 y=0.246 z=0.020 conf=1.000
            parts = obj_line.split()
            obj_id_type = parts[0]  # e.g., block:cube
            obj_id = obj_id_type.split(":")[0]
            obj_type = obj_id_type.split(":")[1] if ":" in obj_id_type else obj_id_type
            
            # Extract x, y, z, conf
            coords = {}
            for part in parts[1:]:
                if '=' in part:
                    key, val = part.split('=', 1)
                    if key in ['x', 'y', 'z']:
                        coords[key] = float(val)
                    elif key == 'conf':
                        coords['conf'] = float(val)
            
            obs_dict["objects"][obj_id] = {
                "type": obj_type,
                "x": coords.get("x", 0.0),
                "y": coords.get("y", 0.0),
                "z": coords.get("z", 0.0),
                "conf": coords.get("conf", 1.0)
            }

        # Parse relations
        for rel_line in relation_lines:
            # Example: block distance_to goal = 0.534
            if " distance_to " in rel_line:
                match = re.match(r"(\w+) distance_to (\w+) = ([0-9.]+)", rel_line)
                if match:
                    obj1, obj2, dist = match.groups()
                    obs_dict["relations"][f"{obj1}_distance_to_{obj2}"] = float(dist)
            elif " near " in rel_line:
                match = re.match(r"(\w+) near (\w+) = ([0-9.]+)", rel_line)
                if match:
                    obj1, obj2, near_val = match.groups()
                    obs_dict["relations"][f"{obj1}_near_{obj2}"] = float(near_val)

        # Parse metadata
        if metadata_line:
            try:
                obs_dict["metadata"] = eval(metadata_line)
            except:
                obs_dict["metadata"] = {}

        return obs_dict

    def init_belief(self) -> dict:
        """
        Initialize a default belief state.
        """
        return {
            "step": 0,
            "objects": {
                "block": {"type": "cube", "x": 0.0, "y": 0.0, "z": 0.0, "conf": 1.0},
                "effector": {"type": "gripper", "x": 0.0, "y": 0.0, "z": 0.0, "conf": 1.0},
                "goal": {"type": "goal_pose", "x": 0.0, "y": 0.0, "z": 0.0, "conf": 1.0}
            },
            "relations": {}
        }

    def init_belief_from_observation(self, obs_text: str) -> dict:
        """
        Initialize belief state from the first observation.
        """
        parsed = self.parse_observation(obs_text)
        belief = {
            "step": parsed["step"],
            "objects": parsed["objects"],
            "relations": parsed["relations"]
        }
        return belief

    def correct_belief(self, belief_prior: dict, obs_text: str) -> dict:
        """
        Update belief with new observation.
        In this deterministic model, we simply replace the belief with the observed state.
        """
        return self.init_belief_from_observation(obs_text)

    def predict_belief(self, belief: dict, action: str) -> dict:
        """
        Predict next belief given an action.
        Action is of the form 'act[a0=...,a1=...,a2=...,a3=...,a4=...,speed=...]'
        """
        # Parse action
        match = re.match(r"act$$a0=([+-]?[0-9.]+),a1=([+-]?[0-9.]+),a2=([+-]?[0-9.]+),a3=([+-]?[0-9.]+),a4=([+-]?[0-9.]+),speed=([+-]?[0-9.]+)$$", action)
        if not match:
            # If action format is not recognized, return unchanged belief
            return belief.copy()
        
        a0, a1, a2, a3, a4, speed = map(float, match.groups())
        
        # Copy belief to avoid mutation
        new_belief = {
            "step": belief["step"] + 1,
            "objects": {k: v.copy() for k, v in belief["objects"].items()},
            "relations": belief["relations"].copy()
        }
        
        # Get current positions
        effector = new_belief["objects"].get("effector", {"x": 0.0, "y": 0.0, "z": 0.0})
        block = new_belief["objects"].get("block", {"x": 0.0, "y": 0.0, "z": 0.0})
        
        # Effector moves based on action and speed
        # a0, a1, a2 likely control x, y, z movement
        dx = a0 * self.EFFECTOR_SPEED * (speed / 1.0)  # Normalize by default speed
        dy = a1 * self.EFFECTOR_SPEED * (speed / 1.0)
        dz = a2 * self.EFFECTOR_SPEED * (speed / 1.0)
        
        # Update effector position
        effector["x"] += dx
        effector["y"] += dy
        effector["z"] += dz
        
        # If effector is close enough to block, move block with it
        dist_effector_block = math.sqrt(
            (effector["x"] - block["x"])**2 +
            (effector["y"] - block["y"])**2 +
            (effector["z"] - block["z"])**2
        )
        
        # Threshold for grasping (from examples, when distance is around 0.01)
        GRASP_THRESHOLD = 0.02
        
        if dist_effector_block <= GRASP_THRESHOLD:
            # Move block with effector
            block["x"] = effector["x"]
            block["y"] = effector["y"]
            block["z"] = effector["z"]
        
        # Update relations
        goal = new_belief["objects"].get("goal", {"x": 0.0, "y": 0.0, "z": 0.0})
        dist_block_goal = math.sqrt(
            (block["x"] - goal["x"])**2 +
            (block["y"] - goal["y"])**2 +
            (block["z"] - goal["z"])**2
        )
        
        new_belief["relations"] = {
            "block_distance_to_goal": dist_block_goal,
            "block_near_goal": 1.0 if dist_block_goal < self.NEAR_THRESHOLD else 0.0,
            "effector_distance_to_block": dist_effector_block
        }
        
        return new_belief

    def readout_observation(self, belief: dict, action: str = "") -> str:
        """
        Generate a textual scene graph from the belief state.
        """
        obs_lines = [
            "Environment: cube",
            "Source: oracle",
            f"Step: {belief.get('step', 0)}",
            "Goal: move block to goal pose"
        ]
        
        obs_lines.append("Objects:")
        for obj_id, obj_data in belief["objects"].items():
            line = f"- {obj_id}:{obj_data['type']} x={obj_data['x']:.3f} y={obj_data['y']:.3f} z={obj_data['z']:.3f} conf={obj_data['conf']:.3f}"
            obs_lines.append(line)
            
        obs_lines.append("Relations:")
        for rel_key, rel_val in belief["relations"].items():
            obs_lines.append(f"- {rel_key} = {rel_val:.3f}")
            
        obs_lines.append('Metadata: {"episode_index":0,"next_raw_index":0,"raw_index":0}')
        
        return "\n".join(obs_lines)

    def extract_valid_action_forms(self) -> dict[str, list[str]]:
        """
        Define valid action templates for this domain.
        """
        return {
            "act": [
                "act[a0=<float>,a1=<float>,a2=<float>,a3=<float>,a4=<float>,speed=<float>]"
            ]
        }