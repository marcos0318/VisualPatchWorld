import re
import math
from typing import Dict, Any, Tuple

from patchworld.worldmodel_base import BaseWorldModel


class CubeWorldModel(BaseWorldModel):
    def init_belief(self) -> dict:
        return {
            "environment": "cube",
            "source": "oracle",
            "step": 0,
            "goal": "",
            "objects": {},
            "relations": {},
            "metadata": {},
            "epistemic_status": {}
        }

    def init_belief_from_observation(self, obs_text: str) -> dict:
        belief = self.init_belief()
        return self.correct_belief(belief, obs_text, "")

    def parse_observation(self, obs_text: str) -> dict:
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

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("Environment:"):
                obs_dict["environment"] = line.split(":", 1)[1].strip()
            elif line.startswith("Source:"):
                obs_dict["source"] = line.split(":", 1)[1].strip()
            elif line.startswith("Step:"):
                obs_dict["step"] = int(line.split(":", 1)[1].strip())
            elif line.startswith("Goal:"):
                obs_dict["goal"] = line.split(":", 1)[1].strip()
            elif line.startswith("Objects:"):
                i += 1
                while i < len(lines) and lines[i].strip().startswith('-'):
                    obj_line = lines[i].strip()
                    # Parse object line like "- block:cube x=0.425 y=0.246 z=0.020 conf=1.000"
                    match = re.match(r'- ([^:]+):([^ ]+) (.+)', obj_line)
                    if match:
                        obj_id, obj_type, attrs_str = match.groups()
                        attrs = {}
                        for attr in attrs_str.split():
                            if '=' in attr:
                                k, v = attr.split('=', 1)
                                if k in ['x', 'y', 'z']:
                                    attrs[k] = float(v)
                                elif k == 'conf':
                                    attrs[k] = float(v)
                        obs_dict["objects"][obj_id] = {
                            "type": obj_type,
                            **attrs
                        }
                    i += 1
                continue
            elif line.startswith("Relations:"):
                i += 1
                while i < len(lines) and lines[i].strip().startswith('-'):
                    rel_line = lines[i].strip()
                    # Parse relation line like "- block distance_to goal = 0.534"
                    match = re.match(r'- ([^ ]+) ([^ ]+) ([^ ]+) = ([0-9.]+)', rel_line)
                    if match:
                        obj1, rel_type, obj2, value = match.groups()
                        key = f"{obj1} {rel_type} {obj2}"
                        obs_dict["relations"][key] = float(value)
                    i += 1
                continue
            elif line.startswith("Metadata:"):
                metadata_str = line.split(":", 1)[1].strip()
                # Simple parsing assuming JSON-like format
                try:
                    # This is a simplified parser for the metadata JSON
                    metadata_str = metadata_str.replace('{', '').replace('}', '').replace('"', '')
                    metadata_parts = metadata_str.split(',')
                    metadata = {}
                    for part in metadata_parts:
                        if '=' in part:
                            k, v = part.split('=', 1)
                            metadata[k.strip()] = int(v.strip())
                    obs_dict["metadata"] = metadata
                except:
                    pass
                i += 1
                continue
            i += 1
        return obs_dict

    def correct_belief(self, belief_prior: dict, obs_text: str, action: str = "") -> dict:
        obs = self.parse_observation(obs_text)
        belief = belief_prior.copy()
        
        # Update visible fields from observation
        belief["environment"] = obs["environment"]
        belief["source"] = obs["source"]
        belief["step"] = obs["step"]
        belief["goal"] = obs["goal"]
        
        # Update objects with observed positions
        for obj_id, obj_data in obs["objects"].items():
            if obj_id not in belief["objects"]:
                belief["objects"][obj_id] = {}
            belief["objects"][obj_id].update(obj_data)
            belief["epistemic_status"][obj_id] = "known"
            
        # Update relations from observation (but we'll recompute them)
        belief["relations_observed"] = obs["relations"].copy()
        
        # Update metadata
        belief["metadata"] = obs["metadata"]
        
        return belief

    def _parse_action(self, action_str: str) -> Dict[str, float]:
        # Parse action like "act[a0=...,a1=...,a2=...,a3=...,a4=...,speed=...]"
        match = re.match(r'act$$a0=([0-9.-]+),a1=([0-9.-]+),a2=([0-9.-]+),a3=([0-9.-]+),a4=([0-9.-]+),speed=([0-9.]+)$$', action_str)
        if not match:
            return {}
        return {
            "a0": float(match.group(1)),
            "a1": float(match.group(2)),
            "a2": float(match.group(3)),
            "a3": float(match.group(4)),
            "a4": float(match.group(5)),
            "speed": float(match.group(6))
        }

    def _apply_action_to_effector(self, effector: Dict[str, Any], action: Dict[str, float]) -> Dict[str, Any]:
        # Apply 5D control vector to effector position
        # a0, a1, a2, a3, a4 are the control dimensions
        # speed is a scaling factor
        new_effector = effector.copy()
        
        # Apply movement based on action vector
        # This is a simplified model based on the examples
        dx = action["a0"] * action["speed"] * 0.1
        dy = action["a1"] * action["speed"] * 0.1
        dz = action["a2"] * action["speed"] * 0.1
        
        new_effector["x"] = effector["x"] + dx
        new_effector["y"] = effector["y"] + dy
        new_effector["z"] = effector["z"] + dz
        
        return new_effector

    def _compute_distance(self, obj1: Dict[str, Any], obj2: Dict[str, Any]) -> float:
        # Compute Euclidean distance between two objects
        dx = obj1["x"] - obj2["x"]
        dy = obj1["y"] - obj2["y"]
        dz = obj1["z"] - obj2["z"]
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def _is_near(self, distance: float, threshold: float = 0.05) -> float:
        # Return 1.0 if near, 0.0 otherwise
        return 1.0 if distance < threshold else 0.0

    def predict_belief(self, belief: dict, action: str) -> dict:
        # Predict step only (no observation correction): b' = f(b, a)
        belief_next = belief.copy()
        belief_next["step"] = belief["step"] + 1
        
        # Parse action
        action_parsed = self._parse_action(action)
        if not action_parsed:
            return belief_next
            
        # Get current objects
        objects = belief["objects"].copy()
        
        # Apply action to effector if it exists
        if "effector" in objects:
            effector = objects["effector"]
            new_effector = self._apply_action_to_effector(effector, action_parsed)
            objects["effector"] = new_effector
            
            # If effector is close to block, move block with it
            if "block" in objects:
                block = objects["block"]
                effector_block_dist = self._compute_distance(new_effector, block)
                # If very close, assume block is grasped and moves with effector
                if effector_block_dist < 0.05:  # Threshold for grasping
                    # Move block with effector
                    block_movement_scale = 0.8  # Not full movement to account for grip
                    dx = (new_effector["x"] - effector["x"]) * block_movement_scale
                    dy = (new_effector["y"] - effector["y"]) * block_movement_scale
                    dz = (new_effector["z"] - effector["z"]) * block_movement_scale
                    
                    objects["block"] = {
                        "type": block["type"],
                        "x": block["x"] + dx,
                        "y": block["y"] + dy,
                        "z": block["z"] + dz,
                        "conf": block.get("conf", 1.0)
                    }
        
        belief_next["objects"] = objects
        
        # Recompute relations
        relations = {}
        if "block" in objects and "goal" in objects:
            block = objects["block"]
            goal = objects["goal"]
            dist = self._compute_distance(block, goal)
            relations["block distance_to goal"] = dist
            relations["block near goal"] = self._is_near(dist)
            
        if "effector" in objects and "block" in objects:
            effector = objects["effector"]
            block = objects["block"]
            dist = self._compute_distance(effector, block)
            relations["effector distance_to block"] = dist
            
        belief_next["relations"] = relations
        
        return belief_next

    def readout_observation(self, belief: dict, action: str = "") -> str:
        # Readout authority.
        # Reuse the same template/sections that parse_observation expects.
        lines = []
        lines.append(f"Environment: {belief.get('environment', 'cube')}")
        lines.append(f"Source: {belief.get('source', 'oracle')}")
        lines.append(f"Step: {belief.get('step', 0)}")
        lines.append(f"Goal: {belief.get('goal', '')}")
        
        lines.append("Objects:")
        for obj_id, obj_data in belief.get("objects", {}).items():
            obj_type = obj_data.get("type", "unknown")
            x = obj_data.get("x", 0.0)
            y = obj_data.get("y", 0.0)
            z = obj_data.get("z", 0.0)
            conf = obj_data.get("conf", 1.0)
            lines.append(f"- {obj_id}:{obj_type} x={x:.3f} y={y:.3f} z={z:.3f} conf={conf:.3f}")
            
        lines.append("Relations:")
        for rel_key, rel_value in belief.get("relations", {}).items():
            lines.append(f"- {rel_key} = {rel_value:.3f}")
            
        lines.append(f"Metadata: {belief.get('metadata', {})}")
        
        return "\n".join(lines)

    def extract_valid_action_forms(self) -> dict[str, list[str]]:
        return {
            "act[<X>]": ["act[a0=...,a1=...,a2=...,a3=...,a4=...,speed=...]"]
        }